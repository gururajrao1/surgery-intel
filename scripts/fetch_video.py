"""Download one or more Cholec80 videos via HTTP Range (central-dir offsets).

Usage: python scripts/fetch_video.py 13 37 70   (downloads videos/videoXX.mp4)
Skips videos already present. Verifies CRC32. Deletes nothing (caller extracts
frames with video2frames.py, then removes the mp4 to save disk).
"""
import os
import struct
import sys
import urllib.request
import zlib

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"


def fetch(lo, hi):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status not in (200, 206):
            raise RuntimeError(f"range {lo}-{hi}: HTTP {r.status}")
        return r.read()


def central_dir():
    total = int(urllib.request.urlopen(
        urllib.request.Request(URL, method="HEAD"), timeout=30).headers["Content-Length"])
    tail = fetch(total - 70000, total - 1)
    eocd = tail.rfind(b"\x50\x4b\x05\x06")
    loc = fetch(total - 70000 + eocd - 20, total - 70000 + eocd - 1)
    (z64_off,) = struct.unpack("<Q", loc[8:16])
    z64 = fetch(z64_off, z64_off + 63)
    n, cd_size, cd_off = struct.unpack("<QQQ", z64[32:56])
    cd = fetch(cd_off, cd_off + cd_size - 1)
    out = {}
    p = 0
    for _ in range(n):
        assert cd[p:p + 4] == b"\x50\x4b\x01\x02"
        method, crc, comp = struct.unpack("<HII", cd[p + 10:p + 12] + cd[p + 16:p + 24])
        fn_len, ex_len, cm_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        (local_off,) = struct.unpack("<I", cd[p + 42:p + 46])
        extra = cd[p + 46 + fn_len:p + 46 + fn_len + ex_len]
        if local_off == 0xFFFFFFFF:
            q = 0
            while q + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[q:q + 4])
                if eid == 0x0001:
                    local_off = struct.unpack(f"<{esz // 8}Q", extra[q + 4:q + 4 + esz])[-1]
                    break
                q += 4 + esz
        name = cd[p + 46:p + 46 + fn_len].decode()
        out[name] = (method, crc, comp, local_off)
        p += 46 + fn_len + ex_len + cm_len
    return out


def get(name, dest, cd):
    method, crc, comp, local_off = cd[name]
    head = fetch(local_off, local_off + 511)
    assert head[:4] == b"\x50\x4b\x03\x04", name
    fn_l, ex_l = struct.unpack("<HH", head[26:30])
    data_off = local_off + 30 + fn_l + ex_l
    # stream in 8MB chunks (server truncates big ranges); retry each chunk
    import binascii
    import http.client
    import time
    import urllib.error
    CH = 1 * 1024 * 1024
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def get_chunk(lo, hi):
        # accumulate partial bytes across dropped connections (true resume)
        buf = bytearray()
        while lo + len(buf) <= hi:
            try:
                blk = fetch(lo + len(buf), hi)
                buf += blk
                if lo + len(buf) > hi:
                    break
                if not blk:
                    break
            except (http.client.IncompleteRead, http.client.RemoteDisconnected,
                    ConnectionError, TimeoutError, urllib.error.URLError) as e:
                part = getattr(e, "partial", b"")
                buf += part
                print(f"  resume @{lo + len(buf)} (+{len(part) // 1024}KB partial)",
                      flush=True)
                time.sleep(2)
        if len(buf) != hi - lo + 1:
            raise RuntimeError(f"chunk {lo}-{hi} incomplete ({len(buf)}B)")
        return bytes(buf)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    tmp = dest + ".deflate" if method == 8 else dest
    lock = threading.Lock()
    fh = open(tmp, "wb")
    fh.truncate(comp)
    done = [0]
    t0 = [__import__("time").time()]

    def worker(off):
        chunk = get_chunk(data_off + off, min(data_off + off + CH - 1, data_off + comp - 1))
        with lock:
            fh.seek(off)
            fh.write(chunk)
            done[0] += len(chunk)
            el = __import__("time").time() - t0[0]
            print(f"  {name}: {done[0] / 1e6:.0f}/{comp / 1e6:.0f} MB "
                  f"({done[0] / 1e6 / max(el, 1):.2f} MB/s)", flush=True)

    ranges = list(range(0, comp, CH))
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(worker, ranges))
    fh.close()
    crc_v = 0
    if method == 8:
        decomp = zlib.decompressobj(-15)
        with open(tmp, "rb") as fi, open(dest, "wb") as fo:
            while True:
                blk = fi.read(8 * 1024 * 1024)
                if not blk:
                    break
                out = decomp.decompress(blk)
                crc_v = binascii.crc32(out, crc_v)
                fo.write(out)
            tail = decomp.flush()
            if tail:
                crc_v = binascii.crc32(tail, crc_v)
                fo.write(tail)
        os.remove(tmp)
    else:
        with open(dest, "rb") as fh:
            while True:
                blk = fh.read(8 * 1024 * 1024)
                if not blk:
                    break
                crc_v = binascii.crc32(blk, crc_v)
    if (crc_v & 0xFFFFFFFF) != crc:
        raise RuntimeError(f"{name}: CRC mismatch")
    print(f"saved {dest} ({comp / 1e9:.2f} GB, CRC ok)")


def main(vids):
    cd = central_dir()
    for v in vids:
        name = f"videos/video{v}.mp4"
        dest = os.path.join("data", "videos", f"video{v}.mp4")
        if os.path.isfile(dest) and os.path.getsize(dest) == cd[name][2]:
            print(f"exists, skipping {dest}")
            continue
        get(name, dest, cd)


if __name__ == "__main__":
    main(sys.argv[1:])
