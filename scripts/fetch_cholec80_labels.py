"""Selectively download Cholec80 annotation files from the 75GB zip via HTTP Range.

The full archive cannot fit on this disk, but phase/tool annotations are small
individual entries. This script reads the ZIP central directory (at the end of
the file), locates annotation entries, and fetches only those byte ranges.
Stdlib only. Verifies CRC32 per entry.
"""
import os
import struct
import urllib.request
import zlib

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"
OUT = os.path.join("data", "cholec80")
WANT = ("phase_annotations", "tool_annotations")


def fetch(lo, hi):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status not in (200, 206):
            raise RuntimeError(f"range {lo}-{hi}: HTTP {r.status}")
        return r.read()


def main():
    total = 74916858781
    tail = fetch(total - 70000, total - 1)
    eocd = tail.rfind(b"\x50\x4b\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    n, cd_size, cd_off = struct.unpack("<HII", tail[eocd + 10:eocd + 12] + tail[eocd + 12:eocd + 20])
    if cd_off == 0xFFFFFFFF:
        # ZIP64: locator sits 20B before EOCD
        loc_off = total - 70000 + eocd - 20
        loc = fetch(loc_off, loc_off + 19)
        if loc[:4] != b"\x50\x4b\x06\x07":
            raise RuntimeError("ZIP64 locator not found")
        (z64_off,) = struct.unpack("<Q", loc[8:16])
        z64 = fetch(z64_off, z64_off + 63)
        if z64[:4] != b"\x50\x4b\x06\x06":
            raise RuntimeError("ZIP64 EOCD not found")
        n, cd_size, cd_off = struct.unpack("<QQQ", z64[32:56])
    print(f"entries={n} central_dir={cd_size}B @ {cd_off}")
    cd = fetch(cd_off, cd_off + cd_size - 1)
    entries = []
    p = 0
    for _ in range(n):
        if cd[p:p + 4] != b"\x50\x4b\x01\x02":
            raise RuntimeError(f"bad central entry @ {p}")
        method, crc, comp = struct.unpack("<HII", cd[p + 10:p + 12] + cd[p + 16:p + 24])
        fn_len, ex_len, cm_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        (local_off,) = struct.unpack("<I", cd[p + 42:p + 46])
        extra = cd[p + 46 + fn_len:p + 46 + fn_len + ex_len]
        if local_off == 0xFFFFFFFF:
            # ZIP64 extra field (id 0x0001): values in order uncomp/comp/offset/disk
            # for each standard field set to its max. Sizes here fit 32-bit, so
            # the single 8-byte value is the local header offset.
            q = 0
            vals = []
            while q + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[q:q + 4])
                if eid == 0x0001:
                    vals = struct.unpack(f"<{esz // 8}Q", extra[q + 4:q + 4 + esz])
                    break
                q += 4 + esz
            if not vals:
                raise RuntimeError(f"{name}: ZIP64 offset missing")
            local_off = vals[-1]
        name = cd[p + 46:p + 46 + fn_len].decode("utf-8", "replace")
        entries.append((name, method, crc, comp, local_off))
        p += 46 + fn_len + ex_len + cm_len
    annots = [e for e in entries if WANT[0] in e[0] or WANT[1] in e[0]]
    print(f"annotation entries: {len(annots)}")
    for name in sorted(set(os.path.dirname(e[0]) for e in annots)):
        print("  dir:", name)
    got = 0
    for name, method, crc, comp, local_off in annots:
        if name.endswith("/") or comp == 0:
            continue  # directory entry, not a file
        head = fetch(local_off, local_off + 511)
        if head[:4] != b"\x50\x4b\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        fn_l, ex_l = struct.unpack("<HH", head[26:30])
        data_off = local_off + 30 + fn_l + ex_l
        raw = fetch(data_off, data_off + comp - 1)
        if method == 8:
            raw = zlib.decompress(raw, -15)
        elif method != 0:
            raise RuntimeError(f"{name}: unsupported method {method}")
        import binascii
        if binascii.crc32(raw) & 0xFFFFFFFF != crc:
            raise RuntimeError(f"{name}: CRC mismatch")
        out = os.path.join(OUT, os.path.basename(os.path.dirname(name)), os.path.basename(name))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(raw)
        got += 1
        if got % 20 == 0:
            print(f"  {got}/{len(annots)}")
    print(f"saved {got} files under {OUT}")
    sizes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"total annotation bytes: {sizes}")


if __name__ == "__main__":
    main()
