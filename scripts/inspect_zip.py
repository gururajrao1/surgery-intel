"""List non-annotation entries (videos) with sizes from the Cholec80 zip central dir."""
import struct
import urllib.request

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"


def fetch(lo, hi):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    total = int(urllib.request.urlopen(
        urllib.request.Request(URL, method="HEAD"), timeout=30).headers["Content-Length"])
    tail = fetch(total - 70000, total - 1)
    eocd = tail.rfind(b"\x50\x4b\x05\x06")
    loc_off = total - 70000 + eocd - 20
    loc = fetch(loc_off, loc_off + 19)
    (z64_off,) = struct.unpack("<Q", loc[8:16])
    z64 = fetch(z64_off, z64_off + 63)
    n, cd_size, cd_off = struct.unpack("<QQQ", z64[32:56])
    cd = fetch(cd_off, cd_off + cd_size - 1)
    print(f"entries={n} total={total}")
    vids = []
    p = 0
    for _ in range(n):
        assert cd[p:p + 4] == b"\x50\x4b\x01\x02"
        comp = struct.unpack("<I", cd[p + 20:p + 24])[0]
        uncomp = struct.unpack("<I", cd[p + 24:p + 28])[0]
        fn_len, ex_len, cm_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        name = cd[p + 46:p + 46 + fn_len].decode()
        if "phase_annotations" not in name and "tool_annotations" not in name \
                and not name.endswith("/"):
            vids.append((name, comp, uncomp))
        p += 46 + fn_len + ex_len + cm_len
    for name, comp, uncomp in sorted(vids):
        print(f"{uncomp / 1e9:6.2f}GB  {name}")


if __name__ == "__main__":
    main()
