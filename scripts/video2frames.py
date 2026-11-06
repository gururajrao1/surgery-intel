"""Convert videos to 1fps frames with FFmpeg (requires ffmpeg on PATH)."""
import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for v in sorted(os.listdir(a.video_dir)):
        if not v.endswith(".mp4"):
            continue
        vid = os.path.splitext(v)[0].replace("video", "")
        out = os.path.join(a.out_dir, vid)
        os.makedirs(out, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i",
                        os.path.join(a.video_dir, v),
                        "-r", str(a.fps), "-start_number", "0",
                        os.path.join(out, "%08d.jpg")],
                       check=True)


if __name__ == "__main__":
    main()
