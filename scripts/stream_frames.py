"""Stream the full Cholec80 set without ever holding 75GB: per video,
download mp4 -> extract 1fps frames -> delete mp4 -> next.

Usage: python scripts/stream_frames.py 13 37 12 28 70 --frames-dir data/frames_1fps
Resume-safe: videos with a complete frame folder are skipped.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fetch_video import central_dir, get


def expected_frames(root, vid):
    pp = os.path.join("data", "cholec80", "phase_annotations", f"video{vid}-phase.txt")
    if not os.path.isfile(pp):
        return None
    with open(pp) as fh:
        n = sum(1 for _ in fh) - 1  # minus header
    return n // 25


def main(vids, frames_dir):
    cd = central_dir()
    for vid in vids:
        out = os.path.join(frames_dir, vid)
        exp = expected_frames(frames_dir, vid)
        have = len([f for f in os.listdir(out) if f.endswith(".jpg")]) if os.path.isdir(out) else 0
        if exp and have >= exp - 2:
            print(f"video{vid}: {have} frames present, skipping")
            continue
        mp4 = os.path.join("data", "videos", f"video{vid}.mp4")
        try:
            get(f"videos/video{vid}.mp4", mp4, cd)
        except Exception as e:
            print(f"video{vid}: download failed ({e}); will retry on next run")
            continue
        os.makedirs(out, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-y", "-i", mp4,
                        "-r", "1", "-start_number", "0",
                        os.path.join(out, "%08d.jpg")], check=True)
        os.remove(mp4)
        print(f"video{vid}: extracted, mp4 deleted")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--frames-dir", default="data/frames_1fps")
    a = ap.parse_args()
    main(a.videos, a.frames_dir)
