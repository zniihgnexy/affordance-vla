"""A contact sheet of evaluation videos: one row per video, N evenly spaced frames, the file name as the row label.

Used to look at failed episodes (robocasa365 environment has imageio and PIL):
  python scripts/rc365/video_montage.py --out sheet.png --frames 6 <video.mp4> [<video.mp4> ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def row(path: Path, frames: int, size: int) -> Image.Image:
    video = imageio.mimread(path, memtest=False)
    picks = np.linspace(0, len(video) - 1, frames).astype(int)
    strip = Image.new("RGB", (frames * size, size + 14), "white")
    for k, index in enumerate(picks):
        strip.paste(Image.fromarray(video[index]).resize((size, size)), (k * size, 14))
    ImageDraw.Draw(strip).text((2, 1), f"{path.stem}  (frames {picks[0]}..{picks[-1]} of {len(video)})", fill="black")
    return strip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=160)
    args = parser.parse_args()
    rows = [row(path, args.frames, args.size) for path in args.videos]
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    top = 0
    for strip in rows:
        sheet.paste(strip, (0, top))
        top += strip.height
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)


if __name__ == "__main__":
    main()
