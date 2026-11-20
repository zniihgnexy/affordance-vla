"""Render the five SVG figures of the layer-level Pipeline v3 page to PNG (light theme, 2x).

Needs Google Chrome, so it runs on the Mac (the server has no browser):
    python scripts/report/render_pipeline_detail.py report/media/pipeline_overview/detail/pipeline_v3_layers.html
Writes v3_detail_<n>_<name>.png next to the page. Each figure is cut out together with the page's own stylesheet,
so the PNG looks like the page; the captions are not in the PNG (the notebook text carries them).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NAMES = ("overview", "layers", "decode", "contact", "counterfactual")  # in page order
PAD = 12  # white margin around a figure, px


def page_figures(page: str) -> tuple[str, list[str]]:
    """The page's stylesheet and its SVG figures in order."""
    style = re.search(r"<style>.*?</style>", page, re.S)
    svgs = re.findall(r"<svg .*?</svg>", page, re.S)
    if style is None or len(svgs) != len(NAMES):
        raise ValueError(f"expected one stylesheet and {len(NAMES)} figures, found {len(svgs)} figures")
    return style.group(0), svgs


def standalone(style: str, svg: str) -> tuple[str, int, int]:
    """One figure as its own light-theme page, and the window size that fits it exactly."""
    found = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if found is None:
        raise ValueError("a figure has no viewBox")
    width, height = (round(float(v)) for v in found.groups())
    html = (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8">{style}<style>'
            f"body{{margin:0;background:#fff}} .scroll{{border:0;border-radius:0;padding:{PAD}px;overflow:visible}}"
            f" .scroll svg{{min-width:0!important;width:{width}px}}</style></head>"
            f'<body><div class="scroll">{svg}</div></body></html>')
    return html, width + 2 * PAD, height + 2 * PAD


def render(page_path: Path, chrome: str, scale: int) -> list[Path]:
    style, svgs = page_figures(page_path.read_text())
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, (name, svg) in enumerate(zip(NAMES, svgs), start=1):
            html, width, height = standalone(style, svg)
            source = Path(tmp) / f"{name}.html"
            source.write_text(html)
            out = page_path.parent / f"v3_detail_{n}_{name}.png"
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
                            f"--screenshot={out}", source.as_uri()], check=True, capture_output=True)
            written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path, help="the layer-level page (HTML)")
    parser.add_argument("--chrome", default=CHROME)
    parser.add_argument("--scale", type=int, default=2, help="device pixels per CSS pixel")
    args = parser.parse_args()
    for path in render(args.page, args.chrome, args.scale):
        print("wrote", path)


if __name__ == "__main__":
    main()
