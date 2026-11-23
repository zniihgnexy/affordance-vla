"""Render a cvpr-figure spec (report/media/pipeline_overview/*.yaml) to SVG, PDF and PNG with Helvetica metrics.

The skill's engine sizes boxes for Helvetica, but LibreOffice on the server only tries the first font of the SVG and
falls back to the wider DejaVu Sans, so English labels overflow. This sets every text in Liberation Sans (same
metrics as Helvetica) before the skill's own PDF and PNG steps.

Usage (server, from the repository root; the skill lives in ~/.claude/skills/cvpr-figure on the Mac):
    python scripts/report/render_figure_spec.py <skill dir> report/media/pipeline_overview/v3_model.yaml --dpi 288
Writes <spec>.svg, <spec>.pdf and <spec>.png next to the spec; the crops the spec names must be in that folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

FONT = "Liberation Sans"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skill", type=Path, help="the cvpr-figure skill folder (holds scripts/render.py)")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--dpi", type=int, default=288)
    args = parser.parse_args()
    render_py = args.skill / "scripts" / "render.py"
    stem = args.spec.with_suffix("")
    subprocess.run([sys.executable, str(render_py), str(args.spec), "-o", str(stem), "-f", "svg"], check=True)
    svg = stem.with_suffix(".svg")
    text, replaced = re.subn(r"font-family='[^']*'", f"font-family='{FONT}'", svg.read_text())
    svg.write_text(text)
    sys.path.insert(0, str(render_py.parent))
    spec = importlib.util.spec_from_file_location("render", render_py)
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    pdf_tool = render.svg_to_pdf(svg, pdf)
    png_tool = render.rasterise(svg, pdf, png, "png", args.dpi)
    print(f"fonts replaced: {replaced}; pdf: {pdf_tool}; png: {png_tool}")


if __name__ == "__main__":
    main()
