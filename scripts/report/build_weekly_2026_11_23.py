"""Build report/2026-11-23_weekly_review.ipynb, the review of the London week 23-29 Nov 2026 (cells only; execute it).

Usage from the repository root (any env with nbformat, jupyter, matplotlib is not needed; the cells only show
tracked images):
    python scripts/report/build_weekly_2026_11_23.py report/2026-11-23_weekly_review.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace report/2026-11-23_weekly_review.ipynb
Reads only tracked files: report/media/weekly_2026_11_23/ (figures of the task records) and
report/media/pipeline_overview/v4_model.png. Nothing runs on a GPU or on the server. In English, short.
Sections: 0 and 8 from weekly_2026_11_23_summary.py, 1 pipeline.py, 2 datasets.py, 3 mug_transfer.py (AFFO-9),
4 robocasa365.py (AFFO-10), 5 hot3d.py (AFFO-11), 6 methods.py, 7 commands.py.
"""

import sys
from pathlib import Path

import nbformat as nbf
from weekly_2026_11_23_commands import command_cells
from weekly_2026_11_23_datasets import dataset_cells
from weekly_2026_11_23_hot3d import hot3d_cells
from weekly_2026_11_23_methods import methods_cells
from weekly_2026_11_23_mug_transfer import mug_transfer_cells
from weekly_2026_11_23_pipeline import pipeline_cells
from weekly_2026_11_23_robocasa365 import robocasa365_cells
from weekly_2026_11_23_summary import summary_cells, timeline_cells

OUT = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Week 3 review · 23–29 Nov 2026 (London time)

**Repository:** `affordance-vla` · written on 26 Nov 2026 (the week is not over)

**Order:** 0 summary · 1 Pipeline v4 · 2 why RoboCasa365 · 3 step 0 ①: mug transfer · 4 step 0 ②: RoboCasa365 on the server · 5 HOT3D · 6 affordance-extraction methods · 7 try it · 8 timeline and next week

- Figures first, then a few lines. Every number is copied from a task record (named in each section); the notebook computes nothing.
- **Simulator ground truth** (object poses, meshes) is used only to make labels and to evaluate, never as a model input; each section says which of its numbers are real runs and which come from the literature.
- **[designed]** = in the plan, not built; **[done]** = code ran and has results.
""")

code("""
from pathlib import Path

from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
""")

summary_cells(md)
pipeline_cells(md, code)
dataset_cells(md, code)
mug_transfer_cells(md, code)
robocasa365_cells(md, code)
hot3d_cells(md, code)
methods_cells(md, code)
command_cells(md)
timeline_cells(md)

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, OUT)
print("wrote", OUT)
