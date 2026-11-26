"""Section 2 of the week-3 review (2026-11-23): why RoboCasa365 (pictures, then a compressed dataset comparison).

Imported by build_weekly_2026_11_23.py, which owns the notebook and provides `ROOT`, `Image` and `display`.
Every number is copied from docs/vla/2026-11-24_VLA_branch_数据集与实验范围调研.md (AFFO-4; abstracts, project pages
and data cards only, nothing downloaded or run), except the two corrections taken from AFFO-10's measurements
(docs/vla/2026-11-25_VLA_branch_第0步_RoboCasa365搭建.md, on the AFFO-10 branch). The picture is a frame of recorded
demonstrations (report/media/weekly_2026_11_23/robocasa365/three_cameras.png, made by AFFO-10).
"""


def dataset_cells(md, code) -> None:
    """Pictures first, then the table, then at most ten lines of conclusions."""
    md(r"""
## 2 · Why RoboCasa365

The project lead's advisor asked (24 Nov) for a VLA that is not limited to one kind of task: it should extend to tasks in indoor rooms and show some video understanding. We needed one simulator with several kinds of interaction, room-level scenes and demonstrations with fine actions. **RoboCasa365** (2026) is that: kitchens, with demonstrations. Three of its tasks as the recorded demonstrations show them (a pick-and-place, a drawer, a knob); three cameras, 256 × 256, 20 fps:
""")
    code('Image(filename=str(ROOT / "report" / "media" / "weekly_2026_11_23" / "robocasa365" / "three_cameras.png"), width=900)')
    md(r"""
**What the survey compared** (AFFO-4, 24 Nov; abstracts, project pages and data cards only, **nothing was downloaded or run**; "not found" = the survey did not find it, not that it does not exist). Compressed from 11 candidates to the 8 that matter:

| Dataset | Scene | What it gives | Scale | Closed-loop test in simulation | Main limit |
| --- | --- | --- | --- | --- | --- |
| **RoboCasa365** | room-level kitchens, mobile manipulation | 3 RGB cameras, language, robot actions; layout and object files per episode | 365 tasks, 2500+ kitchens, 2200+ h | yes; 50-task leaderboard (best VLA 62.6) | kitchens only; no depth or segmentation, no question-answer labels |
| BEHAVIOR-1K (2025 challenge) | whole houses, whole-body mobile | RGB-D, instance segmentation, object states, skill labels | 10,000 episodes, 1200+ h, ~1.5 TB | yes; best q-score 0.26 | object meshes are encrypted (no category-level labels); hard, 8×H200 for a month for the winner |
| MS-HAB | apartments (ReplicaCAD) | RGB-D 128², state | 84 apartments | yes; RL/IL baselines only | no VLA numbers; license not found |
| LIBERO / PRO | tabletop | wrist + third-person cameras | 50 demos × 130 tasks | yes (our current setting) | 2–8 movable objects per scene |
| DROID | 564 real scenes, fixed arm | 3 cameras, language | 76k episodes, 350 h, 1.7 TB (RLDS) | only a separate IsaacLab kitchen set (π0.5: 38.3%) | RLDS has no depth; sim scenes are not the data's scenes |
| AgiBot World | 106 real scenes, mobile dual arm | 8 cameras, sub-step labels | 1M episodes, 2976 h, 48 TB | not found | non-commercial; huge |
| HD-EPIC | 9 real kitchens, first person | digital twin of each kitchen, 3D masks, 26.6K questions | 41 h, 2.3 TB (about 120 GB for video + labels) | none (a question benchmark; Gemini Pro 37.6%) | no robot actions, no grasp labels; non-commercial |
| EgoDex | tabletop, first person | 3D hand poses (68 joints) | 829 h | none | no scene 3D; non-commercial |

(Also in the survey: EPIC-KITCHENS + VISOR, HOI4D, SceneFun3D; none has robot actions.)

- **No dataset has both understanding labels and fine actions** (survey conclusion 2). BEHAVIOR-1K comes closest but is the hardest to use.
- **Why RoboCasa365 (survey, then step 0 ②):** the only candidate with room-level scenes, fine actions, several interaction types, a leaderboard with VLAs and plain MuJoCo. Understanding labels have to be generated from simulator state and sub-task labels (design, not built).
- **Two survey numbers were wrong or unchecked, measured by AFFO-10 (section 4):** the demonstrations are LeRobot v2.1 (the survey said v3.0), and the 50 target bundles are 61.6 GB (the survey said 188 GB; the source of that figure was not found).
- **Not adopted:** BEHAVIOR-1K (encrypted meshes, compute); HD-EPIC is kept for later as an understanding source (Pipeline v4, section 8).
""")
