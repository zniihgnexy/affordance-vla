"""Section 4 of the week-3 review (2026-11-23): step 0 (2), RoboCasa365 set up on the A6000 server (AFFO-10).

Imported by build_weekly_2026_11_23.py, which owns the notebook and provides `ROOT`, `Image` and `display`.
Every number below is copied from docs/vla/2026-11-25_VLA_branch_第0步_RoboCasa365搭建.md, which cites the run
folders on the NAS. The figures are made by scripts/report/make_rc365_figures.py from the replay run and the
downloaded bundles; the replay, training and evaluation numbers come from real simulator runs (software
rendering) and real starVLA training and evaluation jobs; nothing here uses simulator ground truth as a model input.
"""

MEDIA = "report/media/weekly_2026_11_23/robocasa365"


def robocasa365_cells(md, code) -> None:
    """Figures first, then at most ten lines of conclusions, then commands that can be copied and run."""
    md(r"""
## 4 · Step 0 ②: RoboCasa365 runs on the server (AFFO-10)

Three of the 50 target tasks, **CoffeeSetupMug** (pick a mug, place it under the coffee machine), **OpenDrawer** and
**TurnOffStove** (a knob), as the recorded demonstrations show them: three cameras, 256 × 256, 20 fps.
""")
    code(f'Image(filename=str(ROOT / "{MEDIA}/three_cameras.png"), width=900)')
    md(r"""
**Replay.** From the saved MuJoCo model and state of each step, the simulator renders the same view again (software
rendering, no GPU) and adds what the videos do not hold: depth and instance segmentation.
""")
    code(f'Image(filename=str(ROOT / "{MEDIA}/replay_video_vs_rerender.png"), width=1000)')
    md(r"""
**Object meshes.** The category-level map needs surface points of every instance of a category. RoboCasa365 ships
27 mugs, 22 cups and 22 bowls; its own split holds out 10, 7 and 7 of them for the target kitchens.
""")
    code(f'Image(filename=str(ROOT / "{MEDIA}/category_surface_points.png"), width=1000)')
    md(r"""
### What we found

- **It installs and renders without a display or a GPU:** software rendering, 0.10–0.11 s per 256 × 256 camera view, 0.054–0.061 s per physics step; rebuilding one demonstration's kitchen takes 16–19 s.
- **Replay works:** 15 demonstrations (3 tasks × 5) re-rendered from their saved states match the recorded video at 34–36 dB PSNR (one frame off: 26–29 dB) and now come with depth and instance segmentation. Replaying the *actions* open loop does not reproduce the recorded states bit for bit (state error 0.05–0.8 typical, up to 5.9).
- **The data differs from the survey:** LeRobot v2.1 (not v3.0), 514 demonstrations per task, 61.6 GB for the 50 target bundles (not 188 GB; source of that figure not found).
- **Held-out objects are built in:** `split="target"` holds out the last ≤ 5 instances of every category and uses 10 fixed kitchens; the leaderboard protocol and the demonstrations use it. Mugs: 27 instances, 10 held out; cups 22 / 7; bowls 22 / 7; all export surface points.
- **Composite demonstrations carry per-frame subtask labels** (language, skill name, stage, index), as documented; atomic ones do not.
- **No public RoboCasa365 weights** (the public RoboCasa checkpoints are the GR1 tabletop benchmark).
- **QwenOFT trains and evaluates:** frozen VLM, batch 8, 300 steps: 11.4 GiB, 0.445 s/step. Evaluation of that 300-step model, 10 episodes on each of the 3 tasks, 3 clients in parallel: 4.2–5.2 min per episode (4.8 on average, about 0.41 s per environment step), 0/30 successes (a speed check of an under-trained model with a camera and state mismatch, not a capability).
- **QwenPI does not fit one 48 GB card:** its action head alone has 3.35 B parameters (7.79 B in total); out of memory with a frozen VLM at batch 8 and 1, and with all weights at batch 1. Full fine-tuning of QwenOFT (4.5 B) at batch 1 does not fit either (needs roughly 70 GiB, inference). Step 1 needs a frozen QwenOFT VLM, LoRA on frozen weights, or several cards.
""")
    md(r"""
### Try it (server `estar`, repository root of the task branch; all CPU except the last two, which go through the queue)

```bash
bash envs/server/setup_robocasa365_env.sh                    # conda env + pinned code + 22 GB of assets on the NAS
bash scripts/data/fetch_robocasa365.sh                       # 3 tasks, 1.4 GB (--dry-run prints sizes only)
source envs/server/robocasa365_env.sh
"$RC365_PYTHON" scripts/data/replay_robocasa365.py --task CoffeeSetupMug OpenDrawer TurnOffStove --episodes 5 \
    --out $W/runs/robocasa365_replay/my_run                 # re-render, depth, segmentation, PSNR against the videos
"$RC365_PYTHON" scripts/report/make_rc365_figures.py --replay $W/runs/robocasa365_replay/my_run --out /tmp/rc365_figures
bash scripts/server/submit_queue_job.sh 164 rc365_train 30000 1 "cards=6 7" -- \
    env RUN=my_train CONFIG=$PWD/configs/rc365/qwenoft_rc365_3tasks.yaml bash $PWD/scripts/rc365/run_train_gpu.sh
bash scripts/server/submit_queue_job.sh 164 rc365_eval 20000 1 "cards=6 7" -- \
    env CKPT=$W/runs/robocasa365_starvla/my_train/checkpoints/steps_300_pytorch_model.pt \
    bash $PWD/scripts/rc365/run_eval_gpu.sh
```

RoboCasa365 was only run on the server (22 GB of assets, 1.4 GB of demonstrations for these three tasks); no Mac command is given because none was tried.
""")
