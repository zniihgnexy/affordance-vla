"""Section 7 of the week-3 review (2026-11-23): every demo command of the week in one place.

Imported by build_weekly_2026_11_23.py, which owns the notebook. The "run on 26 Nov" marks are the results of
running the commands for this review (AFFO-12): on the server in a fresh worktree of branch
agent/vla-engineer/affo-12 (CPU only, new output folders, nothing overwritten) and on the Mac. Commands that need a
GPU were run by their own tasks (AFFO-9, AFFO-10) and are not re-run here.
"""


def command_cells(md) -> None:
    """Two groups, server and Mac, each command with what happened when it was run for this review."""
    md(r"""
## 7 · Try it

Two groups. **Server**: `ssh estar`, repository root of a worktree of branch `agent/vla-engineer/affo-12` (`git worktree add --detach ~/wt-review origin/agent/vla-engineer/affo-12`), `W=/mnt/nas/xinyizheng/affordance_vla`. **Mac**: repository root of a clone that has run the setup once. "Run" = I ran it for this review on 26 Nov; "not re-run" says why. Output folders are new, so earlier results are never overwritten.

**GPU commands:** the examples written by AFFO-10 use `cards=6 7`, because GPU 0 was full when they ran. **Use GPU 0 first (`cards=0`); if it is full, list the cards that have room at that moment.** GPU jobs go through the queue only (`scripts/server/submit_queue_job.sh`).

### On the server

**① Mug transfer check (section 3).** CPU stages are cheap; the NDF stage needs a GPU.

```bash
bash scripts/data/run_mug_transfer_check.sh collect --set out=runs/mug_transfer/my_check       # 51 s
bash scripts/server/submit_queue_job.sh 164 mug_ndf 8000 1 "cards=0" -- \
    bash $PWD/scripts/data/run_mug_transfer_check.sh ndf --set out=runs/mug_transfer/my_check   # GPU, about 45 min; run by AFFO-9, not re-run
bash scripts/data/run_mug_transfer_check.sh evaluate --set out=runs/mug_transfer/my_check       # table + figures
cat $W/runs/mug_transfer/my_check/results.md
```

Run: `collect` and `evaluate` (with AFFO-9's NDF files copied into the new folder): `results.json` identical to AFFO-9's. The `ndf` stage was not re-run (GPU).

**② HOT3D demo (section 5).** CPU, 30 s.

```bash
bash envs/server/setup_hot3d_env.sh       # once
bash scripts/data/fetch_hot3d_demo.sh     # 2 clips + 3 meshes, 0.6 GB, no sign-up
source envs/server/hot3d_env.sh
PYTHONPATH=. python scripts/data/hot3d_contact_demo.py --out $W/runs/hot3d_demo/my_check   # try --threshold-mm 10
```

Run: the fetch (data already there, nothing downloaded) and the demo (29 s): `summary.json` identical to AFFO-11's. The environment set-up script was not re-run.

**③ RoboCasa365 (sections 2 and 4).** Everything except the last two commands is CPU.

```bash
bash envs/server/setup_robocasa365_env.sh                              # conda env + pinned code + 22 GB assets, safe to repeat
bash scripts/data/fetch_robocasa365.sh --dry-run                       # sizes only; without --dry-run: 1.4 GB for 3 tasks
source envs/server/robocasa365_env.sh
"$RC365_PYTHON" scripts/data/replay_robocasa365.py --task CoffeeSetupMug OpenDrawer TurnOffStove --episodes 5 \
    --out $W/runs/robocasa365_replay/my_run                           # depth, segmentation, PSNR against the recorded video
"$RC365_PYTHON" scripts/report/make_rc365_figures.py --replay $W/runs/robocasa365_replay/my_run --out /tmp/rc365_figures
bash scripts/server/submit_queue_job.sh 164 rc365_train 30000 1 "cards=0" -- \
    env RUN=my_train CONFIG=$PWD/configs/rc365/qwenoft_rc365_3tasks.yaml bash $PWD/scripts/rc365/run_train_gpu.sh   # GPU
bash scripts/server/submit_queue_job.sh 164 rc365_eval 20000 1 "cards=0" -- \
    env CKPT=$W/runs/robocasa365_starvla/my_train/checkpoints/steps_300_pytorch_model.pt \
    bash $PWD/scripts/rc365/run_eval_gpu.sh                            # GPU, about 5 min per episode
```

Run: set-up (11 s, already installed), the dry run (three bundles of 423–479 MB), the replay with `--episodes 1` instead of 5 (3 min 10 s, PSNR 32–39 dB at the recorded frame; section 4 reports 34–36 dB over 15 demonstrations) and the figure script (three PNGs). The 5-episode replay, training and evaluation were run by AFFO-10 and not re-run (GPU, or 15 min).

### On the Mac

**Once:** `bash envs/local/setup_sim_env.sh` (conda env `affordance-vla-sim`), `bash scripts/local/fetch_sim_data.sh` (LIBERO-PRO, task files, evaluation logs, the interaction maps). Neither was re-run: the environment and data were already there. The commands below open the MuJoCo viewer (rotate with the mouse, space pauses, close the window to end) and describe each step in the terminal. A replay re-runs the actions of a server log in the same scene; no model runs on the Mac.

```bash
bash scripts/local/sim.sh list runs/libero_pro/predicted_g1_hold_s10                                    # every logged episode with its outcome
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz  # position perturbation, ours: succeeds
bash scripts/local/sim.sh replay runs/libero_pro/released/libero_object_swap/task2_ep0.npz               # same scene, released checkpoint: fails
bash scripts/local/sim.sh map --suite libero_10 --task 4                                                # the 3D interaction map of the mugs on the objects
bash scripts/local/sim.sh scene --suite libero_object_swap --task 2 --episode 0                         # the test scene with each object's demonstrated grasp
```

Run: `list`, and the first `replay` with `--set viewer.realtime=0 --set viewer.hold=false` (so it ends by itself): success in 128 steps, the same as the server's log, end-effector drift 0.0 mm. **Not re-run this week:** the second replay, `map` and `scene` (they only open a window and wait for you; they were checked for the week-2 review). More demo commands of week 2 are in section 6 of `report/2026-11-16_weekly_review.ipynb`. **There is no Mac command for sections 3 to 5**: they need the server's data (mug maps and NDF), the HOT3D clips or RoboCasa365 assets; their figures are in this notebook.
""")
