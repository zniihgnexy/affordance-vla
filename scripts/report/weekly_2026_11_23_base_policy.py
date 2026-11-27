"""Section of the week-3 review (2026-11-23): step 1 (first half), the base policy (AFFO-16).

Imported by the notebook builder, which owns the notebook and provides `ROOT`, `Image` and `display`. Every number is
copied from docs/vla/2026-11-26_VLA_branch_第1步_底座策略.md, which cites the run folders on the NAS. The policy,
training and evaluation are real (starVLA QwenOFT + LoRA, real MuJoCo episodes with software rendering); no
simulator ground truth is used as a model input. Figures are made by scripts/report/make_base_policy_figures.py.
"""

MEDIA = "report/media/weekly_2026_11_23/base_policy"


def base_policy_cells(md, code) -> None:
    """Figures first, then at most ten lines of conclusions, then commands that can be copied and run."""
    md(r"""
## Step 1 (first half): the base policy, QwenOFT + LoRA on eight RoboCasa365 tasks (AFFO-16)

Every later comparison is "the same backbone, with and without our method", so this is the *without* side. Trained once
(seed 42) on the pretrain split only; evaluated with 20 episodes per task and condition: **A** seen objects in pretrain
kitchens, **B** unseen objects in the same kitchens (same seeds, so the same kitchens as A), **C** unseen objects in the
10 target kitchens.
""")
    code(f'Image(filename=str(ROOT / "{MEDIA}/success_by_task.png"), width=1000)')
    md(r"""
What the failures look like (35 recorded failures; one row each of the seven tasks shown, 8 frames from start to timeout):
""")
    code(f'Image(filename=str(ROOT / "{MEDIA}/failure_examples.png"), width=900)')
    code(f'Image(filename=str(ROOT / "{MEDIA}/failure_classes.png"), width=1000)')
    md(r"""
### What we found

- **Overall 74/160 (46%) with seen objects, 74/160 (46%) with unseen objects, 82/160 (51%) in target kitchens** (Wilson 95 %: 39–54, 39–54, 44–59). Tasks differ much more than conditions: PickPlaceSinkToCounter 90 / 75 / 80 %, TurnOffStove 0 / 5 / 10 %.
- **No measurable cost of unseen object instances:** with the same seed, 21 episodes succeed only with seen objects and 21 only with unseen ones. *Inference:* four of the eight tasks operate a fixture (drawer, cabinet, knob, faucet), so B barely changes what is manipulated there; for the four pick tasks alone it is 46/80 vs 44/80, still no difference. With 20 episodes a cell can only separate differences of about 30 points.
- **Two tasks sit near the floor:** TurnOffStove 3/60 and TurnOnSinkFaucet 12/60 over A+B+C. In 51 of 57 and 39 of 48 failures the end effector stops moving for the last 40 % of the episode (path < 0.1 m): the arm goes up to the stove or faucet and stays there until the timeout. Why is not diagnosed.
- **All 250 failures are timeouts.** In the pick tasks, 92 of 96 failures end with no object in the hand (the trace cannot separate "never reached" from "dropped" from "placed wrong"); only 4 end holding it.
- **Cost:** LoRA + MLP head on one 48 GB card, batch 16: 34.9 GiB allocated (46.0 reserved), 1.81 s/step; 48 000 steps = 24 h 05 min. Evaluation: 480 episodes in 1 h 40 min with 24 simulator clients. **26.5 of the 40 GPU hours used.** One training seed and no mid-training evaluation: seed-to-seed spread is unknown.
- **Usable as the baseline for six tasks (40–90 %); the two near-floor tasks are not informative for the baseline.** Options and their costs are in `docs/vla/2026-11-26_VLA_branch_第1步_底座策略.md`, section 6.
""")
    md(r"""
### Try it (server `estar`, repository root of the task branch; the first two go through the queue and need a GPU)

```bash
bash scripts/server/submit_queue_job.sh 164 rc365_base 40000 1 "cards=0 1 2 3 5 6 7" -- \
    env RUN=affo16_base_oft_lora bash $PWD/scripts/rc365/run_train_base_gpu.sh            # 24 h
python scripts/rc365/merge_lora_checkpoint.py --ckpt $W/runs/robocasa365_starvla/affo16_base_oft_lora/checkpoints/steps_48000_pytorch_model.pt \
    --out $W/runs/robocasa365_starvla/affo16_base_oft_lora_merged_48000                  # CPU, 3 min
bash scripts/server/submit_queue_job.sh 164 rc365_eval_base 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env CKPT=$W/runs/robocasa365_starvla/affo16_base_oft_lora_merged_48000/checkpoints/steps_48000_pytorch_model.pt \
    RUN=my_eval OVERRIDES="conditions=[A,B,C]" bash $PWD/scripts/rc365/run_eval_base_gpu.sh   # 1 h 40 min
O=$W/runs/robocasa365_eval/my_eval
python scripts/rc365/summarise_eval.py $O && python scripts/rc365/classify_failures.py $O      # tables, failure classes (CPU)
```

Quick look without a GPU: the finished run `affo16_eval_base_48000` is on the NAS; the two commands on the last line, run on it, reproduce the tables of this section. `cards=0` first; if GPU 0 is full, list the cards that have room at that moment (the project cap is 40 GPU hours).
""")
