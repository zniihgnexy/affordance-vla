"""Section of the week-3 review (2026-11-23): step 1 (4a), three checks of the base policy (AFFO-19).

Imported by the notebook builder, which owns the notebook and provides `ROOT`, `Image` and `display`. Every number is
copied from docs/vla/2026-11-27_VLA_branch_第1步_底座检查.md, which cites the run folders on the NAS. Policy and
simulator are real (starVLA QwenOFT + LoRA, MuJoCo episodes with software rendering); the part probe reads simulator
state only to *measure* what the arm does, never as a model input. Figures: scripts/report/make_base_checks_figures.py.
"""

MEDIA = "report/media/weekly_2026_11_23/base_checks"


def base_checks_cells(md, code) -> None:
    """Figures first, then at most ten lines of conclusions, then commands that can be copied and run."""
    md(r"""
## Step 1 (4a): three checks of the base policy before any method is added (AFFO-19)

The base policy (section above) is the "without" side of every later comparison, so three things had to be known first:
how many training steps a variant needs, why the arm stops at the stove knob and the faucet, and whether unseen object
instances really cost nothing. 3.5 GPU hours in total, one card, three queue jobs.
""")
    md("**1. Success against training steps** (condition A, 8 tasks x 20 episodes, same seeds, bars = Wilson 95 %)")
    code(f'Image(filename=str(ROOT / "{MEDIA}/success_vs_steps.png"), width=1000)')
    md(
        "**2. Why the arm stops at the stove knob and the faucet** (gripper-to-joint-anchor distance, one line per episode)"
    )
    code(f'Image(filename=str(ROOT / "{MEDIA}/part_distance.png"), width=1000)')
    md("**3. Seen (A) against unseen (B) object instances** (four pick tasks, 75 episodes per bar, same seeds)")
    code(f'Image(filename=str(ROOT / "{MEDIA}/seen_vs_unseen.png"), width=1000)')
    md(r"""
### What we found

- **Success is still rising after 32 000 steps:** 27/160 (17 %), 59/160 (37 %), 74/160 (46 %) at 16 000 / 32 000 / 48 000; 32 000 -> 48 000 paired 15 only-earlier against 30 only-later (sign test p = 0.036). *Inference (medium confidence):* no plateau, so a variant trained for less than 48 000 steps will sit below the base. One training run, three checkpoints, 20 episodes per cell.
- **The two floor tasks do not improve with steps:** TurnOffStove 1, 0, 0 of 20; TurnOnSinkFaucet 7, 7, 5 of 20.
- **The arm does not stop short, it stops at the part:** in the stove failures the gripper gets within 8 cm of the knob in 17 of 20 episodes (median closest distance 3.3 cm); 7 touch it and the joint does not move. At the faucet 8 of 14 failures touch the handle without turning it, 6 never get within 8 cm.
- **Executing 4 steps per query instead of 8 does not fix it:** stove 0/20 and 0/20, faucet 6/20 and 8/20 (a 2-episode difference, inside the noise). *Inference:* what is missing is the contact direction after touching, which is what a plan would have to supply; not tested directly.
- **Seen against unseen instances: no measurable cost.** 170/300 (57 %) against 163/300 (54 %); only-A 47, only-B 40, p = 0.52; A-B = +2.3 points (95 %: -3.8 to +8.4), so gaps above about 8 points are excluded for these four tasks. The target object is a different instance in 220 of 220 pairs; the kitchen is identical in 196 of 220.
- **Not covered:** one training seed; a re-run of the same checkpoint reproduces 39 of 40 outcomes (the evaluation is not bit-deterministic); the four fixture tasks and condition C were not part of check 3.
""")
    md(r"""
### Try it (server `estar`, repository root of the task branch; the three evaluation jobs go through the queue and need a GPU)

```bash
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_steps 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=steps bash $PWD/scripts/rc365/run_base_checks_gpu.sh        # 98 min, two merged checkpoints x 160 episodes
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_stuck 25000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=stuck bash $PWD/scripts/rc365/run_base_checks_gpu.sh        # 21 min, n_act 8 and 4 with the part probe
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_instances 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=instances EXTRA="episodes=75" bash $PWD/scripts/rc365/run_base_checks_gpu.sh   # 90 min, 440 episodes
```

Quick look without a GPU: the finished runs are under `$W/runs/robocasa365_eval/affo19_*`; the tables and the three figures of this section are made from them on the CPU with the four commands in section 8 of `docs/vla/2026-11-27_VLA_branch_第1步_底座检查.md` (`scripts/rc365/summarise_checks.py steps|stuck|instances`, then `scripts/report/make_base_checks_figures.py`).
""")
