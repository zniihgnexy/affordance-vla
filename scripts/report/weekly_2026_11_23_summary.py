"""Sections 0 and 8 of the week-3 review (2026-11-23): the one-page overview, and the timeline with the next steps.

Imported by build_weekly_2026_11_23.py. Facts: docs/HANDOVER.md (version 14, 25 Nov 2026 night) and
docs/vla/2026-11-25_VLA_branch_Pipeline_v4.md; every number is the one printed in the section that owns it
(sections 3 to 5), which copies it from that task's record. Nothing here is new data.
"""


def summary_cells(md) -> None:
    """Section 0: scope, what was done, what was found, what is open, the decisions."""
    md(r"""
## 0 · Summary

**Scope changed (24–25 Nov).** The advisor asked for a VLA that is not limited to one kind of task: it should extend to tasks in indoor rooms and show some video understanding. The project lead picked option 1B (RoboCasa365 kitchens as the main environment, LIBERO kept to connect to earlier results) and said compute is not a constraint for now. **Pipeline v4** (approved 25 Nov) adds an understanding part that names the next object to touch, to the v3 core (interaction map, obedient action).

**What was done this week**

| Section | Work | Kind of evidence |
| --- | --- | --- |
| 1 | Pipeline v4 figure, v3 to v4 changes | design; nothing built for v4 yet |
| 2 | dataset survey, why RoboCasa365 | literature (abstracts, project pages, data cards); nothing downloaded |
| 3 | step 0 ①: carry the mug map to a new mug (AFFO-9) | computed from replayed LIBERO demonstrations; the object poses and meshes are simulator ground truth, the NDF weights are public |
| 4 | step 0 ②: RoboCasa365 on the server (AFFO-10) | real simulator runs (software rendering) and real starVLA training and evaluation jobs; no ground truth fed to a model |
| 5 | HOT3D: can human-hand data give our map (AFFO-11) | real motion-capture annotations (ground truth by construction), no model |
| 6 | survey of affordance-extraction methods | literature; nothing run |

**What we found**

- **Step 0 ①: 1 of 6 groups passes** (red mug held out, NDF). The main reason is in the demonstration data (each mug is always placed in one orientation, so demonstrators grasp from one side), not in the transfer code (inference). With a fixed orientation, "unseen shape" and "unseen orientation" cannot be separated on these mugs.
- **Step 0 ②: the pipeline works end to end.** RoboCasa365 renders with no display or GPU, demonstrations replay from saved states with depth and segmentation added, starVLA trains and evaluates. The **0/30 successes is not a capability result**: a model trained for 300 steps, evaluated with one camera where training used three; it only shows that the pipeline runs, at about 4.8 minutes per episode.
- **A single 48 GB card is too small for QwenPI**, even with the VLM frozen (action head alone 3.35 B parameters). QwenOFT with a frozen VLM needs 11.4 GiB; QwenOFT full fine-tuning did not fit either. **Which backbone and training mode step 1 uses is an open decision for the project lead**, not settled here.
- **HOT3D can fill our map format** from hand and object meshes, but gives "how a person holds it", not "how a gripper should grasp it"; it will not enter the main line. The result is a list of what the data-collection team should record (section 5).
- **Two survey numbers were corrected by measurement** (section 2): RoboCasa365 demonstrations are LeRobot v2.1, and the 50 target bundles are 61.6 GB.

**Where we are, and what is not decided.** Step 0 is finished. **Step 1 has not been approved**, and the plan for next week is not settled: it waits on two questions to the project lead (backbone and training mode; when step 1 starts and how many GPUs it may use). Section 8 lists the candidate next steps, as candidates.

**Decisions this week**

| Date | Decision | By |
| --- | --- | --- |
| 24 Nov | survey data sets and extraction methods first, then pick the final experiment scope | project lead |
| 24 Nov | agents run on the Mac and control the server; one branch and worktree per issue, direct push of `agent/*` branches; GPU 0 first | project lead |
| 25 Nov | scope 1B; compute not a constraint for now | project lead |
| 25 Nov | Pipeline v4 approved; step 0 starts, one GPU through the queue | project lead |
| 25 Nov | this weekly review is the one document the project lead reads | project lead |
""")


def timeline_cells(md) -> None:
    """Section 8: what happened when, and the candidates for next week (not decided)."""
    md(r"""
## 8 · Timeline and next week

**This week (23–29 Nov, London time).** Source: `docs/HANDOVER.md`, section 3, and the task records.

| Day | What happened |
| --- | --- |
| Mon 23 Nov | week-2 review (English, short) for the group meeting; MuJoCo demo commands; pipeline figures redrawn |
| Tue 24 Nov | the advisor's feedback sets a wider scope; the project moves to Multica; two surveys (data sets, extraction methods) done and accepted; hand-over summary written |
| Wed 25 Nov | scope 1B chosen, **Pipeline v4 approved**, step 0 started (AFFO-9, AFFO-10); HOT3D assessment (AFFO-11); step 0 ① ends with 1 of 6 groups passing; step 0 ② passes; this review started |
| Thu 26 Nov | this review built and executed; the commands of section 7 run |
| Fri 27 – Sun 29 Nov | not yet happened |

The account's usage limit cut off agent runs on 24 and 25 Nov (including the first attempt at this review); each was restarted by the lead.

**Next week: candidates, nothing decided.** Step 1 (6–8 atomic tasks: labels, map head and gate, obedience training) is **not approved**. It waits for the project lead's answers on the backbone (question 7: QwenOFT with a LoRA on one card, QwenPI later with more compute, or QwenOFT with full fine-tuning on several cards) and on the start of step 1 with its GPU limit (question 8). If step 1 starts, things the records say must be handled first:

- **Labels:** the map in object coordinates mixes in "which side the robot came from" (demonstrations cover only the side facing the robot). Two candidates: complete a symmetric part into a ring and pick by robot position, or record grasps in a robot-facing frame; decide on RoboCasa, where mugs are placed at random.
- **starVLA evaluation bridge:** training reads 3 cameras, evaluation sends 1 and builds the state unconditionally, and the evaluation environment has no seed. Align before any success rate is reported.
- **Evaluation budget:** every episode takes about 4.8 minutes; a rough estimate (inference) of 27 hours of simulation for 8 tasks × 2 instance splits × 20 episodes, spread over at most 30 CPU processes.
- **Unseen instances:** the target-split demonstrations were recorded with the held-out instances, so "seen and unseen instances" must be designed by us (idea, not checked: train on the pretrain split, evaluate on the target split).
- **One known test problem:** `tests/test_config.py::test_the_resolved_config_records_its_source` fails when pushed from a task worktree on the Mac (the hook exports `GIT_DIR`); it passes when run directly.
""")
