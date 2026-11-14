## What to take from this

**What this run shows**
- **The result of 13 Nov holds on all 50 start states and over three seeds.** Without any oracle plan, the ten-task head with kept plans of the map head solves 0.660 of the position perturbation and 0.681 of the task perturbation (1500 rollouts each), against 0.67 and 0.68 on 20 start states. The spread over seeds is about one point (standard deviation 0.012 for both), so differences of a few points between configurations of this size are real; the earlier runs with one seed and 200 rollouts could not show that.
- **The released policy does not cope with either perturbation:** 0 of 1500 with the target moved, where it grasps another object in 1458 rollouts, and 0.090 with another object asked for, all of it from one task (alphabet soup, 135 of 150), where the object asked for is the one the policy takes in that layout anyway.
- **The oracle plan bounds what a better plan head could add:** 0.920 and 0.753. Under the position perturbation the gap between kept map plans and the oracle plan is the plan head's: with the oracle plan nearly every task is solved (seven of ten tasks at 141 of 150 or more), while the map plans lose most on butter (71 against 150), bbq sauce (78 against 149), orange juice (73 against 121), ketchup (56 against 104) and tomato sauce (92 against 142), and the gripper grasps another object first in 108 rollouts (0 with the oracle plan).
- **Under the task perturbation the gap is in the action head.** Three tasks fail even with the oracle plan: ketchup (7 with map plans, 31 with the oracle plan, of 150), milk (24 and 24) and orange juice (41 and 54). In these rollouts the gripper often does not even come within 2 cm of the grasp point the oracle plan names (it does in 1277 of 1500 rollouts under this perturbation, against 1495 under the other); most failures are "no grasp" (296 of 371 with the oracle plan).
- **Next to the published numbers for this suite:** 0.66 and 0.68 lie above GAM (0.47, 0.50) and CounterAlign (0.51, 0.26) on both perturbations, and below π0.5 with a test-time affordance field on the position perturbation (0.73; it reports no task-perturbation number).

**What it does not show**
- **A like-for-like comparison of methods.** The base models differ: ours is the released starVLA checkpoint (Qwen3-VL-4B), the others build on π0.5 or are not a VLA. GAM uses the simulator's depth and the affordance-field method depth and GPT-4o at test time; our plan head uses the simulator's camera calibration. Our own released policy scores 0.00 under the position perturbation, where π0.5 is reported at 0.17–0.19.
- **The variation over training.** The seeds vary the policy's sampling noise and the simulator, not the training: there is one trained action head and one plan head per configuration.
- **The other three suites** (Spatial, Goal, LIBERO-10) and the object-appearance and language perturbations of LIBERO-PRO.

**Next**
1. Why the action head fails on ketchup, milk and orange juice under the task perturbation even with the oracle plan: where those objects stand in these scenes, and what the gripper does instead.
2. The plan head under the position perturbation: train it also on closed-loop frames, and try a finer map (more cells, or the wrist camera).
3. Variation over training: retrain the action head and the plan head with two more seeds.
4. The other three suites, which need a plan for skills other than pick-and-place; a second architecture; the full fine-tuning comparison agreed on 12 Nov.
