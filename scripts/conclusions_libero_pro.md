## What to take from this

**What this run shows**
- **The released policy survives neither perturbation.** With the target moved: 0 of 200. With another object asked for: 20 of 200, all from one task in which the object asked for happens to be the one the policy takes in that layout anyway. In 370 of its 400 rollouts it grasps another object first.
- **With the right plan, the same frozen VLM solves most of them.** The action head trained with counterfactual plans, given the ORACLE plan, solves 164 of 200 (0.82) with the target moved and 136 of 200 (0.68) with another object asked for. It never grasps a wrong object, and its gripper comes within 2 cm of the planned grasp point in 193 and 187 of 200 rollouts.
- **This also holds for objects the head never trained on.** The bbq sauce, butter and chocolate pudding were never targets in its training; with the target moved it solves 20 of 20 for each. Going to a contact point is not tied to the object. The fourth new object, the orange juice, gets 4 of 20, so it is not free either.
- **What fails is the handling, not where to go.** With the target moved, 36 rollouts fail: in 17 the object is never held, in 16 it is let go outside the basket, in 3 it is still in the gripper at the end. With another object asked for, 64 fail: 28, 22 and 14. The object is knocked over before it is held in 33 and 52 rollouts. The weak objects are the tall ones: ketchup (11 and 6 of 20), orange juice (4 and 4), milk (3 of 20 when asked for).
- **The upper bound lies above what has been published for this suite.** On the Object suite other papers report 0.17–0.19 for π0.5, 0.47 for GAM, 0.51 for CounterAlign and 0.73 for π0.5 with a test-time affordance field under the position perturbation, and 0.01–0.10, 0.50 and 0.26 under the task perturbation. The oracle plan gives 0.82 and 0.68. This is room to work in, not a result: the plan is ground truth.

**What it does not show**
- **No plan is predicted here.** Everything now depends on the plan predictor; its error comes off these numbers.
- **One of four suites, 20 trials per task instead of 50, one run, one seed.**
- **The comparison with published numbers is loose.** Trials and protocol differ, the papers' own baselines disagree, and which instruction they fed under the task perturbation is not known.
- **The action head saw six of the ten tasks.** The head trained on all ten was still training when this was written.
- **Object appearance and language perturbations, and the Spatial, Goal and LIBERO-10 suites, were not run.** Those suites contain skills other than pick-and-place, for which "the contact of the next gripper event" still has to be defined.

**Next**
1. The plan predictor (in training): its contact error on these start frames, then this table again with predicted plans.
2. The ten-task action head as a third column.
3. The handling failures: tall objects that are knocked over, and the carry and release, where no counterfactual plans were used yet.
4. The other three suites, once the plan is defined for their skills.
