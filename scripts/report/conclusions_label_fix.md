## What to take from this

**What this test shows**
- **The detours are gone, and the task perturbation gains the most.** With the oracle plan the head with the new labels solves 166 of 200 under the task perturbation (152 with the labels of 13 Nov), with kept map plans 151 (136). The orange juice, whose labels made the largest detour, rises from 8 to 18 of 20 with the oracle plan and from 7 to 16 with map plans. With the oracle plan the gripper now reaches it in all 20 rollouts (8 never did before) and runs out of time in 3 instead of 12, though it still pauses on the way (a median of 25 steps nearly still, against 32).
- **The milk rises with the oracle plan (3 to 10 of 20), not with kept map plans (4 and 4).** The map plan for the milk is as good as the oracle plan (0.3 cm off, the same approach direction), so the plan is not the reason. The difference is what happens after the milk is bumped: the oracle plan follows the carton, a kept plan stays where the carton stood. With map plans the milk is knocked over in 10 of 20 rollouts.
- **The ketchup is reached sooner but still not delivered** (5 to 1 of 20 with the oracle plan). The gripper now gets to it after a median of 60 steps instead of 86 and holds it more often (12 against 10), but most of those rollouts end with the ketchup outside the basket or knocked over. Its failure has moved from the approach to the grasp and the carry.
- **The new head follows the plan more exactly, which costs where the plan is off.** Under the position perturbation the oracle-plan column is unchanged (186 against 185), but with kept map plans one task collapses: the tomato sauce, from 13 to 2 of 20 (129 against 133 in all). Its map plan is about 3 cm off; the head of 13 Nov still found the can, the new head goes to the planned point and closes on nothing.

**What it does not show**
- **That the differences are more than noise, task by task.** Each task has 20 rollouts from one seed; changes of a few rollouts per task are within noise. The totals of the task perturbation (+14 and +15 of 200) are one and a half to two standard errors of a difference; the orange juice (8 to 18, 7 to 16) and the tomato sauce (13 to 2) are well outside.
- **Which of the two changes does what.** Scaling the path and taking the object's own final approach were tested together.

**Options for the project lead**
1. Make the labels of 15 Nov the default: the gain is on the perturbation where another object is asked for, which is where a plan from the image matters most; the loss is one task where the plan head is 3 cm off.
2. Teach the head to correct small plan errors toward the object it sees: for samples that name another object, put 2 to 3 cm of noise on the plan input while the label still goes to the object's real grasp point (now 1 cm).
3. Renew a kept plan during the approach (the milk with map plans): predict it again now and then while nothing is held, so that a bumped object is followed. This needs a plan head that is good on closed-loop frames too: predicting at every query was worse than keeping the plan (section 4 of the notebook of 13 Nov).
4. Make the plan head more precise (the tomato sauce): more cells or the wrist camera, and training on frames of closed-loop rollouts.
5. The ketchup's grasp and carry, which is now its own problem.
