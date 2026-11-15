## What to take from this

**What the failures are**
- **Mostly running out of time.** In the three tasks the gripper first reaches the grasp point after a median of 98 to 130 steps, against 37 to 69 in the other seven tasks, and 96 to 126 of 150 rollouts run to the step limit. Orange juice and ketchup rollouts spend long stretches nearly still more than 5 cm from the object: the orange juice rollout of section 2 hovers about 20 cm above and 7 cm beside it for 190 steps, then grasps it shortly before the limit. Milk is mostly knocked over (it drops about 4 cm), and ketchup is often touched but not grasped.
- **Not the place.** The orange juice stands at the same place in task 9 of the position perturbation, with the same plan, and is reached by step 54 and solved in 121 of 150 rollouts.

**Why: the training labels for "another object named" in exactly these scenes**
- **They make a detour.** The three tasks are the ones in which the object asked for is closest to where the gripper starts (16 to 19 cm), while the scene's original target is 18 to 30 cm away from it. The label moves the demonstrated path to the original target over to the object asked for. With the rule this action head was trained with, that path is 3.3 to 4.5 times longer than the straight line in these three tasks (1.4 to 2.9 in the other seven), and for ketchup and orange juice it first moves 14 to 18 cm away from the object. The head was trained on these labels for these very scenes and instructions, follows them part of the way, and ends up in states that no demonstration covers. At the place of the position perturbation no such labels were ever made (training shifts stay within 18 cm, this place is 26 cm away), and there the head goes straight.
- **They copy the wrong final approach.** The last part of the moved path keeps the shape of the original target's approach. The ketchup's demonstrations come in low from the side (the last 18 cm start 6 cm above and 8 cm aside, against 12 to 15 cm above for every other object). The milk task is set in the ketchup's scene, so its labels bring the gripper in low beside the tall milk carton, which falls (the labelled path is below the grasp point while still more than 4 cm aside). The ketchup task is set in the bbq sauce's scene and gets its steep approach instead of the ketchup's own. The plan input, by contrast, already gives the approach direction of the object asked for, so plan and label disagree.

**What it does not show**
- That fixing the labels fixes these tasks: that needs a head trained with other labels. A prototype of the first fix below removes the moving away in all ten pairs, but keeps detours of 1.9 to 3.1 for these three, because the moved final approach (18 cm) is as long as the whole straight distance.
- Whether the same holds in the other suites.

**Proposed fix (for the project lead to decide)**
1. Move the free part of the demonstrated path by rotating and scaling it about the gripper's start, so that it heads for the object asked for and shrinks with the distance, instead of adding the shift; the final approach is moved as a whole as before, and shortened when the object is close.
2. Take the final approach from the demonstrations of the object asked for, not from the scene's original target, so that it agrees with the plan's approach direction.
3. Test: one ten-task head with both changes, both perturbations of LIBERO-PRO Object with the oracle plan and with kept map plans, 20 start states; the three tasks should rise and the other seven should not drop.
