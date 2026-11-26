# Affordance-Guided Vision-Language-Action Models for Household Manipulation

*Research proposal · 26 November 2026 · `affordance-vla`*

## 1. Summary

A vision-language-action model (VLA) maps camera images and an instruction to robot actions. Today's VLAs break when an object moves, when the instruction names another object, or when the object is a new instance of a known category. We propose a VLA whose actions are trained to follow an explicit **affordance plan**: where to touch an object and in which direction. An understanding part decides which object comes next, an **interaction map** gives the contact and direction on that object, and the action head is trained to obey the resulting plan. The map is built once per object category and carried to instances never seen in training, so that learning to grasp one mug transfers to any mug. We develop the method in simulated kitchens (RoboCasa365), which hold many kinds of interaction and multi-step tasks, and keep LIBERO to connect to our earlier results.

## 2. Motivation and research questions

Most affordance methods stop at a map or a contact point and hand it to an external planner. A VLA can take such a map as an extra input, yet nothing forces its action to depend on it. Our thesis: **affordance helps only when the action is trained to obey it**.

- **RQ1 (obedience).** If the plan is moved while the image stays the same, does the action follow the plan? Does training for this raise success under perturbation?
- **RQ2 (category-level affordance).** Can one interaction map per category, built from demonstrations, be carried to unseen instances accurately enough to guide a grasp?
- **RQ3 (understanding).** In a multi-step task, can the model tell from recent frames which step is running and which object to interact with next, and does this choice of target improve unseen task compositions?

## 3. Proposed pipeline

![Pipeline: understanding names the next object, the interaction map gives the contact and direction, the action obeys the plan](report/media/pipeline_overview/v4_model.png)

*Gold: open-source starVLA model. Orange: our parts, built and trained on LIBERO. Grey: our steps without weights. Purple dashed: designed, not built yet.*

1. **Understanding: which object next.** The VLM reads the last few frames and the instruction and answers in text which step of the task is running and what the next interaction target is (for example "mug", then "coffee machine"). Labels come for free in simulation: composite demonstrations carry a sub-task label on every frame.
2. **Interaction map: where and which way.** A small head on the VLM features predicts, for every object in the image, a score per cell, a depth, and the direction in which the gripper closes. For handles and knobs it also predicts the direction to move after contact. A semantic gate keeps the map of the target object. The best cell is decoded into a 3D contact point, which becomes a short plan vector (reach, approach, gripper event) held until the gripper acts.
3. **Action obeys the plan.** The action head receives the plan as an extra embedding and is trained with **counterfactual plans**: for the same image, the plan is moved to another object or another point, and the action labels are rewritten to follow the new path. The model cannot ignore the plan and still fit the data.

At test time the model sees RGB images and the instruction only. Simulator ground truth (object poses, meshes, depth) is used to make labels and to evaluate, never as an input. The backbone is starVLA with Qwen3-VL-4B; for now the VLM is adapted with LoRA together with a small action head (QwenOFT), which fits one 48 GB GPU.

## 4. Extracting the affordance map

**Our route: demonstration statistics in a category frame.**

1. **Replay.** Each demonstration is replayed from its saved simulator states, which also yields depth and instance segmentation that the recorded videos do not contain.
2. **Record contacts.** At each gripper closure we store the grasp in the object's frame: the contact position and the closing axis. For articulated parts (drawer handle, door handle, knob) we also store the motion that follows the contact.
3. **Category frame.** All instances of a category are aligned to one frame (for a mug: upright, handle to one side), and the grasps of all training instances are accumulated into one 3D map over the category's surface.
4. **Transfer to a new instance.** The map is carried to an unseen instance by nearest-point matching in the category frame or by a learned descriptor field (NDF).
5. **Project to images.** The 3D map of every object in a frame is projected into the camera views; this gives the per-pixel training target of the map head.

**Alternatives we compared** (literature survey; nothing below was re-run by us):

| Route | Gives | Transfer to new instances | Limit for us |
| --- | --- | --- | --- |
| Demonstration statistics, category-level 3D map (ours) | contact + closing axis, 3D | geometric, testable | needs meshes and several instances per category |
| Descriptor fields: NDF | 6-DoF gripper pose | descriptor matching; 0.88 success on unseen upright mugs | external planner; shapes from ShapeNet |
| Retrieval + feature correspondence: Robo-ABC, RAM | 2D contact point (Robo-ABC); 3D contact + post-contact direction (RAM) | retrieval from a memory of human or robot data | no gripper closing axis |
| Human video: VRB | 2D contact heat-map + wrist path | end-to-end | 2D only, no grasp direction |
| End-to-end: AffordanceVLA | 2D contact heat-map inside a VLA | learned | nearest neighbour; no approach direction, weights not released |

Only AffordanceVLA trains a VLA with affordance; the others produce labels or plans. Our contribution is therefore the training method (obedience), with the category-level map as its label source.

**Human hand-object data.** Data of the HOT3D kind (egocentric video with hand meshes, object meshes and object poses) can supply the same map: from two clips of one mug (300 frames) we obtained five held grasps in our map format. Missing for direct use are a label for "this is a held grasp", a mapping from a human hand to a parallel gripper, and enough shapes per category (HOT3D has 2 mugs, 1 bowl, no cup, nothing articulated). A team collecting such data should record: object pose and mesh per frame, the hand mesh in the same frame, start and end of each hold, 8–10 shapes per category, and handles and knobs.

## 5. Candidate datasets

| Dataset | Scene and scale | Why (not) | Role |
| --- | --- | --- | --- |
| **RoboCasa365** | 2,500+ simulated kitchens, 365 tasks, 2,200+ h of demonstrations | pick-and-place, drawers, doors, knobs and multi-step tasks; saved states allow replay; 27 mugs, 22 cups, 22 bowls with held-out instances; sub-task labels; a 50-task leaderboard with VLAs (best 62.6) | **main environment** |
| LIBERO / LIBERO-PRO | tabletop, 130 tasks; perturbed test suites | our current results are here; only 3 mug meshes | link to earlier results |
| BEHAVIOR-1K | whole houses, 1,200+ h | richest labels, but encrypted meshes; the challenge winner trained on 8×H200 for about a month | not used |
| DROID | 564 real scenes, 350 h | real, but no depth and no matching simulator for closed-loop tests | not used |
| HD-EPIC | 9 real kitchens, 41 h, 26.6K questions, digital twins | strong for understanding; no robot actions or grasp labels | later, as an understanding source |
| HOT3D-type hand-object data | 19 people, 33 rigid objects (HOT3D) | can fill the interaction map (section 4) | affordance source, with the collection team |

No dataset has both understanding labels and fine actions; in RoboCasa365 the understanding labels can be generated from the simulator state and the sub-task labels.

## 6. Evaluation

1. **Unseen instances** of mug, cup and bowl, in kitchens seen in training, so that only the instance is new.
2. **Unseen task compositions**: the 16 held-out composite tasks of RoboCasa365.
3. **Understanding accuracy**: step and next target, on held-out kitchens and objects.
4. **Obedience**: same image, plan moved; does the action follow?
5. **LIBERO-PRO**, to connect to earlier results.

The main comparison is the same backbone with and without our method, with at least three training seeds; leaderboard numbers are reference only. Final numbers follow the official 50-task protocol.

## 7. Preliminary evidence

- **Obedience helps under perturbation (LIBERO-PRO, Object suite).** With a frozen VLM and about 31 M trained parameters, our plan-following action head reaches 0.66 / 0.68 success under position / task perturbation with a predicted plan and 0.92 / 0.75 with the ground-truth plan; the released starVLA checkpoint reaches 0.00 / 0.09. A small-scale diagnostic with one training seed, not a like-for-like comparison with published numbers.
- **Category transfer needs varied data.** On LIBERO's three mugs, carrying the map to a held-out mug passed in 1 of 6 cases: each mug always stands in one orientation, so shape cannot be separated from orientation. The test moves to RoboCasa365, where objects are placed at random.
- **RoboCasa365 works on our server.** Demonstrations replay from saved states and match the recorded video (34–36 dB PSNR), with depth and segmentation added; training and evaluation run end to end; one evaluation episode takes about 4.8 minutes with software rendering.

## 8. Plan, compute and risks

| Step | When | What | Pass criterion |
| --- | --- | --- | --- |
| 1 | now – Jan 2027 | 8 atomic tasks: labels, base policy, map head and gate, obedience training | higher than the same backbone without the method on unseen instances, beyond seed variation |
| Checkpoint | end of Jan | decide: NeurIPS 2027, or a narrower IROS 2027 paper | |
| 2 | Feb – Mar | composite tasks and step understanding | a measurable gain on unseen compositions |
| 3 | Apr – submission | official 50-task protocol, baselines, ablations, paper | |

**Compute.** Development runs on one 48 GB GPU with LoRA. Full fine-tuning of the 4.5 B-parameter model does not fit one such card (about 70 GB needed, our estimate), so the main results need a multi-GPU allocation.

**Risks.** (1) Composite-task scores are low for every published method, so a gain may be hard to measure; step 2 falls back to the step-1 scope. (2) A LoRA-trained base policy may be too weak to compare against; step 1 measures this first. (3) Maps built from demonstrations mix in the side the robot approached from; labels must handle symmetric parts. (4) "Plan, then act" is not new and AffordanceVLA is close; our difference is an action trained to obey the plan, tested by the obedience test.

## References

arXiv identifiers are those checked in our survey records (`docs/vla/2026-11-24_*`). RoboCasa365 [2603.04356] · LIBERO; LIBERO-PRO · starVLA (open-source VLA framework) · BEHAVIOR-1K challenge report [2512.06951] · HD-EPIC [2502.04144] · HOT3D (Meta, CVPR 2025) · NDF [2112.05124] · Robo-ABC [2401.07487] · RAM [2407.04689] · VRB [2304.08488] · AffordanceVLA [2606.06155]
