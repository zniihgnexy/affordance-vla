"""Section 2 of the week-2 review: the VLA we build on, the data, how we use the model, and how the affordance map
is used today. Facts checked on 23 Nov 2026 against the code, the released checkpoint's config.yaml, the LeRobot
datasets' meta files and the metrics.json of the runs named here.

Imported by build_weekly_2026_11_16.py, which owns the notebook. The figure reads the E1 probe
(runs/semantic_probe/e1/probe.npz), which kept the g1 map and the agent view of every first frame it scored.
"""

MAP_FRAMES = [("libero_object_swap", "task2_ep0.npz"), ("libero_object_task", "task2_ep0.npz"),
              ("libero_object_swap", "task5_ep0.npz"), ("libero_spatial_task", "task8_ep0.npz")]


def vla_cells(md, code) -> None:
    """Section 2: model, data, the four ways we use the VLA, and the path from the map to the actions."""
    md(r"""
## 2 · The VLA we build on, the data, and how the affordance map is used

### 2.1 Model: starVLA QwenPI, the released LIBERO checkpoint

| | |
| --- | --- |
| Codebase | [starVLA](https://github.com/starVLA/starVLA) (open source), variant **QwenPI**: a VLM plus a flow-matching action head in the style of π0 |
| Checkpoint | `Qwen3-VL-PI-LIBERO-4in1`, step 100,000, released by starVLA (from its ModelScope mirror, sha256-checked) |
| Size | 7.79B parameters: VLM **Qwen3-VL-4B-Instruct** 4.44B + action head (DiT, 36 blocks × 2560) 3.35B |
| Trained by starVLA on | the four LIBERO suites together (Spatial, Object, Goal, LIBERO-10): 1,693 demonstrations, LeRobot `no_noops` versions |
| Input | agent view + wrist view (224 × 224) and the instruction; we send no robot state |
| Output | 8 actions × 7 numbers (Δposition, Δrotation, gripper) from 4 flow-matching steps; robosuite's controller executes them at 20 Hz |

Layer by layer: section 3.2.

### 2.2 Data

| Data | What it is | What we use it for |
| --- | --- | --- |
| **LIBERO** (training) | simulated tabletop tasks, 50 human teleoperated demonstrations each; suites Spatial, Object, Goal, LIBERO-10, LIBERO-90 | We read only states and actions, replay them in the simulator and record images, gripper and object poses, contacts. **LIBERO-Object** → `dataset10`: 454 demonstrations, 56,852 training frames (5 per task held out) — trains both of our heads. **Spatial** → `dataset_spatial` (E2). Object and LIBERO-10 replays with object poses → the L maps (section 4.3). |
| **LIBERO-PRO** (test only) | the same tasks perturbed: `_swap` objects trade places, `_task` another object is named (also `_object`, `_lan`: appearance, wording) | Zero-shot tests, never trained on: Object, Spatial, LIBERO-10 × swap / task. 20 starts per task (quick) or 50 × 3 seeds (the papers' protocol). |
| LIBERO-90, LIBERO-Plus | more scenes (the red mug); another perturbation benchmark | planned: category-level mug map and G4 (stage 1); task selection (stage 2) |

The released checkpoint was trained on the same LIBERO-Object demonstrations. We add no demonstrations: what we add is the plan pathway and its counterfactual training (section 3.3).

### 2.3 How we use the VLA

Each query runs **one** forward pass of the VLM over the two images and the instruction; our heads read its hidden states.

| Use | What runs | Trained by us | Run |
| --- | --- | --- | --- |
| 1 · baseline | the released checkpoint, unchanged | nothing | `released` |
| 2 · map head | frozen VLM; a small head reads all 36 LLM layers and scores the 8 × 8 image cells | 7.89M | `g1` |
| 3 · plan-following action head | frozen VLM; the released DiT with LoRA (rank 16), its action decoder, and a plan encoder | 30.75M | `s10` |
| 4 · probe | frozen VLM: instruction–cell similarities | nothing | E1 |

- **Frozen VLM = diagnostic scale:** one A6000 (24 GB peak when training s10). The main comparison fine-tunes the whole VLA (decided 12 Nov; stage 3); ablations stay light.
- **Starts as the released policy:** the plan encoder's last layer is zero-initialised, so before training our model gives the released checkpoint's actions (checked to bf16 rounding).
- **Closed loop:** the LIBERO-PRO simulator (CPU) sends the two images, the instruction and the gripper position to a policy server on the GPU (starVLA's websocket server plus the plan, `scripts/vla/serve_plan_policy.py`), executes the 8 returned actions, and asks again.

### 2.4 How the affordance map is used now

The map is the output of the map head **g1**: for each of the 64 cells of the agent view (one per VLM image token, 28 × 28 px) a score for "the contact of the next gripper event is here", plus the position inside the cell and the depth; for the whole image, the event (grasp / release / none) and the approach direction.

**From the map to the actions [implemented]:**

1. **Pick** the highest-scoring cell; its offset and depth give a pixel and a distance along the camera ray.
2. **Lift to 3D** with the fixed camera's calibration → the contact point in the world.
3. **Plan vector**, 10 numbers relative to the gripper: reach / 10 cm, tanh(reach / 2 cm), approach direction, event.
4. **Hold:** the contact stays fixed in the world until the gripper has carried out the event; each query only recomputes the reach. The map is read once per phase (grasp, then release).
5. **Into the action head:** the plan encoder turns the 10 numbers into one 2560-vector, added to each action token of the DiT.

- **The action head never sees the map**, only the decoded point; the other 63 cells are dropped. The map's job is *where*: under position perturbation it finds the object in the image instead of remembering the training layout (section 1.1).
- **Two heads, trained separately, both on ORACLE plans:** the map head on the ORACLE contact (object position + mean demonstrated grasp offset) projected into its cell; the action head on true and counterfactual plans. They meet only at test time.
- **The L 3D maps are not used by the model yet.** They become the labels of E2 / E4: scores on every object, a direction field for the gripper orientation, and the semantic gate picking the object (section 3).

Below: g1's map on first frames of LIBERO-PRO (released runs). Colour = softmax over the 64 cells (transparent = near 0); white box = the best cell, whose point is sent to the action head (its share in the title); orange = cells showing the named object. From left: position perturbation (1.6 cm from the true contact); task perturbation (0.9 cm); tomato sauce, 2.5 cm off (the E0 failure); Spatial, never trained on: it picks another object, not the named bowl.
""")
    code(f"""
PROBE = RUNS / "semantic_probe" / "e1"
K.map_panels(PROBE / "probe.npz", {MAP_FRAMES!r});
""")
