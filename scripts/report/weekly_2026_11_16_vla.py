"""Section 2 of the week-2 review: what the VLA is and which parts are the open-source starVLA model, the links
between it and our parts, the data, how we use the model, and how the affordance map is used today. Facts checked on
23 Nov 2026 against the code (ours and starVLA's QwenPI.py), the released checkpoint's config.yaml and weights, the
LeRobot datasets' meta files and the metrics.json of the runs named here.

Imported by build_weekly_2026_11_16.py, which owns the notebook. The overview figure is rendered from
report/media/pipeline_overview/detail/pipeline_v3_layers.html; the map figure reads the E1 probe
(runs/semantic_probe/e1/probe.npz), which kept the g1 map and the agent view of every first frame it scored.
"""

MAP_FRAMES = [("libero_object_swap", "task2_ep0.npz"), ("libero_object_task", "task2_ep0.npz"),
              ("libero_object_swap", "task5_ep0.npz"), ("libero_spatial_task", "task8_ep0.npz")]


def vla_cells(md, code) -> None:
    """Section 2: the model and its links to our parts, data, the four uses, the path from the map to the actions."""
    md(r"""
## 2 · The VLA we build on, the data, and how the affordance map is used

### 2.1 What the VLA is: starVLA QwenPI

A VLA (vision–language–action model) is a vision–language model (VLM) with an action head: the VLM reads the camera images and the instruction, and the action head turns what the VLM computed into robot motion. We use **QwenPI** from the open-source [starVLA](https://github.com/starVLA/starVLA) codebase (MIT licence): the VLM **Qwen3-VL-4B** plus a **flow-matching action head** (a DiT built after GR00T N1.5) that reads the VLM layer by layer, as π0 does.

One query, every 8 control steps:

1. **Encode:** the two camera images (64 tokens each) and the instruction, inside starVLA's prompt, pass Qwen3-VL's 36 layers; the hidden states of all 36 layers are kept.
2. **Generate:** the action head starts from random noise for the next 8 actions (8 × 7 numbers) and refines it in 4 steps. In each step its block *i* reads VLM layer *i* (cross-attention) and predicts how to move the noise towards an action chunk (flow matching).
3. **Execute:** the chunk is de-normalised with the training data's statistics; the robot's controller executes all 8 actions; then the VLA is queried again.

starVLA trained the whole model on LIBERO demonstrations by imitation (flow-matching loss; VLM learning rate 1e-5, action head 1e-4; 100k steps) and released the weights. We keep them and add our parts around them (below). **Gold zone** = starVLA's model with its released weights, used as is. **Outside** = ours: green border trained by us, black no weights, purple dashed planned. Inside the action head, the LoRA and the plan embedding (green text) are ours too. **①–⑥** = the links between the two (section 2.2).
""")
    code('DETAIL = FIGURES / "detail"   # layer-level figures, rendered from pipeline_v3_layers.html\n'
         'Image(filename=str(DETAIL / "v3_detail_1_overview.png"), width=1000)')
    md(r"""
| | |
| --- | --- |
| Checkpoint | `Qwen3-VL-PI-LIBERO-4in1`, step 100,000 (from starVLA's ModelScope mirror, sha256-checked) |
| Size | 7.79B parameters: VLM 4.44B + action head (36 blocks × 2560) 3.35B |
| Trained by starVLA on | the four LIBERO suites together (Spatial, Object, Goal, LIBERO-10): 1,693 demonstrations, LeRobot `no_noops` versions |
| Input | agent view + wrist view (224 × 224) and the instruction; no robot state |
| Output | 8 actions × 7 numbers (Δposition, Δrotation, gripper); robosuite's controller executes them at 20 Hz |

### 2.2 The links between starVLA and our parts

| # | Link | What flows | Where in the code | How to change it |
| --- | --- | --- | --- | --- |
| ① | images → VLM | agent and wrist view, 224 × 224, rotated 180° as in training; request field `image` | `model_view` in `affordance_vla/sim/episode.py` (simulator side) | which views or crops the simulator side sends |
| ② | instruction → VLM | text in the request field `lang`; starVLA puts it into its prompt | `scripts/eval/rollout_libero_logged.py` | `--instruction`; LIBERO-PRO `_task` and `_lan` change it for you |
| ③ | robot state → action head | nothing: the field `state` is not sent, and starVLA's training config leaves it off | — | only together with a head retrained on the state |
| ④ | VLM → our map head | hidden states of all 36 layers, 2560 numbers per token; the agent view's 64 image tokens sit at known positions | `encode` in `scripts/vla/probe_vla_demo.py`; `image_token_cells` in `affordance_vla/heads/plan_grid.py` | which layers and tokens our heads read (g1 learns a mix of all 36; pre-merge 16 × 16 features planned) |
| ⑤ | our plan → action head | 10 numbers per query, sent by the simulator side (`plan`) or predicted on the server; the plan encoder adds them to the 8 action tokens | `scripts/vla/serve_plan_policy.py`; `affordance_vla/heads/plan_policy.py` | `PLAN=none / oracle / zero / predicted`, `PREDICTOR=g1`, `HOLD=1` in `scripts/eval/run_libero_pro_gpu.sh`; `--deep-plan` in training adds the plan to every block |
| ⑥ | action head → robot | 8 × 7 numbers, de-normalised by starVLA's server; gripper 1 = open, 0 = close, turned into the simulator's −1 / +1 | `env_gripper_command` in `affordance_vla/sim/episode.py` | nothing to set: the chunk length 8 is fixed by the checkpoint |

- **Which weights run:** `ADAPTER` empty → starVLA's own server and weights, nothing of ours (the baseline). `ADAPTER=<run>` → the same weights plus our trained parameters from `adapter.pt`: LoRA (rank 16) on attention and feed-forward of all 36 DiT blocks, the action decoder, the plan encoder.
- **Training our parameters:** `scripts/vla/train_plan_following.py` (action head, VLM and DiT weights frozen) and `scripts/vla/train_plan_predictor.py` (map head, VLM frozen); commands in `EXPERIMENTS.md`.
- **One server flag must stay:** `use_canonical_forward=false`; the checkpoint predates starVLA's newer DiT forward pass.

### 2.3 Data

| Data | What it is | What we use it for |
| --- | --- | --- |
| **LIBERO** (training) | simulated tabletop tasks, 50 human teleoperated demonstrations each; suites Spatial, Object, Goal, LIBERO-10, LIBERO-90 | We read only states and actions, replay them in the simulator and record images, gripper and object poses, contacts. **LIBERO-Object** → `dataset10`: 454 demonstrations, 56,852 training frames (5 per task held out) — trains both of our heads. **Spatial** → `dataset_spatial` (E2). Object and LIBERO-10 replays with object poses → the L maps (section 4.3). |
| **LIBERO-PRO** (test only) | the same tasks perturbed: `_swap` objects trade places, `_task` another object is named (also `_object`, `_lan`: appearance, wording) | Zero-shot tests, never trained on: Object, Spatial, LIBERO-10 × swap / task. 20 starts per task (quick) or 50 × 3 seeds (the papers' protocol). |
| LIBERO-90, LIBERO-Plus | more scenes (the red mug); another perturbation benchmark | planned: category-level mug map and G4 (stage 1); task selection (stage 2) |

The released checkpoint was trained on the same LIBERO-Object demonstrations. We add no demonstrations: what we add is the plan pathway and its counterfactual training (section 3.3).

### 2.4 How we use the VLA

Each query runs **one** forward pass of the VLM over the two images and the instruction; our heads read its hidden states.

| Use | What runs | Trained by us | Run |
| --- | --- | --- | --- |
| 1 · baseline | the released checkpoint, unchanged | nothing | `released` |
| 2 · map head | frozen VLM; a small head reads all 36 LLM layers and scores the 8 × 8 image cells | 7.89M | `g1` |
| 3 · plan-following action head | frozen VLM; the released DiT with LoRA, its action decoder, and a plan encoder | 30.75M | `s10` |
| 4 · probe | frozen VLM: instruction–cell similarities | nothing | E1 |

- **Frozen VLM = diagnostic scale:** one A6000 (24 GB peak when training s10). The main comparison fine-tunes the whole VLA (decided 12 Nov; stage 3); ablations stay light.
- **Starts as the released policy:** the plan encoder's last layer is zero-initialised, so before training our model gives the released checkpoint's actions (checked to bf16 rounding).

### 2.5 How the affordance map is used now

The map is the output of the map head **g1**: for each of the 64 cells of the agent view (one per VLM image token, 28 × 28 px) a score for "the contact of the next gripper event is here", plus the position inside the cell and the depth; for the whole image, the event (grasp / release / none) and the approach direction. In the figures of section 3 it is the orange box "affordance map", drawn on real frames.

| | Now: g1 [implemented] | v3 [planned, E2 / E4] |
| --- | --- | --- |
| What a cell's score means | the next contact is in this cell (softmax over the 64 cells: one target) | how well this part of each object can be grasped (one score per object, several objects high) |
| Also per cell | in-cell offset, depth | in-cell offset, depth, closing direction (cos 2θ, sin 2θ) |
| Which object | learnt inside the head from the instruction | the semantic gate multiplies instruction–cell similarity onto the scores |
| Training label | one cell: the ORACLE contact projected into the image | the L2 / L2c 3D map of each object's category, projected onto every object |
| What reaches the action head | best cell → 3D contact → 10-number plan | best gated cell + its direction → contact and gripper orientation (16 numbers) |

**From the map to the actions [implemented]:**

1. **Pick** the highest-scoring cell; its offset and depth give a pixel and a distance along the camera ray.
2. **Lift to 3D** with the fixed camera's calibration → the contact point in the world.
3. **Plan vector**, 10 numbers relative to the gripper: reach / 10 cm, tanh(reach / 2 cm), approach direction, event.
4. **Hold:** the contact stays fixed in the world until the gripper has carried out the event; each query only recomputes the reach. The map is read once per phase (grasp, then release).
5. **Into the action head** (link ⑤): the plan encoder turns the 10 numbers into one 2560-vector, added to each action token of the DiT.

- **The action head never sees the map**, only the decoded point; the other 63 cells are dropped. The map's job is *where*: under position perturbation it finds the object in the image instead of remembering the training layout (section 1.1).
- **Two heads, trained separately, both on ORACLE plans:** the map head on the ORACLE contact (object position + mean demonstrated grasp offset) projected into its cell; the action head on true and counterfactual plans. They meet only at test time.
- **The L 3D maps are not used by the model yet.** They become the labels of E2 / E4: scores on every object, a direction field for the gripper orientation, and the semantic gate picking the object (section 3).

Below: g1's map on first frames of LIBERO-PRO (released runs). Colour = softmax over the 64 cells (transparent = near 0); white box = the best cell, whose point is sent to the action head (its share in the title); orange = cells showing the named object. From left: position perturbation (1.6 cm from the true contact); task perturbation (0.9 cm); tomato sauce, 2.5 cm off (the E0 failure); Spatial, never trained on: it picks another object, not the named bowl.
""")
    code(f"""
PROBE = RUNS / "semantic_probe" / "e1"
K.map_panels(PROBE / "probe.npz", {MAP_FRAMES!r});
""")
