"""Build report/2026-11-11_affordance_pipeline_walkthrough.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/):
    python ../scripts/report/build_affordance_pipeline_walkthrough.py 2026-11-11_affordance_pipeline_walkthrough.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-11_affordance_pipeline_walkthrough.ipynb
Needs rollouts logged with depth: run_plan_following_gpu.sh with RUN=s2_actions STAGES=eval DEPTH=1
SUFFIX=_depth EPISODES=1 (through the GPU queue), the training set runs/plan_following/dataset, and the
probe of the pipeline demo (runs/vla_pipeline_demo/probe). The closing text is in
conclusions_affordance_pipeline.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_affordance_pipeline.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# The pipeline with an affordance map, one step at a time: what goes in, what comes out
**VLA direction** · LIBERO-Object · released starVLA checkpoint with the plan input of 11 Nov 2026 · a walk-through for discussion, not a result

- **What this is:** one closed-loop rollout, stopped at one moment. That moment is followed through the nine steps of the pipeline, with the input and the output of every step drawn or printed. The last part shows the whole episode.
- **Real:** the images, the depth image and the camera, the VLM and the action head, the plan input sent with each query, the action chunks, the motion of the robot.
- **ORACLE:** the affordance map, the planned contact, the approach direction, the kind of event, and the episode state. They are computed from the simulator. No model predicts them yet.
- **How the rollout was driven:** by the ORACLE contact (object position plus the average grasp point of the demonstrations). The affordance map is drawn next to it; it did not steer the robot.

| step | input | output | in this notebook |
| --- | --- | --- | --- |
| 1 · observation | — | two images, depth, camera, instruction, gripper position | real |
| 2 · VLM | images, instruction | hidden states | real (sizes measured on 10 Nov) |
| 3 · affordance head | hidden states | affordance map over the image | **ORACLE** |
| 4 · contact from the map | map, depth, camera | a 3D point | geometry real, map ORACLE |
| 5 · plan head | hidden states | approach direction, grasp or release | **ORACLE** |
| 6 · plan encoding | contact relative to the gripper, approach, event | ten numbers, then one vector | real |
| 7 · action head | hidden states, plan vector | action chunk, 8 steps × 7 numbers | real |
| 8 · controller | action chunk | motion of the robot | real |
| 9 · episode state | map, depth, camera | one slot per object, fed back into step 2 | **ORACLE**, rule-based, not connected |
""")

code("""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.sim.rollout_log import load_rollout_log
from report import affordance_pipeline_helpers as A
from report import vla_pipeline_demo_helpers as H
from report import vla_pipeline_demo_labels as L

set_matplotlib_formats("jpeg")
plt.rcParams.update({"figure.dpi": 90})
pd.set_option("display.max_colwidth", 140)

RUNS = ROOT / "runs" / "plan_following"
ROLLOUTS = RUNS / "s2_actions" / "rollouts_oracle_depth"   # the policy trained with moved plans, logged with depth
ROLLOUT, STEP = "ask_tomato_sauce_ep1", 24                    # the rollout and the moment that are followed
stats = json.loads((RUNS / "dataset" / "stats.json").read_text())
SCALE, CONTAINER = stats["motion_scale_m_per_unit"], stats["container"]
probe = H.load_probe(ROOT / "runs" / "vla_pipeline_demo" / "probe")  # sizes of the same frozen VLM, 10 Nov

log = load_rollout_log(ROLLOUTS / f"{ROLLOUT}.npz")
plans = A.load_plans(log)
view = A.view_at(log, plans, STEP, CONTAINER)
print(f"instruction: “{log.meta['instruction']}”   scene of the task: “{log.meta['task']}”")
print(f"{log.steps} steps, {len(log.chunk_steps)} queries; the {H.pretty(log.target)} "
      f"{'ended in the basket' if log.delivered else 'did not end in the basket'}")
print(f"followed here: step {view.step} (query {view.query}); plan in force: {A.describe(view)}")
""")

md("""
## 1 · Observation
**This step:** what the simulator hands over before a query: the two camera images the policy sees, the depth image and the camera matrices of the agent view, the instruction, and where the gripper is.

**Status:** real. The last panel uses simulator positions and is only there to read the scene; the policy never sees it. Depth and camera are not inputs of the VLM. They are needed later, to turn a place in the image into a point in space.
""")

code("""
display(A.observation_table(log, view))
fig = A.observation_figure(log, view)
plt.show()
""")

md("""
## 2 · VLM
**This step:** the two images and the instruction go through the vision-language model. It returns hidden states: one vector per token, at every layer. Everything downstream reads these.

**Status:** real. The sizes were measured on 10 Nov; the VLM has been frozen since, so they hold here. Each image is resized to 224 × 224 and becomes 64 tokens, drawn below as an 8 × 8 grid: a cell is the finest position the VLM's own tokens can name.
""")

code("""
display(A.vlm_table(probe, log))
fig = A.token_grid_figure(log, view, probe.facts["tokens_image"] // 2)
plt.show()
""")

md("""
## 3 · Affordance head → affordance map
**This step:** from the hidden states, mark in the image where the gripper should act on the object the instruction is about: for a grasp, the part of the object to hold; for a release, the place to put it.

**Status:** **ORACLE.** No such head exists yet. The map below is built from the simulator and is one possible definition of the label:
- take the pixels that show the object (simulator segmentation);
- lift each pixel to a 3D point with the depth image;
- a pixel scores high when its point is close to where the demonstrations hold this object (a Gaussian, 3 cm wide). For a release the distance is measured in the ground plane to the release point over the basket (4 cm wide).
""")

code("""
fig = A.map_figure(log, view)
plt.show()
print(A.object_size_text(log, view, probe.facts["tokens_image"] // 2))
""")

md("""
## 4 · From the map to a 3D point
**This step:** a map is a picture; the robot needs a point in space. Every pixel of the map is lifted to 3D with the depth image and the camera matrices, and the points are averaged with the map as weights.

**Status:** the geometry is real; the map is ORACLE. The yellow dot is the map's point. The white plus is the planned contact used in the rollout: where the gripper's centre should be at the grasp. The two differ, because a camera sees the surface of the object and the gripper's centre ends up between the fingers, inside or above it. Both are ORACLE here, so the distance between them is not the error of a model: it is how far the two definitions lie apart.
""")

code("""
fig = A.contact_figure(log, view)
plt.show()
horizontal, above = A.gap(view)
print("the map's point (m):      ", np.round(view.map_point, 3))
print("planned contact (m):      ", np.round(view.contact, 3))
print(f"the planned contact is {horizontal:.1f} cm away horizontally and {above:+.1f} cm in height")
""")

md("""
## 5 · Plan head → approach direction and kind of event
**This step:** besides the contact, the plan says from which direction the gripper should arrive and whether the next event is a grasp or a release.

**Status:** **ORACLE.** The direction is the average direction of the demonstrations' last motion before the event, for this object. In the method these come from a plan head on the VLM.
""")

code("""
fig = A.approach_figure(log, view)
plt.show()
print("approach direction (unit vector, world):", np.round(view.approach, 2), "  event:", view.event)
""")

md("""
## 6 · Plan encoding
**This step:** the plan becomes ten numbers: the vector from the gripper to the contact (in units of 10 cm, and again squashed so that the last centimetres count), the approach direction, and +1 for a grasp or −1 for a release. A small trained network turns them into one vector of 2,560 numbers that is added to each of the eight action tokens.

**Status:** real. The first column is what was sent in this rollout. The second column shows what the numbers would be if the contact were the map's point of step 4: the difference is the gap seen there.
""")

code("""
display(A.plan_table(log, view))
""")

md("""
## 7 · Action head → action chunk
**This step:** the action head reads the hidden states and the plan vector and returns the next eight steps. Each step has seven numbers: how far to move the gripper in x, y and z, how much to turn it, and whether it is open (1) or closed (0).

**Status:** real; this is the output of the trained model for this query. Summing the translations and multiplying by the measured motion per unit gives the path the chunk asks for.
""")

code("""
display(A.chunk_table(view).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"))
fig = A.chunk_figure(log, view, SCALE)
plt.show()
""")

md("""
## 8 · Controller → motion of the robot
**This step:** the simulator's controller turns each of the eight commands into joint torques. It only follows the small step it is given; it does not plan a path and does not avoid obstacles. After the eight steps (0.4 s) the policy is queried again with a new image.

**Status:** real.
""")

code("""
fig = A.execution_figure(log, view, SCALE)
plt.show()
display(A.execution_table(log, view, SCALE))
""")

md("""
## 9 · Episode state
**This step:** after the chunk, update one slot per object: where it was last seen, whether it is visible, hidden, or in the gripper, and for how long it has not been seen. In the method these slots become a few tokens for the VLM at the next query.

**Status:** **ORACLE**, rule-based, and not connected to the model. The slots are filled from simulator positions and visibility. In this scene almost nothing is ever hidden, so the state has little to do.
""")

code("""
history = L.episode_state_history(log)
display(A.slot_table(history, min(view.query_step + 8, log.steps)))
fig = L.state_overlay_figure(log, history, L.state_key_steps(log))
plt.show()
""")

md("""
## The whole episode
The same quantities at six queries, drawn into the agent view, then as curves over all queries, then as a video. In the video the contact stays where the latest query put it for the eight steps of a chunk, while the map follows the current image. When the object is in the gripper the plan switches to the release, and the map moves from the object to the basket. At the very last query the plan flips back to the grasp: the object has been let go but does not yet count as lying in the basket.
""")

code("""
views = [A.view_at(log, plans, int(step), CONTAINER) for step in log.chunk_steps]
picks = [views[k] for k in np.unique(np.linspace(0, len(views) - 1, 6).astype(int))]
fig = A.film_strip(log, picks, SCALE)
plt.show()
fig = A.timeline_figure(log, views)
plt.show()
display(A.gap_table(views))
""")

code("""
CLIP = ROOT / "report" / "media" / "affordance_pipeline_walkthrough.mp4"  # tracked in git
A.render_overlay_video(log, plans, CONTAINER, SCALE, CLIP)
Video(str(CLIP), embed=True, width=560)
""")

md("""
**Another condition, same policy:** the task's own instruction, with the salad dressing moved 9 cm from its place. The map follows the bottle.
""")

code("""
other = load_rollout_log(ROLLOUTS / "shift_9cm_ep1.npz")
other_plans = A.load_plans(other)
other_views = [A.view_at(other, other_plans, int(step), CONTAINER) for step in other.chunk_steps]
fig = A.film_strip(other, [other_views[k] for k in np.unique(np.linspace(0, len(other_views) - 1, 6).astype(int))], SCALE)
plt.show()
print(f"instruction: “{other.meta['instruction']}”; shifts: {other.meta['shifts']}; "
      f"the {H.pretty(other.target)} {'ended in the basket' if other.delivered else 'did not end in the basket'}")
display(A.gap_table(other_views))
""")

md("""
**All 13 rollouts logged with depth** (same policy: 5 standard starts, 3 with the target moved, 5 that name another object). The gap columns are medians over the queries of a rollout, in cm: how far the planned contact lies from the map's point, both ORACLE.
""")

code("""
display(A.gap_overview(ROLLOUTS, CONTAINER))
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
