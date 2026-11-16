## What to take from this

**What the walk-through shows**
- **Where the affordance map sits.** It is an image-sized output between the VLM and the plan. It says where on the object, or where in the basket, the gripper should act. It reaches the robot only through the contact point of the plan.
- **A map gives a point on a surface; the plan needs the centre of the gripper.** For grasps the two are 0.9–2.1 cm apart horizontally and within 1 cm in height (13 rollouts). For releases the map's point lies on the basket and the release point 8.4–9.3 cm above it. Something has to add that offset: a plan head that predicts it, or an action head trained on map points directly.
- **The plan changes inside an episode.** Once the object is in the gripper the plan switches from the grasp to the release, and the map moves from the object to the basket. Something has to decide that switch. Here the simulator's contact flags do.
- **What the action head adds.** Everything between the gripper and the contact: the path of the chunk, the descent, the moment the gripper closes. The controller only follows each small step; in the chunk shown the gripper moved 6.9 cm where the chunk asked for 7.9 cm.
- **How coarse the VLM's own tokens are.** Each image becomes 8 × 8 tokens, so one token covers 32 × 32 pixels. The tomato sauce is 18 × 25 pixels: smaller than one token. A map that marks part of such an object needs a head that works at a finer resolution than the tokens.

**What is not real here**
- **The map, the contact, the approach direction, the switch and the episode state are ORACLE.** The rollout was driven by the ORACLE contact. Driving it with the map's point (1–2 cm off for grasps) has not been run.
- **The definition of the map is my choice for this picture.** A Gaussian, 3 cm wide, around the place where the demonstrations hold the object. It is not a decision.
- **The size is small.** One policy and 13 rollouts logged with depth, of which 9 delivered the named object; two are shown in detail.

**Questions this raises**
1. **What should the map mark?** The part of the object the fingers hold (as here), the whole target object, or what your earlier affordance work produces?
2. **Where does the 3D contact come from?** The map's point plus a predicted offset, or a contact regressed directly with the map as a side output?
3. **Who decides between grasp and release,** and when the plan switches?
4. **Is 8 × 8 tokens per image enough** to place a map on small objects, or does the affordance head need finer image features?
