## Conclusions

- **E0 does not pass on LIBERO-Object.** We compared each rollout's first close command near the named object with
  the nearest demonstrated grasp of that object.
  - Failed first grasps are only 1.4 degrees further from that demonstration than held ones with map plans kept,
    and 1.6 degrees with the oracle plan. The criterion asks for 10 degrees.
  - The closing direction about the vertical is within 0.5 degrees of a demonstration in every group (median).
  - The tilt differs by 2 to 3 degrees.
- **The demonstrations hardly vary either.** Every object is grasped from above:
  - the median tilt is 4 to 12 degrees;
  - the closing direction varies by 7 to 24 degrees between demonstrations of the same object (90th percentile).

  On this suite there is about one way to hold each object, and the policies reproduce it.
- **What differs is where the gripper closes.** Median distance from the gripper to the object's centre at the close
  command:

  | | held, task solved | not held |
  | --- | --- | --- |
  | map plans kept | 2.0 cm | 4.7 cm |
  | oracle plan | 1.8 cm | 4.5 cm |

  In the wrist images of failed closes, the fingers shut above or beside the object, not around it. Two objects
  stand out:
  - **ketchup:** in the two examples shown, the bottle is already lying on its side;
  - **tomato sauce:** about 10 degrees in every group, because its demonstrations vary most.
- **Consequence, by the rule agreed on 19 Nov (Pipeline v3, section 8):**
  - The orientation branch pauses on LIBERO-Object.
  - E3 (orientation upper bound) and E5 (orientation from the map) move to LIBERO-10, where mugs and moka pots have
    handles. The range of the "turn the orientation" counterfactual will be set there; on this suite ±45 degrees
    would lie far outside anything the demonstrations or the failures show.
  - On LIBERO-Object the position part of the interaction map comes first: interaction score, depth and finer
    features, then E1 and E2.
- **Not shown:**
  - Angles are in world coordinates, because object orientations are not logged yet. Step L adds them.
  - The closing distance is measured to the object's centre, so it includes the object's size. It is compared only
    between outcomes of the same objects.
