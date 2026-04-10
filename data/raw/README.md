# Raw Dataset Layout

This folder stores captured data for gesture collection.

## Locked-in training schema (before preprocessing/training implementation)

- Classes: **9 total** = 8 gesture classes + `idle`.
- One take folder = **one labeled sample**.
- One sample is expected to contain **90 frames**.
- First model input per frame: **upper-body BODY_25 subset x,y only** (15 keypoints × 2) = **30 features**.
- Therefore the target sample tensor shape is **(90, 30)**.

## Capture organization

- OpenPose JSON takes go under:
  - `data/raw/openpose_json/<gesture>/<person>/<session>/<take>/`
- Optional matching RGB video goes under:
  - `data/raw/rgb_video/<gesture>/<person>/<session>/`
- Suggested take naming: `take_001`, `take_002`, etc.
- Keep one person in frame for each take.
- Start and end every take in a neutral pose.

> For the first classifier version, `idle` is a normal label and should be captured under the same folder pattern.
