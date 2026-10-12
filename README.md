# shared-region-bursts

Groups a folder of frames into scenes that share a region, aligns each scene to its
reference frame with a homography and crops it to the area every frame covers.

## Install

Needs Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`pip` installs CPU PyTorch on Windows. For GPU, install a CUDA build of `torch` from
https://pytorch.org first.

## Use

```
python group_bursts.py --in_dir frames --out_dir scenes
python group_bursts.py --in_dir frames --out_dir scenes --dry_run          # print scenes only
python group_bursts.py --in_dir frames --out_dir scenes --refs a.png b.png # pick the references yourself
python group_bursts.py --self_test
```

`--matcher` switches the matching method. Any [vismatch](https://github.com/gmberton/vismatch)
model name works:

| `--matcher` | Method |
|---|---|
| `orb-nn` | ORB baseline |
| `sift-nn` | SIFT baseline |
| `xfeat` | XFeat |
| `aliked-lightglue` | ALIKED + LightGlue (default) |
| `roma` | RoMa (dense, slow, most robust) |

Each model has its own license; check it before commercial use.

## Output

```
scenes/
  scene_000/000_<name>.png ... NNN_<name>.png
  groups.csv
```

- Frames in a scene are the same size, 8-bit PNG, warped into the reference view.
- The reference is the middle file (index `n // 2`); the rest are ordered by how much of it they cover.
- A scene holds at most `--burst_size` frames (default 14) and is dropped below `--min_frames`.
- A frame can appear in more than one scene.
- `groups.csv` lists every frame with its scene, inliers, overlap and the matcher used.

## Limits

- A homography is exact only for a near-planar scene; tall structures show parallax.
- Matching cost grows with frames x scenes.
- Alignment is about as accurate as the matcher at `--match_size`; raise it for finer registration.
