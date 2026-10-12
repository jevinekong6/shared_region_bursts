"""
Group frames into scenes that share a region, align every scene to its reference
frame with a homography and crop it to the region all of its frames cover.

  1. Pick references: a frame that covers no existing reference becomes a new one
     (or pass --refs to name the reference frames yourself).
  2. Fill scenes: a frame joins every scene whose reference it covers. "Covers" is
     the fraction of the *reference* the frame overlaps, so a close-up of a small
     part of a wide reference stays out of the wide scene (it would shrink its crop)
     and becomes its own scene instead.
  3. Per scene, the reference plus the --burst_size - 1 best-covering frames are
     warped into the reference view and cropped to the largest shared rectangle.

--matcher is any vismatch model name: orb-nn / sift-nn (classical baselines), xfeat,
aliked-lightglue, superpoint-lightglue, roma, ...

Outputs in --out_dir:
  scene_XXX/NNN_<original name>.png   aligned, cropped, same-size 8-bit frames. The
                                      reference is the middle file (index n // 2).
  groups.csv                          one row per frame per scene, plus unused frames

Usage:
    python group_bursts.py --in_dir frames --out_dir scenes
    python group_bursts.py --in_dir frames --out_dir scenes_orb --matcher orb-nn
    python group_bursts.py --in_dir frames --out_dir scenes --refs frame_000120.png
    python group_bursts.py --self_test [--matcher xfeat]
"""

import argparse
import csv
import os
import tempfile

import cv2
import numpy as np
import torch
from vismatch import get_matcher

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
BORDER = 5      # px kept away from warped frame borders
MIN_CROP = 64   # scenes whose shared crop is smaller than this are dropped


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def load_small(path, size):
    """-> (CxHxW RGB float tensor in [0, 1] with longest side <= size, 3x3 scale full-res -> small)."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit('Could not read image: {}'.format(path))
    h, w = img.shape[:2]
    s = min(1.0, size / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    return t, np.diag([img.shape[1] / w, img.shape[0] / h, 1.0])


######################################### Matching ###########################################################

def homography_is_sane(H, max_scale_change=4.0, max_perspective=2e-3):
    """Rejects degenerate fits: flips, huge scale changes, extreme perspective."""
    A = H[:2, :2] / H[2, 2]
    det = np.linalg.det(A)
    if det <= 0 or not (1 / max_scale_change ** 2 < det < max_scale_change ** 2):
        return False
    return np.abs(H[2, :2] / H[2, 2]).max() < max_perspective


def cover(matcher, src, ref, args):
    """src, ref = load_small() results. Returns (H full-res src -> ref, inliers, fraction of ref covered) or None."""
    res = matcher(src[0], ref[0])
    H, n = res['H'], int(res['num_inliers'])
    if H is None or n < args.min_inliers:
        return None
    H = np.asarray(H, dtype=np.float64)
    if not homography_is_sane(H):
        return None
    (hs, ws), (hr, wr) = src[0].shape[1:], ref[0].shape[1:]
    overlap = float(cv2.warpPerspective(np.ones((hs, ws), np.uint8), H, (wr, hr), flags=cv2.INTER_NEAREST).mean())
    if overlap < args.min_overlap:
        return None
    H = np.linalg.inv(ref[1]) @ H @ src[1]
    return H / H[2, 2], n, overlap


def group(paths, matcher, args):
    """Returns (sorted reference indices, hits) with hits[(frame, ref)] = cover() result."""
    # ponytail: ~frames x refs matcher calls and the first frame seen is the reference. For many thousands
    # of frames, shortlist candidates with a global descriptor (MegaLoc, DINOv2-SALAD) before matching.
    names = [os.path.basename(p) for p in paths]
    refs, hits = {}, {}
    for r in args.refs or []:
        if os.path.basename(r) not in names:
            raise SystemExit('--refs: {} is not in {}'.format(r, args.in_dir))
        i = names.index(os.path.basename(r))
        refs[i] = load_small(paths[i], args.match_size)

    # the second pass matches frames against references that were created after them
    for final in (False, True):
        for i, path in enumerate(paths):
            todo = [r for r in refs if r != i and (i, r) not in hits]
            if not todo and (final or args.refs or i in refs):
                continue
            src = load_small(path, args.match_size)
            for r in todo:
                hits[(i, r)] = cover(matcher, src, refs[r], args)
            if not final and not args.refs and not any(hits.get((i, r)) for r in refs):
                refs[i] = src
            if not final:
                print('  [{}/{}] {} ({} references)'.format(i + 1, len(paths), names[i], len(refs)))
    return sorted(refs), hits


######################################### Shared crop ########################################################

def translate_mat(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def keep_subpixel_shift(H, cx, cy):
    """Puts back the sub-pixel part of H's local translation at (cx, cy). A pure translation then warps as an
    integer shift (pixels copied, not interpolated) and the frame keeps its real +/- 0.5 px offset to the
    reference. Rotation/scale/perspective are still removed."""
    c = np.array([cx, cy, 1.0])
    p = np.linalg.solve(H, c)
    t = c[:2] - p[:2] / p[2]  # local translation src -> ref at (cx, cy)
    return translate_mat(*(np.round(t) - t)) @ H


def largest_rectangle(mask):
    """Largest axis-aligned all-True rectangle in a 2D bool mask -> (x, y, w, h)."""
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int64)
    best = (0, 0, 0, 0, 0)  # area, x, y, w, h
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []  # column indices with increasing heights
        for c in range(cols + 1):
            cur = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= cur:
                idx = stack.pop()
                hgt = heights[idx]
                left = stack[-1] + 1 if stack else 0
                area = hgt * (c - left)
                if area > best[0]:
                    best = (area, left, r - hgt + 1, c - left, hgt)
            stack.append(c)
    return best[1:]


def shared_crop(masks, work_size=800):
    """Largest rectangle covered by every mask (found on a downscaled mask) -> (x, y, w, h)."""
    inter = np.logical_and.reduce(masks)
    H, W = inter.shape
    s = min(1.0, work_size / max(H, W))
    small = cv2.resize(inter.astype(np.uint8), (max(1, int(W * s)), max(1, int(H * s))),
                       interpolation=cv2.INTER_AREA) == 1  # pixel is in only if fully covered
    x, y, w, h = largest_rectangle(small)
    # back to full resolution, staying conservatively inside
    x0, y0 = int(np.ceil(x / s)), int(np.ceil(y / s))
    return x0, y0, int(np.floor((x + w) / s)) - x0, int(np.floor((y + h) / s)) - y0


def align_scene(frames):
    """frames = [(image, H full-res image -> reference)], with H = None for the reference itself.
    Returns the warped, cropped frames in the same order, or None if the shared crop is too small."""
    rh, rw = next(img.shape[:2] for img, H in frames if H is None)
    frames = [(img, np.eye(3) if H is None else H) for img, H in frames]
    k = np.ones((2 * BORDER + 1, 2 * BORDER + 1), np.uint8)
    masks = []
    for img, H in frames:
        m = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), H, (rw, rh), flags=cv2.INTER_NEAREST)
        masks.append(cv2.erode(m, k, borderValue=0) > 0)
    x0, y0, cw, ch = shared_crop(masks)
    if min(cw, ch) < MIN_CROP:
        return None
    T = translate_mat(-x0, -y0)
    # ponytail: H is only as accurate as the matcher at --match_size (roughly 1 px there). For sub-pixel
    # registration raise --match_size or add an ECC refinement step here.
    return [cv2.warpPerspective(img, T @ keep_subpixel_shift(H, x0 + cw / 2.0, y0 + ch / 2.0), (cw, ch),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for img, H in frames]


######################################### Main ###############################################################

def run(args):
    files = list_images(args.in_dir)
    if not files:
        raise SystemExit('No images found in {}'.format(args.in_dir))
    paths = [os.path.join(args.in_dir, f) for f in files]

    matcher = get_matcher(args.matcher, device='cuda' if torch.cuda.is_available() else 'cpu')
    print('Matching {} frames with {}'.format(len(files), args.matcher))
    refs, hits = group(paths, matcher, args)

    rows, used = [], set()
    for n, r in enumerate(refs):
        scene = 'scene_{:03d}'.format(n)
        members = sorted(((i, h) for (i, rr), h in hits.items() if rr == r and h), key=lambda m: -m[1][2])
        members = members[:args.burst_size - 1]
        half = (len(members) + 1) // 2  # run_inference-style bursts take the middle frame as the reference
        order = members[:half] + [(r, (None, '', 1.0))] + members[half:]

        status = 'ok'
        if len(order) < args.min_frames:
            status = 'dropped: only {} frames'.format(len(order))
        elif not args.dry_run:
            crops = align_scene([(cv2.imread(paths[i]), h[0]) for i, h in order])
            if crops is None:
                status = 'dropped: shared crop too small'
            else:
                os.makedirs(os.path.join(args.out_dir, scene), exist_ok=True)
                for seq, ((i, _), crop) in enumerate(zip(order, crops)):
                    out_name = '{:03d}_{}.png'.format(seq, os.path.splitext(files[i])[0])
                    cv2.imwrite(os.path.join(args.out_dir, scene, out_name), crop)
                status = 'ok: {}x{}'.format(crops[0].shape[1], crops[0].shape[0])

        print('  {}: reference {}, {} frames, {}'.format(scene, files[r], len(order), status))
        for seq, (i, (_, inliers, overlap)) in enumerate(order):
            if status.startswith('ok'):
                used.add(i)
            rows.append(dict(file=files[i], scene=scene, seq=seq, role='reference' if i == r else 'member',
                             inliers=inliers, overlap=round(overlap, 4), status=status, matcher=args.matcher))
    rows += [dict(file=f, scene='', seq='', role='unused', inliers='', overlap='', status='', matcher=args.matcher)
             for i, f in enumerate(files) if i not in used]

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'groups.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Done. Scenes in {} (see groups.csv)'.format(args.out_dir))
    return rows


def self_test(args):
    """Two synthetic scenes, three shifted views each -> two aligned scenes of three frames."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        args.in_dir, args.out_dir = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
        args.refs, args.burst_size, args.min_frames, args.dry_run = None, 3, 3, False
        os.makedirs(args.in_dir)
        for s in 'ab':
            canvas = np.full((1200, 1400, 3), 128, np.uint8)
            for _ in range(600):
                x, y = int(rng.integers(0, 1400)), int(rng.integers(0, 1200))
                w, h = (int(v) for v in rng.integers(15, 120, 2))
                cv2.rectangle(canvas, (x, y), (x + w, y + h), [int(c) for c in rng.integers(0, 256, 3)], -1)
            for k, (ox, oy) in enumerate([(100, 100), (130, 117), (85, 140)]):
                cv2.imwrite(os.path.join(args.in_dir, '{}{}.png'.format(s, k)), canvas[oy:oy + 900, ox:ox + 1100])

        rows = run(args)
        scenes = sorted({r['scene'] for r in rows if r['status'].startswith('ok')})
        assert len(scenes) == 2, 'expected 2 scenes, got {}'.format(scenes)
        assert not [r for r in rows if r['role'] == 'unused'], 'some frames were left unused'
        for scene, s in zip(scenes, 'ab'):
            out = list_images(os.path.join(args.out_dir, scene))
            assert len(out) == 3 and out[1] == '001_{}0.png'.format(s), out  # reference in the middle
            imgs = [cv2.imread(os.path.join(args.out_dir, scene, f)).astype(np.float64) for f in out]
            assert imgs[0].shape == imgs[1].shape == imgs[2].shape and imgs[1].shape[0] > 700, imgs[1].shape
            for f, img in zip(out, imgs):
                err = np.abs(img - imgs[1]).mean()  # ~40 when misaligned by a few px
                assert err < 3, '{} is not aligned to the reference (mean abs diff {:.1f})'.format(f, err)
    print('self test passed ({})'.format(args.matcher))


def main():
    parser = argparse.ArgumentParser(description='Group frames into shared-region scenes, align and crop them')
    parser.add_argument('--in_dir', type=str, help='Flat folder of frames')
    parser.add_argument('--out_dir', type=str, help='Output folder')
    parser.add_argument('--refs', nargs='+', default=None,
                        help='File names in --in_dir to use as the scene references (default: found automatically)')
    parser.add_argument('--matcher', default='aliked-lightglue', type=str,
                        help='Any vismatch model: orb-nn, sift-nn, xfeat, aliked-lightglue, roma, ...')
    parser.add_argument('--match_size', default=1024, type=int,
                        help='Longest image side used for matching. The warp is always applied at full resolution.')
    parser.add_argument('--burst_size', default=14, type=int, help='Keep at most this many frames per scene')
    parser.add_argument('--min_frames', default=None, type=int,
                        help='Drop scenes with fewer frames than this (default: --burst_size)')
    parser.add_argument('--min_overlap', default=0.5, type=float,
                        help='A frame joins a scene if it covers at least this fraction of the reference')
    parser.add_argument('--min_inliers', default=40, type=int)
    parser.add_argument('--dry_run', action='store_true', help='Only print the scenes, write nothing')
    parser.add_argument('--self_test', action='store_true', help='Run on synthetic frames and check the result')
    args = parser.parse_args()

    if args.self_test:
        return self_test(args)
    if not args.in_dir or not args.out_dir:
        parser.error('--in_dir and --out_dir are required')
    if args.min_frames is None:
        args.min_frames = args.burst_size
    run(args)


if __name__ == '__main__':
    main()
