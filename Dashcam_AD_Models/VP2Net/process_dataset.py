"""
process_dataset.py
==================

Generate a *synthetic* ADED-style dataset so that `scripts/train.sh` and
`scripts/test.sh` can run end-to-end without the real data.

The layout produced here is exactly the one `utils/dataset.py` expects:

    <data_root>/
        images/train_val/<video_id>/<frame>.jpg    # RGB traffic frame
        maps/train_val/<video_id>/<frame>.jpg      # grayscale saliency map
        images/test/<video_id>/<frame>.jpg
        maps/test/<video_id>/<frame>.jpg

The exact set of `<video_id>/<frame>.jpg` files is read from the annotation
lists (`lists/aded/untrimmed_16_training.txt` / `untrimmed_16_test.txt`), so
every path referenced by a training/testing sample is guaranteed to exist.

Frames are not random noise: each video id gets a deterministic, seeded
pseudo driving scene (sky gradient, perspective road, lane markings, moving
vehicles) whose motion pattern depends on the class label of that video, and
the saliency map is a Gaussian blob following the salient object. This gives
the pipeline a weak but learnable signal instead of pure noise.

Usage
-----
    python process_dataset.py --config configs/aded/v2pnet.yaml
    python process_dataset.py --config configs/aded/v2pnet.yaml --verify-only
"""

import argparse
import ast
import os
import sys
import zlib
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import config


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

CLASS_NAMES = {0: "DN", 1: "ACP", 2: "WVA", 3: "SRL", 4: "SSS", 5: "ALC"}

# split name -> (images sub dir, maps sub dir) used by utils/dataset.py
SPLIT_DIRS = {
    "train": ("images/train_val", "maps/train_val"),
    "test": ("images/test", "maps/test"),
}


# --------------------------------------------------------------------------- #
# annotation lists -> file manifest
# --------------------------------------------------------------------------- #

def read_list_file(txt_path):
    """Read an ADED annotation list.

    The first line is skipped exactly like `utils/dataset.py` does.
    Returns a list of ``(label_str, [rel_path, ...])`` tuples.
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError('annotation list not found: {}'.format(txt_path))

    entries = []
    with open(txt_path) as f:
        f.readline()  # header / blank line, dropped by the dataset as well
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError) as err:
                raise ValueError('{}:{}: cannot parse line ({})'.format(txt_path, line_no, err))
            entries.append((record['label'], record['sequence']))
    return entries


def build_manifest(entries, num_classes):
    """Collapse the sample list into ``{video_id: {'label': int, 'frames': [str]}}``."""
    manifest = OrderedDict()
    for label_str, sequence in entries:
        label = int(str(label_str)[-1:])
        if not 0 <= label < num_classes:
            raise ValueError('label {} out of range for {} classes'.format(label_str, num_classes))
        for rel_path in sequence:
            parts = rel_path.strip('/').split('/')
            if len(parts) != 2:
                raise ValueError('unexpected sequence path: {}'.format(rel_path))
            video_id, frame_name = parts
            item = manifest.get(video_id)
            if item is None:
                item = {'label': label, 'frames': set()}
                manifest[video_id] = item
            item['frames'].add(frame_name)

    for video_id, item in manifest.items():
        item['frames'] = sorted(item['frames'])
    return manifest


def frame_index(frame_name):
    """'0006.jpg' -> 6."""
    return int(os.path.splitext(frame_name)[0])


# --------------------------------------------------------------------------- #
# synthetic scene
# --------------------------------------------------------------------------- #

def scene_params(split, video_id, label):
    """Deterministic per-video appearance parameters."""
    seed = zlib.crc32('{}/{}'.format(split, video_id).encode('utf-8')) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    n_extra = int(rng.integers(1, 3))
    return {
        'label': label,
        'sky_top': rng.uniform(0.30, 0.70, size=3),
        'sky_bottom': rng.uniform(0.55, 0.95, size=3),
        'ground': rng.uniform(0.18, 0.38, size=3),
        'road': float(rng.uniform(0.22, 0.34)),
        'horizon': float(rng.uniform(0.38, 0.50)),
        'road_half': float(rng.uniform(0.55, 0.85)),
        'lane_phase': float(rng.uniform(0.0, 1.0)),
        'lane_speed': float(rng.uniform(1.6, 3.2)),
        'extra_colors': rng.uniform(0.25, 0.85, size=(n_extra, 3)),
        'extra_x': rng.uniform(-0.32, 0.32, size=n_extra),
        'extra_scale': rng.uniform(0.10, 0.18, size=n_extra),
        'extra_speed': rng.uniform(0.15, 0.55, size=n_extra),
        'hero_color': rng.uniform(0.30, 0.90, size=3),
        'noise': float(rng.uniform(0.010, 0.028)),
        'seed': seed,
    }


def hero_object(label, t, params):
    """Class-conditioned trajectory of the salient object.

    Returns ``(cx, cy, w, h, vp_shift)`` in normalized image coordinates.
    """
    horizon = params['horizon']
    vp_shift = 0.0

    if label == 0:      # DN - normal driving, lead car far ahead, barely moving
        cx = 0.50 + 0.03 * np.sin(2.0 * np.pi * t)
        scale = 0.13 + 0.04 * t
        cy = horizon + 0.55 * scale
    elif label == 1:    # ACP - object crossing the road from left to right
        cx = -0.08 + 1.16 * t
        scale = 0.14 + 0.10 * t
        cy = horizon + 0.30 + 0.10 * np.sin(np.pi * t)
    elif label == 2:    # WVA - vehicle ahead approaching fast (looms in centre)
        cx = 0.50 + 0.05 * np.sin(np.pi * t)
        scale = 0.12 + 0.34 * (t ** 1.4)
        cy = horizon + 0.60 * scale
    elif label == 3:    # SRL - vehicle coming in from the right lane
        cx = 0.94 - 0.34 * t
        scale = 0.13 + 0.16 * t
        cy = horizon + 0.16 + 0.16 * t
    elif label == 4:    # SSS - vehicle sticking to the left side
        cx = 0.06 + 0.30 * t
        scale = 0.13 + 0.16 * t
        cy = horizon + 0.16 + 0.16 * t
    else:               # ALC - ego lane change, whole scene drifts sideways
        direction = 1.0 if (params['seed'] % 2 == 0) else -1.0
        vp_shift = direction * 0.22 * (t ** 1.2)
        cx = 0.50 - 0.6 * vp_shift
        scale = 0.14 + 0.08 * t
        cy = horizon + 0.60 * scale

    w = scale
    h = scale * 0.78
    return float(cx), float(cy), float(w), float(h), float(vp_shift)


def render_frame(t, params, grid):
    """Render one RGB frame and its saliency map.

    ``t`` is the normalized time in [0, 1].
    Returns ``(rgb_uint8[H, W, 3], sal_uint8[H, W])``.
    """
    xx, yy, height, width = grid
    label = params['label']
    horizon = params['horizon']

    _, _, _, _, vp_shift = hero_object(label, t, params)
    vp_x = 0.5 + vp_shift

    # --- sky / ground background -----------------------------------------
    img = np.empty((height, width, 3), dtype=np.float32)
    sky_ramp = np.clip(yy / max(horizon, 1e-6), 0.0, 1.0)[..., None]
    sky = params['sky_top'][None, None, :] * (1.0 - sky_ramp) + params['sky_bottom'][None, None, :] * sky_ramp
    ground = np.broadcast_to(params['ground'][None, None, :], (height, width, 3))
    is_ground = (yy >= horizon)[..., None]
    img[:] = np.where(is_ground, ground, sky)

    # --- perspective road -------------------------------------------------
    depth = np.clip((yy - horizon) / max(1.0 - horizon, 1e-6), 0.0, 1.0)
    half_width = 0.012 + params['road_half'] * (depth ** 1.35)
    on_road = (yy >= horizon) & (np.abs(xx - vp_x) <= half_width)
    road_shade = params['road'] * (0.72 + 0.42 * depth)
    img[on_road] = road_shade[on_road][..., None]

    # --- scrolling lane markings -----------------------------------------
    z = 1.0 / np.maximum(depth, 1e-3)
    dash = ((z * params['lane_speed'] + params['lane_phase'] + 3.0 * t) % 1.0) < 0.45
    offset = np.abs(xx - vp_x)
    centre_line = on_road & dash & (offset < 0.028 * (0.25 + depth))
    edge_line = on_road & (np.abs(offset - half_width) < 0.016 * (0.25 + depth))
    img[centre_line | edge_line] = 0.92

    # --- background vehicles ---------------------------------------------
    for color, x0, scale, speed in zip(params['extra_colors'], params['extra_x'],
                                       params['extra_scale'], params['extra_speed']):
        cx = vp_x + x0 * (0.35 + 0.65 * t)
        s = scale * (0.55 + speed * t)
        cy = horizon + 0.55 * s
        _paint_box(img, xx, yy, cx, cy, s, s * 0.75, color)

    # --- salient (hero) object -------------------------------------------
    cx, cy, w, h, _ = hero_object(label, t, params)
    _paint_box(img, xx, yy, cx, cy, w, h, params['hero_color'])

    # --- sensor noise -----------------------------------------------------
    rng = np.random.default_rng((params['seed'] + int(t * 100003)) & 0xFFFFFFFF)
    img += rng.normal(0.0, params['noise'], size=img.shape).astype(np.float32)
    np.clip(img, 0.0, 1.0, out=img)

    # --- saliency map: blob on the hero + weak blob at the vanishing point -
    sigma_x = max(w * 0.62, 0.035)
    sigma_y = max(h * 0.62, 0.035)
    sal = np.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x ** 2) + ((yy - cy) ** 2) / (2.0 * sigma_y ** 2)))
    sal += 0.35 * np.exp(-(((xx - vp_x) ** 2) / (2.0 * 0.075 ** 2) + ((yy - horizon) ** 2) / (2.0 * 0.055 ** 2)))
    sal /= max(float(sal.max()), 1e-6)

    return (img * 255.0).astype(np.uint8), (sal * 255.0).astype(np.uint8)


def _paint_box(img, xx, yy, cx, cy, w, h, color):
    """Draw a simple vehicle-like box (body + darker window band)."""
    half_w, half_h = w * 0.5, h * 0.5
    body = (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_h)
    if not body.any():
        return
    img[body] = np.asarray(color, dtype=np.float32)[None, :]
    window = body & (np.abs(yy - (cy - 0.22 * h)) <= 0.20 * h) & (np.abs(xx - cx) <= 0.72 * half_w)
    img[window] = np.asarray(color, dtype=np.float32)[None, :] * 0.45


def make_grid(height, width):
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return xx, yy, height, width


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #

_JOB_CFG = {}


def _init_worker(cfg):
    _JOB_CFG.update(cfg)
    _JOB_CFG['grid'] = make_grid(cfg['gen_h'], cfg['gen_w'])


def _process_video(job):
    """Render every requested frame of one video id. Returns (written, skipped)."""
    split, video_id, label, frames = job
    cfg = _JOB_CFG
    img_dir = os.path.join(cfg['data_root'], SPLIT_DIRS[split][0], video_id)
    sal_dir = os.path.join(cfg['data_root'], SPLIT_DIRS[split][1], video_id)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sal_dir, exist_ok=True)

    params = scene_params(split, video_id, label)
    indices = [frame_index(name) for name in frames]
    lo, hi = min(indices), max(indices)
    span = float(hi - lo) if hi > lo else 1.0

    written = skipped = 0
    for name, idx in zip(frames, indices):
        img_path = os.path.join(img_dir, name)
        sal_path = os.path.join(sal_dir, name)
        if not cfg['overwrite'] and os.path.isfile(img_path) and os.path.isfile(sal_path):
            skipped += 1
            continue

        rgb, sal = render_frame((idx - lo) / span, params, cfg['grid'])
        Image.fromarray(rgb, mode='RGB').save(img_path, quality=cfg['quality'])
        Image.fromarray(sal, mode='L').save(sal_path, quality=cfg['quality'])
        written += 1

    return written, skipped


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def collect_jobs(manifests, limit_videos):
    jobs = []
    for split, manifest in manifests.items():
        items = list(manifest.items())
        if limit_videos:
            items = items[:limit_videos]
        for video_id, item in items:
            jobs.append((split, video_id, item['label'], item['frames']))
    return jobs


def verify(manifests, data_root):
    """Report every file referenced by the lists that is missing on disk."""
    missing = []
    total = 0
    for split, manifest in manifests.items():
        img_sub, sal_sub = SPLIT_DIRS[split]
        for video_id, item in manifest.items():
            for name in item['frames']:
                for sub in (img_sub, sal_sub):
                    path = os.path.join(data_root, sub, video_id, name)
                    total += 1
                    if not os.path.isfile(path):
                        missing.append(path)
    return total, missing


def summarize(manifests):
    for split, manifest in manifests.items():
        per_class = {}
        n_frames = 0
        for item in manifest.values():
            per_class[item['label']] = per_class.get(item['label'], 0) + 1
            n_frames += len(item['frames'])
        dist = ', '.join('{}({}): {}'.format(CLASS_NAMES.get(k, k), k, per_class[k])
                         for k in sorted(per_class))
        print('  [{:5s}] videos: {:4d} | frames: {:6d} | {}'.format(
            split, len(manifest), n_frames, dist))


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate a synthetic ADED-style dataset for VP2Net train/test scripts')
    parser.add_argument('--config', type=str, default='configs/aded/v2pnet.yaml',
                        help='config file used by train.py / test.py')
    parser.add_argument('--data-root', type=str, default=None,
                        help='output root (default: Data.data_root from the config)')
    parser.add_argument('--split', type=str, default='all', choices=['all', 'train', 'test'],
                        help='which split to generate')
    parser.add_argument('--gen-h', type=int, default=None,
                        help='rendered frame height (default: img_h + 20)')
    parser.add_argument('--gen-w', type=int, default=None,
                        help='rendered frame width (default: img_w + 40)')
    parser.add_argument('--quality', type=int, default=88, help='JPEG quality')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='number of worker processes')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-render frames that already exist')
    parser.add_argument('--limit-videos', type=int, default=0,
                        help='debug only: render just the first N video ids per split '
                             '(produces a dataset the full lists cannot be run on)')
    parser.add_argument('--verify-only', action='store_true',
                        help='do not render, only check that every listed file exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='print what would be generated and exit')
    return parser.parse_args()


def main():
    args = get_parser()
    cfg = config.load_cfg_from_cfg_file(args.config)

    data_root = args.data_root if args.data_root else cfg.data_root
    gen_h = args.gen_h if args.gen_h else cfg.img_h + 20
    gen_w = args.gen_w if args.gen_w else cfg.img_w + 40

    wanted = ['train', 'test'] if args.split == 'all' else [args.split]
    list_of = {'train': cfg.train_list, 'test': cfg.test_list}

    manifests = OrderedDict()
    for split in wanted:
        entries = read_list_file(list_of[split])
        manifests[split] = build_manifest(entries, cfg.classes)

    print('=========== SYNTHETIC ADED DATASET ===========')
    print('  config     : {}'.format(args.config))
    print('  data_root  : {}'.format(os.path.abspath(data_root)))
    print('  frame size : {}x{} (train crop target {}x{})'.format(gen_h, gen_w, cfg.img_h, cfg.img_w))
    print('  classes    : {}'.format(cfg.classes))
    summarize(manifests)

    if args.verify_only:
        total, missing = verify(manifests, data_root)
        print('  checked {} files, {} missing'.format(total, len(missing)))
        for path in missing[:20]:
            print('    missing: {}'.format(path))
        if len(missing) > 20:
            print('    ... and {} more'.format(len(missing) - 20))
        return 0 if not missing else 1

    jobs = collect_jobs(manifests, args.limit_videos)
    n_files = 2 * sum(len(job[3]) for job in jobs)
    print('  to render  : {} videos -> {} jpg files'.format(len(jobs), n_files))
    print('==============================================')

    if args.dry_run:
        return 0

    worker_cfg = {
        'data_root': data_root,
        'gen_h': gen_h,
        'gen_w': gen_w,
        'quality': args.quality,
        'overwrite': args.overwrite,
    }

    written = skipped = 0
    if args.workers > 1:
        with Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_cfg,)) as pool:
            for w, s in tqdm(pool.imap_unordered(_process_video, jobs, chunksize=4),
                             total=len(jobs), desc='Rendering'):
                written += w
                skipped += s
    else:
        _init_worker(worker_cfg)
        for job in tqdm(jobs, desc='Rendering'):
            w, s = _process_video(job)
            written += w
            skipped += s

    print('  rendered {} frames ({} pairs skipped, already present)'.format(written, skipped))

    total, missing = verify(manifests, data_root)
    if missing:
        print('  WARNING: {}/{} listed files are still missing'.format(len(missing), total))
        for path in missing[:10]:
            print('    missing: {}'.format(path))
        return 1

    print('  all {} listed files are present.'.format(total))
    return 0


if __name__ == '__main__':
    sys.exit(main())
