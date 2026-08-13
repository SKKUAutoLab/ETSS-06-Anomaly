"""
Generate a synthetic ego-view traffic accident dataset similar to DoTA / DADA.

Each video sample mimics the processed representation used by MYModel (model.py):
  - features  : [T, 1 + N_OBJ, 512]  frame-level feature (slot 0) + 19 object features
  - detection : [T, N_OBJ, 6]        (x1, y1, x2, y2, class_id, score); padded rows use class_id = 2
  - depth     : [T, 72, 128]         uint8 depth map at 1/10 image resolution, values in [0, 255]
  - label     : [2]                  one-hot, [1, 0] = negative video, [0, 1] = accident video
  - toa       : scalar               time-of-accident frame (10 fps, like DoTA); T + 1 for negatives

Positive videos contain two "actor" objects whose trajectories converge to a
collision point at frame `toa`; their appearance features (and the frame-level
feature) drift along a fixed "risk" direction as the accident approaches, so
the anticipation model has a learnable signal.

Usage:
    python process_dataset.py                          # default: 100 train / 30 test videos
    python process_dataset.py --num_train 10 --num_test 10
"""

import argparse
import os

import numpy as np

# Image geometry (DoTA-like ego-view resolution)
IMG_W, IMG_H = 1280, 720
DEPTH_SCALE = 10                       # depth map is indexed at center // 10 in model.py
DEPTH_H, DEPTH_W = IMG_H // DEPTH_SCALE, IMG_W // DEPTH_SCALE   # 72 x 128

NUM_FRAMES = 100                       # fixed by AdaptiveAdjacencyMatrix(19, 100, 10)
NUM_OBJECTS = 19
FEATURE_DIM = 512
FPS = 10

PAD_CLASS = 2                          # class id filtered out by clear_and_rearrange()

# Fixed unit vector along which accident-related features drift (same for the
# whole dataset so that the signal is learnable).
_risk_rng = np.random.default_rng(1234)
RISK_DIR = _risk_rng.standard_normal(FEATURE_DIM).astype(np.float32)
RISK_DIR /= np.linalg.norm(RISK_DIR)


def _make_trajectories(rng, is_positive, toa):
    """Simulate smooth object trajectories; returns boxes [T, N_OBJ, 4] and n_active."""
    n_active = int(rng.integers(6, NUM_OBJECTS + 1))

    # Per-object box size (vehicles / pedestrians)
    widths = rng.uniform(50, 220, size=n_active)
    heights = rng.uniform(45, 180, size=n_active)

    # Start centers and constant velocities (pixels / frame) with small noise
    cx0 = rng.uniform(120, IMG_W - 120, size=n_active)
    cy0 = rng.uniform(120, IMG_H - 120, size=n_active)
    vx = rng.uniform(-4, 4, size=n_active)
    vy = rng.uniform(-2.5, 2.5, size=n_active)

    t = np.arange(NUM_FRAMES)[:, None]                       # [T, 1]
    cx = cx0[None, :] + vx[None, :] * t + rng.normal(0, 1.0, (NUM_FRAMES, n_active))
    cy = cy0[None, :] + vy[None, :] * t + rng.normal(0, 1.0, (NUM_FRAMES, n_active))

    actors = None
    if is_positive:
        # Two actors converge linearly to a shared collision point at frame toa
        actors = rng.choice(n_active, size=2, replace=False)
        col_x = rng.uniform(IMG_W * 0.3, IMG_W * 0.7)
        col_y = rng.uniform(IMG_H * 0.35, IMG_H * 0.75)
        for a in actors:
            ramp = np.clip(np.arange(NUM_FRAMES) / max(toa, 1), 0, 1)  # 0 -> 1 at toa
            cx[:, a] = (1 - ramp) * cx0[a] + ramp * col_x + rng.normal(0, 1.0, NUM_FRAMES)
            cy[:, a] = (1 - ramp) * cy0[a] + ramp * col_y + rng.normal(0, 1.0, NUM_FRAMES)
            # After the collision the actors stay entangled around the crash point
            cx[toa:, a] = col_x + rng.normal(0, 2.0, NUM_FRAMES - toa)
            cy[toa:, a] = col_y + rng.normal(0, 2.0, NUM_FRAMES - toa)

    cx = np.clip(cx, 1, IMG_W - 2)
    cy = np.clip(cy, 1, IMG_H - 2)

    boxes = np.zeros((NUM_FRAMES, NUM_OBJECTS, 4), dtype=np.float32)
    for i in range(n_active):
        x1 = np.clip(cx[:, i] - widths[i] / 2, 0, IMG_W - 1)
        x2 = np.clip(cx[:, i] + widths[i] / 2, 0, IMG_W - 1)
        y1 = np.clip(cy[:, i] - heights[i] / 2, 0, IMG_H - 1)
        y2 = np.clip(cy[:, i] + heights[i] / 2, 0, IMG_H - 1)
        boxes[:, i] = np.stack([x1, y1, x2, y2], axis=1)

    return boxes, n_active, actors


def _make_depth(rng, boxes, n_active):
    """Road-like depth map: vertical gradient + smooth noise + object patches."""
    # Bottom of the image is close (large value), top is far
    base = np.linspace(40, 230, DEPTH_H, dtype=np.float32)[:, None]
    base = np.repeat(base, DEPTH_W, axis=1)                  # [72, 128]

    depth = np.zeros((NUM_FRAMES, DEPTH_H, DEPTH_W), dtype=np.float32)
    for f in range(NUM_FRAMES):
        noise = rng.normal(0, 4, (DEPTH_H // 8, DEPTH_W // 8))
        noise = np.kron(noise, np.ones((8, 8)))[:DEPTH_H, :DEPTH_W]
        frame = base + noise
        # Stamp each object with its own depth (proportional to bbox bottom edge)
        for i in range(n_active):
            x1, y1, x2, y2 = boxes[f, i]
            obj_depth = 255.0 * (y2 / IMG_H)
            gx1, gy1 = int(x1) // DEPTH_SCALE, int(y1) // DEPTH_SCALE
            gx2, gy2 = int(x2) // DEPTH_SCALE + 1, int(y2) // DEPTH_SCALE + 1
            frame[gy1:gy2, gx1:gx2] = 0.5 * frame[gy1:gy2, gx1:gx2] + 0.5 * obj_depth
        depth[f] = frame

    return np.clip(depth, 0, 255).astype(np.uint8)


def _make_features(rng, is_positive, toa, n_active, actors):
    """Frame + object appearance features with an accident-correlated drift."""
    features = np.zeros((NUM_FRAMES, 1 + NUM_OBJECTS, FEATURE_DIM), dtype=np.float32)

    # Temporally coherent embeddings: per-slot base vector + small per-frame noise
    base = rng.standard_normal((1 + NUM_OBJECTS, FEATURE_DIM)).astype(np.float32)
    noise = rng.normal(0, 0.1, (NUM_FRAMES, 1 + NUM_OBJECTS, FEATURE_DIM)).astype(np.float32)
    features[:] = base[None, :, :] + noise

    # Inactive object slots carry near-zero features
    features[:, 1 + n_active:, :] *= 0.05

    if is_positive:
        # Risk signal ramps up over the 2 s before the accident and persists after
        ramp = np.clip((np.arange(NUM_FRAMES) - (toa - 2 * FPS)) / (2.0 * FPS), 0, 1)
        ramp = ramp.astype(np.float32)[:, None]
        for a in actors:
            features[:, 1 + a, :] += 2.0 * ramp * RISK_DIR[None, :]
        # Weaker global cue on the frame-level slots (indices 0 and 1)
        features[:, 0, :] += 0.5 * ramp * RISK_DIR[None, :]
        features[:, 1, :] += 0.5 * ramp * RISK_DIR[None, :]

    return features


def generate_video(rng, is_positive):
    toa = int(rng.integers(55, 91)) if is_positive else NUM_FRAMES + 1

    boxes, n_active, actors = _make_trajectories(rng, is_positive, toa if is_positive else 0)
    depth = _make_depth(rng, boxes, n_active)
    features = _make_features(rng, is_positive, toa, n_active, actors)

    detection = np.zeros((NUM_FRAMES, NUM_OBJECTS, 6), dtype=np.float32)
    detection[:, :, :4] = boxes
    classes = rng.integers(0, 2, size=n_active)              # 0 / 1 = valid classes
    detection[:, :n_active, 4] = classes[None, :]
    detection[:, :n_active, 5] = rng.uniform(0.5, 1.0, (NUM_FRAMES, n_active))
    detection[:, n_active:, 4] = PAD_CLASS                   # padding rows -> filtered out

    label = np.array([0.0, 1.0] if is_positive else [1.0, 0.0], dtype=np.float32)

    return {
        'features': features.astype(np.float16),             # stored compact, cast back on load
        'detection': detection,
        'depth': depth,
        'label': label,
        'toa': np.float32(toa),
    }


def generate_split(out_dir, num_videos, rng, split_name):
    os.makedirs(out_dir, exist_ok=True)
    num_pos = num_videos // 2
    flags = np.array([True] * num_pos + [False] * (num_videos - num_pos))
    rng.shuffle(flags)

    for idx, is_positive in enumerate(flags):
        sample = generate_video(rng, bool(is_positive))
        path = os.path.join(out_dir, f'{split_name}_{idx:04d}.npz')
        np.savez_compressed(path, **sample)

    print(f'[{split_name}] wrote {num_videos} videos '
          f'({int(flags.sum())} positive / {int((~flags).sum())} negative) -> {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic DoTA/DADA-style accident dataset')
    parser.add_argument('--data_root', type=str, default='data/synthetic')
    parser.add_argument('--num_train', type=int, default=100,
                        help='number of training videos (keep it a multiple of 10)')
    parser.add_argument('--num_test', type=int, default=30,
                        help='number of test videos (keep it a multiple of 10)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    generate_split(os.path.join(args.data_root, 'train'), args.num_train, rng, 'train')
    generate_split(os.path.join(args.data_root, 'test'), args.num_test, rng, 'test')
    print('Done.')


if __name__ == '__main__':
    main()
