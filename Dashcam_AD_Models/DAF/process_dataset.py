import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 10 classes expected by evaluation.py: c0 ~ c9
CLASSES = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
DOMAINS = ['S', 'L']
IMG_SIZE = 256

# One distinctive base color per class so the classes are learnable
CLASS_COLORS = [
    (220, 60, 60),    # c0 red
    (60, 180, 60),    # c1 green
    (60, 90, 220),    # c2 blue
    (230, 200, 50),   # c3 yellow
    (170, 60, 200),   # c4 purple
    (60, 200, 200),   # c5 cyan
    (240, 140, 40),   # c6 orange
    (240, 120, 180),  # c7 pink
    (120, 90, 50),    # c8 brown
    (140, 140, 140),  # c9 gray
]

# One shape pattern per class as a second learnable cue
CLASS_SHAPES = ['circle', 'square', 'triangle', 'hstripes', 'vstripes',
                'cross', 'ring', 'diamond', 'dots', 'diag']


def draw_shape(draw, shape, color, rng):
    cx = rng.integers(80, IMG_SIZE - 80)
    cy = rng.integers(80, IMG_SIZE - 80)
    r = rng.integers(40, 70)
    if shape == 'circle':
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == 'square':
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == 'triangle':
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == 'hstripes':
        for y in range(0, IMG_SIZE, 32):
            draw.rectangle([0, y, IMG_SIZE, y + 14], fill=color)
    elif shape == 'vstripes':
        for x in range(0, IMG_SIZE, 32):
            draw.rectangle([x, 0, x + 14, IMG_SIZE], fill=color)
    elif shape == 'cross':
        w = r // 2
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=color)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=color)
    elif shape == 'ring':
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=16)
    elif shape == 'diamond':
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
    elif shape == 'dots':
        for _ in range(12):
            px = rng.integers(20, IMG_SIZE - 20)
            py = rng.integers(20, IMG_SIZE - 20)
            draw.ellipse([px - 12, py - 12, px + 12, py + 12], fill=color)
    elif shape == 'diag':
        for d in range(-IMG_SIZE, IMG_SIZE, 40):
            draw.line([(d, 0), (d + IMG_SIZE, IMG_SIZE)], fill=color, width=12)


def make_image(class_idx, domain, rng):
    color = CLASS_COLORS[class_idx]
    # domain shift: S has a dark background, L has a light background
    if domain == 'S':
        bg = tuple(int(rng.integers(10, 60)) for _ in range(3))
    else:
        bg = tuple(int(rng.integers(180, 240)) for _ in range(3))

    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # jitter the class color a little
    jitter = rng.integers(-25, 26, size=3)
    jittered = tuple(int(np.clip(color[i] + jitter[i], 0, 255)) for i in range(3))
    draw_shape(draw, CLASS_SHAPES[class_idx], jittered, rng)

    # domain shift: L images are blurred, S images get gaussian noise
    if domain == 'L':
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    else:
        arr = np.asarray(img, dtype=np.int16)
        noise = rng.normal(0, 12, arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img


def main():
    parser = argparse.ArgumentParser(description='Generate a synthetic ImageFolder dataset')
    parser.add_argument('--out_dir', type=str, default='datasets')
    parser.add_argument('--n_per_class', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for domain in DOMAINS:
        for class_idx, class_name in enumerate(CLASSES):
            folder = os.path.join(args.out_dir, domain, class_name)
            os.makedirs(folder, exist_ok=True)
            for i in range(args.n_per_class):
                img = make_image(class_idx, domain, rng)
                img.save(os.path.join(folder, '{}_{}_{:04d}.jpg'.format(domain, class_name, i)))
        print('Domain {}: {} classes x {} images -> {}'.format(
            domain, len(CLASSES), args.n_per_class, os.path.join(args.out_dir, domain)))


if __name__ == '__main__':
    main()
