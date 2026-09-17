#!/usr/bin/env python
"""
process_dataset.py
==================

Convert the raw DAD download under ``./datasets`` into the per-video feature
files that TIME-VAD's ``dataset.py`` consumes, then rewrite the ``.list`` files
under ``./data`` so they point at the newly produced files.

Why re-extraction is required
-----------------------------
The archive shipped by the dataset authors contains **VGG-16** features:

    datasets/DAD/features/{training,testing}/batch_XXX.npz -> data (10, 100, 20, 4096)

TIME-VAD instead consumes one ``.npy`` per video with shape ``[T, 10, F]``:

    dataset.py            features.transpose(1, 0, 2)      -> [10, T, F]
    test_10crop.py        torch.randn(1, 100, 10, 768)     # [batch, time, crops, features]
    option.py             --feature-size 768

and ``train.py`` dots those features against ``data/clip_features{1,2}.npy``
(shapes ``(19, 768)`` / ``(27, 768)``, i.e. CLIP ViT-L/14 *text* embeddings)::

    train.py:103   positive_similarity = torch.matmul(features, self.positive_anchor.t())

So the video features must live in the CLIP ViT-L/14 image-embedding space
(768-d).  4096-d VGG-16 features cannot be mapped into it, therefore this script
re-extracts from the raw ``.mp4`` videos with CLIP ViT-L/14 using the standard
10-crop protocol.

Output
------
    datasets/DAD/clip_features/training/{negative,positive}/*.npy   [100, 10, 768] float32
    datasets/DAD/clip_features/testing/{negative,positive}/*.npy    [100, 10, 768] float32

The ``.list`` files are rewritten **line by line, order preserved**.  Order is
load-bearing: ``test_10crop.py`` walks the test loader sequentially and slices
``ground_truth.npy`` as it goes, and ``option.py``'s ``normal_train_end`` index
(829) addresses the train list positionally.  Originals are backed up to
``<name>.list.bak`` before the first overwrite.

Rewriting is idempotent: the output layout keeps the same
``<split>/<class>/<id>`` tail, so re-parsing an already-rewritten list yields
the same destinations.

Usage
-----
    # 1. inspect what is present / missing, verify the mapping, no GPU needed
    python process_dataset.py --check

    # 2. extract features (writes ~5.4 GB)
    python process_dataset.py

    # 3. rewrite the .list files only (no extraction)
    python process_dataset.py --lists-only

    # 4. confirm the produced files load with the right shape
    python process_dataset.py --verify

Notes
-----
* Requires the OpenAI ``clip`` package.  Model weights (~890 MB for ViT-L/14)
  are fetched on first use into ``~/.cache/clip`` unless
  ``--clip-download-root`` is given.
* Peak host RAM is roughly ``num_workers * 300 MB`` while decoding 720p clips.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# CLIP ViT-L/14 image-encoder input statistics.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_INPUT_RES = 224
RESIZE_SHORT_SIDE = 256
NUM_CROPS = 10
FEATURE_DIM = 768


# --------------------------------------------------------------------------- #
# Dataset specifications
# --------------------------------------------------------------------------- #
# ``num_frames`` is fixed by the ground-truth file:
#   data/DAD/ground_truth.npy -> (46600,) == 466 test videos * 100 frames
DATASET_SPECS = {
    "dad": {
        "name": "DAD",
        "num_frames": 100,
        "out_root": PROJECT_ROOT / "datasets" / "DAD" / "clip_features",
        "ground_truth": PROJECT_ROOT / "data" / "DAD" / "ground_truth.npy",
        "lists": {
            "train": PROJECT_ROOT / "data" / "DAD" / "train.list",
            "test": PROJECT_ROOT / "data" / "DAD" / "test.list",
        },
        # DAD videos already carry the training/testing split in their path.
        "video_root": PROJECT_ROOT / "datasets" / "DAD" / "videos",
        "class_dirs": {
            "negative": ["negative"],
            "positive": ["positive"],
        },
        "split_aware": True,
    },
}


# --------------------------------------------------------------------------- #
# List parsing / path mapping
# --------------------------------------------------------------------------- #
def read_list(list_path):
    """Read a .list file into a list of stripped, non-empty lines."""
    with open(list_path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def parse_entry(line):
    """
    Split a list entry into ``(split_dir, class_dir, video_id)``.

    Entries look like ``/home/sumit/RTFM-main/outDAD/training/negative/000198.npy``;
    only the last three components carry meaning.
    """
    parts = Path(line).parts
    if len(parts) < 3:
        raise ValueError(f"Cannot parse list entry: {line!r}")
    split_dir, class_dir, filename = parts[-3], parts[-2], parts[-1]
    return split_dir, class_dir, Path(filename).stem


def output_path(spec, split_dir, class_dir, video_id):
    """Destination .npy for one list entry, mirroring the original layout."""
    return spec["out_root"] / split_dir / class_dir / f"{video_id}.npy"


def resolve_source_video(spec, split_dir, class_dir, video_id):
    """
    Locate the raw .mp4 for one list entry, or ``None`` if it is not present.

    Folder namings come from ``class_dirs`` in the dataset spec, so a renamed
    layout works without editing this function.
    """
    candidates = spec["class_dirs"].get(class_dir)
    if candidates is None:
        raise ValueError(
            f"Unknown class directory {class_dir!r} for dataset {spec['name']}"
        )

    roots = []
    if spec["split_aware"]:
        roots.append(spec["video_root"] / split_dir)
    else:
        roots.append(spec["video_root"])

    for root in roots:
        for sub in candidates:
            path = root / sub / f"{video_id}.mp4"
            if path.is_file():
                return path
        # Also allow the videos to sit directly in the root, un-nested.
        flat = root / f"{video_id}.mp4"
        if flat.is_file():
            return flat
    return None


def build_plan(spec):
    """
    Build the full work plan for a dataset.

    Returns ``{split: [entry, ...]}`` where each entry is a dict with the source
    video, the destination .npy and the original list line.  List order is
    preserved exactly.
    """
    plan = {}
    for split, list_path in spec["lists"].items():
        if not list_path.is_file():
            raise FileNotFoundError(f"List file not found: {list_path}")
        entries = []
        for line in read_list(list_path):
            split_dir, class_dir, video_id = parse_entry(line)
            entries.append(
                {
                    "line": line,
                    "split_dir": split_dir,
                    "class_dir": class_dir,
                    "video_id": video_id,
                    "src": resolve_source_video(spec, split_dir, class_dir, video_id),
                    "dst": output_path(spec, split_dir, class_dir, video_id),
                }
            )
        plan[split] = entries
    return plan


# --------------------------------------------------------------------------- #
# Video decoding + 10-crop
# --------------------------------------------------------------------------- #
def _make_clip_dataset(num_frames):
    """
    Build the torch Dataset class lazily so ``--check`` works without torch
    having to import CUDA machinery first.
    """
    import cv2
    import torch
    import torch.utils.data as tdata
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    class VideoClipDataset(tdata.Dataset):
        """Decode a video and return a resized uint8 clip ``[T, 3, H', W']``."""

        def __init__(self, items, target_frames):
            self.items = items
            self.target_frames = target_frames

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            # OpenCV's own threading fights the DataLoader workers.
            cv2.setNumThreads(0)
            item = self.items[index]
            try:
                frames = self._decode(str(item["src"]))
                if not frames:
                    raise RuntimeError("no frames decoded")
                clip = self._to_tensor(frames)
                return index, clip, ""
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                return index, torch.zeros(0), f"{type(exc).__name__}: {exc}"

        def _decode(self, path):
            cap = cv2.VideoCapture(path)
            frames = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # cv2 decodes BGR; CLIP expects RGB.
                    frames.append(frame[:, :, ::-1])
            finally:
                cap.release()
            return frames

        def _to_tensor(self, frames):
            frames = self._resample(frames, self.target_frames)
            arr = np.ascontiguousarray(np.stack(frames, axis=0))  # [T, H, W, 3]
            clip = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]
            # Short side -> 256, bicubic + antialias to match CLIP's PIL resize.
            clip = TF.resize(
                clip,
                RESIZE_SHORT_SIDE,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return clip

        @staticmethod
        def _resample(frames, target):
            """Force the clip to exactly ``target`` frames."""
            n = len(frames)
            if n == target:
                return frames
            idx = np.linspace(0, n - 1, target).round().astype(int)
            return [frames[i] for i in idx]

    return VideoClipDataset


def ten_crop_normalize(clip_uint8, device):
    """
    ``[T, 3, H, W]`` uint8 -> ``[T * 10, 3, 224, 224]`` normalized float.

    Crop order follows torchvision's convention: the five crops
    (top-left, top-right, bottom-left, bottom-right, center) of the frame,
    followed by the same five crops taken from the horizontally flipped frame.
    Note this means crop 5 mirrors crop 1, not crop 0.

    Output is frame-major / crop-minor, so ``out[i * 10 + c]`` is crop ``c`` of
    frame ``i`` and the result reshapes cleanly to ``[T, 10, 768]``.
    """
    import torch
    import torchvision.transforms.functional as TF

    x = clip_uint8.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    crops = TF.ten_crop(x, [CLIP_INPUT_RES, CLIP_INPUT_RES])  # 10 x [T, 3, 224, 224]
    x = torch.stack(crops, dim=1)  # [T, 10, 3, 224, 224]
    t = x.shape[0]
    return x.reshape(t * NUM_CROPS, 3, CLIP_INPUT_RES, CLIP_INPUT_RES), t


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def load_clip_model(args):
    """Load CLIP ViT-L/14 and return ``(model, device)``."""
    import torch

    try:
        import clip
    except ImportError as exc:
        raise SystemExit(
            "The 'clip' package is required for feature extraction.\n"
            "Install it with:  pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print(f"Loading CLIP {args.clip_model} on {device} ...")
    model, _ = clip.load(
        args.clip_model,
        device=device,
        jit=False,
        download_root=args.clip_download_root,
    )
    model.eval()
    if args.fp32:
        model.float()

    # CLIP.encode_image already casts its input to self.dtype, but check the
    # embedding width matches what the rest of the repo hard-codes.
    with torch.no_grad():
        probe = torch.zeros(1, 3, CLIP_INPUT_RES, CLIP_INPUT_RES, device=device)
        dim = model.encode_image(probe).shape[-1]
    if dim != FEATURE_DIM:
        raise SystemExit(
            f"{args.clip_model} produces {dim}-d image embeddings, but TIME-VAD "
            f"expects {FEATURE_DIM}-d (option.py --feature-size, model.py fc1). "
            f"Use ViT-L/14."
        )
    print(f"CLIP loaded: image embedding dim = {dim}")
    return model, device


def extract_split(spec, entries, model, device, args, split_name):
    """Extract features for one split. Returns a stats dict."""
    import torch
    from torch.utils.data import DataLoader

    todo = []
    skipped_existing = 0
    missing_source = []

    for entry in entries:
        if entry["src"] is None:
            missing_source.append(entry)
            continue
        if entry["dst"].is_file() and not args.overwrite:
            if _is_valid_feature(entry["dst"], spec["num_frames"]):
                skipped_existing += 1
                continue
        todo.append(entry)

    # De-duplicate: the DAD lists reference each source video once, but a video
    # can legitimately appear in more than one entry in a custom list.
    unique_todo, seen = [], set()
    for entry in todo:
        key = str(entry["dst"])
        if key in seen:
            continue
        seen.add(key)
        unique_todo.append(entry)
    todo = unique_todo

    if args.limit:
        todo = todo[: args.limit]

    print(
        f"[{spec['name']}/{split_name}] {len(entries)} entries | "
        f"{len(todo)} to extract | {skipped_existing} already done | "
        f"{len(missing_source)} missing source video"
    )
    if not todo:
        return {
            "entries": len(entries),
            "extracted": 0,
            "skipped_existing": skipped_existing,
            "missing_source": [e["line"] for e in missing_source],
            "failed": [],
        }

    for entry in todo:
        entry["dst"].parent.mkdir(parents=True, exist_ok=True)

    dataset_cls = _make_clip_dataset(spec["num_frames"])
    loader = DataLoader(
        dataset_cls(todo, spec["num_frames"]),
        batch_size=None,  # dataset yields one whole clip per item
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    failed = []
    done = 0
    started = time.time()

    for index, clip_uint8, error in loader:
        entry = todo[int(index)]
        if error:
            failed.append({"line": entry["line"], "error": error})
            print(f"  !! {entry['src']}: {error}")
            continue
        try:
            frames, num_frames = ten_crop_normalize(clip_uint8, device)
            chunks = []
            with torch.no_grad():
                for start in range(0, frames.shape[0], args.chunk_size):
                    feats = model.encode_image(frames[start : start + args.chunk_size])
                    chunks.append(feats.float())
            feat = torch.cat(chunks, dim=0)
            feat = feat.reshape(num_frames, NUM_CROPS, FEATURE_DIM)
            np.save(entry["dst"], feat.cpu().numpy().astype(np.float32))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            failed.append({"line": entry["line"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {entry['src']}: {exc}")
            continue

        done += 1
        if done % args.log_every == 0 or done == len(todo):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (len(todo) - done) / rate if rate else 0.0
            print(
                f"  [{spec['name']}/{split_name}] {done}/{len(todo)} "
                f"({rate:.2f} vid/s, ~{remaining / 60:.1f} min left)"
            )

    return {
        "entries": len(entries),
        "extracted": done,
        "skipped_existing": skipped_existing,
        "missing_source": [e["line"] for e in missing_source],
        "failed": failed,
    }


def _is_valid_feature(path, num_frames):
    """
    Header + size check, so a resume cannot silently skip a truncated file.

    Reading the header alone is not enough: ``np.save`` writes the header before
    the data, so a run killed mid-write leaves a file whose header still
    declares the full shape.  That would pass a header-only check, get skipped
    by the resume path, and only fail much later inside ``np.load`` at training
    time.  Comparing the on-disk size against the declared payload catches it.
    """
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _, dtype = np.lib.format._read_array_header(fh, version)
            data_offset = fh.tell()
            fh.seek(0, os.SEEK_END)
            actual_size = fh.tell()
    except Exception:  # noqa: BLE001 - a corrupt file just needs redoing
        return False

    if shape != (num_frames, NUM_CROPS, FEATURE_DIM) or dtype != np.dtype("float32"):
        return False

    expected_size = data_offset + int(np.prod(shape)) * dtype.itemsize
    return actual_size == expected_size


# --------------------------------------------------------------------------- #
# List rewriting
# --------------------------------------------------------------------------- #
def rewrite_lists(spec, plan, args):
    """
    Point every .list entry at its new .npy, preserving line order and count.

    Backs the original up to ``<name>.list.bak`` before the first overwrite.
    """
    for split, list_path in spec["lists"].items():
        entries = plan[split]
        original = read_list(list_path)
        if len(original) != len(entries):
            raise RuntimeError(
                f"{list_path}: parsed {len(entries)} entries but file has "
                f"{len(original)} lines - refusing to rewrite"
            )

        new_lines = []
        for entry in entries:
            dst = entry["dst"]
            if args.relative_paths:
                new_lines.append(os.path.relpath(dst, PROJECT_ROOT))
            else:
                new_lines.append(str(dst))

        backup = list_path.with_suffix(list_path.suffix + ".bak")
        if not backup.exists():
            backup.write_text("\n".join(original))
            print(f"  backed up {list_path.name} -> {backup.name}")

        # Match the originals: no trailing newline, one path per line.
        list_path.write_text("\n".join(new_lines))

        missing = sum(1 for e in entries if not e["dst"].is_file())
        status = "all present" if missing == 0 else f"{missing} target(s) not yet extracted"
        print(f"  wrote {list_path} ({len(new_lines)} entries, {status})")


# --------------------------------------------------------------------------- #
# Reporting modes
# --------------------------------------------------------------------------- #
def report_check(spec, plan):
    """Print coverage without touching the GPU or the videos themselves."""
    print(f"\n=== {spec['name']} ===")
    print(f"  frames per video : {spec['num_frames']}")
    print(f"  output root      : {spec['out_root']}")

    grand_missing = 0
    for split, entries in plan.items():
        by_class = {}
        for entry in entries:
            bucket = by_class.setdefault(entry["class_dir"], {"n": 0, "src": 0, "dst": 0})
            bucket["n"] += 1
            bucket["src"] += entry["src"] is not None
            bucket["dst"] += entry["dst"].is_file()

        total = len(entries)
        print(f"  {split} list: {spec['lists'][split]}  ({total} entries)")
        for class_dir, bucket in sorted(by_class.items()):
            missing = bucket["n"] - bucket["src"]
            grand_missing += missing
            flag = "" if missing == 0 else f"   <-- {missing} MISSING"
            print(
                f"    {class_dir:<9} entries={bucket['n']:<5} "
                f"videos_found={bucket['src']:<5} features_done={bucket['dst']:<5}{flag}"
            )

    # Cross-check the test list against the ground truth length.
    gt_path = spec["ground_truth"]
    if gt_path.is_file():
        gt = np.load(gt_path)
        expected = len(plan["test"]) * spec["num_frames"]
        ok = "OK" if gt.shape[0] == expected else "MISMATCH"
        print(
            f"  ground truth     : {gt.shape[0]} labels vs "
            f"{len(plan['test'])} x {spec['num_frames']} = {expected}  [{ok}]"
        )
    else:
        print(f"  ground truth     : NOT FOUND at {gt_path}")

    if grand_missing:
        print(f"  >> {grand_missing} source video(s) missing for {spec['name']}")
    return grand_missing


def report_verify(spec, plan):
    """Load every produced .npy header and confirm the shape/dtype."""
    print(f"\n=== verify {spec['name']} ===")
    bad = 0
    for split, entries in plan.items():
        missing = 0
        wrong = 0
        for entry in entries:
            if not entry["dst"].is_file():
                missing += 1
                continue
            if not _is_valid_feature(entry["dst"], spec["num_frames"]):
                wrong += 1
                if wrong <= 5:
                    print(f"    bad shape/dtype: {entry['dst']}")
        total = len(entries)
        ok = total - missing - wrong
        bad += missing + wrong
        print(
            f"  {split}: {ok}/{total} valid "
            f"[{spec['num_frames']}, {NUM_CROPS}, {FEATURE_DIM}] float32, "
            f"{missing} missing, {wrong} malformed"
        )
    return bad


def selftest():
    """Static self-check of the crop/resample maths - no dataset access."""
    import torch

    print("=== self-test ===")

    dataset_cls = _make_clip_dataset(50)
    inst = dataset_cls([], 50)

    # Frame resampling forces the exact target length in both directions.
    for n, target in [(50, 50), (37, 50), (100, 50), (100, 100), (250, 100)]:
        frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]
        out = inst._resample(frames, target)
        assert len(out) == target, (n, target, len(out))
    print("  frame resampling      OK")

    # Resize keeps the short side at 256 and preserves aspect ratio.
    frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(4)]
    clip = inst._to_tensor(frames)
    assert clip.shape[0] == 50, clip.shape
    assert clip.shape[1] == 3, clip.shape
    assert min(clip.shape[2], clip.shape[3]) == RESIZE_SHORT_SIDE, clip.shape
    assert clip.shape[3] == round(1280 * 256 / 720), clip.shape
    print(f"  resize short side     OK  {tuple(clip.shape)}")

    # Ten-crop produces 10 distinct 224x224 views per frame.
    small = clip[:4]
    frames_out, t = ten_crop_normalize(small, torch.device("cpu"))
    assert t == 4, t
    assert frames_out.shape == (4 * NUM_CROPS, 3, CLIP_INPUT_RES, CLIP_INPUT_RES), frames_out.shape
    print(f"  ten-crop + normalize  OK  {tuple(frames_out.shape)}")

    # The reshape used after encoding must keep frame-major, crop-minor order.
    fake = torch.arange(4 * NUM_CROPS * FEATURE_DIM, dtype=torch.float32)
    fake = fake.reshape(4 * NUM_CROPS, FEATURE_DIM)
    feat = fake.reshape(4, NUM_CROPS, FEATURE_DIM)
    assert torch.equal(feat[2, 3], fake[2 * NUM_CROPS + 3])
    print("  [T,10,768] reshape    OK")

    # dataset.py transposes (1, 0, 2); confirm that yields [10, T, 768].
    arr = np.zeros((50, NUM_CROPS, FEATURE_DIM), dtype=np.float32)
    assert arr.transpose(1, 0, 2).shape == (NUM_CROPS, 50, FEATURE_DIM)
    print("  dataset.py transpose  OK  -> (10, 50, 768)")

    # List parsing on the real repo lists.
    for key, spec in DATASET_SPECS.items():
        for split, list_path in spec["lists"].items():
            if not list_path.is_file():
                print(f"  !! list not found: {list_path}")
                continue
            lines = read_list(list_path)
            parsed = [parse_entry(line) for line in lines]
            classes = sorted({c for _, c, _ in parsed})
            unknown = [c for c in classes if c not in spec["class_dirs"]]
            assert not unknown, f"{list_path}: unknown classes {unknown}"
            print(
                f"  parse {key}/{split:<5}     OK  {len(lines)} lines, classes={classes}"
            )
    print("=== self-test passed ===")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Extract CLIP ViT-L/14 10-crop features for DAD and "
        "repoint the TIME-VAD .list files at them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="dad", choices=["dad"],
                   help="Which dataset to process")

    mode = p.add_argument_group("modes")
    mode.add_argument("--check", action="store_true",
                      help="Report source/feature coverage and exit (no GPU needed)")
    mode.add_argument("--verify", action="store_true",
                      help="Validate the produced .npy files and exit")
    mode.add_argument("--self-test", action="store_true",
                      help="Run internal shape/parsing checks and exit")
    mode.add_argument("--lists-only", action="store_true",
                      help="Rewrite the .list files without extracting features")
    mode.add_argument("--no-update-lists", action="store_true",
                      help="Extract features but leave the .list files untouched")

    ext = p.add_argument_group("extraction")
    ext.add_argument("--clip-model", default="ViT-L/14",
                     help="CLIP variant; must yield 768-d embeddings")
    ext.add_argument("--clip-download-root", default=None,
                     help="Where to cache CLIP weights (default: ~/.cache/clip)")
    ext.add_argument("--gpu-id", type=int, default=0,
                     help="CUDA device index, -1 for CPU")
    ext.add_argument("--num-workers", type=int, default=4,
                     help="Video decoding workers (~300 MB RAM each)")
    ext.add_argument("--chunk-size", type=int, default=200,
                     help="Images per CLIP forward pass")
    ext.add_argument("--fp32", action="store_true",
                     help="Run CLIP in fp32 instead of its native fp16")
    ext.add_argument("--overwrite", action="store_true",
                     help="Re-extract features that already exist")
    ext.add_argument("--limit", type=int, default=0,
                     help="Only process the first N videos per split (0 = all)")
    ext.add_argument("--log-every", type=int, default=50,
                     help="Progress print interval in videos")

    out = p.add_argument_group("output")
    out.add_argument("--relative-paths", action="store_true",
                     help="Write repo-relative paths into the .list files")
    out.add_argument("--report", default=None,
                     help="Write a JSON run report to this path")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Validate before loading CLIP, so a bad value fails in a second rather than
    # after a ~890 MB model load.
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0 (0 means no limit)")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if args.lists_only and args.no_update_lists:
        raise SystemExit("--lists-only and --no-update-lists cancel out; nothing to do")

    if args.self_test:
        selftest()
        return 0

    keys = [args.dataset]

    plans = {}
    for key in keys:
        plans[key] = build_plan(DATASET_SPECS[key])

    if args.check:
        total_missing = sum(report_check(DATASET_SPECS[k], plans[k]) for k in keys)
        if total_missing:
            print(
                f"\n{total_missing} source video(s) are missing. Features cannot be "
                f"produced for those entries until the videos are downloaded."
            )
        return 0

    if args.verify:
        bad = sum(report_verify(DATASET_SPECS[k], plans[k]) for k in keys)
        print(f"\n{'All feature files valid.' if bad == 0 else f'{bad} problem(s) found.'}")
        return 0 if bad == 0 else 1

    report = {}

    if not args.lists_only:
        model, device = load_clip_model(args)
        for key in keys:
            spec = DATASET_SPECS[key]
            report[key] = {}
            for split in ("train", "test"):
                report[key][split] = extract_split(
                    spec, plans[key][split], model, device, args, split
                )

    if not args.no_update_lists:
        for key in keys:
            print(f"\nUpdating {DATASET_SPECS[key]['name']} list files ...")
            rewrite_lists(DATASET_SPECS[key], plans[key], args)

    # Summary
    print("\n" + "=" * 62)
    for key in keys:
        spec = DATASET_SPECS[key]
        stats = report.get(key)
        if stats is None:
            continue
        extracted = sum(s["extracted"] for s in stats.values())
        skipped = sum(s["skipped_existing"] for s in stats.values())
        missing = sum(len(s["missing_source"]) for s in stats.values())
        failed = sum(len(s["failed"]) for s in stats.values())
        print(
            f"{spec['name']}: extracted={extracted} already_done={skipped} "
            f"missing_source={missing} failed={failed}"
        )
    print("=" * 62)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"Report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
