import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2


def extract_frames_from_video(video_path, out_dir, image_ext=".jpg", jpeg_quality=100, overwrite=False):
    video_path = Path(video_path)
    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = list(images_dir.glob(f"*{image_ext}"))
        if len(existing) > 0:
            return len(existing)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        frame_idx += 1
        out_path = images_dir / f"{frame_idx:06d}{image_ext}"
        if image_ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        else:
            cv2.imwrite(str(out_path), frame)
        ok, frame = cap.read()

    cap.release()
    return frame_idx


def collect_jobs(videos_root, splits):
    jobs = []
    for split in splits:
        for cls_name in ["positive", "negative"]:
            cls_root = videos_root / split / cls_name
            if not cls_root.exists():
                print(f"[Skip] class folder does not exist: {cls_root}")
                continue
            videos = sorted(cls_root.glob("*.mp4"))
            print(f"[{split}/{cls_name}] found {len(videos)} videos")
            for video_path in videos:
                jobs.append((video_path, cls_root / video_path.stem))
    return jobs


def process_dad(root, splits, overwrite, image_ext, jpeg_quality, workers):
    videos_root = Path(root) / "videos"
    jobs = collect_jobs(videos_root, splits)

    total_videos = 0
    total_frames = 0
    failed = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_frames_from_video, video_path, out_dir, image_ext, jpeg_quality, overwrite): video_path
            for video_path, out_dir in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            video_path = futures[future]
            try:
                n = future.result()
                total_videos += 1
                total_frames += n
            except Exception as e:
                failed.append(str(video_path))
                print(f"[Error] {video_path}: {e}")
            if i % 50 == 0 or i == len(jobs):
                print(f"processed {i}/{len(jobs)}")

    print(f"Done. videos={total_videos}, frames={total_frames}, failed={len(failed)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/DAD")
    parser.add_argument("--splits", type=str, nargs="+", default=["training", "testing"])
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dad(
        root=args.root,
        splits=args.splits,
        overwrite=args.overwrite,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
    )
