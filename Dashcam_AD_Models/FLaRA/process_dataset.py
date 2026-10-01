#!/usr/bin/env python3
"""
Generate a small synthetic dataset matching the directory layout in org.txt.

The generated files are aligned with the paths used by scripts/train.sh and
scripts/test.sh:

  datasets/nexar/train.csv
  datasets/nexar/val.csv
  datasets/nexar/test.csv
  datasets/dad/test.csv
  datasets/dada2000/test.csv
  datasets/dota/test.csv

The DADA-2000 metadata lives under datasets/dada2000, while scripts/test.sh
currently points --dada2000_data_root to datasets/dad2000. To keep that script
working without changing it, this generator writes DADA-style videos to both
datasets/dada2000/videos and datasets/dad2000/videos.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import cv2
import numpy as np


NEXAR_COLUMNS = [
    "file_name",
    "time_of_event",
    "time_of_alert",
    "light_conditions",
    "weather",
    "scene",
    "time_to_accident",
    "label",
    "split",
    "test_split",
]

EVENT_COLUMNS = [
    "id",
    "batch",
    "Event-type",
    "Nexar-vehicle-involved",
    "min_support",
    "Time-of-alert",
    "Time-of-collision",
    "Diff",
    "article_category",
    "label",
]

DOTA_COLUMNS = [
    "id",
    "batch",
    "Event-type",
    "Nexar-vehicle-involved",
    "min_support",
    "Time-of-alert",
    "Time-of-collision",
    "Diff",
    "article_category",
    "ego_discrepancy",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic videos and metadata for the FLaRA scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets"),
        help="Root directory where synthetic dataset folders will be created.",
    )
    parser.add_argument(
        "--nexar-train-per-class",
        type=int,
        default=4,
        help="Number of positive and negative Nexar training videos.",
    )
    parser.add_argument(
        "--eval-per-class",
        type=int,
        default=2,
        help="Number of positive and negative examples for each validation/test set.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="FPS used for all generated videos.",
    )
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=7.0,
        help="Duration of each generated video in seconds.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Synthetic video width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Synthetic video height.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing video and CSV files.",
    )
    return parser.parse_args()


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    if path.suffix.lower() == ".avi":
        codecs = ("MJPG", "XVID", "mp4v")
    else:
        codecs = ("mp4v", "avc1", "MJPG")

    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Could not open a video writer for {path}")


def draw_frame(
    frame_index: int,
    total_frames: int,
    width: int,
    height: int,
    positive: bool,
    variant: int,
) -> np.ndarray:
    progress = frame_index / max(total_frames - 1, 1)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    sky = int(65 + 35 * np.sin(progress * np.pi + variant))
    road = int(55 + 12 * np.cos(progress * np.pi * 2.0 + variant))
    frame[: height // 2, :] = (sky + 30, sky + 8, sky)
    frame[height // 2 :, :] = (road, road, road)

    horizon = height // 2
    lane_color = (235, 235, 210)
    cv2.line(frame, (width // 2 - 20, height), (width // 2 - 5, horizon), lane_color, 2)
    cv2.line(frame, (width // 2 + 20, height), (width // 2 + 5, horizon), lane_color, 2)

    for offset in range(-1, 2):
        x = int((progress * width * 1.4 + offset * width / 2 + variant * 17) % width)
        cv2.rectangle(frame, (x - 18, horizon - 30), (x + 18, horizon - 10), (60, 110, 190), -1)
        cv2.rectangle(frame, (x - 12, horizon - 24), (x + 12, horizon - 14), (180, 210, 235), -1)

    ego_y = int(height * 0.70)
    ego_color = (40, 150, 70) if not positive else (30, 95, 210)
    cv2.rectangle(frame, (width // 2 - 42, ego_y), (width // 2 + 42, ego_y + 34), ego_color, -1)
    cv2.circle(frame, (width // 2 - 30, ego_y + 35), 8, (18, 18, 18), -1)
    cv2.circle(frame, (width // 2 + 30, ego_y + 35), 8, (18, 18, 18), -1)

    if positive:
        hazard_y = int(height * (0.24 + 0.38 * progress))
        hazard_size = int(22 + 34 * progress)
        hazard_x = int(width * (0.62 - 0.18 * progress + 0.03 * np.sin(variant + progress * 7)))
        cv2.rectangle(
            frame,
            (hazard_x - hazard_size, hazard_y - hazard_size // 2),
            (hazard_x + hazard_size, hazard_y + hazard_size // 2),
            (30, 30, 220),
            -1,
        )
        if progress > 0.68:
            alpha = min((progress - 0.68) / 0.32, 1.0)
            overlay = frame.copy()
            cv2.circle(overlay, (hazard_x, hazard_y), int(24 + 90 * alpha), (40, 40, 255), 5)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    else:
        signal_x = int(width * 0.78)
        cv2.circle(frame, (signal_x, horizon - 35), 12, (30, 180, 60), -1)
        cv2.rectangle(frame, (signal_x - 4, horizon - 22), (signal_x + 4, horizon + 24), (25, 25, 25), -1)

    label = "synthetic crash" if positive else "synthetic normal"
    cv2.putText(
        frame,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def write_video(
    path: Path,
    positive: bool,
    variant: int,
    fps: float,
    seconds: float,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(int(round(fps * seconds)), 48)
    writer = open_video_writer(path, fps, width, height)
    try:
        for frame_index in range(total_frames):
            frame = draw_frame(frame_index, total_frames, width, height, positive, variant)
            writer.write(frame)
    finally:
        writer.release()


def nexar_row(
    file_name: str,
    label: str,
    split: str,
    test_split: str = "",
    event_time: object = 4.0,
    alert_time: object = 2.0,
    time_to_accident: object = "",
) -> dict:
    return {
        "file_name": file_name,
        "time_of_event": event_time,
        "time_of_alert": alert_time,
        "light_conditions": "Normal",
        "weather": "Clear",
        "scene": "Urban",
        "time_to_accident": time_to_accident,
        "label": label,
        "split": split,
        "test_split": test_split,
    }


def event_row(video_id: str, label: object, positive: bool) -> dict:
    collision = 4.0 if positive else ""
    alert = 2.0 if positive else ""
    diff = 2.0 if positive else ""
    return {
        "id": video_id,
        "batch": "",
        "Event-type": "Collision",
        "Nexar-vehicle-involved": "yes",
        "min_support": 10,
        "Time-of-alert": alert,
        "Time-of-collision": collision,
        "Diff": diff,
        "article_category": "pos-ego",
        "label": label,
    }


def dota_row(video_id: str, label: object, positive: bool) -> dict:
    row = event_row(video_id, label, positive)
    row["ego_discrepancy"] = ""
    return row


def generate_nexar(args: argparse.Namespace) -> None:
    root = args.output_root / "nexar"
    ensure_dirs(
        [
            root / "train" / "positive",
            root / "train" / "negative",
            root / "test-public" / "positive",
            root / "test-public" / "negative",
            root / "test-private" / "positive",
            root / "test-private" / "negative",
        ]
    )

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []

    for class_index, (label, label_dir, positive) in enumerate(
        [
            ("crash", "positive", True),
            ("normal_driving", "negative", False),
        ]
    ):
        for i in range(args.nexar_train_per_class):
            name = f"nx_train_{label_dir}_{i:03d}.mp4"
            write_video(
                root / "train" / label_dir / name,
                positive,
                class_index * 100 + i,
                args.fps,
                args.video_seconds,
                args.width,
                args.height,
                args.overwrite,
            )
            train_rows.append(nexar_row(name, label, "train"))

        for i in range(args.eval_per_class):
            name = f"nx_val_{label_dir}_{i:03d}.mp4"
            write_video(
                root / "test-public" / label_dir / name,
                positive,
                class_index * 200 + i,
                args.fps,
                args.video_seconds,
                args.width,
                args.height,
                args.overwrite,
            )
            val_rows.append(
                nexar_row(
                    name,
                    label,
                    "test",
                    test_split="public",
                    event_time=4.0 if positive else "",
                    alert_time=2.0 if positive else "",
                    time_to_accident=1.5,
                )
            )

        for i in range(args.eval_per_class):
            test_split = "public" if i % 2 == 0 else "private"
            name = f"nx_test_{label_dir}_{i:03d}.mp4"
            write_video(
                root / f"test-{test_split}" / label_dir / name,
                positive,
                class_index * 300 + i,
                args.fps,
                args.video_seconds,
                args.width,
                args.height,
                args.overwrite,
            )
            test_rows.append(
                nexar_row(
                    name,
                    label,
                    "test",
                    test_split=test_split,
                    event_time=4.0 if positive else "",
                    alert_time=2.0 if positive else "",
                    time_to_accident=1.5,
                )
            )

    write_csv(root / "train.csv", NEXAR_COLUMNS, train_rows, args.overwrite)
    write_csv(root / "val.csv", NEXAR_COLUMNS, val_rows, args.overwrite)
    write_csv(root / "test.csv", NEXAR_COLUMNS, test_rows, args.overwrite)


def generate_dad(args: argparse.Namespace) -> None:
    root = args.output_root / "dad"
    ensure_dirs([root / "testing" / "positive", root / "testing" / "negative"])

    rows: List[dict] = []
    for label, label_dir, positive in [
        ("positive", "positive", True),
        ("negative", "negative", False),
    ]:
        for i in range(args.eval_per_class):
            name = f"dad_{label_dir}_{i:03d}.mp4"
            write_video(
                root / "testing" / label_dir / name,
                positive,
                400 + i + (0 if positive else 100),
                args.fps,
                args.video_seconds,
                args.width,
                args.height,
                args.overwrite,
            )
            rows.append(event_row(name, label, positive))

    write_csv(root / "test.csv", EVENT_COLUMNS, rows, args.overwrite)


def generate_dada2000(args: argparse.Namespace) -> None:
    metadata_root = args.output_root / "dada2000"
    video_roots = [args.output_root / "dada2000", args.output_root / "dad2000"]
    ensure_dirs([video_root / "videos" for video_root in video_roots])

    rows: List[dict] = []
    for label, label_name, positive in [(1, "positive", True), (0, "negative", False)]:
        for i in range(args.eval_per_class):
            video_id = f"dada_{label_name}_{i:03d}.mp4"
            stem = Path(video_id).stem
            for video_root in video_roots:
                write_video(
                    video_root / "videos" / f"images_{stem}.avi",
                    positive,
                    600 + i + (0 if positive else 100),
                    args.fps,
                    args.video_seconds,
                    args.width,
                    args.height,
                    args.overwrite,
                )
            rows.append(event_row(video_id, label, positive))

    write_csv(metadata_root / "test.csv", EVENT_COLUMNS, rows, args.overwrite)


def generate_dota(args: argparse.Namespace) -> None:
    root = args.output_root / "dota"
    rows: List[dict] = []

    for label, label_name, positive in [(1, "positive", True), (0, "negative", False)]:
        for i in range(args.eval_per_class):
            video_id = f"dota_{label_name}_{i:03d}.mp4"
            stem = Path(video_id).stem
            write_video(
                root / "dota_annotated" / stem / f"{stem}.mp4",
                positive,
                800 + i + (0 if positive else 100),
                args.fps,
                args.video_seconds,
                args.width,
                args.height,
                args.overwrite,
            )
            rows.append(dota_row(video_id, label, positive))

    write_csv(root / "test.csv", DOTA_COLUMNS, rows, args.overwrite)


def main() -> int:
    args = parse_args()

    if args.nexar_train_per_class < 1:
        raise ValueError("--nexar-train-per-class must be at least 1")
    if args.eval_per_class < 1:
        raise ValueError("--eval-per-class must be at least 1")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.video_seconds <= 0:
        raise ValueError("--video-seconds must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")

    generate_nexar(args)
    generate_dad(args)
    generate_dada2000(args)
    generate_dota(args)

    print(f"Synthetic dataset is ready under {args.output_root}")
    print("Run scripts/train.sh and scripts/test.sh from the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
