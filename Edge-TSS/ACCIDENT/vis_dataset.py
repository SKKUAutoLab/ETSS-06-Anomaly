import argparse
import sys
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent))
from dataset.accident_dataset import default_dataset_root, resolve_dataset_root

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = default_dataset_root(REPO_ROOT)

def safe_dir_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")

def metadata_path_for_kind(dataset_root: Path, kind: str) -> Path:
    if kind == "real":
        return dataset_root / "metadata-real.csv"
    if kind == "synthetic":
        return dataset_root / "metadata-synthetic.csv"
    raise ValueError(f"Unknown dataset kind: {kind}")

def path_column_for_kind(kind: str) -> str:
    if kind == "real":
        return "path"
    if kind == "synthetic":
        return "rgb_path"
    raise ValueError(f"Unknown dataset kind: {kind}")

def output_root_for_kind(dataset_root: Path, kind: str) -> Path:
    if kind == "real":
        return dataset_root / "vis_real"
    if kind == "synthetic":
        return dataset_root / "vis_synthetic"
    raise ValueError(f"Unknown dataset kind: {kind}")

def resolve_video_path(row: pd.Series, dataset_root: Path, kind: str) -> Path:
    path_value = str(row[path_column_for_kind(kind)])
    video_path = dataset_root / path_value
    if video_path.exists():
        return video_path
    if kind == "real":
        fallback_path = dataset_root / "real_videos" / Path(path_value).name
    else:
        fallback_path = dataset_root / "synthetic_videos" / "videos" / str(row["type"]) / Path(path_value).name

    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Video not found for metadata path: {path_value}")

def draw_label(frame, text: str, origin: tuple[int, int] = (16, 34), font_scale: float = 0.85, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    padding = 8
    cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), (x + text_size[0] + padding, y + baseline + padding), (0, 0, 0), -1)
    cv2.putText(frame, text, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def normalized_to_pixel(value: float, size: int) -> int:
    pixel = int(round(float(value) * size))
    return max(0, min(size - 1, pixel))

def draw_ground_truth(frame, row: pd.Series) -> None:
    height, width = frame.shape[:2]
    x1 = normalized_to_pixel(row["x1"], width)
    y1 = normalized_to_pixel(row["y1"], height)
    x2 = normalized_to_pixel(row["x2"], width)
    y2 = normalized_to_pixel(row["y2"], height)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    cx = normalized_to_pixel(row["center_x"], width)
    cy = normalized_to_pixel(row["center_y"], height)
    box_thickness = max(2, int(round(min(width, height) * 0.003)))
    dot_radius = max(8, int(round(min(width, height) * 0.012)))
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), box_thickness, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), dot_radius + 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), dot_radius, (0, 0, 255), -1, cv2.LINE_AA)

def should_draw_ground_truth(frame_index: int, fps: float, row: pd.Series, window_seconds: float) -> bool:
    half_window = window_seconds / 2
    draw_from_time = False
    draw_from_frame = False
    if "accident_time" in row and pd.notna(row["accident_time"]):
        current_time = frame_index / fps
        draw_from_time = abs(current_time - float(row["accident_time"])) <= half_window
    if "accident_frame" in row and pd.notna(row["accident_frame"]):
        frame_window = max(1, int(round(half_window * fps)))
        accident_frame = int(round(float(row["accident_frame"])))
        draw_from_frame = abs(frame_index - accident_frame) <= frame_window
    if pd.notna(row.get("accident_time")) or pd.notna(row.get("accident_frame")):
        return draw_from_time or draw_from_frame
    return True

def visualize_video(row: pd.Series, dataset_root: Path, kind: str, overwrite: bool, window_seconds: float) -> Path:
    video_path = resolve_video_path(row, dataset_root, kind)
    output_dir = output_root_for_kind(dataset_root, kind) / safe_dir_name(str(row["type"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_gt.mp4"
    if output_path.exists() and not overwrite:
        return output_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    label = (f"type: {row['type']} | "
             f"time: {float(row['accident_time']):.2f}s | "
             f"frame: {int(round(float(row['accident_frame'])))}")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if should_draw_ground_truth(frame_index, fps, row, window_seconds):
            draw_ground_truth(frame, row)
        draw_label(frame, label)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    return output_path

def visualize_kind(dataset_root: Path, kind: str, take: int | None, overwrite: bool, window_seconds: float) -> list[Path]:
    metadata = pd.read_csv(metadata_path_for_kind(dataset_root, kind))
    if take is not None:
        metadata = metadata.head(take)
    written_paths = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Visualizing {kind} ground truth"):
        written_paths.append(visualize_video(row=row, dataset_root=dataset_root, kind=kind, overwrite=overwrite, window_seconds=window_seconds))
    return written_paths

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ACCIDENT real and synthetic ground-truth annotations.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_ROOT, help="Path to dataset/ (default: ./dataset)")
    parser.add_argument("--kind", choices=["real", "synthetic", "both"], default="both", help="Which dataset split to visualize.")
    parser.add_argument("--take", type=int, default=None, help="Visualize only the first N rows for each selected kind.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualization videos.")
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Draw dot and bbox only within this many seconds centered on the accident.")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_path)
    kinds = ["real", "synthetic"] if args.kind == "both" else [args.kind]
    total = 0
    for kind in kinds:
        written_paths = visualize_kind(dataset_root=dataset_root, kind=kind, take=args.take, overwrite=args.overwrite, window_seconds=args.window_seconds)
        total += len(written_paths)
        print(f"Saved {len(written_paths)} {kind} visualization videos.")
    print(f"Saved {total} visualization videos under {dataset_root}")

if __name__ == "__main__":
    main()