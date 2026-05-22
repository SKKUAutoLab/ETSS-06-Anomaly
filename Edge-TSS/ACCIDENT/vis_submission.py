import argparse
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMISSION = REPO_ROOT / "submission.csv"
DEFAULT_VIDEO_DIR = REPO_ROOT / "datasets" / "videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "vis_submission"

def resolve_video_path(path: str, video_dir: Path) -> Path:
    relative_path = Path(path)
    candidates = [video_dir / relative_path.name]
    if relative_path.parts and relative_path.parts[0] != "videos":
        candidates.append(video_dir / relative_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(video_dir.rglob(relative_path.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Video not found for submission path: {path}")

def draw_label(frame, text: str, origin: tuple[int, int] = (16, 34), font_scale: float = 0.9, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    padding = 8
    cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), (x + text_size[0] + padding, y + baseline + padding), (0, 0, 0), -1)
    cv2.putText(frame, text, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_location_dot(frame, center_x: float, center_y: float) -> None:
    height, width = frame.shape[:2]
    px = int(round(center_x * width))
    py = int(round(center_y * height))
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    radius = max(8, int(round(min(width, height) * 0.012)))
    cv2.circle(frame, (px, py), radius + 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), radius, (0, 0, 255), -1, cv2.LINE_AA)

def should_draw_dot(frame_index: int, fps: float, accident_time: float, dot_window_seconds: float) -> bool:
    current_time = frame_index / fps
    return abs(current_time - accident_time) <= dot_window_seconds / 2

def visualize_video(row: pd.Series, video_dir: Path, output_dir: Path, overwrite: bool, dot_window_seconds: float) -> Path:
    video_path = resolve_video_path(str(row["path"]), video_dir)
    output_path = output_dir / f"{video_path.stem}_submission.mp4"
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
    accident_time = float(row["accident_time"])
    center_x = float(row["center_x"])
    center_y = float(row["center_y"])
    label = f"type: {row['type']} | time: {accident_time:.2f}s"
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if should_draw_dot(frame_index, fps, accident_time, dot_window_seconds):
            draw_location_dot(frame, center_x, center_y)
        draw_label(frame, label)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize challenge submission predictions.")
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION, help="Path to submission.csv.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Path to the challenge videos directory (default: ./datasets/videos).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for annotated videos.")
    parser.add_argument("--take", type=int, default=None, help="Visualize only the first N videos.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualization videos.")
    parser.add_argument("--dot-window-seconds", type=float, default=1.0, help="Show the localization dot only within this many seconds centered on accident_time.")
    args = parser.parse_args()

    predictions = pd.read_csv(args.submission_path)
    if args.take is not None:
        predictions = predictions.head(args.take)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for _, row in tqdm(predictions.iterrows(), total=len(predictions), desc="Visualizing submission predictions"):
        written_paths.append(visualize_video(row=row, video_dir=args.video_dir, output_dir=args.output_dir, overwrite=args.overwrite, dot_window_seconds=args.dot_window_seconds))
    print(f"Saved {len(written_paths)} visualization videos to {args.output_dir}")

if __name__ == "__main__":
    main()