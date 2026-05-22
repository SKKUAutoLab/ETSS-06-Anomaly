import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from train import DEFAULT_CLASS_NAMES, RFDETR_VARIANTS, create_rfdetr
from PIL import Image

def find_best_checkpoint(ckpt_dir: Path) -> Path:
    for name in ["checkpoint_best_total.pth", "checkpoint_best_regular.pth"]:
        path = ckpt_dir / name
        if path.is_file():
            return path
    pths = sorted(ckpt_dir.glob("*.pth"))
    if pths:
        return pths[-1]
    ckpts = sorted(ckpt_dir.glob("**/*.ckpt"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

def load_rfdetr(checkpoint_path: Path, class_names: list[str], variant: str):
    rfdetr = create_rfdetr(variant, num_classes=len(class_names))
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data.get("model", checkpoint_data.get("state_dict", checkpoint_data))
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items() if key.startswith("model.")}
    rfdetr.model.model.load_state_dict(state_dict, strict=False)
    rfdetr.model.class_names = class_names
    return rfdetr

def extract_frames(video_path: Path, max_frames: int) -> list[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, total // max_frames)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append((idx, idx / fps, frame))
        idx += 1
    cap.release()
    return frames

def run_inference(test_df: pd.DataFrame, model, dataset_root: Path, class_names: list[str], max_frames: int, threshold: float, fallback_type: str) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        video_path = dataset_root / str(row["path"])
        frames = extract_frames(video_path, max_frames=max_frames)
        best_conf = 0.0
        best_time = float(row["duration"]) / 2.0
        best_cx = 0.5
        best_cy = 0.5
        best_type = fallback_type
        if frames:
            h, w = frames[0][2].shape[:2]
            for _, timestamp, bgr in frames:
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                detections = model.predict(image, threshold=threshold)
                if (len(detections) == 0 or detections.confidence is None or len(detections.confidence) == 0):
                    continue
                top = int(np.argmax(detections.confidence))
                confidence = float(detections.confidence[top])
                if confidence <= best_conf:
                    continue
                best_conf = confidence
                best_time = float(timestamp)
                box = detections.xyxy[top]
                best_cx = float(np.clip((box[0] + box[2]) / 2.0 / w, 0, 1))
                best_cy = float(np.clip((box[1] + box[3]) / 2.0 / h, 0, 1))
                if detections.class_id is not None:
                    class_id = int(detections.class_id[top])
                    if 0 <= class_id < len(class_names):
                        best_type = class_names[class_id]
        rows.append({"path": row["path"], "accident_time": round(best_time, 2), "center_x": round(best_cx, 3), "center_y": round(best_cy, 3), "type": best_type})
    return pd.DataFrame(rows)

def gaussian_score(pred: float, gt: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((pred - gt) / sigma) ** 2))

def evaluate(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    merged = predictions.merge(truth, on="path", suffixes=("_pred", "_gt"))
    if merged.empty:
        raise ValueError("No matching paths between predictions and ground truth.")
    temporal = []
    spatial = []
    classification = []
    abs_time_error = []
    acc_05 = []
    acc_1 = []
    acc_2 = []
    for _, row in merged.iterrows():
        dt = abs(float(row["accident_time_pred"]) - float(row["accident_time_gt"]))
        abs_time_error.append(dt)
        acc_05.append(dt <= 0.5)
        acc_1.append(dt <= 1.0)
        acc_2.append(dt <= 2.0)
        temporal.append(gaussian_score(row["accident_time_pred"], row["accident_time_gt"], sigma=2.0))
        dist = np.hypot(float(row["center_x_pred"]) - float(row["center_x_gt"]), float(row["center_y_pred"]) - float(row["center_y_gt"]))
        spatial.append(gaussian_score(dist, 0.0, sigma=0.1))
        classification.append(float(row["type_pred"] == row["type_gt"]))
    temporal_score = float(np.mean(temporal))
    spatial_score = float(np.mean(spatial))
    classification_score = float(np.mean(classification))
    harmonic = 3.0 / (1 / max(temporal_score, 1e-9) + 1 / max(spatial_score, 1e-9) + 1 / max(classification_score, 1e-9))
    return {"n": float(len(merged)), "temporal_gaussian_sigma_2": temporal_score, "spatial_gaussian_sigma_0.1": spatial_score, "classification_accuracy": classification_score,
            "harmonic_mean": harmonic, "time_mae_seconds": float(np.mean(abs_time_error)), "time_accuracy_at_0.5s": float(np.mean(acc_05)),
            "time_accuracy_at_1s": float(np.mean(acc_1)), "time_accuracy_at_2s": float(np.mean(acc_2))}

def resolve_class_names(metadata: pd.DataFrame) -> list[str]:
    present = set(metadata["type"].astype(str))
    if not (present & set(DEFAULT_CLASS_NAMES)):
        raise ValueError("No known accident types found in metadata.")
    return list(DEFAULT_CLASS_NAMES)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test RF-DETR on real ACCIDENT split.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--metadata", type=Path, default=Path("dataset/metadata-test-real.csv"))
    parser.add_argument("--train-metadata", type=Path, default=Path("dataset/metadata-train-real.csv"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("rfdetr_real_ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("rfdetr_real_test"))
    parser.add_argument("--submission-name", default="submission.csv")
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--variant", default="large", choices=sorted(RFDETR_VARIANTS), help="RF-DETR variant used by the checkpoint. Default keeps the notebook's large model.")
    args = parser.parse_args()

    test_df = pd.read_csv(args.metadata)
    train_df = pd.read_csv(args.train_metadata)
    class_names = resolve_class_names(test_df)
    fallback_type = train_df["type"].value_counts().index[0]
    checkpoint = args.checkpoint if args.checkpoint is not None else find_best_checkpoint(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint={checkpoint}")
    print(f"variant={args.variant}")
    print(f"classes={class_names}")
    print(f"fallback_type={fallback_type}")
    print(f"test rows={len(test_df)} cuda={torch.cuda.is_available()}")
    model = load_rfdetr(checkpoint, class_names, args.variant)
    submission = run_inference(test_df=test_df, model=model, dataset_root=args.dataset_root, class_names=class_names, max_frames=args.max_frames_per_video, threshold=args.threshold, fallback_type=fallback_type)
    submission_path = args.output_dir / args.submission_name
    submission.to_csv(submission_path, index=False)
    metrics = evaluate(submission, test_df[["path", "accident_time", "center_x", "center_y", "type"]])
    metrics_path = args.output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved {submission_path}")
    print(f"Saved {metrics_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()