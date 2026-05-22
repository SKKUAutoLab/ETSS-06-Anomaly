import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from test import evaluate, find_best_checkpoint, load_rfdetr, resolve_class_names

Signal = QtCore.Signal
DATASET_ROOT = Path("dataset")
TEST_METADATA = Path("dataset/metadata-test-real.csv")
TRAIN_METADATA = Path("dataset/metadata-train-real.csv")
CHECKPOINT_DIR = Path("rfdetr_real_ckpt")
OUTPUT_DIR = Path("rfdetr_real_test")
SUBMISSION_NAME = "submission.csv"
VIS_DIR = Path("vis_qt")
VARIANT = "large"
MAX_FRAMES_PER_VIDEO = 120
THRESHOLD = 0.25
VIS_WINDOW_SECONDS = 0.5
PLAYBACK_FPS_FALLBACK = 25.0
DISPLAY_TIME_ACCURACY_SECONDS = 3.5

def draw_prediction_overlay(frame_bgr: np.ndarray, center_x: float, center_y: float, accident_time: float, accident_type: str, confidence: float) -> np.ndarray:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    x = int(np.clip(center_x, 0.0, 1.0) * max(w - 1, 1))
    y = int(np.clip(center_y, 0.0, 1.0) * max(h - 1, 1))
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 16, (255, 255, 255), 2)
    label = f"type: {accident_type}  time: {accident_time:.2f}s  conf: {confidence:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (8, 8), (tw + 20, th + baseline + 20), (0, 0, 0), -1)
    cv2.putText(frame, label, (14, th + 14), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def extract_frames(video_path: Path, max_frames: int) -> list[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or PLAYBACK_FPS_FALLBACK
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

def predict_one_video(row, model, class_names: list[str], fallback_type: str):
    video_path = DATASET_ROOT / str(row["path"])
    frames = extract_frames(video_path, max_frames=MAX_FRAMES_PER_VIDEO)
    best_conf = 0.0
    best_frame_idx = -1
    best_time = float(row["duration"]) / 2.0
    best_cx = 0.5
    best_cy = 0.5
    best_type = fallback_type
    best_bgr = None
    if frames:
        h, w = frames[0][2].shape[:2]
        best_frame_idx, _, best_bgr = frames[0]
        for frame_idx, timestamp, bgr in frames:
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            detections = model.predict(image, threshold=THRESHOLD)
            if (len(detections) == 0 or detections.confidence is None or len(detections.confidence) == 0):
                continue
            top = int(np.argmax(detections.confidence))
            confidence = float(detections.confidence[top])
            if confidence <= best_conf:
                continue
            best_conf = confidence
            best_frame_idx = int(frame_idx)
            best_time = float(timestamp)
            best_bgr = bgr
            box = detections.xyxy[top]
            best_cx = float(np.clip((box[0] + box[2]) / 2.0 / w, 0, 1))
            best_cy = float(np.clip((box[1] + box[3]) / 2.0 / h, 0, 1))
            if detections.class_id is not None:
                class_id = int(detections.class_id[top])
                if 0 <= class_id < len(class_names):
                    best_type = class_names[class_id]
    if best_bgr is None:
        best_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    prediction = {"path": row["path"], "accident_time": round(best_time, 2), "center_x": round(best_cx, 3), "center_y": round(best_cy, 3), "type": best_type}
    details = {"frame_idx": best_frame_idx, "confidence": best_conf, "display_bgr": best_bgr, "video_path": video_path}
    return prediction, details

def save_visualization_video(video_path: Path, prediction: dict, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or PLAYBACK_FPS_FALLBACK
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        if abs(timestamp - float(prediction["accident_time"])) <= VIS_WINDOW_SECONDS:
            frame = draw_prediction_overlay(frame, float(prediction["center_x"]), float(prediction["center_y"]), float(prediction["accident_time"]), str(prediction["type"]), 0.0)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()

class InferenceWorker(QtCore.QObject):
    log = Signal(str)
    frame = Signal(object)
    finished = Signal(dict, list)
    error = Signal(str)

    @QtCore.Slot()
    def run(self):
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            VIS_DIR.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Loading test metadata: {TEST_METADATA}")
            test_df = pd.read_csv(TEST_METADATA)
            train_df = pd.read_csv(TRAIN_METADATA)
            class_names = resolve_class_names(test_df)
            fallback_type = train_df["type"].value_counts().index[0]
            checkpoint = find_best_checkpoint(CHECKPOINT_DIR)
            self.log.emit(f"Loading checkpoint: {checkpoint}")
            self.log.emit(f"Variant: {VARIANT}, CUDA: {torch.cuda.is_available()}")
            self.log.emit(f"Classes: {class_names}")
            self.log.emit(f"Fallback type from train metadata: {fallback_type}")
            model = load_rfdetr(checkpoint, class_names, VARIANT)
            rows = []
            vis_paths = []
            total = len(test_df)
            for idx, row in test_df.iterrows():
                video_start_time = time.time()
                prediction, details = predict_one_video(row, model, class_names, fallback_type)
                rows.append(prediction)
                display_bgr = draw_prediction_overlay(details["display_bgr"], float(prediction["center_x"]), float(prediction["center_y"]), float(prediction["accident_time"]),
                                                      str(prediction["type"]), float(details["confidence"]))
                self.frame.emit(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB))
                video_name = Path(str(row["path"])).stem + ".mp4"
                vis_path = VIS_DIR / video_name
                save_visualization_video(details["video_path"], prediction, vis_path)
                if vis_path.is_file():
                    vis_paths.append(str(vis_path))
                process_time = time.time() - video_start_time
                self.log.emit(f"[{idx + 1}/{total}], Frame: {row['path'].split('/')[-1]}, "
                              f"Pred Accident Time: {prediction['accident_time']:.2f}s, "
                              f"Center: ({prediction['center_x']:.3f}, {prediction['center_y']:.3f}), "
                              f"Accident Type: {prediction['type']}, Confidence={details['confidence']:.3f}, "
                              f"Processed Time: {process_time:.2f}s")
            submission = pd.DataFrame(rows)
            submission_path = OUTPUT_DIR / SUBMISSION_NAME
            submission.to_csv(submission_path, index=False)
            metrics = evaluate(submission, test_df[["path", "accident_time", "center_x", "center_y", "type"]])
            merged = submission.merge(test_df[["path", "accident_time"]], on="path", suffixes=("_pred", "_gt"))
            time_error = (merged["accident_time_pred"].astype(float) - merged["accident_time_gt"].astype(float)).abs()
            display_accuracy_key = f"time_accuracy_at_{DISPLAY_TIME_ACCURACY_SECONDS:.1f}s"
            metrics[display_accuracy_key] = float((time_error <= DISPLAY_TIME_ACCURACY_SECONDS).mean())
            metrics_path = OUTPUT_DIR / "metrics.csv"
            pd.DataFrame([{"Accuracy": metrics[display_accuracy_key]}]).to_csv(metrics_path, index=False)
            self.log.emit(f"Saved submission: {submission_path}")
            self.log.emit(f"Saved metrics: {metrics_path}")
            self.log.emit(f"Saved visualizations: {VIS_DIR}")
            self.log.emit(f"Accuracy: {metrics[display_accuracy_key] * 100.0:.2f}%")
            self.finished.emit(metrics, vis_paths)
        except Exception as exc:
            self.error.emit(str(exc))

class ClickableVideoLabel(QtWidgets.QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ClickableSlider(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            ratio = event.position().x() / max(self.width(), 1)
            value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
            self.setValue(int(np.clip(value, self.minimum(), self.maximum())))
        super().mousePressEvent(event)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Accident Detection")
        self.resize(1280, 920)
        self.thread = None
        self.worker = None
        self.video_capture = None
        self.video_paths = []
        self.video_index = 0
        self.video_frame_count = 0
        self.video_fps = PLAYBACK_FPS_FALLBACK
        self.video_is_playing = False
        self.slider_is_updating = False
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.play_next_frame)
        self.run_button = QtWidgets.QPushButton("Run Testing")
        self.run_button.setMinimumHeight(44)
        self.run_button.clicked.connect(self.start_inference)
        self.image_label = ClickableVideoLabel("Click Run Testing button to start")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setStyleSheet("background-color: #111; color: white;")
        self.image_label.clicked.connect(self.toggle_video_playback)
        self.play_button = QtWidgets.QPushButton("Start")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_video_playback)
        self.previous_button = QtWidgets.QPushButton("Previous Video")
        self.previous_button.setEnabled(False)
        self.previous_button.clicked.connect(self.play_previous_video)
        self.next_button = QtWidgets.QPushButton("Next Video")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.play_next_video)
        self.video_slider = ClickableSlider(QtCore.Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        self.video_slider.valueChanged.connect(self.seek_video)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        self.metrics_label = QtWidgets.QLabel("The metric will appear here after testing finishes.")
        self.metrics_label.setMinimumHeight(72)
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        self.metrics_label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.image_label, stretch=1)
        playback_layout = QtWidgets.QHBoxLayout()
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.previous_button)
        playback_layout.addWidget(self.next_button)
        playback_layout.addWidget(self.video_slider, stretch=1)
        layout.addLayout(playback_layout)
        layout.addWidget(self.log_box)
        layout.addWidget(self.metrics_label)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def append_log(self, text: str):
        self.log_box.append(text)

    def start_inference(self):
        self.run_button.setEnabled(False)
        self.log_box.clear()
        self.metrics_label.setText("Testing is running...")
        self.stop_playback()
        self.play_button.setEnabled(False)
        self.play_button.setText("Start")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        self.video_paths = []
        self.video_index = 0
        self.thread = QtCore.QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.frame.connect(self.show_numpy_frame)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def show_numpy_frame(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimage = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def on_finished(self, metrics: dict, vis_paths: list):
        self.run_button.setEnabled(True)
        self.video_paths = list(vis_paths)
        self.video_index = 0
        self.metrics_label.setText(self.format_metrics(metrics))
        self.append_log(f"Testing finished. Visualization videos: {len(self.video_paths)}")
        if self.video_paths:
            self.start_video_playback(0)
        else:
            self.append_log("No visualization videos were saved.")

    def on_error(self, message: str):
        self.append_log(f"ERROR: {message}")
        self.metrics_label.setText("Testing failed. See log for details.")
        self.run_button.setEnabled(True)
        self.play_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.video_slider.setEnabled(False)

    def format_metrics(self, metrics: dict) -> str:
        key = f"time_accuracy_at_{DISPLAY_TIME_ACCURACY_SECONDS:.1f}s"
        return f"Accuracy: {float(metrics[key]) * 100.0:.2f}%"

    def stop_playback(self):
        self.video_timer.stop()
        self.video_is_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def start_video_playback(self, index: int):
        self.stop_playback()
        if not self.video_paths:
            return
        self.video_index = index % len(self.video_paths)
        path = self.video_paths[self.video_index]
        self.video_capture = cv2.VideoCapture(path)
        if not self.video_capture.isOpened():
            self.append_log(f"Could not open visualization video: {path}")
            self.play_next_video()
            return
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or PLAYBACK_FPS_FALLBACK
        self.slider_is_updating = True
        self.video_slider.setRange(0, max(self.video_frame_count - 1, 0))
        self.video_slider.setValue(0)
        self.slider_is_updating = False
        self.video_slider.setEnabled(self.video_frame_count > 0)
        self.play_button.setEnabled(True)
        self.previous_button.setEnabled(len(self.video_paths) > 1)
        self.next_button.setEnabled(len(self.video_paths) > 1)
        self.play_button.setText("Stop")
        self.video_is_playing = True
        self.append_log(f"Playing visualization: {Path(path).name}")
        self.play_next_frame()
        self.video_timer.start(max(int(1000 / self.video_fps), 1))

    def toggle_video_playback(self):
        if self.video_capture is None:
            return
        if self.video_is_playing:
            self.video_timer.stop()
            self.video_is_playing = False
            self.play_button.setText("Start")
        else:
            self.video_is_playing = True
            self.play_button.setText("Stop")
            self.video_timer.start(max(int(1000 / self.video_fps), 1))

    def seek_video(self, frame_index: int):
        if self.slider_is_updating or self.video_capture is None:
            return
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = self.video_capture.read()
        if not ok:
            return
        self.show_numpy_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def play_next_video(self):
        if not self.video_paths:
            return
        self.start_video_playback((self.video_index + 1) % len(self.video_paths))

    def play_previous_video(self):
        if not self.video_paths:
            return
        self.start_video_playback((self.video_index - 1) % len(self.video_paths))

    def play_next_frame(self):
        if self.video_capture is None:
            return
        ok, frame_bgr = self.video_capture.read()
        if not ok:
            self.play_next_video()
            return
        self.show_numpy_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.slider_is_updating = True
        self.video_slider.setValue(max(current_frame, 0))
        self.slider_is_updating = False

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()