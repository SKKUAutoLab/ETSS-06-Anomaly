import argparse
import random
import shutil
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import rfdetr
from rfdetr.config import TrainConfig
from rfdetr.training import RFDETRModelModule, build_trainer

SEED = 42
DEFAULT_CLASS_NAMES = ["head-on", "rear-end", "sideswipe", "single", "t-bone"]
RFDETR_VARIANTS = {"n": "RFDETRNano", "nano": "RFDETRNano", "s": "RFDETRSmall", "small": "RFDETRSmall", "m": "RFDETRMedium", "medium": "RFDETRMedium", "l": "RFDETRLarge",
                   "large": "RFDETRLarge", "xl": "RFDETRXLarge", "xlarge": "RFDETRXLarge", "2xl": "RFDETR2XLarge", "2xlarge": "RFDETR2XLarge"}

def create_rfdetr(variant: str, **kwargs):
    key = variant.lower()
    if key not in RFDETR_VARIANTS:
        choices = ", ".join(sorted(RFDETR_VARIANTS))
        raise ValueError(f"Unknown RF-DETR variant '{variant}'. Choices: {choices}")
    class_name = RFDETR_VARIANTS[key]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        available = ", ".join(name for name in sorted(dir(rfdetr)) if name.startswith("RFDETR"))
        raise ValueError(f"{class_name} is not available in the installed rfdetr package. "
                         f"Available RF-DETR classes: {available}")
    return model_cls(**kwargs)

def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RealAccidentDataset(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, records: list[dict], dataset_root: Path, resolution: int, augment: bool = False):
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        transforms = []
        if augment:
            transforms.extend([A.HorizontalFlip(p=0.5), A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6), A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                               A.ImageCompression(quality_range=(40, 95), p=0.3)])
        transforms.append(A.Resize(resolution, resolution))
        self.transform = A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.2))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        video_path = self.dataset_root / record["path"]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame"])
        ok, bgr = cap.read()
        cap.release()
        if ok:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        h0, w0 = image.shape[:2]
        transformed = self.transform(image=image, bboxes=record["bboxes"], cls=record["cls"])
        image = transformed["image"]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(torch.tensor(self.MEAN).view(3, 1, 1))
        tensor.div_(torch.tensor(self.STD).view(3, 1, 1))
        bboxes = transformed["bboxes"]
        cls = transformed["cls"]
        if bboxes:
            boxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(cls, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        return tensor, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "orig_size": torch.tensor([h0, w0]), "size": torch.tensor([self.resolution, self.resolution])}

class RealAccidentDataModule(LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame, dataset_root: Path, resolution: int, class_names: list[str], batch_size: int, neg_per_video: int = 2,
                 neg_margin_seconds: float = 2.0, num_workers: int = 2):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        self.class_names = list(class_names)
        self.type_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.batch_size = batch_size
        self.neg_per_video = neg_per_video
        self.neg_margin_seconds = neg_margin_seconds
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:
        self.train_ds = RealAccidentDataset(self._build_records(self.train_df), self.dataset_root, self.resolution, augment=True)
        self.val_ds = RealAccidentDataset(self._build_records(self.val_df), self.dataset_root, self.resolution, augment=False)
        print(f"DataModule train={len(self.train_ds)} val={len(self.val_ds)}")

    def _build_records(self, metadata: pd.DataFrame) -> list[dict]:
        records = []
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Building records"):
            video_path = self.dataset_root / str(row["path"])
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total <= 0:
                continue
            accident_frame = int(round(float(row["accident_frame"])))
            accident_frame = max(0, min(accident_frame, total - 1))
            cx = float((row["x1"] + row["x2"]) / 2.0)
            cy = float((row["y1"] + row["y2"]) / 2.0)
            bw = float(max(row["x2"] - row["x1"], 0.01))
            bh = float(max(row["y2"] - row["y1"], 0.01))
            class_id = self.type_to_id[str(row["type"])]
            records.append({"path": str(row["path"]), "frame": accident_frame, "bboxes": [[cx, cy, bw, bh]], "cls": [class_id]})
            margin = int(round(self.neg_margin_seconds * fps))
            negatives = []
            if accident_frame - margin > 5:
                negatives.append((accident_frame - margin) // 2)
            if accident_frame + margin < total - 5:
                negatives.append((accident_frame + margin + total) // 2)
            while len(negatives) < self.neg_per_video:
                negatives.append(min(3, total - 1))
            for frame in negatives[: self.neg_per_video]:
                records.append({"path": str(row["path"]), "frame": int(frame), "bboxes": [], "cls": []})
        return records

    @staticmethod
    def _collate(batch):
        from rfdetr.utilities import nested_tensor_from_tensor_list
        images, targets = zip(*batch)
        return nested_tensor_from_tensor_list(list(images)), list(targets)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF-DETR on real ACCIDENT split.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-metadata", type=Path, default=Path("dataset/metadata-train-real.csv"))
    parser.add_argument("--val-metadata", type=Path, default=Path("dataset/metadata-test-real.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("rfdetr_real_ckpt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--variant", default="large", choices=sorted(RFDETR_VARIANTS), help="RF-DETR variant to train. Default keeps the notebook's large model.")
    parser.add_argument("--keep-output-dir", action="store_true")
    args = parser.parse_args()

    set_reproducibility(args.seed)
    train_df = pd.read_csv(args.train_metadata)
    val_df = pd.read_csv(args.val_metadata)
    class_names = [name for name in DEFAULT_CLASS_NAMES if name in set(train_df["type"])]
    rfdetr_model = create_rfdetr(args.variant)
    model_config = rfdetr_model.model_config
    model_config.num_classes = len(class_names)
    resolution = model_config.resolution
    if args.output_dir.exists() and not args.keep_output_dir:
        shutil.rmtree(args.output_dir)
    train_config = TrainConfig(dataset_dir=str(args.dataset_root), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_dir=str(args.output_dir),
                               class_names=class_names, lr_scheduler="cosine", checkpoint_interval=max(1, args.val_interval), eval_interval=max(1, args.val_interval),
                               compute_val_loss=True, progress_bar="tqdm", devices=1, grad_accum_steps=args.grad_accum_steps)
    module = RFDETRModelModule(model_config, train_config)
    datamodule = RealAccidentDataModule(train_df=train_df, val_df=val_df, dataset_root=args.dataset_root, resolution=resolution, class_names=class_names,
                                        batch_size=args.batch_size, num_workers=args.num_workers)
    trainer = build_trainer(train_config, model_config, check_val_every_n_epoch=max(1, args.val_interval))
    print(f"classes={class_names}")
    print(f"train rows={len(train_df)} val rows={len(val_df)}")
    print(f"variant={args.variant} epochs={args.epochs} val_interval={args.val_interval} "
          f"batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
          f"resolution={resolution} cuda={torch.cuda.is_available()}")
    trainer.fit(module, datamodule)
    print(f"Training complete. Checkpoints saved in {args.output_dir}")

if __name__ == "__main__":
    main()