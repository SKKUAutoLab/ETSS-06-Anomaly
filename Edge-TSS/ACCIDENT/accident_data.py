from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

class SyntheticAccidentDataset(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, records, syn_root, resolution, augment=False):
        self.records = records
        self.syn_root = Path(syn_root)
        self.resolution = resolution
        tfms = []
        if augment:
            tfms.extend([A.HorizontalFlip(p=0.5), A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6), A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                         A.ImageCompression(quality_range=(40, 95), p=0.3)])
        tfms.append(A.Resize(resolution, resolution))
        self._tfm = A.Compose(tfms, bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.2))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        cap = cv2.VideoCapture(str(self.syn_root / rec["vpath"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
        ok, bgr = cap.read()
        cap.release()
        if ok:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        h0, w0 = img.shape[:2]
        out = self._tfm(image=img, bboxes=rec["bboxes"], cls=rec["cls"])
        img = out["image"]
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        t.sub_(torch.tensor(self.MEAN).view(3, 1, 1))
        t.div_(torch.tensor(self.STD).view(3, 1, 1))
        bboxes = out["bboxes"]
        cls = out["cls"]
        if bboxes:
            boxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(cls, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        return t, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "orig_size": torch.tensor([h0, w0]), "size": torch.tensor([self.resolution, self.resolution])}

class AccidentDataModule(LightningDataModule):
    def __init__(self, syn_df, syn_root, resolution, collision_types, batch_size=4, val_split=0.1, neg_per_vid=2, neg_margin=2.0, num_workers=2):
        super().__init__()
        self.syn_df = syn_df
        self.syn_root = Path(syn_root)
        self.resolution = resolution
        self.class_names = list(collision_types)
        self.type_to_id = {t: i for i, t in enumerate(collision_types)}
        self.batch_size = batch_size
        self.val_split = val_split
        self.neg_per_vid = neg_per_vid
        self.neg_margin = neg_margin
        self.num_workers = num_workers

    def setup(self, stage=None):
        shuffled = self.syn_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = max(1, int(len(shuffled) * self.val_split))
        train_recs, val_recs = [], []
        for idx, row in tqdm(shuffled.iterrows(), total=len(shuffled), desc="Building index"):
            recs = val_recs if idx < n_val else train_recs
            vpath = row["rgb_path"]
            cap = cv2.VideoCapture(str(self.syn_root / vpath))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            acc_f = min(int(row["accident_time"] * fps), total - 1)
            cx = (row["x1"] + row["x2"]) / 2.0
            cy = (row["y1"] + row["y2"]) / 2.0
            bw = max(row["x2"] - row["x1"], 0.01)
            bh = max(row["y2"] - row["y1"], 0.01)
            cid = self.type_to_id.get(row["type"], 0)
            recs.append({"vpath": vpath, "frame": acc_f, "bboxes": [[cx, cy, bw, bh]], "cls": [cid]})
            margin = int(self.neg_margin * fps)
            negs = []
            if acc_f - margin > 5:
                negs.append((acc_f - margin) // 2)
            if acc_f + margin < total - 5:
                negs.append((acc_f + margin + total) // 2)
            while len(negs) < self.neg_per_vid:
                negs.append(min(3, total - 1))
            for nf in negs[:self.neg_per_vid]:
                recs.append({"vpath": vpath, "frame": nf, "bboxes": [], "cls": []})
        self.train_ds = SyntheticAccidentDataset(train_recs, self.syn_root, self.resolution, augment=True)
        self.val_ds = SyntheticAccidentDataset(val_recs, self.syn_root, self.resolution, augment=False)
        print(f"  DataModule — "
              f"train: {len(self.train_ds)}  val: {len(self.val_ds)}")

    @staticmethod
    def _collate(batch):
        from rfdetr.utilities import nested_tensor_from_tensor_list
        imgs, tgts = zip(*batch)
        return (nested_tensor_from_tensor_list(list(imgs)), list(tgts))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate, pin_memory=True)