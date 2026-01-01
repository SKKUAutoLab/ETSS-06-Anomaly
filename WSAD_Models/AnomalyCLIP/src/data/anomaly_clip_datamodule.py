from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.augmentations import get_augmentations

class AnomalyCLIPDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/", train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000), batch_size: int = 64, num_workers: int = 0,
                 pin_memory: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.load_from_features:
            self.transform_train = None
            self.transform_val = None
        else:
            self.transform_train = get_augmentations(self.hparams.input_size, self.hparams.ncrops)
            self.transform_val = get_augmentations(self.hparams.input_size, self.hparams.ncrops)
        self.train_data_normal: Optional[Dataset] = None
        self.train_data_anomaly: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.train_data_normal_test_mode: Optional[Dataset] = None

    @property
    def num_classes(self):
        return self.hparams.num_classes

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.hparams.load_from_features:
            from data.components.feature_dataset import VideoFrameDataset
        else:
            from data.components.video_dataset import VideoFrameDataset
        if not self.train_data_normal and not self.train_data_anomaly and not self.test_data:
            self.train_data_normal = VideoFrameDataset(root_path=self.hparams.frames_root, annotationfile_path=self.hparams.annotation_file_normal, normal_id=self.hparams.normal_id,
                                                       num_segments=self.hparams.num_segments, frames_per_segment=self.hparams.seg_length, imagefile_template=self.hparams.image_tmpl,
                                                       transform=self.transform_train, ncrops=self.hparams.ncrops, stride=self.hparams.stride)
            self.train_data_anomaly = VideoFrameDataset(root_path=self.hparams.frames_root, annotationfile_path=self.hparams.annotation_file_anomaly, normal_id=self.hparams.normal_id,
                                                        num_segments=self.hparams.num_segments, frames_per_segment=self.hparams.seg_length, imagefile_template=self.hparams.image_tmpl,
                                                        transform=self.transform_train, ncrops=self.hparams.ncrops, stride=self.hparams.stride, spatialannotationdir_path=self.hparams.spatialannotationdir_path)
            self.test_data = VideoFrameDataset(root_path=self.hparams.frames_root, annotationfile_path=self.hparams.annotation_file_test, normal_id=self.hparams.normal_id,
                                               num_segments=self.hparams.num_segments, frames_per_segment=self.hparams.seg_length, imagefile_template=self.hparams.image_tmpl,
                                               transform=self.transform_val, test_mode=True, ncrops=self.hparams.ncrops, temporal_annotation_file=self.hparams.annotation_file_temporal_test,
                                               labels_file=self.hparams.labels_file, stride=self.hparams.stride)
            self.train_data_normal_test_mode = VideoFrameDataset(root_path=self.hparams.frames_root, annotationfile_path=self.hparams.annotation_file_normal, normal_id=self.hparams.normal_id,
                                                                 num_segments=self.hparams.num_segments, frames_per_segment=self.hparams.seg_length, imagefile_template=self.hparams.image_tmpl,
                                                                 transform=self.transform_val, test_mode=True, ncrops=self.hparams.ncrops, stride=self.hparams.stride)

    def train_dataloader(self):
        train_loader_normal = DataLoader(dataset=self.train_data_normal, batch_size=self.hparams.batch_size // 2, num_workers=self.hparams.num_workers // 2,
                                         pin_memory=self.hparams.pin_memory, shuffle=True, drop_last=True)
        train_loader_abnormal = DataLoader(dataset=self.train_data_anomaly, batch_size=self.hparams.batch_size // 2, num_workers=self.hparams.num_workers // 2,
                                           pin_memory=self.hparams.pin_memory, shuffle=True, drop_last=True)
        return [train_loader_normal, train_loader_abnormal]

    def val_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.hparams.batch_size_test, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.hparams.batch_size_test, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, drop_last=False)

    def train_dataloader_test_mode(self):
        return DataLoader(dataset=self.train_data_normal_test_mode, batch_size=self.hparams.batch_size_test, num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, drop_last=False)

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

if __name__ == "__main__":
    _ = AnomalyCLIPDataModule()