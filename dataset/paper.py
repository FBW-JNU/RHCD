"""Distinct-secret sampling and explicit crop/target geometry for RHCD."""

import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from dataset.div2k import imread, to_float_tensor
from dataset.transform import augment, center_crop_or_resize, random_crop


class RHCDDataset(Dataset):
    def __init__(self, data_list, cfg, training=False):
        self.cfg, self.training, self.epoch = cfg, training, 0
        root = Path(cfg.data_root).expanduser().resolve()
        self.images = []
        for line in Path(data_list).read_text().splitlines():
            if line.strip():
                path = Path(line.strip()).expanduser()
                self.images.append(str((path if path.is_absolute() else root / path).resolve()))
        self.pool = sorted(set(self.images))
        self.pool_index = {path: index for index, path in enumerate(self.pool)}
        if len(self.pool) < int(cfg.rounds) + 1:
            raise ValueError("Each split needs at least rounds+1 unique source images")
        missing = [path for path in self.pool if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(missing[:5])

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _seed(self, index):
        seed = int(self.cfg.manual_seed if self.training else self.cfg.fixed_val_secret_seed)
        return seed * 1000000007 + (self.epoch if self.training else 0) * 1000003 + int(index)

    def sample_paths(self, index):
        rng = random.Random(self._seed(index))
        cover = self.images[index]
        excluded = self.pool_index[cover]
        selected = rng.sample(range(len(self.pool) - 1), int(self.cfg.rounds))
        return [cover] + [self.pool[value + (value >= excluded)] for value in selected]

    def __getitem__(self, index):
        state = random.getstate()
        random.seed(self._seed(index))
        try:
            size = int(self.cfg.train_crop_size if self.training else self.cfg.val_crop_size)
            images = []
            for path in self.sample_paths(index):
                image = imread(path)
                if self.training:
                    image = random_crop(image, size)
                    image = augment(image, hflip=bool(self.cfg.hflip), rotation=bool(self.cfg.rotation))
                else:
                    image = center_crop_or_resize(image, size)
                images.append(to_float_tensor(np.ascontiguousarray(image)))
        finally:
            random.setstate(state)
        keys = ["img_gt", "img_sec"] + [f"img_sec_{index}" for index in range(2, int(self.cfg.rounds) + 1)]
        return dict(zip(keys, images), sample_index=index)
