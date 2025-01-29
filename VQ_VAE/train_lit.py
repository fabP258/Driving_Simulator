import os
import random
from pathlib import Path

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import DrivingDataset
from lit_vq_vae import LitVqVae
from vq_vae import VqVaeConfig


def split_list(list_to_split: list, shuffle: bool = False, ratio: float = 0.5):
    if shuffle:
        random.seed(10)
        random.shuffle(list_to_split)
    split_index = int(len(list_to_split) * ratio)
    if split_index == 0:
        raise ValueError(
            f"List of length {len(list_to_split)} can't be splitted with ratio {ratio}."
        )
    front_sublist = list_to_split[:split_index]
    back_sublist = list_to_split[split_index:]
    return front_sublist, back_sublist


def get_segment_folders():
    segment_folders = []

    for chunk_idx in range(4, 5):
        chunk_path = "/home/fabio/comma2k19/Chunk_" + f"{chunk_idx}"
        print(chunk_path)
        baselevel = len(chunk_path.split(os.path.sep))
        for subdirs, dirs, files in os.walk(chunk_path):
            curlevel = len(subdirs.split(os.path.sep))
            if (curlevel - baselevel) == 2:
                segment_folders.append(Path(subdirs))

    return segment_folders


# Setup logger
log_path = Path(__file__).parent / "logs"
log_path.mkdir(exist_ok=True, parents=True)
logger = TensorBoardLogger(log_path, name="vq_gan_logs")

vq_vae = LitVqVae(VqVaeConfig())
segment_folders = get_segment_folders()
segment_folders_train, segment_folders_test = split_list(
    segment_folders[:20], shuffle=True, ratio=0.8
)
train_dataloader = DataLoader(
    DrivingDataset(segment_folders_train),
    batch_size=4,
    shuffle=True,
    num_workers=20,
)
val_dataloader = DataLoader(
    DrivingDataset(segment_folders_test),
    batch_size=4,
    shuffle=False,
    num_workers=20,
)
trainer = lightning.Trainer(
    accelerator="gpu", fast_dev_run=False, max_steps=400000, logger=logger
)
trainer.fit(
    model=vq_vae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)
