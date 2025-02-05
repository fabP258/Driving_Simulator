import os
import torch
from typing import List
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalizes image from range [0, 1] to [-1,1]"""
    return x * 2.0 - 1.0


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    """Denormalizes image from range [-1, 1] to [0,1]"""
    return (x + 1.0) / 2.0


class DrivingDataset(Dataset):
    def __init__(self, segment_path_list: List[Path]):
        self.segment_path_list: List[Path] = []
        self.transforms = transforms.Compose(
            [transforms.Resize((256, 512)), transforms.ToTensor()]
        )

        self.frame_start_indices = []
        self.frame_count = 0
        for seg_path in segment_path_list:

            # check for data availability
            if not (seg_path / "images").is_dir():
                continue

            # get number of images
            image_filenames = [
                fn for fn in os.listdir(seg_path / "images") if fn.endswith(".png")
            ]

            # skip segment if image directory does not contain any images
            if len(image_filenames) == 0:
                continue

            self.segment_path_list.append(seg_path)
            self.frame_start_indices.append(self.frame_count)
            self.frame_count += len(image_filenames)

    def __len__(self):
        return self.frame_count

    def read_image(self, image_path: Path):
        raw_image = Image.open(image_path)
        rgb_image = raw_image.convert("RGB")
        return rgb_image

    def get_segment_index(self, index: int) -> int:
        if index < self.frame_start_indices[-1]:
            segment_index = next(
                x[0] for x in enumerate(self.frame_start_indices) if x[1] > index
            )
        else:
            segment_index = len(self.frame_start_indices)
        segment_index -= 1
        return segment_index

    def __getitem__(self, index):
        segment_index = self.get_segment_index(index)
        frame_index = index - self.frame_start_indices[segment_index]
        image_path = (
            self.segment_path_list[segment_index] / f"images/{frame_index:04}.png"
        )
        image_tensor = self.transforms(self.read_image(image_path))
        image_tensor = normalize_image(image_tensor)
        return image_tensor
