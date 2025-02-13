import random
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader

from trainer import VqVaeTrainer
from dataset import NpDataset


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


if __name__ == "__main__":

    # Setup logger
    log_path = (
        Path(__file__).parent.parent
        / "runs"
        / datetime.now().strftime("%Y%m%d_%H_%M_%S")
    )
    log_path.mkdir(exist_ok=True, parents=True)

    checkpoint_path = None

    checkpoint_path = (
        Path(__file__).parent.parent
        / "runs"
        / "20250212_18_19_38"
        / "checkpoint_step30000.pth"
    )

    tensor_paths = list(Path("/home/fabio/comma2k19").rglob("*.npy"))
    tensor_paths_train, tensor_paths_test = split_list(
        tensor_paths, shuffle=True, ratio=0.9
    )
    train_dataloader = DataLoader(
        NpDataset(tensor_paths_train),
        batch_size=4,
        shuffle=True,
        num_workers=1,
    )
    print(f"Training dataset contains {len(train_dataloader.dataset)} images.")
    val_dataloader = DataLoader(
        NpDataset(tensor_paths_test),
        batch_size=4,
        shuffle=False,
        num_workers=20,
    )
    if checkpoint_path is not None:
        trainer = VqVaeTrainer.from_checkpoint(checkpoint_path, log_path)
    else:
        trainer = VqVaeTrainer(log_path, gan_start_steps=100000)
    trainer.train(train_dataloader)
