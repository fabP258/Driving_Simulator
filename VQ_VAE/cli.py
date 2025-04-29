import random
from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader

from trainer import VqVaeTrainer
from dataset import ImageDataset


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


def train(
    image_root_path: str,
    checkpoint_path: str | None,
    logs_directory: str | None,
    train_test_ratio: float,
    batch_size: int,
    num_workers: int,
):
    log_path = logs_directory
    if log_path is None:
        log_path = Path(__file__).parent.parent
    log_path = log_path / "runs" / datetime.now().strftime("%Y%m%d_%H_%M_%S")
    log_path.mkdir(exist_ok=True, parents=True)

    image_paths = list(Path(image_root_path).rglob("*.png"))
    image_paths_train, image_paths_test = split_list(
        image_paths, shuffle=True, ratio=train_test_ratio
    )
    train_dataloader = DataLoader(
        ImageDataset(image_paths_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    print(
        f"Training dataset contains {len(train_dataloader.dataset)} images."
    )  # TODO: Use logger
    val_dataloader = DataLoader(
        ImageDataset(image_paths_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    if checkpoint_path is None:
        trainer = VqVaeTrainer(
            log_path, gan_start_steps=75000
        )  # TODO: Add config for c'tor args
    else:
        trainer = VqVaeTrainer.from_checkpoint(checkpoint_path, log_path)
    trainer.train(train_dataloader)


def _create_train_subparser(subparsers: _SubParsersAction) -> None:
    parser = subparsers.add_parser("train", description="Train a VQ-VAE model.")
    parser.add_argument(
        "--image_root_path",
        type=str,
        required=True,
        help="Path to the root folder containing images (png) for training.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--logs_directory",
        type=str,
        required=False,
        help="Directory where the logs shall be stored.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=10,
        help="Size of mini-batches used for training and validation. Default: 10.",
    )
    parser.add_argument("--num_workers", type=int, required=False, default=2)
    parser.add_argument(
        "--train_test_ratio",
        type=float,
        required=False,
        default=0.9,
        help="Ratio between training and test examples.",
    )
    parser.set_defaults(command=train)


def _create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="VQ-VAE Command Line Interface")

    subparsers = parser.add_subparsers()
    _create_train_subparser(subparsers)

    return parser


def main() -> int:
    """Main function"""

    parser = _create_parser()
    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")

    _ = command(**kwargs)

    return 0


if __name__ == "__main__":
    main()
