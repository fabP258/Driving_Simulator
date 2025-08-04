from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tokenizer.models.vqgan import VQModel
from tokenizer.data.dataset import ImageDataset
from tokenizer.data.token_dataset import VideoTokenDataset
from tokenizer.engine.trainer import Trainer as CustomTrainer
from tokenizer.models.world_model import WorldModel


def train(
    train_image_root_path: str,
    val_image_root_path: str | None,
    config_path: str | None,
    checkpoint_path: str | None,
    batch_size: int,
    num_workers: int,
    grad_acc_steps: int,
):
    if config_path:
        config = OmegaConf.load(config_path)
        model = VQModel(
            **config.model.params,
            base_learning_rate=config.model.base_learning_rate,
            grad_acc_steps=grad_acc_steps,
        )
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
    else:
        if not checkpoint_path:
            assert ValueError(
                "Unable to instantiate VQModel with both missing config and checkpoint."
            )
        model = VQModel.load_from_checkpoint(checkpoint_path)

    train_ds = ImageDataset(folder_path=train_image_root_path)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = None
    if val_image_root_path:
        val_ds = ImageDataset(folder_path=val_image_root_path)
        val_dl = DataLoader(
            val_ds, batch_size=8 * batch_size, shuffle=False, num_workers=num_workers
        )

    dt = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("checkpoints") / dt,
        filename="vqgan_driving_{step}",
        save_top_k=-1,
        every_n_train_steps=2000,
    )
    trainer = Trainer(
        max_epochs=20,
        devices=1,
        accelerator="gpu",
        logger=TensorBoardLogger(f"./logs/{dt}", name="vqgan"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=checkpoint_path,
    )


def train_wm(
    video_token_root_path: str,
    config_path: str,
    checkpoint_path: str,
    batch_size: int,
    num_workers: int,
    grad_acc_steps: int,
):
    ctor_args = {}
    if config_path:
        config = OmegaConf.load(config_path)
        ctor_args.update(**config.model)
    if checkpoint_path:
        module = WorldModel.from_checkpoint(checkpoint_path, None, ctor_args)
    else:
        module = WorldModel(**ctor_args)

    train_ds = VideoTokenDataset(
        video_token_root_path,
        module.hparams["n_cond_frames"],
        module.hparams["H"],
        module.hparams["W"],
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    dt = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    trainer = CustomTrainer(
        max_epochs=100,
        log_dir=f"./logs/{dt}",
        checkpoint_dir=Path("checkpoints") / dt,
        checkpoint_every_n_steps=10000,
    )
    trainer.fit(module, train_dl, grad_acc_steps=grad_acc_steps)


def _create_train_subparser(subparsers: _SubParsersAction) -> None:
    parser = subparsers.add_parser("train", description="Train a VQ-GAN model.")
    parser.add_argument(
        "--train_image_root_path",
        type=str,
        required=True,
        help="Path to the root folder containing images (png) for training.",
    )
    parser.add_argument(
        "--val_image_root_path",
        type=str,
        required=True,
        help="Path to the root folder containing images (png) for validation.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to the hyperparameter config.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=10,
        help="Size of mini-batches used for training and validation. Default: 10.",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        required=False,
        default=1,
        help="Number of gradient accumulation steps. Default: 1.",
    )
    parser.add_argument("--num_workers", type=int, required=False, default=2)
    parser.set_defaults(command=train)


def _create_train_wm_parser(subparsers: _SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train_wm", description="Train a transformer World Model."
    )
    parser.add_argument(
        "--video_token_root_path",
        type=str,
        required=True,
        help="Path to the root folder containing tokenized video tensors.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="Path to the model config yaml file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=10,
        help="Size of mini-batches used for training and validation. Default: 10.",
    )
    parser.add_argument("--num_workers", type=int, required=False, default=2)
    parser.add_argument("--grad_acc_steps", type=int, required=False, default=1)
    parser.set_defaults(command=train_wm)


def _create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="VQ-VAE Command Line Interface")

    subparsers = parser.add_subparsers()
    _create_train_subparser(subparsers)
    _create_train_wm_parser(subparsers)

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
