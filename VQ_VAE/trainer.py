import random
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import DrivingDataset
from vq_vae import VQVAE


class VqVaeTrainer:

    def __init__(
        self,
        segment_folders: List[Path],
        output_path: Path,
        train_test_ratio: float = 0.8,
        batch_size: int = 10,
        num_workers: int = 12,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-8,
    ):

        # split the segment folders into test and training data
        self._segment_folders_train, self._segment_folders_test = (
            VqVaeTrainer.split_list(
                segment_folders, shuffle=True, ratio=train_test_ratio
            )
        )

        self._train_dataloader = DataLoader(
            DrivingDataset(self._segment_folders_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self._test_dataloader = DataLoader(
            DrivingDataset(self._segment_folders_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self._output_path = Path(output_path)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = VQVAE(
            in_channels=3,
            num_hiddens=256,
            num_downsampling_layers=4,
            num_residual_layers=2,
            num_residual_hiddens=256,
            embedding_dim=16,
            num_embeddings=512,
        )
        self._model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self._loss_fn = torch.nn.MSELoss()

        self._train_loss = []
        self._test_loss = []

    def train(self, epochs: int):

        self._train_loss = []
        self._test_loss = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._train_loss.append(self.train_single_epoch(epoch))
            # TODO: Run test
            self.generate_loss_plot(suffix=f"ep{epoch}")

    def train_single_epoch(self, epoch: int):
        self._model.train()

        epoch_loss = 0

        for batch, x in enumerate(self._train_dataloader):
            self._optimizer.zero_grad()
            x = x.to(self._device)
            x_recon = self._model(x)
            loss = self._loss_fn(x_recon, x)
            # TODO: Add commitment loss
            epoch_loss += loss.item()
            if batch % 100 == 0:
                print(
                    f"batch loss: {loss.item():>7f} [{batch:>5d}/{len(self._train_dataloader):>5d}]"
                )
                self.plot_sample(x[0, :], x_recon[0, :], epoch, batch, False)
            loss.backward()
            self._optimizer.step()

        epoch_loss /= len(self._train_dataloader)
        print(f"Mean epoch train loss: {epoch_loss}")

        return epoch_loss

    def generate_loss_plot(self, suffix: str = ""):
        fig, ax = plt.subplots()
        ax.plot(self._test_loss, label="Train")
        ax.legend()
        ax.set_xlabel("Epoch [-]")
        ax.set_ylabel("Loss")

        fig.savefig(self._output_path / f"loss_plot_{suffix}.png")

    @staticmethod
    def image_tensor_to_array(image: torch.Tensor):
        arr = image.cpu().detach().numpy()
        return np.transpose(arr, (1, 2, 0))

    def plot_sample(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        epoch: int,
        batch: int,
        validation: bool,
    ):
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(VqVaeTrainer.image_tensor_to_array(x))
        ax[0].axis("off")
        ax[0].set_title("Input image")
        ax[1].imshow(VqVaeTrainer.image_tensor_to_array(x_recon))
        ax[1].axis("off")
        ax[1].set_title("Reconstructed image")

        output_path = (
            self._output_path / "validation"
            if validation
            else self._output_path / "train"
        )
        output_path = output_path / f"ep{epoch:02}"
        output_path.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path / f"image_reconstruction_b{batch:06}.png")

    @staticmethod
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
