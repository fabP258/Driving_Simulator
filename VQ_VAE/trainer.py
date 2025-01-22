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
        num_workers: int = 18,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-8,
        loss_beta: float = 0.25,
        use_l2_normalization: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9,
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
            embedding_dim=256,
            num_embeddings=1024,
            use_l2_normalization=use_l2_normalization,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )
        print(
            f"Model parameters: {sum(p.numel() for p in self._model.parameters() if p.requires_grad)}"
        )
        self._model.to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=5, gamma=0.1
        )

        self._loss_fn = torch.nn.MSELoss()
        self._loss_beta = loss_beta

        self.initialize_loss_containers()

    def initialize_loss_containers(self):
        self._train_loss = {
            "reconstruction_loss": [],
            "commitment_loss": [],
        }
        self._test_loss = {"reconstruction_loss": []}
        self._embedding_entropy = []

    def train(self, epochs: int):

        self.initialize_loss_containers()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            recon_loss, commitment_loss = self.train_single_epoch(epoch)
            self._train_loss["commitment_loss"].append(commitment_loss)
            self._train_loss["reconstruction_loss"].append(recon_loss)
            self._lr_scheduler.step()
            self._test_loss["reconstruction_loss"].append(
                self.calculate_test_loss(epoch)
            )
            self._embedding_entropy.append(self._model.codebook_selection_entropy())
            self.generate_loss_plot(suffix=f"ep{epoch}")
            self.plot_codebook_usage(suffix=f"ep{epoch}")
            self._model.reset_codebook_usage_counts()
            self._model.save_checkpoint(self._output_path, epoch)

    def train_single_epoch(self, epoch: int):
        self._model.train()

        commitment_epoch_loss = 0
        recon_epoch_loss = 0

        for batch, x in enumerate(self._train_dataloader):
            self._optimizer.zero_grad()
            x = x.to(self._device)
            out = self._model(x)
            recon_loss = self._loss_fn(out["x_recon"], x)
            commitment_loss = self._loss_beta * out["commitment_loss"]
            loss = recon_loss + commitment_loss
            dictionary_loss = out["dictionary_loss"]
            if dictionary_loss is not None:
                loss += out["dictionary_loss"]
            commitment_epoch_loss += commitment_loss.item()
            recon_epoch_loss += recon_loss.item()
            if batch % 1000 == 0:
                print(
                    f"batch loss: {loss.item():>7f} [{batch:>5d}/{len(self._train_dataloader):>5d}]"
                )
                self.plot_sample(x[0, :], out["x_recon"][0, :], epoch, batch, False)
            loss.backward()
            self._optimizer.step()

        commitment_epoch_loss /= len(self._train_dataloader)
        recon_epoch_loss /= len(self._train_dataloader)
        print(f"Mean epoch train reconstruction loss: {recon_epoch_loss}")

        return recon_epoch_loss, commitment_epoch_loss

    @torch.no_grad
    def calculate_test_loss(self, epoch: int):
        self._model.eval()

        recon_test_loss = 0

        for batch, x in enumerate(self._test_dataloader):
            x = x.to(self._device)
            out = self._model(x)
            recon_test_loss += self._loss_fn(out["x_recon"], x).item()

            if batch % 500 == 0:
                self.plot_sample(x[0, :], out["x_recon"][0, :], epoch, batch, True)

        recon_test_loss /= len(self._test_dataloader)
        print(f"Mean test reconstruction loss: {recon_test_loss:>8f} \n")

        return recon_test_loss

    def generate_loss_plot(self, suffix: str = ""):
        fig, ax = plt.subplots()
        (plt1,) = ax.plot(
            self._train_loss["reconstruction_loss"], label="Train (recon.) loss"
        )
        (plt2,) = ax.plot(
            self._train_loss["commitment_loss"], label="Train (commitment) loss"
        )
        (plt3,) = ax.plot(
            self._test_loss["reconstruction_loss"], label="Test (recon.) loss"
        )
        ax.set_xlabel("Epoch [-]")
        ax.set_ylabel("Loss")

        axr = ax.twinx()
        (plt4,) = axr.plot(
            self._embedding_entropy, label="Embedding selection entropy", color="k"
        )
        axr.set_ylabel("Entropy")

        lines = [plt1, plt2, plt3, plt4]
        ax.legend(lines, [line.get_label() for line in lines])

        fig.savefig(self._output_path / f"loss_plot_{suffix}.png")
        plt.close(fig)

    def plot_codebook_usage(self, suffix: str = ""):
        codebook_usage_counts = self._model.get_codebook_usage_counts()
        fig, ax = plt.subplots()
        ax.scatter(
            x=list(range(len(codebook_usage_counts))),
            y=codebook_usage_counts,
        )
        ax.set_xlabel("Codebook index")
        ax.set_ylabel("Count")

        fig.savefig(self._output_path / f"codebook_usage_counts_{suffix}.png")
        plt.close(fig)

    @staticmethod
    def image_tensor_to_array(image: torch.Tensor):
        arr = image.cpu().detach().numpy()
        np.clip(arr, 0, 1, out=arr)
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
        plt.close(fig)

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
