import random
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import DrivingDataset
from vq_vae import VqVae, VqVaeConfig
from loss import GanLoss


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
        l1_loss_weight: float = 0.2,
        vq_vae_config: VqVaeConfig = VqVaeConfig(),
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

        self._vq_vae = VqVae(config=vq_vae_config)
        print(
            f"VQ-VAE parameters: {sum(p.numel() for p in self._vq_vae.parameters() if p.requires_grad)}"
        )
        self._vq_vae.to(self._device)

        self._vae_optimizer = torch.optim.AdamW(
            self._vq_vae.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.5, 0.9),
        )
        self._vae_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._vae_optimizer, step_size=5, gamma=0.1
        )

        self._gan_loss = GanLoss(num_disc_layers=4, num_disc_hiddens=64)
        self._gan_loss.to(self._device)

        self._disc_optimizer = torch.optim.AdamW(
            self._gan_loss.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.5, 0.9),
        )

        self._l2_loss_fn = torch.nn.MSELoss()
        self._l1_loss_fn = torch.nn.L1Loss()
        self._loss_beta = loss_beta
        self._l1_loss_weight = l1_loss_weight
        self._l2_loss_weight = 2.0
        self._gan_loss_weight = 0.1

        self.initialize_loss_containers()

    def initialize_loss_containers(self):
        self._train_loss = {
            "l1_loss": [],
            "l2_loss": [],
            "commitment_loss": [],
            "gan_loss": {"discriminator": [], "generator": []},
        }
        self._test_loss = {"reconstruction_loss": []}
        self._embedding_entropy = []

    def train(self, epochs: int):

        self.initialize_loss_containers()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            l1_loss, l2_loss, commitment_loss, gan_loss = self.train_single_epoch(epoch)
            self._train_loss["commitment_loss"].append(commitment_loss)
            self._train_loss["l1_loss"].append(l1_loss)
            self._train_loss["l2_loss"].append(l2_loss)
            self._train_loss["gan_loss"]["discriminator"].append(
                gan_loss["discriminator"]
            )
            self._train_loss["gan_loss"]["generator"].append(gan_loss["generator"])

            self._vae_lr_scheduler.step()
            self._test_loss["reconstruction_loss"].append(
                self.calculate_test_loss(epoch)
            )
            self._embedding_entropy.append(self._vq_vae.codebook_selection_entropy())
            self.generate_loss_plot(suffix=f"ep{epoch}")
            self.plot_codebook_usage(suffix=f"ep{epoch}")
            self._vq_vae.reset_codebook_usage_counts()
            self._vq_vae.save_checkpoint(self._output_path, epoch)

    def train_single_epoch(self, epoch: int):
        self._vq_vae.train()
        self._gan_loss.train()

        commitment_epoch_loss = 0
        l1_epoch_loss = 0
        l2_epoch_loss = 0
        gan_epoch_loss = {"discriminator": 0, "generator": 0}

        for batch, x in enumerate(self._train_dataloader):
            for i in range(2):
                self._vae_optimizer.zero_grad()
                self._disc_optimizer.zero_grad()
                x = x.to(self._device)
                out = self._vq_vae(x)

                # sum up loss
                l1_loss = self._l1_loss_fn(out["x_recon"], x)
                l2_loss = self._l2_loss_fn(out["x_recon"], x)
                recon_loss = (
                    self._l1_loss_weight * l1_loss + self._l2_loss_weight * l2_loss
                )
                gan_loss = self._gan_loss_weight * self._gan_loss(x, out["x_recon"], i)
                commitment_loss = self._loss_beta * out["commitment_loss"]
                loss = recon_loss + commitment_loss + gan_loss
                dictionary_loss = out["dictionary_loss"]
                if dictionary_loss is not None:
                    loss += out["dictionary_loss"]

                loss.backward()

                if i == 0:
                    # "Generator update"
                    self._vae_optimizer.step()

                    commitment_epoch_loss += commitment_loss.item()
                    l1_epoch_loss += l1_loss.item()
                    l2_epoch_loss += l2_loss.item()
                    gan_epoch_loss["generator"] += gan_loss.item()
                    if batch % 1000 == 0:
                        print(
                            f"batch loss: {loss.item():>7f} [{batch:>5d}/{len(self._train_dataloader):>5d}]"
                        )
                        self.plot_sample(
                            x[0, :], out["x_recon"][0, :], epoch, batch, False
                        )

                if i == 1:
                    # Discriminator update
                    self._disc_optimizer.step()
                    gan_epoch_loss["discriminator"] += gan_loss.item()

        commitment_epoch_loss /= len(self._train_dataloader)
        l1_epoch_loss /= len(self._train_dataloader)
        l2_epoch_loss /= len(self._train_dataloader)
        gan_epoch_loss["generator"] /= len(self._train_dataloader)
        gan_epoch_loss["discriminator"] /= len(self._train_dataloader)
        print(f"Mean epoch train reconstruction loss (MSE): {l2_epoch_loss}")

        return l1_epoch_loss, l2_epoch_loss, commitment_epoch_loss, gan_epoch_loss

    @torch.no_grad
    def calculate_test_loss(self, epoch: int):
        self._vq_vae.eval()
        self._gan_loss.eval()

        recon_test_loss = 0

        for batch, x in enumerate(self._test_dataloader):
            x = x.to(self._device)
            out = self._vq_vae(x)
            recon_test_loss += self._l2_loss_fn(out["x_recon"], x).item()

            if batch % 500 == 0:
                self.plot_sample(x[0, :], out["x_recon"][0, :], epoch, batch, True)

        recon_test_loss /= len(self._test_dataloader)
        print(f"Mean test reconstruction loss: {recon_test_loss:>8f} \n")

        return recon_test_loss

    def generate_loss_plot(self, suffix: str = ""):
        fig, ax = plt.subplots()
        (plt1,) = ax.plot(self._train_loss["l1_loss"], label="Train L1 (recon.) loss")
        (plt2,) = ax.plot(self._train_loss["l2_loss"], label="Train L2 (recon.) loss")
        (plt3,) = ax.plot(
            self._train_loss["commitment_loss"], label="Train (commitment) loss"
        )
        (plt4,) = ax.plot(
            self._train_loss["gan_loss"]["generator"], label="Generator GAN loss"
        )
        (plt5,) = ax.plot(
            self._train_loss["gan_loss"]["discriminator"],
            label="Discriminator GAN loss",
        )

        (plt6,) = ax.plot(
            self._test_loss["reconstruction_loss"], label="Test L2 (recon.) loss"
        )
        ax.set_xlabel("Epoch [-]")
        ax.set_ylabel("Loss")

        axr = ax.twinx()
        (plt7,) = axr.plot(
            self._embedding_entropy, label="Embedding selection entropy", color="k"
        )
        axr.set_ylabel("Entropy")

        lines = [plt1, plt2, plt3, plt4, plt5, plt6, plt7]
        ax.legend(lines, [line.get_label() for line in lines])

        fig.savefig(self._output_path / f"loss_plot_{suffix}.png")
        plt.close(fig)

    def plot_codebook_usage(self, suffix: str = ""):
        codebook_usage_counts = self._vq_vae.get_codebook_usage_counts()
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
