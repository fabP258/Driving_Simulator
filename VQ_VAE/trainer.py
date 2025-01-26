import random
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from lpips import LPIPS

from dataset import DrivingDataset
from vq_vae import VqVae, VqVaeConfig
from discriminator import Discriminator


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalizes image from range [0, 1] to [-1,1]"""
    return x * 2 - 1


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
        VqVaeTrainer.print_num_parameters("VQ-VAE", self._vq_vae)
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

        self._discriminator = Discriminator(
            in_channels=3, num_layers=4, num_hiddens=128
        )
        VqVaeTrainer.print_num_parameters("Discriminator", self._discriminator)
        self._discriminator.to(self._device)

        self._disc_optimizer = torch.optim.AdamW(
            self._discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.5, 0.9),
        )
        self._disc_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._disc_optimizer, step_size=5, gamma=0.1
        )

        self._l2_loss_fn = torch.nn.MSELoss()
        self._l1_loss_fn = torch.nn.L1Loss()
        self._perceptual_loss = LPIPS(net="vgg").eval().to(self._device)
        self._gan_loss_fn = torch.nn.BCEWithLogitsLoss()
        self._loss_beta = loss_beta
        self._l1_loss_weight = l1_loss_weight
        self._l2_loss_weight = 2.0
        self._perceptual_loss_weight = 0.1
        self._gan_loss_weight = 1.0

        self.initialize_loss_containers()

    def initialize_loss_containers(self):
        self._train_loss = {
            "l1_loss": [],
            "l2_loss": [],
            "perceptual_loss": [],
            "commitment_loss": [],
            "gan_loss": {"discriminator": [], "generator": []},
        }
        self._test_loss = {"reconstruction_loss": []}
        self._embedding_entropy = []

    @staticmethod
    def print_num_parameters(name: str, model: torch.nn.Module):
        """Print the name and the number of paramaters for a given model."""
        print(
            f"{name} parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    def train(self, epochs: int):

        self.initialize_loss_containers()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            l1_loss, l2_loss, perceptual_loss, commitment_loss, gan_loss = (
                self.train_single_epoch(epoch)
            )
            self._train_loss["commitment_loss"].append(commitment_loss)
            self._train_loss["l1_loss"].append(l1_loss)
            self._train_loss["l2_loss"].append(l2_loss)
            self._train_loss["perceptual_loss"].append(perceptual_loss)
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
        self._discriminator.train()

        commitment_epoch_loss = 0
        l1_epoch_loss = 0
        l2_epoch_loss = 0
        perceptual_epoch_loss = 0
        gan_epoch_loss = {"discriminator": 0, "generator": 0}

        for batch, x in enumerate(self._train_dataloader):
            x = x.to(self._device)

            # forward pass (VAE + Discriminator)
            out = self._vq_vae(x)
            disc_logits_real = self._discriminator(x)
            disc_logits_fake = self._discriminator(out["x_recon"])

            # sum up loss
            l1_loss = self._l1_loss_fn(out["x_recon"], x)
            l2_loss = self._l2_loss_fn(out["x_recon"], x)
            perceptual_loss = torch.squeeze(
                self._perceptual_loss(
                    normalize_image(out["x_recon"]), normalize_image(x)
                )
            ).mean()
            recon_loss = (
                self._l1_loss_weight * l1_loss
                + self._l2_loss_weight * l2_loss
                + self._perceptual_loss_weight * perceptual_loss
            )

            commitment_loss = self._loss_beta * out["commitment_loss"]

            generator_loss = self._gan_loss_fn(
                disc_logits_fake, torch.zeros_like(disc_logits_fake)
            )
            gan_loss_weight = self._vq_vae.calculate_adaptive_weight(
                recon_loss, generator_loss
            )

            vae_loss = (
                recon_loss
                + commitment_loss
                + generator_loss * gan_loss_weight * self._gan_loss_weight
            )
            dictionary_loss = out["dictionary_loss"]
            if dictionary_loss is not None:
                vae_loss += out["dictionary_loss"]

            # optimize VAE
            self._vae_optimizer.zero_grad()
            vae_loss.backward()
            self._vae_optimizer.step()

            # Discriminator loss
            disc_loss_real = self._gan_loss_fn(
                disc_logits_real, torch.ones_like(disc_logits_real)
            )
            disc_logits_fake = self._discriminator(out["x_recon"].detach())
            disc_loss_fake = self._gan_loss_fn(
                disc_logits_fake, torch.zeros_like(disc_logits_fake)
            )
            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

            # Optimize discriminator
            self._disc_optimizer.zero_grad()
            disc_loss.backward()
            self._disc_optimizer.step()

            # Logging
            commitment_epoch_loss += commitment_loss.item()
            l1_epoch_loss += l1_loss.item()
            l2_epoch_loss += l2_loss.item()
            perceptual_epoch_loss += perceptual_loss.item()
            gan_epoch_loss["generator"] += generator_loss.item()
            gan_epoch_loss["discriminator"] += disc_loss.item()

            if batch % 1000 == 0:
                print(
                    f"batch loss: {recon_loss.item():>7f} [{batch:>5d}/{len(self._train_dataloader):>5d}]"
                )
                self.plot_sample(
                    x[0, :],
                    out["x_recon"][0, :],
                    torch.sigmoid(disc_logits_real[0, 0, :]),
                    torch.sigmoid(disc_logits_fake[0, 0, :]),
                    epoch,
                    batch,
                    False,
                )

        commitment_epoch_loss /= len(self._train_dataloader)
        l1_epoch_loss /= len(self._train_dataloader)
        l2_epoch_loss /= len(self._train_dataloader)
        perceptual_epoch_loss /= len(self._train_dataloader)
        gan_epoch_loss["generator"] /= len(self._train_dataloader)
        gan_epoch_loss["discriminator"] /= len(self._train_dataloader)
        print(f"Mean epoch train reconstruction loss (L1): {l1_epoch_loss}")

        return (
            l1_epoch_loss,
            l2_epoch_loss,
            perceptual_epoch_loss,
            commitment_epoch_loss,
            gan_epoch_loss,
        )

    @torch.no_grad
    def calculate_test_loss(self, epoch: int):
        self._vq_vae.eval()
        self._discriminator.eval()

        recon_test_loss = 0

        for batch, x in enumerate(self._test_dataloader):
            x = x.to(self._device)
            out = self._vq_vae(x)
            recon_test_loss += self._l1_loss_fn(out["x_recon"], x).item()

            disc_logits_real = self._discriminator(x)
            disc_logits_fake = self._discriminator(out["x_recon"])

            if batch % 500 == 0:
                self.plot_sample(
                    x[0, :],
                    out["x_recon"][0, :],
                    torch.sigmoid(disc_logits_real[0, 0, :]),
                    torch.sigmoid(disc_logits_fake[0, 0, :]),
                    epoch,
                    batch,
                    True,
                )

        recon_test_loss /= len(self._test_dataloader)
        print(f"Mean test reconstruction loss: {recon_test_loss:>8f} \n")

        return recon_test_loss

    def generate_loss_plot(self, suffix: str = ""):
        fig, ax = plt.subplots()
        (plt1,) = ax.plot(self._train_loss["l1_loss"], label="Train L1 (recon.) loss")
        (plt2,) = ax.plot(self._train_loss["l2_loss"], label="Train L2 (recon.) loss")
        (plt3,) = ax.plot(self._train_loss["perceptual_loss"], label="Perceptual loss")
        (plt4,) = ax.plot(
            self._train_loss["commitment_loss"], label="Train (commitment) loss"
        )
        (plt5,) = ax.plot(
            self._train_loss["gan_loss"]["generator"], label="Generator GAN loss"
        )
        (plt6,) = ax.plot(
            self._train_loss["gan_loss"]["discriminator"],
            label="Discriminator GAN loss",
        )

        (plt7,) = ax.plot(
            self._test_loss["reconstruction_loss"], label="Test L1 (recon.) loss"
        )
        ax.set_xlabel("Epoch [-]")
        ax.set_ylabel("Loss")

        axr = ax.twinx()
        (plt8,) = axr.plot(
            self._embedding_entropy, label="Embedding selection entropy", color="k"
        )
        axr.set_ylabel("Entropy")

        lines = [plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8]
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
        real_probs_real: torch.Tensor,
        real_probs_fake: torch.Tensor,
        epoch: int,
        batch: int,
        validation: bool,
    ):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(VqVaeTrainer.image_tensor_to_array(x))
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Input image")
        ax[1, 0].imshow(VqVaeTrainer.image_tensor_to_array(x_recon))
        ax[1, 0].axis("off")
        ax[1, 0].set_title("Reconstructed image")

        im1 = ax[0, 1].imshow(
            np.clip(real_probs_real.cpu().detach().numpy(), 0, 1), vmin=0, vmax=1
        )
        ax[0, 1].axis("off")
        cb1 = plt.colorbar(im1, ax=ax[0, 1])
        cb1.set_label("Real probability")

        im2 = ax[1, 1].imshow(
            np.clip(real_probs_fake.cpu().detach().numpy(), 0, 1), vmin=0, vmax=1
        )
        ax[1, 1].axis("off")
        cb2 = plt.colorbar(im2, ax=ax[1, 1])
        cb2.set_label("Real probability")

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
