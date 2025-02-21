from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import asdict
from typing import Union
from tqdm import tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from lpips import LPIPS

# from taming_model import Encoder, Decoder, VqVaeConfig
from model import Encoder, Decoder
from quantizer import VectorQuantizer
from discriminator import Discriminator, weights_init
from dataset import denormalize_image, normalize_image


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Implements a horizontally shifted ReLU."""
    if global_step < threshold:
        weight = value
    return weight


class Trainer(ABC):

    def __init__(
        self,
        log_dir: Union[str, Path],
        device: str,
        max_epochs: int = 100,
        max_steps: int = 1000000,
    ):
        self.log_dir = Path(log_dir)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.epoch = 0
        self.global_step = 0
        self.device = (
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.to_device()
        self.print_parameter_counts()

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: Union[None, DataLoader] = None,
    ):
        while self.epoch < self.max_epochs:
            self.set_train_mode()
            for data in tqdm(train_dataloader, desc=f"Train Epoch {self.epoch}"):
                # to device
                if isinstance(data, tuple):
                    for sub_data in data:
                        if isinstance(sub_data, torch.Tensor):
                            sub_data = sub_data.to(self.device)
                elif isinstance(data, torch.Tensor):
                    data = data.to(self.device)
                self.training_step(data)
                if self.global_step % 10000 == 0:
                    self.save_checkpoint(
                        self.log_dir / f"checkpoint_step{self.global_step}.pth"
                    )

                self.global_step += 1
                self.logger.flush()

            # TODO: Iterate over validation dataloader every x (epochs, batches)?

            self.epoch += 1

    def to_device(self):
        for member in vars(self):
            attr = getattr(self, member)
            if isinstance(attr, nn.Module):
                attr.to(self.device)

    def set_train_mode(self):
        for member in vars(self):
            attr = getattr(self, member)
            if isinstance(attr, nn.Module):
                attr.train()

    def set_eval_mode(self):
        for member in vars(self):
            attr = getattr(self, member)
            if isinstance(attr, nn.Module):
                attr.eval()

    def print_parameter_counts(self):
        print("--- MODEL PARAMETERS ---")
        for member in vars(self):
            attr = getattr(self, member)
            if isinstance(attr, nn.Module):
                n_parameters = sum(
                    p.numel() for p in attr.parameters() if p.requires_grad
                )
                if n_parameters < 100000:
                    print(f"{member}: {n_parameters * 1e-3:.2f} K")
                else:
                    print(f"{member}: {n_parameters * 1e-6:.2f} M")

    @abstractmethod
    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_path: str):
        raise NotImplementedError


class VqVaeTrainer(Trainer):

    def __init__(
        self,
        log_dir: str,
        perceptual_loss_weight: float = 0.1,
        gan_weight: float = 1.0,
        gan_start_steps: int = 50000,
        commitment_loss_weight: float = 1.0,
        l1_loss_weight: float = 0.1,
        l2_loss_weight: float = 2.0,
        initial_lr: float = 2.5e-6,
        final_lr: float = 2.5e-7,
        weight_decay: float = 0.01,
        cosine_decay_steps: int = 50000,
        disc_updates_per_step: int = 3,
        device: str = "cuda",
    ):
        # nn.Modules used for training
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(
            embedding_dim=1024,
            num_embeddings=8192,
            use_l2_normalization=True,
            use_ema=True,
            ema_decay=0.99,
            ema_eps=1e-5,
        )
        self.decoder = Decoder()
        self.discriminator = Discriminator(
            in_channels=3, num_layers=4, num_hiddens=128
        ).apply(weights_init)
        self.perceptual = LPIPS(net="vgg")
        self.perceptual_loss_weight = perceptual_loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gan_weight = gan_weight
        self.gan_start_steps = gan_start_steps
        self.commitment_loss_weight = commitment_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.cosine_decay_steps = cosine_decay_steps
        self.disc_updates_per_step = disc_updates_per_step
        self.optimizers, self.lr_schedulers = self.configure_optimizers(
            initial_lr, final_lr, cosine_decay_steps, weight_decay
        )
        self.generator_step = 0
        self.discriminator_step = 0
        super().__init__(log_dir, device)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        quant, dict_loss, commitment_loss, entropy, enc_indices = self.quantizer(h)
        return quant, dict_loss, commitment_loss, entropy, enc_indices

    def decode(self, x):
        x_recon = self.decoder(x)
        return x_recon

    def decode_code(self, code):
        raise NotImplementedError

    @staticmethod
    def l1_reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor):
        return torch.mean(torch.abs(x - x_recon))

    @staticmethod
    def l2_reconstruction_loss(x: torch.Tensor, x_recon: torch.tensor):
        return torch.mean((x - x_recon) ** 2)

    def calc_discriminator_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, log_images: bool
    ):
        logits_real = self.discriminator(x)
        logits_fake = self.discriminator(x_recon.detach())
        disc_loss_real = self.bce_loss(logits_real, torch.ones_like(logits_real))
        disc_loss_fake = self.bce_loss(logits_fake, torch.zeros_like(logits_fake))
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        self.logger.add_scalar("train_discriminator_loss", disc_loss, self.global_step)
        self.logger.add_scalar(
            "train_discriminator_real_loss", disc_loss_real, self.global_step
        )
        self.logger.add_scalar(
            "train_discriminator_fake_loss", disc_loss_fake, self.global_step
        )
        if log_images:
            self.log_images(
                nn.functional.sigmoid(logits_real),
                nn.functional.sigmoid(logits_fake),
                "train_discriminator",
                self.discriminator_step,
                normalize=False,
            )
        return disc_loss

    def training_step_generator(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        commitment_loss: torch.Tensor,
        dict_loss: Union[None, torch.tensor],
        gan_weight: float,
    ):
        # L1 & L2 reconstruction loss
        l1_reconstruction_loss = self.l1_reconstruction_loss(x, x_recon)
        l2_reconstruction_loss = self.l2_reconstruction_loss(x, x_recon)
        reconstruction_loss = (
            self.l1_loss_weight * l1_reconstruction_loss
            + self.l2_loss_weight * l2_reconstruction_loss
        )

        # Perceptual loss
        self.perceptual.eval()
        perceptual_loss = torch.squeeze(
            self.perceptual(normalize_image(x), normalize_image(x_recon))
        ).mean()
        nll_loss = reconstruction_loss + self.perceptual_loss_weight * perceptual_loss

        # Generator GAN loss
        disc_logits_fake = self.discriminator(x_recon)
        generator_loss = self.bce_loss(
            disc_logits_fake, torch.ones_like(disc_logits_fake)
        )
        generator_adaptive_weight = self.calculate_adaptive_weight(
            nll_loss, generator_loss
        )
        generator_loss_weight = generator_adaptive_weight * gan_weight
        vae_loss = (
            nll_loss
            + generator_loss_weight * generator_loss
            + self.commitment_loss_weight * commitment_loss
        )
        if dict_loss is not None:
            vae_loss += dict_loss

        self.optimizers["autoencoder"].zero_grad()
        vae_loss.backward()
        self.optimizers["autoencoder"].step()

        self.logger.add_scalar(
            "train_l1_reconstruction_loss", l1_reconstruction_loss, self.generator_step
        )
        self.logger.add_scalar(
            "train_l2_reconstruction_loss",
            l2_reconstruction_loss,
            self.generator_step,
        )
        self.logger.add_scalar(
            "train_perceptual_loss", perceptual_loss, self.generator_step
        )
        self.logger.add_scalar(
            "train_generator_loss", generator_loss, self.generator_step
        )
        self.logger.add_scalar(
            "train_generator_adaptive_weight",
            generator_adaptive_weight,
            self.generator_step,
        )

    def training_step(self, x: torch.Tensor):
        z, dict_loss, commitment_loss, entropy, _ = self.encode(x)
        x_recon = self.decode(z)

        # TODO: Group with train/<scalar_name>
        self.logger.add_scalar(
            "train_commitment_loss", commitment_loss, self.generator_step
        )
        self.logger.add_scalar("train_codebook_entropy", entropy, self.generator_step)

        gan_weight = adopt_weight(
            self.gan_weight,
            self.generator_step,
            threshold=self.gan_start_steps,
        )
        self.logger.add_scalar("train_gan_weight", gan_weight, self.generator_step)

        log_images = self.generator_step % 100 == 0
        if log_images:
            self.log_images(
                x, x_recon, "train_samples", self.generator_step, normalize=False
            )

        if (
            self.global_step % (self.disc_updates_per_step + 1) == 0
            or self.generator_step < self.gan_start_steps
        ):
            # Optimize GEN
            self.training_step_generator(
                x, x_recon, commitment_loss, dict_loss, gan_weight
            )
            self.generator_step += 1
        else:
            # Optimize DISC
            disc_loss = self.calc_discriminator_loss(x, x_recon, log_images=log_images)
            self.logger.add_scalar(
                "train_discriminator_loss", disc_loss, self.discriminator_step
            )
            disc_loss *= gan_weight
            self.optimizers["discriminator"].zero_grad()
            disc_loss.backward()
            self.optimizers["discriminator"].step()
            self.discriminator_step += 1

        self.update_lr_schedulers()

    def update_lr_schedulers(self):
        self.logger.add_scalar(
            "vae_lr",
            self.lr_schedulers["autoencoder"].get_last_lr()[0],
            self.global_step,
        )
        self.logger.add_scalar(
            "disc_lr",
            self.lr_schedulers["discriminator"].get_last_lr()[0],
            self.global_step,
        )
        if (
            self.gan_start_steps
            < self.global_step
            < (self.gan_start_steps + self.cosine_decay_steps)
        ):
            self.lr_schedulers["autoencoder"].step()
            self.lr_schedulers["discriminator"].step()

    def log_images(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        log_name: str,
        step: int,
        normalize: bool = True,
    ):
        images = []
        images.extend(
            [
                denormalize_image(x_real[i, :]) if normalize else x_real[i, :]
                for i in range(x_real.shape[0])
            ]
        )
        images.extend(
            [
                denormalize_image(x_fake[i, :]) if normalize else x_fake[i, :]
                for i in range(x_fake.shape[0])
            ]
        )
        self.logger.add_image(
            log_name,
            make_grid(
                images, nrow=x_real.shape[0]
            ),  # TODO: Use the normalize flag here
            step,
        )

    def configure_optimizers(
        self,
        initial_lr: float,
        final_lr: float,
        cosine_decay_steps: int,
        weight_decay: float,
    ):
        # Autoencoder
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=initial_lr,
            betas=(0.5, 0.9),
            weight_decay=weight_decay,
        )
        lr_sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ae, T_max=cosine_decay_steps, eta_min=final_lr
        )

        # Discriminator
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=initial_lr,
            betas=(0.5, 0.9),
        )
        lr_sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_disc, T_max=cosine_decay_steps, eta_min=final_lr
        )
        return {"autoencoder": opt_ae, "discriminator": opt_disc}, {
            "autoencoder": lr_sched_ae,
            "discriminator": lr_sched_disc,
        }

    def save_checkpoint(self, checkpoint_path: str):
        checkpoint = {}
        # save model state dicts
        for member_name in vars(self):
            attr = getattr(self, member_name)
            if isinstance(attr, nn.Module):
                checkpoint.update({member_name: attr.state_dict()})

        # save optimizer states
        checkpoint.update(
            {
                "optimizer_states": {
                    "autoencoder": self.optimizers["autoencoder"].state_dict(),
                    "discriminator": self.optimizers["discriminator"].state_dict(),
                }
            }
        )

        # save lr scheduler states
        checkpoint.update(
            {
                "scheduler_states": {
                    "autoencoder": self.lr_schedulers["autoencoder"].state_dict(),
                    "discriminator": self.lr_schedulers["discriminator"].state_dict(),
                }
            }
        )

        # save c'tor args
        checkpoint.update(
            {
                "ctor_args": {
                    "perceptual_loss_weight": self.perceptual_loss_weight,
                    "gan_weight": self.gan_weight,
                    "gan_start_steps": self.gan_start_steps,
                    "commitment_loss_weight": self.commitment_loss_weight,
                    "cosine_decay_steps": self.cosine_decay_steps,
                    "l1_loss_weight": self.l1_loss_weight,
                    "l2_loss_weight": self.l2_loss_weight,
                    "disc_updates_per_step": self.disc_updates_per_step,
                }
            }
        )

        # save class state
        checkpoint.update(
            {
                "class_state": {
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "generator_step": self.generator_step,
                    "discriminator_step": self.discriminator_step,
                }
            }
        )

        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, log_path: Union[str, Path], device: str = "cuda"
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ctor_args = checkpoint["ctor_args"]
        trainer = cls(log_dir=log_path, **ctor_args, device=device)

        for member_name in vars(trainer):
            attr = getattr(trainer, member_name)
            if isinstance(attr, nn.Module):
                attr.load_state_dict(checkpoint[member_name])

        for opt_type in ["autoencoder", "discriminator"]:
            trainer.optimizers[opt_type].load_state_dict(
                checkpoint["optimizer_states"][opt_type]
            )
            trainer.lr_schedulers[opt_type].load_state_dict(
                checkpoint["scheduler_states"][opt_type]
            )

        trainer.global_step = checkpoint["class_state"]["global_step"]
        trainer.epoch = checkpoint["class_state"]["epoch"]
        trainer.generator_step = checkpoint["class_state"]["generator_step"]
        trainer.discriminator_step = checkpoint["class_state"]["discriminator_step"]

        return trainer

    def get_last_decoder_layer(self):
        return self.decoder.out_conv.weight

    def calculate_adaptive_weight(self, rec_loss, gen_loss):
        last_layer = self.get_last_decoder_layer()
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
