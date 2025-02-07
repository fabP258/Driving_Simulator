from abc import ABC, abstractmethod
from typing import Union
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from lpips import LPIPS

from taming_model import Encoder, Decoder, VqVaeConfig
from quantizer import VectorQuantizer
from discriminator import Discriminator, weights_init
from dataset import denormalize_image


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Implements a horizontally shifted ReLU."""
    if global_step < threshold:
        weight = value
    return weight


class Trainer(ABC):
    def __init__(
        self,
        train_dataloader: DataLoader,
        logger: SummaryWriter,
        validation_dataloader: Union[None, DataLoader] = None,
        max_epochs: int = 100,
        max_steps: int = 1000000,
    ):
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.validation_dataloader = validation_dataloader
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.epoch = 0
        self.global_step = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to_device()

    def train(self):
        while self.epoch < self.max_epochs:
            self.set_train_mode()
            for data in tqdm(self.train_dataloader, desc=f"Train Epoch {self.epoch}"):
                # to device
                if isinstance(data, tuple):
                    for sub_data in data:
                        if isinstance(sub_data, torch.Tensor):
                            sub_data = sub_data.to(self.device)
                elif isinstance(data, torch.Tensor):
                    data = data.to(self.device)
                self.training_step(data)
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

    @abstractmethod
    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs):
        raise NotImplementedError


class VqVaeTrainer(Trainer):

    def __init__(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: Union[None, DataLoader] = None,
        vae_config: VqVaeConfig = VqVaeConfig(),
        logger: SummaryWriter = SummaryWriter(),
        perceptual_loss_weight: float = 0.1,
        gan_weight: float = 1.0,
        gan_start_steps: int = 50000,
        commitment_loss_weight: float = 1.0,
        initial_lr: float = 2.5e-6,
        final_lr: float = 2.5e-7,
        cosine_decay_steps: int = 50000,
    ):
        # nn.Modules used for training
        self.encoder = Encoder(**vae_config.encoder_decoder_args())
        self.pre_quant_conv = nn.Conv2d(
            in_channels=(
                2 * vae_config.z_channels
                if vae_config.double_z
                else vae_config.z_channels
            ),
            out_channels=vae_config.embedding_dim,
            kernel_size=1,
        )
        self.quantizer = VectorQuantizer(**vae_config.quantizer_args())
        self.post_quant_conv = nn.Conv2d(
            in_channels=vae_config.embedding_dim,
            out_channels=vae_config.z_channels,
            kernel_size=3,
            padding=1,
        )
        self.decoder = Decoder(**vae_config.encoder_decoder_args())
        self.discriminator = Discriminator(
            in_channels=3, num_layers=2, num_hiddens=64
        ).apply(weights_init)
        self.perceptual = LPIPS(net="vgg")
        self.perceptual_loss_weight = perceptual_loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gan_weight = gan_weight
        self.gan_start_steps = gan_start_steps
        self.commitment_loss_weight = commitment_loss_weight
        self.optimizers, self.lr_schedulers = self.configure_optimizers(
            initial_lr, final_lr, cosine_decay_steps
        )
        super().__init__(train_dataloader, logger, validation_dataloader)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.pre_quant_conv(h)
        quant, dict_loss, commitment_loss, entropy, enc_indices = self.quantizer(h)
        return quant, dict_loss, commitment_loss, entropy, enc_indices

    def decode(self, x):
        h = self.post_quant_conv(x)
        x_recon = self.decoder(h)
        return x_recon

    def decode_code(self, code):
        raise NotImplementedError

    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor):
        return torch.mean(torch.abs(x - x_recon))

    def discriminator_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, log_output: bool
    ):
        disc_logits_real = self.discriminator(x)
        disc_loss_real = self.bce_loss(
            disc_logits_real, torch.ones_like(disc_logits_real)
        )
        disc_logits_fake = self.discriminator(x_recon.detach())
        disc_loss_fake = self.bce_loss(
            disc_logits_fake, torch.zeros_like(disc_logits_fake)
        )
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        self.logger.add_scalar("train_discriminator_loss", disc_loss, self.global_step)
        self.logger.add_scalar(
            "train_discriminator_real_loss", disc_loss_real, self.global_step
        )
        self.logger.add_scalar(
            "train_discriminator_fake_loss", disc_loss_fake, self.global_step
        )
        if log_output:
            self.log_images(
                nn.functional.sigmoid(disc_logits_real),
                nn.functional.sigmoid(disc_logits_fake),
                "train_discriminator",
                normalize=False,
            )
        return disc_loss

    def training_step(self, x: torch.Tensor):
        z, dict_loss, commitment_loss, entropy, _ = self.encode(x)
        x_recon = self.decode(z)

        # L1 reconstruction loss
        reconstruction_loss = self.reconstruction_loss(x, x_recon)

        # Perceptual loss
        self.perceptual.eval()
        perceptual_loss = torch.squeeze(self.perceptual(x, x_recon)).mean()
        nll_loss = reconstruction_loss + self.perceptual_loss_weight * perceptual_loss

        # Genrator GAN loss
        disc_logits_fake = self.discriminator(x_recon)
        generator_loss = self.bce_loss(
            disc_logits_fake, torch.ones_like(disc_logits_fake)
        )
        generator_adaptive_weight = self.calculate_adaptive_weight(
            nll_loss, generator_loss
        )
        gan_weight = adopt_weight(
            self.gan_weight,
            self.global_step,
            threshold=self.gan_start_steps,
        )
        generator_loss_weight = generator_adaptive_weight * gan_weight
        vae_loss = (
            nll_loss
            + generator_loss_weight * generator_loss
            + self.commitment_loss_weight * commitment_loss
        )
        if dict_loss is not None:
            vae_loss += dict_loss

        log_images = self.global_step % 100 == 0

        # Optimize VAE
        self.optimizers["autoencoder"].zero_grad()
        vae_loss.backward()
        self.optimizers["autoencoder"].step()

        # Optimize Discriminator
        disc_loss = self.discriminator_loss(x, x_recon, log_images) * gan_weight
        self.optimizers["discriminator"].zero_grad()
        disc_loss.backward()
        self.optimizers["discriminator"].step()

        # log images
        if log_images:
            self.log_images(x, x_recon, "train_samples")

        # log
        self.logger.add_scalar(
            "train_l1_reconstruction_loss", reconstruction_loss, self.global_step
        )
        self.logger.add_scalar(
            "train_perceptual_loss", perceptual_loss, self.global_step
        )
        self.logger.add_scalar("train_generator_loss", generator_loss, self.global_step)
        self.logger.add_scalar(
            "train_generator_adaptive_weight",
            generator_adaptive_weight,
            self.global_step,
        )
        self.logger.add_scalar(
            "train_commitment_loss", commitment_loss, self.global_step
        )
        self.logger.add_scalar("train_codebook_entropy", entropy, self.global_step)

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
            self.global_step,
        )

    def configure_optimizers(
        self, initial_lr: float, final_lr: float, cosine_decay_steps: int
    ):
        # Autoencoder
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.pre_quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=initial_lr,
            betas=(0.5, 0.9),
        )
        lr_sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ae, T_max=cosine_decay_steps, eta_min=final_lr
        )

        # Discriminator
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=initial_lr, betas=(0.5, 0.9)
        )
        lr_sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_disc, T_max=cosine_decay_steps, eta_min=final_lr
        )
        return {"autoencoder": opt_ae, "discriminator": opt_disc}, {
            "autoencoder": lr_sched_ae,
            "discriminator": lr_sched_disc,
        }

    def get_last_decoder_layer(self):
        return self.decoder.conv_out.weight

    def calculate_adaptive_weight(self, rec_loss, gen_loss):
        last_layer = self.get_last_decoder_layer()
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
