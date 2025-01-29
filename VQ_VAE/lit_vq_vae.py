import lightning

import torch
from torch import nn
from torchvision.utils import make_grid

from lpips import LPIPS

from taming_model import Encoder, Decoder


from vq_vae import VqVaeConfig
from quantizer import VectorQuantizer
from discriminator import Discriminator


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalizes image from range [0, 1] to [-1,1]"""
    return x * 2 - 1


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Implements a horizontally shifted ReLU."""
    if global_step < threshold:
        weight = value
    return weight


class LitVqVae(lightning.LightningModule):

    def __init__(self, config: VqVaeConfig, lr: float = 1e-4):
        super().__init__()

        # disable automatic optimization since we have multiple optimizers
        self.automatic_optimization = False

        self.encoder = Encoder(**config.encoder_decoder_args())
        self.pre_quant_conv = nn.Conv2d(
            in_channels=2 * config.z_channels if config.double_z else config.z_channels,
            out_channels=config.embedding_dim,
            kernel_size=1,
        )
        self.quantizer = VectorQuantizer(**config.quantizer_args())
        self.post_quant_conv = nn.Conv2d(
            in_channels=config.embedding_dim,
            out_channels=config.z_channels,
            kernel_size=3,
            padding=1,
        )
        self.decoder = Decoder(**config.encoder_decoder_args())
        self.discriminator = Discriminator(in_channels=3, num_layers=2, num_hiddens=64)
        self.perceptual = LPIPS(net="vgg")
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.learning_rate = lr

        # loss weights
        self.codebook_weight = 0.25  # beta
        self.perceptual_weight = 0.1
        self.gan_weight = 1.0
        self.gan_start_steps = 50000  # in batches/steps, each forward pass is 2 steps

    def encode(self, x):
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

    def forward(self, x):
        quant, dict_loss, commitment_loss, entropy, _ = self.encode(x)
        x_recon = self.decode(quant)
        return {
            "dictionary_loss": dict_loss,
            "commitment_loss": commitment_loss,
            "entropy": entropy,
            "x_recon": x_recon,
        }

    def training_step(self, batch, batch_idx):
        # TODO: refactor this method

        x = batch
        out = self(x)
        opt_ae, opt_disc = self.optimizers()

        # VQ-VAE
        self.toggle_optimizer(opt_ae)

        # Discriminator
        disc_logits_real = self.discriminator(x)
        disc_logits_fake = self.discriminator(out["x_recon"])

        # L1 reconstruction loss
        recon_loss = torch.mean(torch.abs(x.contiguous() - out["x_recon"].contiguous()))

        # LPIPS perceptual loss
        self.perceptual.eval()
        perceptual_loss = torch.squeeze(
            self.perceptual(normalize_image(x), normalize_image(out["x_recon"]))
        ).mean()
        nll_loss = recon_loss + self.perceptual_weight * perceptual_loss

        # Generator BCE loss
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
        gan_weight *= generator_adaptive_weight
        vae_loss = (
            nll_loss
            + gan_weight * generator_loss
            + self.codebook_weight * out["commitment_loss"]
        )
        dictionary_loss = out["dictionary_loss"]
        if dictionary_loss is not None:
            # no ema
            vae_loss += dictionary_loss

        opt_ae.zero_grad()
        self.manual_backward(vae_loss)
        opt_ae.step()

        self.untoggle_optimizer(opt_ae)

        # Discriminator
        self.toggle_optimizer(opt_disc)

        disc_loss_real = self.bce_loss(
            disc_logits_real, torch.ones_like(disc_logits_real)
        )
        disc_logits_fake = self.discriminator(out["x_recon"].detach())
        disc_loss_fake = self.bce_loss(
            disc_logits_fake, torch.zeros_like(disc_logits_fake)
        )
        gan_weight = adopt_weight(
            self.gan_weight,
            self.global_step,
            threshold=self.gan_start_steps,
        )
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        opt_disc.zero_grad()
        self.manual_backward(gan_weight * disc_loss)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        self.log("train_l1_recon_loss", recon_loss, prog_bar=True)
        self.log("train_perceptual_loss", perceptual_loss)
        self.log("train_generator_loss", generator_loss)
        self.log("train_generator_adaptive_weight", generator_adaptive_weight)
        self.log("train_commitment_loss", out["commitment_loss"])
        self.log("train_discriminator_loss", disc_loss)
        self.log("train_codebook_entropy", out["entropy"])

    def validation_step(self, batch, batch_idx):
        x = batch
        out = self(x)

        # Generator loss
        disc_logits_fake = self.discriminator(out["x_recon"])
        generator_loss = self.bce_loss(
            disc_logits_fake, torch.ones_like(disc_logits_fake)
        )

        # L1 reconstruction loss
        recon_loss = torch.mean(torch.abs(x.contiguous() - out["x_recon"].contiguous()))

        # LPIPS perceptual loss
        perceptual_loss = torch.squeeze(
            self.perceptual(normalize_image(x), normalize_image(out["x_recon"]))
        ).mean()

        disc_logits_real = self.discriminator(x)
        disc_loss_real = self.bce_loss(
            disc_logits_real, torch.ones_like(disc_logits_real)
        )
        disc_loss_fake = self.bce_loss(
            disc_logits_fake, torch.zeros_like(disc_logits_fake)
        )
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

        if batch_idx % 1000 == 0:
            self.logger.experiment.add_image(
                f"reconstructions_ep{self.global_step}",
                make_grid(out["x_recon"]),
                batch_idx,
            )

        self.log("test_l1_recon_loss", recon_loss, prog_bar=True)
        self.log("test_perceptual_loss", perceptual_loss)
        self.log("test_generator_loss", generator_loss)
        self.log("test_commitment_loss", out["commitment_loss"])
        self.log("test_discriminator_loss", disc_loss)
        self.log("test_codebook_entropy", out["entropy"])

    def configure_optimizers(self):
        lr = self.learning_rate

        # Autoencoder
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.pre_quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )

        # Discriminator
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )

        return [opt_ae, opt_disc], []

    def get_last_decoder_layer(self):
        return self.decoder.conv_out.weight

    def calculate_adaptive_weight(self, rec_loss, gen_loss):
        last_layer = self.get_last_decoder_layer()
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
