from typing import Sequence
import torch
from torchvision.utils import make_grid

from tokenizer.engine.module import TrainableModule
from tokenizer.modules.autoencoder.model import Encoder, Decoder
from tokenizer.modules.quantization.quantize import FSQuantizer
from tokenizer.modules.loss.vqperceptual import VQLPIPSWithDiscriminator


class ImageTokenizer(TrainableModule):

    @TrainableModule.save_hyperparameters
    def __init__(
        self,
        lr: int,
        ch: int,
        ch_mult: Sequence,
        num_res_blocks: int,
        attn_resolutions: Sequence,
        resolution: int,
        z_channels: int,
        double_z: bool,
        fsq_levels: Sequence,
        latent_channels: int,
        disc_start: int,
        disc_factor: float,
        disc_weight: float,
        pixelloss_weight: float,
        perceptual_weight: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.learning_rate = lr
        self.encoder = Encoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=3,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
        )
        self.decoder = Decoder(
            ch=ch,
            out_ch=3,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            resolution=resolution,
            z_channels=z_channels,
            give_pre_end=False,
        )
        self.quantizer = FSQuantizer(levels=fsq_levels, dtype=torch.float32)
        # TODO: move these into Encoder/Decoder
        self.quant_conv = torch.nn.Conv2d(
            in_channels=z_channels, out_channels=latent_channels, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            in_channels=latent_channels, out_channels=z_channels, kernel_size=1
        )
        self.loss = VQLPIPSWithDiscriminator(
            disc_start=disc_start,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
            pixelloss_weight=pixelloss_weight,
            perceptual_weight=perceptual_weight,
        )

    # TODO: encode() & decode() can simply be the forward calls
    #       if quant_conv and post_quant_conv are moved to Encoder & Decoder
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # TODO: refactor this -> (quant, indices)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def decode(self, x):
        h = self.post_quant_conv(x)
        dec = self.decoder(h)
        return dec

    def forward(self, x):
        quant, emb_loss, info = self.encode(x)
        x_recon = self.decode(quant)
        return x_recon, emb_loss, info

    def _training_step(self, batch, batch_idx, opt_idx):
        x_recon, _, info = self(batch)
        indices = info[2]
        loss, log_dict = self.loss(
            batch,
            x_recon,
            opt_idx,
            self.macro_step[0],
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log_dict(log_dict, self.micro_step[opt_idx])
        if opt_idx == 0 and batch_idx % 100 == 0:
            self.log_batch_reconstructions(
                batch, x_recon, step=self.micro_step[opt_idx]
            )
            self.logger.add_histogram(
                "train/codebook_distribution",
                indices.view(-1),
                global_step=self.micro_step[opt_idx],
            )
        return loss

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
            weight_decay=0.01,
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], [None, None]

    def log_batch_reconstructions(self, x, x_rec, name="train/recon", step=0):
        # Normalize to [0, 1]
        x = (x + 1.0) / 2.0
        x_rec = (x_rec + 1.0) / 2.0

        # Make a horizontal grid of each row
        top_row = make_grid(x.detach().cpu(), nrow=len(x), padding=2)
        bottom_row = make_grid(x_rec.detach().cpu(), nrow=len(x_rec), padding=2)

        # Stack rows vertically
        both = torch.cat([top_row, bottom_row], dim=1)  # stack along height (dim=1)

        # TODO: wrap add_image call
        self.logger.add_image(name, both, step)
