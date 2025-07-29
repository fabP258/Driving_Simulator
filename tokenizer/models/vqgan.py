import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
from typing import Dict, Union, Tuple, List

from tokenizer.modules.autoencoder.model import Encoder, Decoder
from tokenizer.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from tokenizer.modules.vqvae.quantize import FSQuantizer
from tokenizer.modules.loss.vqperceptual import VQLPIPSWithDiscriminator


class VQModel(pl.LightningModule):
    def __init__(
        self,
        ddconfig: Dict[str, Union[bool, int, float, Tuple[int], List[int]]],
        lossconfig: Dict[str, Union[bool, int, float, str]],
        n_embed: int,
        embed_dim: int,
        base_learning_rate: float,
        grad_acc_steps: int,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        **extra_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grad_acc_steps = grad_acc_steps
        self.automatic_optimization = False
        self.learning_rate = base_learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # only a single loss target is supported for now, breaks compatibility with some configs
        self.loss = VQLPIPSWithDiscriminator(**lossconfig.get("params", dict()))
        # self.quantize = VectorQuantizer(
        #    n_embed,
        #    embed_dim,
        #    remap=remap,
        #    sane_index_shape=sane_index_shape,
        #    legacy=True,
        # )
        self.quantize = FSQuantizer(levels=[8, 8, 8, 6, 5], dtype=torch.float32)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # codebook utilization tracking
        self.register_buffer(
            "codebook_usage_counts",
            torch.zeros(n_embed, dtype=torch.long),
            persistent=False,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_tokens(self, indices):
        embeddings = self.quantize.embedding(indices.view(indices.shape[0], -1))
        B, _, C = embeddings.shape
        H, W = indices.shape[1], indices.shape[2]
        z_q = embeddings.permute(0, 2, 1).contiguous().view(B, C, H, W)
        dec = self.decode(z_q)
        return dec

    def forward(self, input):
        quant, diff, info = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, info

    def training_step(self, batch, batch_idx):
        x = batch
        xrec, qloss, info = self(x)

        # log the codebook usage counts
        codebook_indices = info[2]
        self.codebook_usage_counts += torch.bincount(
            codebook_indices.view(-1), minlength=self.codebook_usage_counts.shape[0]
        )
        self.log(
            "train/codebook_usage",
            torch.count_nonzero(self.codebook_usage_counts)
            / self.codebook_usage_counts.shape[0],
            logger=True,
            on_step=True,
        )
        probs = self.codebook_usage_counts.float() / torch.sum(
            self.codebook_usage_counts.float()
        )
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        self.log("train/codebook_entropy", entropy, logger=True, on_step=True)

        opt_ae, opt_disc = self.optimizers()

        # ==== Autoencoder loss ====
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "train/aeloss",
            float(aeloss.detach().cpu()),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        self.manual_backward(aeloss)

        # ==== Discriminator loss ==
        discloss, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "train/discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        self.manual_backward(discloss)

        if ((batch_idx + 1) % self.grad_acc_steps) == 0:
            self.log_batch_reconstructions(x, xrec, step=self.global_step)
            opt_ae.step()
            opt_disc.step()
            opt_ae.zero_grad()
            opt_disc.zero_grad()

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss, _ = self(x)
        aeloss, log_dict_ae = self.loss(
            qloss,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        rec_loss = log_dict_ae.pop("val/rec_loss")
        self.log(
            "val/rec_loss",
            rec_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
        return self.log_dict  # TODO: Why?

    def on_train_epoch_end(self):
        # Reset buffer to zeros at the end of each training epoch
        self.codebook_usage_counts.zero_()

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_batch_reconstructions(self, x, x_rec, name="train/recon", step=0):
        # Normalize to [0, 1]
        x = (x + 1.0) / 2.0
        x_rec = (x_rec + 1.0) / 2.0

        # Make a horizontal grid of each row
        top_row = make_grid(x.detach().cpu(), nrow=len(x), padding=2)
        bottom_row = make_grid(x_rec.detach().cpu(), nrow=len(x_rec), padding=2)

        # Stack rows vertically
        both = torch.cat([top_row, bottom_row], dim=1)  # stack along height (dim=1)

        self.logger.experiment.add_image(name, both, step)
