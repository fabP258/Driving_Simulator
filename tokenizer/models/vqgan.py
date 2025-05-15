import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
from typing import Dict, Union, Tuple, List

from tokenizer.modules.autoencoder.model import Encoder, Decoder
from tokenizer.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from tokenizer.modules.loss.vqperceptual import VQLPIPSWithDiscriminator


class VQModel(pl.LightningModule):
    def __init__(
        self,
        ddconfig: Dict[str, Union[bool, int, float, Tuple[int], List[int]]],
        lossconfig: Dict[str, Union[bool, int, float, str]],
        n_embed: int,
        embed_dim: int,
        base_learning_rate: float,
        ckpt_path=None,  # TODO: Needed?
        ignore_keys=[],
        colorize_nlabels=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.learning_rate = base_learning_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # only a single loss target is supported for now, breaks compatibility with some configs
        self.loss = VQLPIPSWithDiscriminator(**lossconfig.get("params", dict()))
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx):
        x = batch
        xrec, qloss = self(x)

        opt_ae, opt_disc = self.optimizers()

        if batch_idx % 100 == 0:
            self.log_batch_reconstructions(x, xrec, step=self.global_step)

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
        # NOTE: on_epoch=True can't be used with pytorch-lightning<1.1.0 since it accumulates values on gpu
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
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

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
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss = self(x)
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

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
