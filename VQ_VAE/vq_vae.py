from pathlib import Path
from pathlib import Path

import torch
import numpy as np
from torch import nn
from dataclasses import dataclass, asdict

# from encoder import Encoder
# from decoder import Decoder
from taming_model import Encoder, Decoder, CompressorConfig

from quantizer import VectorQuantizer


@dataclass
class VqVaeConfig:
    in_channels: int = 3
    out_channels: int = 3
    ch_mult: tuple[int] = (1, 1, 2, 2, 4)
    attn_resolutions: tuple[int] = (16,)
    resolution: int = 256
    num_res_blocks: int = 2
    z_channels: int = 256
    vocab_size: int = 1024
    ch: int = 128
    dropout: float = 0.0
    double_z: bool = True
    embedding_dim: int = 128
    use_l2_normalization: bool = True
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_eps: float = 1e-5

    def encoder_decoder_args(self) -> dict:
        kw_args = {
            "ch": self.ch,
            "out_ch": self.out_channels,
            "ch_mult": self.ch_mult,
            "num_res_blocks": self.num_res_blocks,
            "attn_resolutions": self.attn_resolutions,
            "dropout": self.dropout,
            "in_channels": self.in_channels,
            "resolution": self.resolution,
            "z_channels": self.z_channels,
        }
        return kw_args

    def quantizer_args(self) -> dict:
        kw_args = {
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.vocab_size,
            "use_l2_normalization": self.use_l2_normalization,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_eps": self.ema_eps,
        }
        return kw_args


class VqVae(nn.Module):

    def __init__(self, config: VqVaeConfig):
        super().__init__()
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

        # log codebook vector usage counts
        self._codebook_usage_counts = torch.zeros(config.vocab_size, dtype=torch.int32)

        self.config: VqVaeConfig = config

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
        quant, dict_loss, commitment_loss, entropy, enc_indices = self.encode(x)
        self.update_codebook_usage_counts(enc_indices)
        x_recon = self.decode(quant)
        return {
            "dictionary_loss": dict_loss,
            "commitment_loss": commitment_loss,
            "entropy_loss": entropy,
            "x_recon": x_recon,
        }

    def calculate_adaptive_weight(self, rec_loss, gen_loss):
        last_layer = self.get_last_decoder_layer()
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gen_grads = torch.autograd.grad(gen_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def get_last_decoder_layer(self):
        return self.decoder.conv_out.weight

    def update_codebook_usage_counts(self, encoding_indices: torch.Tensor):
        flattened_encoding_indices = encoding_indices.flatten().cpu()
        self._codebook_usage_counts += torch.bincount(
            flattened_encoding_indices, minlength=self.config.vocab_size
        )

    def get_codebook_usage_counts(self, normalize: bool = True):
        usage_counts = self._codebook_usage_counts.numpy()
        if normalize:
            usage_counts = usage_counts / np.sum(usage_counts)
        return usage_counts

    def reset_codebook_usage_counts(self):
        self._codebook_usage_counts = torch.zeros(
            self.config.vocab_size, dtype=torch.int32
        )

    def codebook_selection_entropy(self):
        probabilities = self.get_codebook_usage_counts(normalize=True)
        entropy = -np.sum(probabilities * (np.log(probabilities + 1e-10)))
        return entropy

    def save_checkpoint(
        self,
        output_path: str | Path,
        epoch: int,
        postfix: str = None,
    ):
        output = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
        }

        file_name = type(self).__name__
        if postfix:
            file_name += postfix
        file_name += f"_ep{epoch:03}"
        file_name += ".pth"

        torch.save(output, Path(output_path) / file_name)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path)
        model = cls(VqVaeConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
