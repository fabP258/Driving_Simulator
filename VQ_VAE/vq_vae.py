from pathlib import Path

import torch
import numpy as np
from torch import nn
from dataclasses import dataclass, asdict

from encoder import Encoder
from decoder import Decoder
from quantizer import VectorQuantizer


@dataclass
class VqVaeConfig:
    in_channels: int = 3
    num_hiddens: int = 256
    num_downsampling_layers: int = 4
    num_residual_layers: int = 4
    num_residual_hiddens: int = 256
    embedding_dim: int = 256
    num_embeddings: int = 1024
    use_l2_normalization: bool = False
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_eps: float = 1e-5

    def encoder_args(self) -> dict:
        kw_args = {
            "in_channels": self.in_channels,
            "num_hiddens": self.num_hiddens,
            "num_downsampling_layers": self.num_downsampling_layers,
            "num_residual_layers": self.num_residual_layers,
            "num_residual_hiddens": self.num_residual_hiddens,
        }
        return kw_args

    def decoder_args(self) -> dict:
        kw_args = {
            "num_hiddens": self.num_hiddens,
            "num_upsampling_layers": self.num_downsampling_layers,
            "num_residual_layers": self.num_residual_layers,
            "num_residual_hiddens": self.num_residual_hiddens,
        }
        return kw_args

    def quantizer_args(self) -> dict:
        kw_args = {
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "use_l2_normalization": self.use_l2_normalization,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_eps": self.ema_eps,
        }
        return kw_args


class VqVae(nn.Module):

    def __init__(self, config: VqVaeConfig):
        super().__init__()
        self.encoder = Encoder(**config.encoder_args())
        self.pre_quant_conv = nn.Conv2d(
            in_channels=config.num_hiddens,
            out_channels=config.embedding_dim,
            kernel_size=1,
        )
        self.quantizer = VectorQuantizer(**config.quantizer_args())
        self.post_quant_conv = nn.Conv2d(
            in_channels=config.embedding_dim,
            out_channels=config.num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.decoder = Decoder(**config.decoder_args())

        # log codebook vector usage counts
        self._codebook_usage_counts = torch.zeros(
            config.num_embeddings, dtype=torch.int32
        )

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

    def get_last_decoder_layer(self):
        return self.decoder.upconv[-1].weight

    def update_codebook_usage_counts(self, encoding_indices: torch.Tensor):
        flattened_encoding_indices = encoding_indices.flatten().cpu()
        self._codebook_usage_counts += torch.bincount(
            flattened_encoding_indices, minlength=self.config.num_embeddings
        )

    def get_codebook_usage_counts(self, normalize: bool = True):
        usage_counts = self._codebook_usage_counts.numpy()
        if normalize:
            usage_counts = usage_counts / np.sum(usage_counts)
        return usage_counts

    def reset_codebook_usage_counts(self):
        self._codebook_usage_counts = torch.zeros(
            self.config.num_embeddings, dtype=torch.int32
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
