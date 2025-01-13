import torch
from pathlib import Path
from typing import Union

from torch import nn
from torch.nn import functional as F


class ResidualStack(nn.Module):
    def __init__(
        self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int
    ):
        super().__init__()
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        return torch.relu(h)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_downsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()

        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)
            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_hiddens: int,
        num_upsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)
            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)
            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())
        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Dictionary embeddings
        limit = 3**0.5
        self.embedding_table = nn.Parameter(
            torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit)
        )

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C] -> [B x H x W, C]
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x**2).sum(1, keepdim=True)
            - 2 * flat_x @ self.embedding_table
            + (self.embedding_table**2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]),
            self.embedding_table.transpose(0, 1),
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()

        # Straight-through gradient.
        quantized_x = x + (quantized_x - x).detach()

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )


class VQVAE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_downsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        embedding_dim: int,
        num_embeddings: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(embedding_dim, num_embeddings)
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }

    def save_checkpoint(
        self,
        output_path: str | Path,
        epoch: int,
        postfix: str = None,
        metadata: dict = None,
    ):
        output = {"epoch": epoch, "model_state_dict": self.state_dict()}

        if metadata:
            output.update(metadata)

        file_name = type(self).__name__
        if postfix:
            file_name += postfix
        file_name += f"_ep{epoch:03}"
        file_name += ".pth"

        torch.save(output, Path(output_path) / file_name)

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> Union[dict, None]:
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint.pop("model_state_dict"))

        if len(checkpoint.keys()) > 0:
            return checkpoint

        return None
