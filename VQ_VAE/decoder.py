from torch import nn
from resnet import ResidualStack


class Decoder(nn.Module):
    def __init__(
        self,
        # embedding_dim: int,
        num_hiddens: int,
        num_upsampling_layers: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super().__init__()
        # self.conv = nn.Conv2d(
        #    in_channels=embedding_dim,
        #    out_channels=num_hiddens,
        #    kernel_size=3,
        #    padding=1,
        # )
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
        # h = self.conv(x)
        h = self.residual_stack(x)
        x_recon = self.upconv(h)
        return x_recon
