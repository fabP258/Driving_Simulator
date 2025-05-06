from torch import nn
from model import NonLocalBlock2D


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    """Patch-based discriminator mapping images to 1-channel logits"""

    def __init__(self, in_channels: int, num_layers: int, num_hiddens: int):
        super().__init__()

        sequence = []
        sequence += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # 2, 4, 8, 8, 8, 8 ...
            sequence += [
                nn.Conv2d(
                    num_hiddens * nf_mult_prev,
                    num_hiddens * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(num_groups=32, num_channels=num_hiddens * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            if n > 1:
                sequence.append(NonLocalBlock2D(in_channels=num_hiddens * nf_mult))

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, 8)
        sequence += [
            nn.Conv2d(
                num_hiddens * nf_mult_prev,
                num_hiddens * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=32, num_channels=num_hiddens * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(num_hiddens * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.conv = nn.Sequential(*sequence)

    def forward(self, x):
        return self.conv(x)
