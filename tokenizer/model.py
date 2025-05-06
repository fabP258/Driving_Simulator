import torch
from torch import nn


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)

    def forward(self, x):
        h = x
        h = self.conv1(nonlinearity(self.norm1(h)))
        h = self.conv2(nonlinearity(self.norm2(h)))
        # TODO: dropout?
        return h + x


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        theta = self.theta(x).view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
        g = self.g(x).view(batch_size, self.inter_channels, -1)  # (B, C', H*W)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (B, H*W, H*W)
        attention = self.softmax(theta_phi)

        out = torch.bmm(g, attention.transpose(1, 2))  # (B, C', H*W)
        out = out.view(batch_size, self.inter_channels, H, W)
        out = self.out_conv(out)

        return x + out  # Residual connection


class Encoder(nn.Module):

    def __init__(
        self,
        base_channels: int = 128,
        depth: int = 4,
        z_channels: int = 1024,
        n_res_blocks: int = 2,
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(
            in_channels=3,
            out_channels=base_channels,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        # downsampling
        self.down = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth + 1):
            for _ in range(n_res_blocks):
                self.down.append(ResidualBlock(in_channels))
            if i == depth:
                self.down.append(NonLocalBlock2D(in_channels))
                break
            out_channels = base_channels * (2**i)
            self.down.append(Downsample(in_channels, out_channels))
            in_channels = out_channels

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResidualBlock(in_channels))
        self.mid.append(NonLocalBlock2D(in_channels))
        self.mid.append(ResidualBlock(in_channels))

        # end
        # TODO: GroupNorm?
        self.out_conv = nn.Conv2d(
            in_channels, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.in_conv(x)
        for down in self.down:
            x = down(x)

        for mid in self.mid:
            x = mid(x)

        return self.out_conv(x)


class Decoder(nn.Module):

    def __init__(
        self,
        base_channels: int = 128,
        depth: int = 4,
        z_channels: int = 1024,
        n_res_blocks: int = 2,
    ):
        super().__init__()

        out_channels = base_channels * (2 ** (depth - 1))
        self.in_conv = nn.Conv2d(
            z_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid = nn.ModuleList()
        self.mid.append(ResidualBlock(out_channels))
        self.mid.append(NonLocalBlock2D(out_channels))
        self.mid.append(ResidualBlock(out_channels))

        self.up = nn.ModuleList()
        in_channels = out_channels
        for i in range(depth, -1, -1):
            for _ in range(n_res_blocks):
                self.up.append(ResidualBlock(in_channels))
            if i == depth:
                self.up.append(NonLocalBlock2D(in_channels))
                continue
            out_channels = base_channels * (2**i)
            self.up.append(Upsample(in_channels, out_channels))
            in_channels = out_channels
        # TODO: GroupNorm?
        self.out_conv = nn.Conv2d(in_channels, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.in_conv(x)

        for mid in self.mid:
            x = mid(x)

        for up in self.up:
            x = up(x)

        return self.out_conv(x)
