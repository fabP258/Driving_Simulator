from torch import nn


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.nonlinearity(x)


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.convt(x)
        x = self.norm(x)
        return self.nonlinearity(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.nonlinearity = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.nonlinearity(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.nonlinearity(x + residual)


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
        self.nonlinearity = nn.ReLU()

        # downsampling
        self.down = nn.ModuleList()
        in_channels = base_channels
        for i in range(depth):
            out_channels = base_channels * (2**i)
            self.down.append(Downsample(in_channels, out_channels))
            for _ in range(n_res_blocks):
                self.down.append(ResidualBlock(out_channels))
            # TODO: non-local block / attention
            in_channels = out_channels

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResidualBlock(in_channels))
        self.mid.append(ResidualBlock(in_channels))

        # end
        # TODO: GroupNorm?
        self.out_conv = nn.Conv2d(
            in_channels, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.nonlinearity(self.in_conv(x))
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
        self.nonlinearity = nn.ReLU()

        self.mid = nn.ModuleList()
        self.mid.append(ResidualBlock(out_channels))
        self.mid.append(ResidualBlock(out_channels))

        self.up = nn.ModuleList()
        in_channels = out_channels
        for i in range(depth - 2, -1, -1):
            out_channels = base_channels * (2**i)
            self.up.append(Upsample(in_channels, out_channels))
            for _ in range(n_res_blocks):
                self.up.append(ResidualBlock(out_channels))
            in_channels = out_channels
        self.up.append(Upsample(in_channels, in_channels))

        self.out_conv = nn.Conv2d(in_channels, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.nonlinearity(self.in_conv(x))

        for mid in self.mid:
            x = mid(x)

        for up in self.up:
            x = up(x)

        return self.out_conv(x)
