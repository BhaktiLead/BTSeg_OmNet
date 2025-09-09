import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block: two Conv2D + BatchNorm + ReLU layers.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """
    Upsampling block using ConvTranspose2D.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):
    """
    Attention gate to refine skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class OMNet(nn.Module):
    """
    OMNet-inspired encoder-decoder with attention gates.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = ConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = ConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Decoder with attention
        self.up4 = UpConv(features[3] * 2, features[3])
        self.att4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[3] // 2)
        self.dec4 = ConvBlock(features[3] * 2, features[3])

        self.up3 = UpConv(features[3], features[2])
        self.att3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[2] // 2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])

        self.up2 = UpConv(features[2], features[1])
        self.att2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[1] // 2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])

        self.up1 = UpConv(features[1], features[0])
        self.att1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0] // 2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        # Final output
        out = self.final_conv(d1)
        return torch.sigmoid(out)
