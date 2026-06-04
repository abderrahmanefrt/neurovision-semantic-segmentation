import torch
import torch.nn as nn


# ==========================================
# Convolution Block
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# Encoder
# ==========================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):

        x1 = self.conv1(x)      # 256x256x64
        p1 = self.pool1(x1)     # 128x128x64

        x2 = self.conv2(p1)     # 128x128x128
        p2 = self.pool2(x2)     # 64x64x128

        x3 = self.conv3(p2)     # 64x64x256
        p3 = self.pool3(x3)     # 32x32x256

        return x1, x2, x3, p3


# ==========================================
# Decoder
# ==========================================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 32x32x256 -> 64x64x128
        self.up3 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        # concat avec x3 :
        # 128 + 256 = 384
        self.dec3 = ConvBlock(384, 128)

        # 64x64x128 -> 128x128x64
        self.up2 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        # concat avec x2 :
        # 64 + 128 = 192
        self.dec2 = ConvBlock(192, 64)

        # 128x128x64 -> 256x256x64
        self.up1 = nn.ConvTranspose2d(
            64, 64,
            kernel_size=2,
            stride=2
        )

        # concat avec x1 :
        # 64 + 64 = 128
        self.dec1 = ConvBlock(128, 64)

    def forward(self, bottleneck, x1, x2, x3):

        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        return d1


# ==========================================
# FCN Model
# ==========================================
class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        # Pixel-wise classifier
        self.classifier = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        x1, x2, x3, bottleneck = self.encoder(x)

        x = self.decoder(
            bottleneck,
            x1,
            x2,
            x3
        )

        x = self.classifier(x)

        return x


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":

    model = FCN(num_classes=4)

    x = torch.randn(1, 3, 256, 256)

    out = model(x)

    print("Output shape :", out.shape)