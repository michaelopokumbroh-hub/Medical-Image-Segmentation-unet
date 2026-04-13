# ResNet34 backbone + UNet decoder.

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
# ResNet BasicBlock
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet34Encoder(nn.Module):
# ResNet34 encoder for UNet
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(*[BasicBlock(64 if i==0 else 64, 64) for i in range(3)])
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            *[BasicBlock(128, 128) for _ in range(3)]
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            *[BasicBlock(256, 256) for _ in range(5)]
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            *[BasicBlock(512, 512) for _ in range(2)]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [x, c1, c2, c3, c4, c5]

class UNetDecoder(nn.Module):
# UNet decoder block for ResNet34 + UNet
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, skip_ch in zip(enc_channels, dec_channels, enc_channels[1:]):
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, 2))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
    def forward(self, features):
        x = features[-1]
        skips = features[-2::-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
        return x

class ResNet34UNet(nn.Module):
# Full ResNet34-UNet for segmentation
    def __init__(self):
        super().__init__()
        self.encoder = ResNet34Encoder()
        enc_channels = [512, 256, 128, 64, 64, 3]
        dec_channels = [256, 128, 64, 64, 32]
        self.decoder = UNetDecoder(enc_channels, dec_channels)
        self.final = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        feats = self.encoder(x)
        d = self.decoder(feats)
        out = self.final(d)
        return out
