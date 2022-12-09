import torch
from torch import nn


class ae(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = torch.nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        # )
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.unpool1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.decoder = torch.nn.Sequential(
        #
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        #     #nn.ConvTranspose2d(in_channels=128, out_channels=64, output_padding=1, kernel_size=3, stride=2, padding=1),
        #     #nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     #nn.ConvTranspose2d(in_channels=64, out_channels=3, output_padding=1, kernel_size=3, stride=2, padding=1),
        # )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.pool1(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.pool2(x1)

        x2 = self.unpool1(x1)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)
        x2 = self.unpool2(x2)
        x2 = self.conv7(x2)
        x2 = self.conv8(x2)
        #x1 = self.encoder(x)
        #x2 = self.decoder(x1)
        return x1, x2
