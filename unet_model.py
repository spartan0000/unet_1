#unet model
import torch
import torch.nn as nn
import torch.nn.functional as F




class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2) #skip connections take the 512 output here, add 512 and input 1024 in the next line's conv layer
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dec1 = conv_block(128, 64)

        self.output = nn.Conv2d(64, 3, kernel_size = 1)

    def forward(self, x):
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim = 1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim = 1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim = 1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim = 1)
        d1 = self.dec1(d1)

        out = self.output(d1)

        return out

