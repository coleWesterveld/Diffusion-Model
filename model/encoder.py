import torch
from torch import nn
from decoder import VAE_AttentionBlock, VAE_ResidualBlock
from torch.nn import functional as F

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch size, channels, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height /2, width/2)
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),

            # (batch_size, 128, height /2, width/2) -> (batch_size, 128, height /2, width/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 128, height /2, width/2) -> (batch_size, 128, height /2, width/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height /2, width/2) -> (batch_size, 256, height /4, width/4)
            nn.Conv2d(256, 128, kernel_size = 3, stride = 2, padding = 0),

            # (batch_size, 256, height /2, width/2) -> (batch_size, 512, height /4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height /2, width/2) -> (batch_size, 512, height /4, width/4)
            VAE_ResidualBlock(512, 512), 

            # (batch_size, 512, height /4, width/4) -> (batch_size, 512, height /8, width/8)
            nn.Conv2d(512, 128, kernel_size = 3, stride = 2, padding = 0),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            VAE_ResidualBlock(512, 512),
         
            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 512, height /8, width/8)
            nn.GroupNorm(32, 512),

            # like a curvy ReLU, I guess it can in some cases converge faster and have marginally better performance
            # but ion think its that big of a deal but hey what do I know
            nn.SiLU(),

            # (batch_size, 512, height /8, width/8) -> (batch_size, 8, height /8, width/8)
            nn.Conv2d(512, 8, kernel_size = 3, padding = 0),

            # (batch_size, 8, height /8, width/8) -> (batch_size, 8, height /8, width/8)
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0) 
        )

        def forward(self, x: torch.Tensor, nosie:torch.Tensor) -> torch.Tensor:
            # x: (batch size, channels, height, width)
            # nosier: (batch size, out_channels,. height / 8, width / 8)

            # apply all layers of model in order
            for module in self:
                if getattr(module, 'stride', None) == (2, 2):
                    # padding: left, right, top, bottom
                    x = F.pad(x, (0, 1, 0, 1))
