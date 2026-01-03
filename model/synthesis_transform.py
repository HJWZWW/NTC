import torch.nn as nn
from model.layers import GDN, deconv


class SynthesisTransform(nn.Module):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__()
        self.decoder = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        return self.decoder(x)