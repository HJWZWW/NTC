import torch.nn as nn
from model.layers import GDN, conv

class AnalysisTransform(nn.Module):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )


    def forward(self, x):
        return self.encoder(x)