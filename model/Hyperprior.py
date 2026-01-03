import numpy as np
import torch
import math
import torch.nn as nn
from loss.distortion import Distortion
from model.layers import Mlp
from compressai.entropy_models.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste
from model.analysis_transform import AnalysisTransform as ga
from model.synthesis_transform import SynthesisTransform as gs
from utils.utils import BCHW2BLN, BLN2BCHW
from model.layers import conv, deconv
from compressai.ans import BufferedRansEncoder, RansDecoder
import warnings
import torch.nn.functional as F


class Hyperprior(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config['N']
        M = config['M']
        self.ga = ga(N=N, M=M)
        self.gs = gs(N=N, M=M)

        self.ha = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.hs = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M, stride=1, kernel_size=3),
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

        self.distortion = Distortion(config)
        self.H = self.W = 0

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.02)


    def update(self, force=True):
        scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        self.entropy_bottleneck.update(force=force)
        self.gaussian_conditional.update_scale_table(scale_table)
        return self.entropy_bottleneck.quantized_cdf.size()

    def forward(self, x, require_probs=False):
        B, C, H, W = x.shape
        y = self.ga(x)
        z = self.ha(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales_hat = self.hs(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)

        x_hat = self.gs(y_hat)
        mse_loss = self.distortion(x, x_hat)
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W) / B

        if require_probs:
            return mse_loss, bpp_y, bpp_z, x_hat, y, y_likelihoods, scales_hat
        else:
            return mse_loss, bpp_y, bpp_z, x_hat


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss