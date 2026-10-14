"""
Architecture variants of the variational UNet, tried in the search
before anything is moved into models/.

- residual blocks: every VariationalDoubleConv gets a shortcut, as in
  models/residual_unet.py (identity, or a variational 1x1 conv when the
  channel count changes).
- global residual: the network predicts a correction to its input,
  prediction = x + f(x).
"""

from torch import nn

from models.layers import get_activation, get_normalization
from models.variational import VariationalConv2d
from models.variational_unet import VariationalDoubleConv


class VariationalResidualDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation,
                 normalization, num_groups, prior_standard_deviation,
                 initial_rho):
        super().__init__()
        padding = kernel_size // 2
        v = {"prior_standard_deviation": prior_standard_deviation,
             "initial_rho": initial_rho}
        self.conv1 = VariationalConv2d(in_channels, out_channels, kernel_size,
                                       padding=padding, **v)
        self.post1 = nn.Sequential(
            get_normalization(normalization, out_channels, num_groups),
            get_activation(activation),
        )
        self.conv2 = VariationalConv2d(out_channels, out_channels, kernel_size,
                                       padding=padding, **v)
        self.norm2 = get_normalization(normalization, out_channels, num_groups)
        self.shortcut = (
            None if in_channels == out_channels
            else VariationalConv2d(in_channels, out_channels, 1, **v)
        )
        self.activation = get_activation(activation)

    def forward(self, x, sample=True):
        y, kl1 = self.conv1(x, sample=sample)
        y = self.post1(y)
        y, kl2 = self.conv2(y, sample=sample)
        y = self.norm2(y)
        kl = kl1 + kl2
        if self.shortcut is None:
            s = x
        else:
            s, kl3 = self.shortcut(x, sample=sample)
            kl = kl + kl3
        return self.activation(y + s), kl


def use_residual_blocks(model, p):
    def convert(block):
        return VariationalResidualDoubleConv(
            block.conv1.in_channels, block.conv2.out_channels,
            p["kernel_size"], p["activation"], p["normalization"],
            p["num_groups"], float(p["prior_standard_deviation"]),
            float(p["initial_rho"]),
        )

    for blocks in [model.encoders, model.decoders]:
        for i, b in enumerate(blocks):
            if isinstance(b, VariationalDoubleConv):
                blocks[i] = convert(b)
    model.bottleneck = convert(model.bottleneck)
    return model


class GlobalResidual(nn.Module):

    def __init__(self, net, variational):
        super().__init__()
        self.net = net
        self.variational = variational

    def forward(self, x, sample=True):
        if self.variational:
            y, kl = self.net(x, sample=sample)
            return x + y, kl
        return x + self.net(x)
