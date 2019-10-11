import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNNDecoder(nn.Module):
    def __init__(self, output_channels, args):
        super(ImpalaCNNDecoder, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [32, 32, 32, 16]
        self.downsample = not args.no_downsample
        if self.downsample:
            self.initial_conv_size = 32 * 9 * 9
            self.initial_conv_shape = (32, 9, 9)
        else:
            self.initial_conv_size = 32 * 12 * 9
            self.initial_conv_shape = (32, 12, 9)
        self.initial_linear = nn.Linear(self.hidden_size, self.initial_conv_size)

        """
        torch.Size([256, 32, 12, 9])
        torch.Size([256, 32, 25, 19])
        torch.Size([256, 32, 51, 39])
        torch.Size([256, 16, 103, 79])
        torch.Size([256, 1, 207, 159])
        """

        self.layer1 = self._make_layer(self.depths[0], self.depths[1], (1, 1))
        self.layer2 = self._make_layer(self.depths[1], self.depths[2])
        self.layer3 = self._make_layer(self.depths[2], self.depths[3])
        self.layer4 = self._make_layer(self.depths[3], output_channels)
        self.flatten = Flatten()
        self.train()

    def _make_layer(self, in_channels, depth, padding=(0, 0)):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, depth, 3, stride=2, output_padding=padding),
            # nn.ReflectionPad2d(padding),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    def forward(self, inputs):
        out = self.initial_linear(inputs)
        out = F.relu(out.view(-1, *self.initial_conv_shape))
        if self.downsample:
            out = self.layer4(self.layer3(self.layer2(out)))
        else:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                print(out.size())
                out = layer(out)
        print(out.size())
        return out

