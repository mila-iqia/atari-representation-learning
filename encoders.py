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


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, hidden_size=512):
        super(ImpalaCNN, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.depths = [16, 32, 32]
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.flatten = Flatten()
        self.final_linear = init_(nn.Linear(self.depths[2] * 7 * 7, hidden_size))

        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    def forward(self, inputs):
        out = inputs
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.relu(self.final_linear(self.flatten(out)))

        return out
