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
    def __init__(self, input_channels, args):
        super(ImpalaCNN, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [32, 64, 128, 256]
        self.downsample = not args.no_downsample
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])
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
        if self.downsample:
            out = self.layer3(self.layer2(self.layer1(out)))
        else:
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.avg_pool2d(out, out.size()[2:], stride=1).squeeze()


class NatureCNN(nn.Module):
    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size # redundant
        self.downsample = not args.no_downsample
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
            )
        else:
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 256, 3, stride=1)),
            )
        self.train()

    def forward(self, inputs):
        out = self.main(inputs)
        return F.relu(F.avg_pool2d(out, out.size()[2:], stride=1)).squeeze()
