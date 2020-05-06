class cbr(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding, dilation, relu=True):
        super().__init__()

        self.add_module(
            "conv", nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation, bias=False)
            )
        self.add_module(
            "bn", nn.BatchNorm2d(out_channels)
        )
        
        if relu:
            self.add_module(
                "relu", nn.ReLU()
            )

class bottleneck(nn.Module):

    BOTTLENECK_EXPANSION = 4

    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super().__init__()

        mid_channels = out_channels // self.BOTTLENECK_EXPANSION

        self.reduce = cbr(in_channels, mid_channels, 1, stride, 0, 1, True)
        self.conv3x3 = cbr(mid_channels, mid_channels, 3, 1, dilation, dilation, True)
        self.increase = cbr(mid_channels, out_channels, 1, 1, 0, 1, False)
        self.shortcut = cbr(in_channels, out_channels, 1, stride, 0, 1, False) if downsample else lambda x:x #identity

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class ResidualBlock(nn.Sequential):
    def __init__(self, n_layers, in_channels, out_channels, stride, dilation, multi_grids=None):
        super().__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        for i in range(n_layers):
            self.add_module(
                "block{}".format(i +1),
                bottleneck(in_channels=(in_channels if i==0 else out_channels),
                           out_channels=out_channels,
                           stride = (stride if i==0 else 1),
                           dilation = dilation * multi_grids[i],
                           downsample = (True if i==0 else False))
            )

class Stem(nn.Sequential):
    def __init__(self, out_channels):
        super().__init__()

        self.add_module("conv1", cbr(3, out_channels, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResNet(nn.Sequential):
    def __init__(self, n_classes, n_blocks):
        super().__init__()

        ch = [64 * 2 ** p for p in range(6)]

        self.add_module("layer1", Stem(ch[0]))
        self.add_module("layer2", ResidualBlock(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", ResidualBlock(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", ResidualBlock(n_blocks[2], ch[3], ch[4], 2, 1))
        self.add_module("layer5", ResidualBlock(n_blocks[3], ch[4], ch[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", flatten())
        self.add_module("fc", nn.Linear(ch[5], n_classes))
    