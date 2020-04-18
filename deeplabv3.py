'''
deeplabv3 = DeeplabV3(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
    )
'''

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


class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) # GAP
        self.conv = cbr(in_channels, out_channels, 1, 1, 0, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        x = self.pool(x)
        x = self.conv(x)
        out = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()

        self.stages = nn.Module()
        self.stages.add_module("c0", cbr(in_channels, out_channels, 1, 1, 0, 1))

        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                cbr(in_channels, out_channels, 3, 1, padding=rate, dilation=rate)
            )
        
        self.stages.add_module("imagepool", ImagePool(in_channels, out_channels))

    def forward(self, x):
        out = torch.cat([stage(x) for stage in self.stages.children()], dim=1)
        return out



class DeeplabV3(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super().__init__()
        
        #Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        
        self.add_module("layer1", Stem(ch[0]))
        self.add_module("layer2", ResidualBlock(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer3", ResidualBlock(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer4", ResidualBlock(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module("layer5", ResidualBlock(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids))
        self.add_module("aspp", ASPP(ch[5], 256, atrous_rates))

        concat_ch = 256 * (len(atrous_rates) + 2) #cbr, rates, imagepool

        self.add_module("fc1", cbr(concat_ch, 256, 1, 1, 0, 1)) 
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))