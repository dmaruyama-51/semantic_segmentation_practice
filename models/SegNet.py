class SegNet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3):
        super().__init__()

        self.in_channels = in_channels

        self.down1 = Down2(self.in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down3(128, 256)
        self.down4 = Down3(256, 512)
        self.down5 = Down3(512, 512)

        self.up5 = Up3(512, 512)
        self.up4 = Up3(512, 256)
        self.up3 = Up3(256, 128)
        self.up2 = Up2(128, 64)
        self.up1 = Up2(64, n_classes)

    def forward(self, x):
        x, indices_1, unpool_shape1 = self.down1(x)
        x, indices_2, unpool_shape2 = self.down2(x)
        x, indices_3, unpool_shape3 = self.down3(x)
        x, indices_4, unpool_shape4 = self.down4(x)
        x, indices_5, unpool_shape5 = self.down5(x)

        x = self.up5(x, indices_5, unpool_shape5)
        x = self.up4(x, indices_4, unpool_shape4)
        x = self.up3(x, indices_3, unpool_shape3)
        x = self.up2(x, indices_2, unpool_shape2)
        out = self.up1(x, indices_1, unpool_shape1)

        return out
        

class cbr(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding):
        super().__init__()

        module = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=k_size, padding=padding, stride=stride),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.module(x)
        return out
        

class Down2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv3 = cbr(out_channels, out_channels, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpooled_shape = x.size()
        out, indices = self.pool(x)

        return out, indices, unpooled_shape

class Down3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = cbr(in_channels, out_channels, 3, 1, 1)
        self.conv2 = cbr(out_channels, out_channels, 3, 1, 1)
        self.conv3 = cbr(out_channels, out_channels, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpooled_shape = x.size()
        out, indices = self.pool(x)

        return out, indices, unpooled_shape

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = cbr(in_channels, in_channels, 3, 1, 1)
        self.conv2 = cbr(in_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices, output_size):
        x = self.unpool(in_channels, indices=indices, output_size=output_size)
        x = self.conv1(x)
        out = self.conv2(x)
        return out

class Up3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = cbr(in_channels, in_channels, 3, 1, 1)
        self.conv2 = cbr(in_channels, in_channels, 3, 1, 1)
        self.conv3 = cbr(in_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices, output_size):
        x = self.unpool(in_channels, indices=indices, output_size=output_size)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

