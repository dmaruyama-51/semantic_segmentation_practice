class PSPNet(nn.Module):
    def __init__(self, n_classes=21, block_config=[3, 4, 23, 3], input_size=(473, 473)):
        super().__init__()

        self.block_config = block_config
        self.n_classes = n_classes
        self.input_size = input_size

        # Encoder
        self.cbr1 = conv2dBatchNormRelu(in_channels=3, n_filters=64, k_size=3, padding=1, stride=2)
        self.cbr2 = conv2dBatchNormRelu(64, 64, 3, 1, 1)
        self.cbr3 = conv2dBatchNormRelu(64, 128, 3, 1, 1)

        # Vanilla Residual Blocks
        self.res_block2 = residualBlock(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlock(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlock(self.block_config[2], 512, 256, 1024, 1, dilation=2)
        self.res_block5 = residualBlock(self.block_config[3], 1024, 512, 2048, 1, dilation=4)

        # Pyramid Pooling Module
        self.pyramid_pool = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv
        self.cbr_final = conv2dBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.cls = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.cbr_aux = conv2dBatchNormRelu(in_channels=1024, n_filters=256, k_size=3, padding=1, stride=1)
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

        # Auxiliary loss
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x):
        h, w = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Auxiliary layers for training
        if self.training:
            x_aux = self.cbr_aux(x)
            x_aux = self.dropout(x_aux)
            x_aux = self.aux_cls(x_aux)

        x = self.res_block5(x)

        x = self.pyramid_pool(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return (x, x_aux)

        else: #eval
            return x


class conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding):
        super().__init__()

        module = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=k_size, padding=padding, stride=stride),
            nn.BatchNorm2d(n_filters)
        )
    
    def forward(self, x):
        out = self.module(x)
        return out


class conv2dBatchNormRelu(nn.Module):
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

class bottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation=1):
        super().__init__()

        self.cbr1 = conv2dBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0)

        if dilation > 1:
            self.cbr2 = conv2dBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=dilation, dilation=dilation)

        else:
            self.cbr2 = conv2dBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=1, dilation=1)

        self.cb3 = conv2dBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0)
        self.cb4 = conv2dBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)

        return F.relu(conv + residual, inplace=True)

class bottleNeckIdentity(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1):
        super().__init__()

        self.bnr1 = conv2dBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0)

        if dilation > 1:
            self.bnr2 = conv2dBatchNormRelu(mid_channels, mid_channels, 3, stride=1, padding=dilation, dilation=dilation)
        else:
            self.bnr2 = conv2dBatchNormRelu(mid_channels, mid_channels, 3, stride=1, padding=1, dilation=1)

        self.cb3 = conv2dBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))

        return F.relu(x + residual, inplace=True)



class residualBlock(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1):
        super().__init__()

        if dilation > 1:
            stride = 1

        
        # residualBlock = convBlock + identityBlock
        layers = []
        layers.append(
            bottleNeck(in_channels, mid_channels, out_channels, stride, dilation)
        )
        for i in range(n_blocks - 1):
            layers.append(
                bottleNeckIdentity(out_channels, mid_channels, stride, dilation)
            )
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class pyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()

        out_channels = int(in_channels / len(pool_sizes))
        self.cbr = conv2dBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool_sizes = pool_sizes
        
    def forward(self, x):
        h, w = x.shape[2:]

        out1 = self.cbr(F.adaptive_avg_pool2d(x, output_size=(pool_sizes[0], pool_sizes[0])))
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear", align_corners=True)

        out2 = self.cbr(F.adaptive_avg_pool2d(x, output_size=(pool_sizes[1], pool_sizes[1])))
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear", align_corners=True)

        out3 = self.cbr(F.adaptive_avg_pool2d(x, output_size=(pool_sizes[2], pool_sizes[2])))
        out3 = F.interpolate(out3, size=(h, w), mode="bilinear", align_corners=True)

        out4 = self.cbr(F.adaptive_avg_pool2d(x, output_size=(pool_sizes[3], pool_sizes[3])))
        out4 = F.interpolate(out4, size=(h, w), mode="bilinear", align_corners=True)

        out = torch.cat([x, out1, out2, out3, out4], dim=1)

        return out

