'''
わかりやすい。VGGから実装
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py
'''

class FCN32s(nn.Module):
    def __init__(self, n_classes=21):
        super().__init__()

        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
             nn.Conv2d(4096, self.n_classes, 1)
        )

    def forward(self, x):
        conv1 = self.block1(x)
        conv2 = self.block2(conv1) 
        conv3 = self.block3(conv2)
        conv4 = self.block4(conv3)
        conv5 = self.block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample(score, x.size()[2:])

        return out

class FCN16s(nn.Module):
    def __init__(self, n_classes=21):
        super().__init__()

        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
             nn.Conv2d(4096, self.n_classes, 1)
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.block1(x)
        conv2 = self.block2(conv1) 
        conv3 = self.block3(conv2)
        conv4 = self.block4(conv3)
        conv5 = self.block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4

        out = F.upsample(score, x.size()[2:])

        return out

class FCN8s(nn.Module):
    def __init__(self, n_classes=21):
        super().__init__()

        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
             nn.Conv2d(4096, self.n_classes, 1)
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.block1(x)
        conv2 = self.block2(conv1) 
        conv3 = self.block3(conv2)
        conv4 = self.block4(conv3)
        conv5 = self.block5(conv4)

        score = self.classifier(conv5)


        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4

        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3

        out = F.upsample(score, x.size()[2:])

        return out




'''
https://github.com/petko-nikolov/pysemseg/blob/master/pysemseg/models/fcn.py
'''

class FCN32s(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.vgg = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.vgg.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

class FCN16s(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.vgg = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1)
        )

        #追加
        self.conv4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.up5 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # VGGの中間の特徴マップ
        pool4 = self.vgg.features[:-7](x)
        pool5 = self.vgg.features[-7:](pool4)

        # pool5は２倍にアップサンプリング
        pool5_up = self.up5(self.classifier(pool5))
        
        pool4 = self.conv4(pool4)


        # 和を取る
        x = pool4 + pool5_up

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

class FCN８s(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.vgg = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1)
        )

        #追加
        self.conv3 = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(512, n_classes, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # VGGの中間の特徴マップ
        pool3 = self.vgg.features[:-14](x)
        pool4 = self.vgg.features[-14:-7](pool3)
        pool5 = self.vgg.features[-7:](pool4)

        # pool5は２倍にアップサンプリング
        pool5_up = self.up5(self.classifier(pool5))
        
        pool4 = self.conv4(pool4)


        # 和を取る
        x = pool4 + pool5_up

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

