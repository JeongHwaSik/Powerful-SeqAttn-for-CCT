import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze and Excitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))

        self.excite = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape # (B, C, H, W)

        # .view() is 20% faster than .reshape()
        out = self.squeeze(x).view(B, C) # (B, C, 1, 1) -> (B, C)
        out = self.excite(out).view(B, C, 1, 1).expand_as(x) # (B, C, 1, 1) -> (B, C, H, W)

        return x * out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride, downsample=None, se=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SEBlock(planes, reduction=16) if se is True else nn.Identity()

        # fitting channels and (H, W) for skip connections
        self.skip_connection = downsample


    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.skip_connection is not None:
            residual = self.skip_connection(x)

        out += residual
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride, downsample=None, se=False):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.se = SEBlock(planes * self.expansion, reduction=16) if se is True else nn.Identity()

        # fitting channels and (H, W) for skip connections
        self.skip_connection = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.skip_connection is not None:
            residual = self.skip_connection(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    """
    ResNet model for CIFAR dataset (32×32×3)
    """

    def __init__(self,
                 num_blocks: list,
                 channels: list = [64, 128, 256, 512],
                 strides: list = [1 ,2, 2, 2],
                 in_channel = 3,
                 num_classes = 100,
                 bottleneck = False,
                 se = False
        ):
        super(ResNet, self).__init__()

        # BottleNeck Blocks vs. Basic Blocks
        if bottleneck:
            block_type = BottleNeck
        else:
            block_type = BasicBlock

        # add SEBlock or not
        self.se = se

        # self.in_channel: input channel for each Blocks
        # n_channel: channel inside of Blocks (e.g. BottleNeck Block's bottleneck channel)
        self.in_channel = channels[0]


        # aggressive stem layer (downsampling 4×)
        self.conv1 = nn.Conv2d(in_channel, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        # make BIG Blocks
        self.layers = [self._make_layer(block_type, num_blocks[i], channels[i], strides[i]) for i in range(len(num_blocks))]

        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        self.classifier = nn.Linear(channels[-1] * block_type.expansion, num_classes)

        # weight initialization
        self.init_weights()

        # to add BIG Blocks into the model (e.g. print(model))
        self._register_layer()

    def _make_layer(self, block_type, n_block:int, n_channel:int, stride:int):

        # Convolution for skip connections
        if self.in_channel != n_channel * block_type.expansion or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, n_channel * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_channel * block_type.expansion)
            )
        else:
            downsample = None

        layers = []
        layers.append(block_type(self.in_channel, n_channel, stride, downsample, se=self.se))

        self.in_channel = n_channel * block_type.expansion
        for _ in range(1, n_block):
            layers.append(block_type(self.in_channel, n_channel, stride=1, downsample=None, se=self.se))

        return nn.Sequential(*layers)


    def _register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i+1, 'layer'))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        for layer in self.layers:
            x = layer(x)

        x = nn.Flatten(1)(self.glob_avg_pool(x))

        return self.classifier(x)


model_config = {
    # ResNet
    'resnet18' : {'parameter':dict(num_blocks=[2, 2, 2, 2], bottleneck=False), 'etc':{}},
    'resnet34': {'parameter':dict(num_blocks=[3, 4, 6, 3], bottleneck=False), 'etc':{}},
    'resnet50': {'parameter':dict(num_blocks=[3, 4, 6, 3], bottleneck=True), 'etc':{}},
    'resnet101': {'parameter': dict(num_blocks=[3, 4, 23, 3], bottleneck=True), 'etc': {}},
    'resnet152': {'parameter': dict(num_blocks=[3, 8, 36, 3], bottleneck=True), 'etc': {}},

    # SE-ResNet
    'seresnet18' : {'parameter':dict(num_blocks=[2, 2, 2, 2], bottleneck=False, se=True), 'etc':{}},
    'seresnet34': {'parameter': dict(num_blocks=[3, 4, 6, 3], bottleneck=False, se=True), 'etc': {}},
    'seresnet50': {'parameter': dict(num_blocks=[3, 4, 6, 3], bottleneck=True, se=True), 'etc': {}},
    'seresnet101': {'parameter': dict(num_blocks=[3, 4, 23, 3], bottleneck=True, se=True), 'etc': {}},
    'seresnet152': {'parameter': dict(num_blocks=[3, 8, 36, 3], bottleneck=True, se=True), 'etc': {}},
}

def create_resnet_for_cifar(model_name, num_classes):
    config = model_config[model_name]['parameter']
    return ResNet(num_classes=num_classes, **config)