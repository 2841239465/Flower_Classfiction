from torch import nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        """请在此处作答 构造残差网络模型基本块"""
        ## 基本块基本构成 卷积1+BN1+ReLU+卷积2+BN2
        ## 由卷积1输入+BN2输出的downsample，请注意卷积1输入和BN2输出的特征尺寸变化
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """请在此处作答 构造残差网络模型基本块"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """The resnet model."""

    def __init__(self, infeat, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        """请在此处作答 补充完整输入ResNet的前几层"""
        ## ResNet模型前几层结构：卷基层+BN+ReLU+池化层
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        infeats = [num_blocks[i][1] for i in range(len(num_blocks))]
        infeats.insert(0, infeat)
        self.layers = nn.Sequential(*[
            self._make_layer(
                block,
                infeats[i],
                num_blocks[i][1],
                num_blocks[i][0],
                stride=num_blocks[i][2]) for i in range(len(num_blocks))
        ])

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_blocks[-1][1], num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks,
                    stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)

        out = self.avgpool1(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet10(num_classes):
    """请在此处作答 利用给定的ResNet类和补充完成功能的BasicBlock构造resnet10"""
    resnet10 = ResNet(infeat=64, block=BasicBlock, num_blocks=[[1,128,1], [1,256,2], [2,512,2]], num_classes=num_classes)
    return resnet10

def resnet12(num_classes):
    """请在此处作答 利用给定的ResNet类和补充完成功能的BasicBlock构造resnet12"""
    resnet12 = ResNet(infeat=64, block=BasicBlock, num_blocks=[[1,128,1], [2,256,2], [2,512,2]], num_classes=num_classes)
    return resnet12

def resnet20(num_classes):
    """请在此处作答 利用给定的ResNet类和补充完成功能的BasicBlock构造resnet20"""
    resnet20 = ResNet(infeat=64, block=BasicBlock, num_blocks=[[3,128,1], [3,256,2], [3,512,2]], num_classes=num_classes)
    return resnet20
