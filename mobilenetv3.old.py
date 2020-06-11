'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

IdleBlock applied by Szh in 2020/6/11
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#TODO:

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),#TODO:SENet论文里是两次全连接
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, Idle):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.idle=Idle

        channel_x0=int(in_size*Idle)
        channel_x1=in_size-channel_x0
        channel_y1=out_size-int(out_size*Idle)

        #注意当使用Idle剪枝后，expand_size应根据参与卷积运算的通道数量进行调整
        self.conv1 = nn.Conv2d(channel_x1, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)#group=in_size时相当于做dw卷积
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, channel_y1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channel_y1)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),#当残差结构的输入和数出维度不匹配时，用conv+bn的方法将输入的尺寸变成输出尺寸
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        if(self.idle!=0):
            x0, x1 = torch.split(x, [int(x.shape[1] * self.idle), x.shape[1] - int(x.shape[1] * self.idle)],
                                 dim=1)  # Idle左右通道拆分
            out1 = self.nolinear1(self.bn1(self.conv1(x1)))# 右通道卷积操作，需要左边通道参与卷积的话就输入x0，分别对应论文中的R-Idle和L-Idle
            out1 = self.nolinear2(self.bn2(self.conv2(out1)))
            out1 = self.bn3(self.conv3(out1))
            out = torch.cat([x0, out1], 1)  # Idle左右通道拼接

        else:
            out = self.nolinear1(self.bn1(self.conv1(x)))
            out = self.nolinear2(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out#如果stride==1,out=out+shortcut(x);如果stride!=1,out=out。即stride==1时使用残差结构
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1, 0),#对应文中表1，(核大小，输入通道数，exp size，输出通道数，激活函数，se模块，stride大小,Idle剪枝比例)
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2, 0),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1, 0),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2, 0),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1, 0.75),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1, 0.5),
            Block(3, 40, 240, 80, hswish(), None, 2, 0),
            Block(3, 80, 200, 80, hswish(), None, 1, 0.5),
            Block(3, 80, 184, 80, hswish(), None, 1, 0.5),
            Block(3, 80, 184, 80, hswish(), None, 1, 0.75),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1, 0),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1, 0.75),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1, 0),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2, 0),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1, 0),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))#论文中该层没有bn
	#out = self.hs2(self.conv2(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)#将输出拉伸为一维
        out = self.hs3(self.bn3(self.linear3(out)))#论文中该层没有bn
	#out = self.hs3(self.linear3(out))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2, 0),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2, 0),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1, 0),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2, 0),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1, 0.5),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1, 0.5),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1, 0),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1, 0),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2, 0),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1, 0),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1, 0),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



def test():
    net = MobileNetV3_Small()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

# test()
