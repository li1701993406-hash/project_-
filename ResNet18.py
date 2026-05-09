import torch
import torch.nn as nn
import torch.nn.functional as F

    #获取GPU
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #定义ResNet块
class ResNet_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1,kernel_size = 3,padding = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride = stride,padding = padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size,1,padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
            #定义残差连接
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels,out_channels,stride = stride,kernel_size = 1)
        else:
            self.shortcut = nn.Identity()

        #定义ResNet块的向前传播
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        return y + self.shortcut(x) #残差连接


    #定义网络层结构
class ResNet(nn.Module):
    def __init__(self,):
        super().__init__()
            #定义stage1层
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(kernel_size = 3,stride = 1)
            #定义主残差连接
        self.layer1 = nn.Sequential(ResNet_block(64,64,stride = 2,kernel_size = 3,))
        self.layer2 = nn.Sequential(ResNet_block(64,128,stride = 2,kernel_size = 3,))
        self.layer3 = nn.Sequential(ResNet_block(128,256,stride = 2,kernel_size = 3,))
        self.layer4 = nn.Sequential(ResNet_block(256,512,stride = 2,kernel_size = 3,))

            #全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512,14)

        #ResNet层向前传播
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.MaxPool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  #放入展平层后传入全连接层
        x = self.fc1(self.flatten(x))
        return x
