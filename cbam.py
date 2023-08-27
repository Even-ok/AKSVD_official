"""
Channel & space attention module
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avgOut = self.fc(self.avg_pool(x).view(b, c))  #avg_pool:[32, 256, 1, 1]  avg:[32, 256]
        maxOut = self.fc(self.max_pool(x).view(b, c))  #maxOut:[32, 256]
        y = self.sigmoid(avgOut + maxOut).view(b, c, 1, 1)  #y:[32, 256,1,1]
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding= (kernel_size -1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, h, w = x.size()
        avgOut = torch.mean(x, dim=1, keepdim=True)  #avgOut[32, 1, 32, 32]
        maxOut, _ = torch.max(x, dim=1, keepdim=True)  #maxOut[32, 1, 32, 32]
        y = torch.cat([avgOut, maxOut], dim=1)  #y[32, 2, 32, 32]   
        y = self.sigmoid(self.conv(y))  #y[32, 1, 32, 32]  
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(channel, reduction)
        self.SpatialAtt = SpatialAttention(kernel_size) 

    def forward(self, x):
        x = self.ChannelAtt(x)
        x = self.SpatialAtt(x)
        return x