#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from copy import deepcopy
from torch import nn

# class Dense_Block(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, dropout_rate):
#         super(Dense_Block, self).__init__()
#         self.conv_list = nn.ModuleList()
#         self.planes = planes
#         self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=1, dilation=1)
#         self.conv_2 = nn.Conv2d((inplanes + planes), planes, kernel_size=kernel_size, padding=2,  dilation=2)
#         self.conv_3 = nn.Conv2d((inplanes + 2 * planes), planes, kernel_size=kernel_size, padding=3, dilation=3)

#         self.norm = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#         self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#         self.conv_1_1 = nn.Conv2d((inplanes + 3 * planes), planes, kernel_size=[1 for _ in kernel_size],
#                                   padding=[0 for i in kernel_size])
#         self.norm_1_1 = nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#         self.nonlin_1_1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
#         # dropout
#         self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

#     def forward(self, x): # x : (4,32,96,96,64)
#         out_1 = self.nonlin(self.norm(self.dropout(self.conv_1(x))))  # out_1 : (4,32,94,94,62)
#         residual_1 = torch.cat((x, out_1), dim=1)  # 32 + 32

#         out_2 = self.nonlin(self.norm(self.dropout(self.conv_2(residual_1))))  # 32
#         residual_2 = torch.cat((out_2, residual_1), dim=1)  # 32 + 64

#         out_3 = self.nonlin(self.norm(self.dropout(self.conv_3(residual_2))))  # 32
#         out = torch.cat((residual_2, out_3), dim=1)  # 96 + 32

#         out = self.nonlin_1_1(self.norm_1_1(self.conv_1_1(out)))  # 32
#         return out


class DenseDownBlock_first(nn.Module):

    def __init__(self, in_planes, kernel_size, dropout_rate):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_planes*2, in_planes, kernel_size=kernel_size, stride=1, padding=1)

        self.norm = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.pool_op = nn.MaxPool2d(2,stride=2)

    def forward(self, x):
        residual_1 = x  # 32
        f = self.conv1(x)
        out_1 = self.nonlin(self.norm(self.dropout(self.conv1(x))))
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2
        residual_out = self.nonlin(self.norm(self.dropout(self.conv2(concat_1))))  # 32 * 2
        out = self.pool_op(residual_out)

        return out , residual_out



# 여기서 DDenseBlock 구현

class DenseDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropout_rate):

        super().__init__()

        self.kernel_size = kernel_size
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        # maxpooling 구현
        self.pool_op = nn.MaxPool2d(2,stride=2)

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_planes*2, in_planes*2, kernel_size=kernel_size, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.norm2 = nn.BatchNorm2d(in_planes*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

        # conv3 = 1*1 conv
        self.conv3 = nn.Conv2d(in_planes * 4, in_planes * 2, [1 for _ in kernel_size], padding=[0 for i in kernel_size])
        self.norm3 = nn.BatchNorm2d(in_planes * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    
    def forward(self, x):


        # before version
        residual_1 = x  # 32
        out_1 = self.nonlin(self.norm1(self.dropout(self.conv1(x))))  # 32
        residual_2 = out_1
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 32 * 2

        out = self.nonlin(self.norm2(self.dropout(self.conv2(concat_1))))  # 32 * 2

        concat_2 = torch.cat((out, residual_1), dim=1)  # 32*2 + 32*1 = 32 * 3
        concat_2 = torch.cat((concat_2, residual_2), dim = 1) # 32*3 + 32* = 32 * 4

        residual_out = self.nonlin3(self.norm3(self.conv3(concat_2)))
        out = self.pool_op(residual_out)

        return out ,residual_out


class DenseDownLayer_first(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, block=DenseDownBlock_first):
        super().__init__()
        self.convs = nn.Sequential(
            block(in_channel, kernel_size, drop_rate)
        )
    def forward(self, x):
        return self.convs(x)

class DenseDownLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, block=DenseDownBlock):
        super().__init__()
        self.convs = nn.Sequential(
            block(in_channel, out_channel, kernel_size, drop_rate)
        )
    def forward(self, x):
        return self.convs(x)









# 여기는 Dense_Up_Block 구현하기

class DenseUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dropout_rate):

        super().__init__()

        aim_planes = in_planes // 2  # 256
        self.conv0 = nn.Conv2d(in_planes, aim_planes, [1 for _ in kernel_size],
                                      padding=[0 for i in kernel_size])

        self.norm0 = nn.BatchNorm2d(aim_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin0 = nn.LeakyReLU(negative_slope=0.01, inplace=True)


        self.conv1 = nn.Conv2d(aim_planes, aim_planes, kernel_size=kernel_size, stride=1, padding=1)

        self.conv2 = nn.Conv2d(aim_planes*2, aim_planes, kernel_size=kernel_size, stride=1, padding=1)

        self.conv3 = nn.Conv2d(aim_planes * 3, aim_planes, [1 for _ in kernel_size],padding=[0 for i in kernel_size])
        self.norm3 = nn.BatchNorm2d(aim_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)


    def forward(self, x):
        x = self.nonlin0(self.norm0(self.conv0(x)))  # 256
        residual_1 = x  # 256

        out_1 = self.conv1(x)  # 256
        residual_2 = out_1  # 256
        concat_1 = torch.cat((out_1, residual_1), dim=1)  # 512

        out = self.conv2(concat_1)  # 256

        concat_2 = torch.cat((out, residual_1), dim=1)  # 512
        concat_2 = torch.cat((concat_2,residual_2), dim = 1) # 512 + 256

        out = self.norm3(self.conv3(concat_2))
        out = self.nonlin3(out)

        return out


class DenseUpLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, drop_rate, block=DenseUpBlock):
        super().__init__()

        self.convs = nn.Sequential(
            block(in_channel, out_channel, kernel_size, drop_rate)
        )

    def forward(self, x):
        return self.convs(x)






class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
