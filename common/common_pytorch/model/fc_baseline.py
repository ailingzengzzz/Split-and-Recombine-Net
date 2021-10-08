# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from common.arguments.basic_args import parse_args
args = parse_args()

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]

        x_out = x

        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        y = self._forward_blocks(x)

        y = y.permute(0, 2, 1) #[1024,1,3K]

        y = y.view(sz[0], -1, self.num_joints_out, 3)
        if args.norm == 'lcn':
            pose_2d = x_out + y[...,:2]
            y = torch.cat([pose_2d, y[...,2:3]], dim=-1)
        return y


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0],groups=1, bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,groups=1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False,groups=1))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], stride=filter_widths[0], groups=1, bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], groups=1,bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1,groups=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x



class Same_Model(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    Change the padding number and type, padding before putting into convolution (self.zeropadding, self.rep_pad with ReplicationPad1d, nn.ConstantPad1d, nn.ReflectionPad1d):
        if padding == 0: The same setting as the TemporalModel
        if padding == number of dilation or stride: it will keep the same temporal size as 2d inputs
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.

        New Arguments:
        FlexGroupLayer: Use this function with different group strategies
        self.rep_pad: Recommend use nn.ReflectionPad1d to make the same temporal size as 2d inputs.

        """
        mode = 'replicate'   #padding mode: reflect, replicate, zeros
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, kernel_size=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.ref_pad = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)
            layers_conv.append(nn.Conv1d(channels, channels, kernel_size=filter_widths[0], dilation=next_dilation, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, kernel_size=1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            self.ref_pad.append(nn.ReplicationPad1d(next_dilation))
            #self.ref_pad.append(nn.ReflectionPad1d(next_dilation))
            next_dilation *= filter_widths[i]
        #self.reflec = nn.ReflectionPad1d(1)
        self.reflec = nn.ReplicationPad1d(1)
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_layer = nn.Conv1d(channels, num_joints_out * 3, kernel_size=1, bias=True)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(self.reflec(x)))))
        for i in range(len(self.pad) - 1):
            res = x
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](self.ref_pad[i](x)))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
        x = self.final_layer(x)
        return x

