# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch
from common.arguments.basic_args import parse_args
from common.common_pytorch.model.srnet_utils.flex_layer import FlexGroupLayer
from common.common_pytorch.model.srnet_utils.group_index import get_input, shrink_output

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
        self.replication_padding = nn.ReplicationPad1d(1) # For replication padding
        self.padding = 1# For zero padding
        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        self.conv_inc = [1] * 34
        self.conv_seq, self.final_outc = get_input(args.group)

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
        self.frames= frames
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

    def _get_next_seq(self, inc, channels, out_seq):
        """
        Generate input information of the next layer
        :param inc: input sequence of each group, type:list e.g [455, 569]
        :param channels: output channel size of the whole layer, type:int e.g 1024
        :param out_seq: output sequence index of each each group, which decides how to group with those indexes. type:list. e.g [[0],[1]]
        :return: Next input sequence and Next output sequence index
        """
        in_ch_sum = 0
        for index, i in enumerate(out_seq):
            in_ch_sum += sum(map(lambda x: inc[x], i))
        out_chs = []
        next_seq = []
        for index, i in enumerate(out_seq):
            in_ch = sum(map(lambda x:inc[x],i))
            if len(out_seq) == 1:
                out_ch = channels
            elif index == len(out_seq)-1:
                out_ch = channels-sum(out_chs)
            else:
                out_ch = int(in_ch / in_ch_sum * channels)
            out_chs.append(out_ch)
            next_seq.append([index])
        return out_chs, next_seq

    def _get_all_seq(self, inc, channels, out_seq, filter_widths):
        """
        :return: Get all sequence info. for a model.
        """
        in_out_seq = []
        in_out_seq.append(self._get_next_seq(inc, channels, out_seq))
        # Generate input sequence and output sequence of each layer
        for i in range(1, len(filter_widths)):
            in_out_seq.append(self._get_next_seq(in_out_seq[2*i-2][0], channels, in_out_seq[2*i-2][1]))
            in_out_seq.append(self._get_next_seq(in_out_seq[2*i-1][0], channels, in_out_seq[2*i-1][1]))
        # For Final layer:
        in_out_seq.append(self._get_next_seq(in_out_seq[2*i][0], channels, in_out_seq[2*i][1]))
        return in_out_seq

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
        y = shrink_output(y)

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
        self.expand_conv = FlexGroupLayer(self.conv_inc, channels, self.conv_seq, kernel_size=filter_widths[0],
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func=args.mean_func,
                                          ups_mean=args.ups_mean, bias=False)

        in_out_seq = self._get_all_seq(self.conv_inc, channels, self.conv_seq, filter_widths)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]

        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)
            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-2][0], channels, in_out_seq[2*i-2][1], kernel_size=filter_widths[0], dilation=next_dilation,
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))

            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-1][0], channels, in_out_seq[2*i-1][1], kernel_size=1, dilation=1,
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))

            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.final_layer = FlexGroupLayer(in_out_seq[-1][0], self.final_outc, in_out_seq[2*i][1], kernel_size=1, dilation=1,
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func=args.mean_func,
                                          ups_mean=args.ups_mean, bias=True)

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]  # Drop left&right with length of pad
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
        x = self.final_layer(x)

        return x

class Same_Model(TemporalModelBase):
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
        self.expand_conv = FlexGroupLayer(self.conv_inc, channels, self.conv_seq, kernel_size=filter_widths[0],
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func=args.mean_func,
                                          ups_mean=args.ups_mean, bias=False)

        in_out_seq = self._get_all_seq(self.conv_inc, channels, self.conv_seq, filter_widths)

        layers_conv = []
        layers_bn = []
        self.ref_pad = []
        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]

        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)
            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-2][0], channels, in_out_seq[2*i-2][1], kernel_size=filter_widths[0], dilation=next_dilation,
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))

            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-1][0], channels, in_out_seq[2*i-1][1], kernel_size=1, dilation=1,
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))
            #self.ref_pad.append(nn.ReplicationPad1d(next_dilation))
            self.ref_pad.append(nn.ReflectionPad1d(next_dilation))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.final_layer = FlexGroupLayer(in_out_seq[-1][0], self.final_outc, in_out_seq[2*i][1], kernel_size=1, dilation=1,
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func=args.mean_func,
                                          ups_mean=args.ups_mean, bias=True)


        self.reflec = nn.ReflectionPad1d(1)
        #self.reflec = nn.ReplicationPad1d(1)
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(self.reflec(x)))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x # Drop left&right with length of pad
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](self.ref_pad[i](x)))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
        x = self.final_layer(x)

        return x


# 243-81-27-27-9-9-3-3-1
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

        self.expand_conv = FlexGroupLayer(self.conv_inc, channels, self.conv_seq, kernel_size=filter_widths[0], stride=filter_widths[0],
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func = args.mean_func, ups_mean=args.ups_mean, bias=False)
        in_out_seq = self._get_all_seq(self.conv_inc, channels, self.conv_seq, filter_widths)
        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)
            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-2][0], channels, in_out_seq[2*i-2][1], kernel_size=filter_widths[0], stride=filter_widths[0],
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(FlexGroupLayer(in_out_seq[2*i-1][0], channels, in_out_seq[2*i-1][1], kernel_size=1, dilation=1,
                                              feature_split=args.split, recombine=args.recombine,
                                              fix_seq=self.conv_seq, mean_func=args.mean_func,
                                              ups_mean=args.ups_mean, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
        self.final_layer = FlexGroupLayer(in_out_seq[-1][0], self.final_outc, in_out_seq[2*i][1], kernel_size=1, dilation=1,
                                          feature_split=args.split, recombine=args.recombine,
                                          fix_seq=self.conv_seq, mean_func=args.mean_func,
                                          ups_mean=args.ups_mean, bias=True)
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))
        x = self.final_layer(x)
        return x


