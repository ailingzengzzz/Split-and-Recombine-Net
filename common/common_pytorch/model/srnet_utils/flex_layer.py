import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import math

from common.arguments.basic_args import parse_args
from common.common_pytorch.model.srnet_utils.global_module import With_all_joints, With_other_joints

args = parse_args()

class FlexGroupLayer(nn.Module):
    def __init__(self, inc, outc, out_seq, kernel_size=3, padding=0, dilation=1, stride=1,
                 feature_split='others', recombine='multiply', repeat_concat=False, fix_seq=None, mean_func=False, ups_mean=False, bias=None):
        super(FlexGroupLayer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.in_channel_group = inc
        self.in_channel = sum(inc)
        self.out_channel = outc
        self.out_seq = out_seq
        self.in_group_num = len(inc)
        self.out_group_num = len(out_seq)

        # Experiment params
        self.feature_split = feature_split

        # Operators of combination info.
        self.recombine = recombine
        self.fix_seq = fix_seq #Used for concat by the [1st layer ratio]
        self.mean_func = mean_func
        self.ups_mean = ups_mean
        in_ch_sum = 0
        for index, i in enumerate(out_seq):
            in_ch_sum += sum(map(lambda x:self.in_channel_group[x],i))
        in_group_accmulate = [0]
        self.groups = []
        for i in range(self.in_group_num):
            in_group_accmulate.append(sum(self.in_channel_group[:i+1]))
        self.out_chs = [] #record each out channels of each conv to computer last group channel number
        group_conv = []
        # Prepare concat
        cat_num = []
        part_in = [] #For partial inputs
        in_cat = [0]
        in_ch_cat_list = []
        self.concat_group = []
        for index,i in enumerate(out_seq):
            in_ch = sum(map(lambda x:self.in_channel_group[x],i))
            if self.out_group_num == 1:
                out_ch = self.out_channel
            elif index == len(out_seq)-1:
                out_ch = self.out_channel-sum(self.out_chs)
            else:
                out_ch = int(in_ch/in_ch_sum*self.out_channel)
            self.out_chs.append(out_ch)
            part_in.append(in_ch_sum-in_ch)
            if self.recombine == 'concat':
                in_ch_cat, cat_num_ = self._keep_ratio(in_ch, self.fix_seq, index, added_dim=1, by_ratio=False)
                in_ch_cat_list.append(in_ch_cat)
                in_cat.append(sum(in_ch_cat_list))
                cat_num.append(cat_num_) # Store How many additional values can be included?
                cat_index = (list(map(lambda x:[in_cat[x], in_cat[x+1]],[index])))
                self.concat_group.append(cat_index)
                group_conv.append(nn.Conv1d(in_ch_cat, out_ch, kernel_size=self.kernel_size, stride=self.kernel_size, padding=0, dilation=1, groups=1, bias=False))
            else:
                group_conv.append(nn.Conv1d(in_ch,out_ch,kernel_size=self.kernel_size,stride=self.kernel_size,padding=0, dilation=1,groups=1,bias=False))
            indexes = (list(map(lambda x:[in_group_accmulate[x],in_group_accmulate[x+1]],i)))
            self.groups.append(indexes)
        self.group_conv = nn.ModuleList(group_conv)

        # Get information from all joints or other joints
        if self.feature_split == 'all':
            self.get_all_info = With_all_joints(inc, outc, out_seq, kernel_size, padding, dilation, stride,
                                                modulation=args.modulation, group_modulation=args.group_modulation,split_modulation=args.split_modulation,
                                                channelwise=args.channelwise, recombine=self.recombine, repeat_concat=args.repeat_concat, mean_dim=cat_num, global_info=feature_split, bias=None)
        elif self.feature_split == 'others':
            self.get_part_info = With_other_joints(inc, outc, out_seq, kernel_size, padding, dilation, stride,
                                                   split_modulation=args.split_modulation, recombine=self.recombine, repeat_concat=args.repeat_concat,
                                                   in_c=part_in, mean_func=mean_func, mean_dim=cat_num, ups_mean=ups_mean)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))


    def forward(self, x):
        dtype = x.data.type()
        # Rerank and group input following the joint indexes
        x_group, x_full = self._split_fc(x, dtype)

        # Divide the info. source from [all joints] or [other joints] that is out of the group
        if self.feature_split == 'all':
            x_out = self._get_global_input(x_full)
            # Get additional info from all joints
            m = self.get_all_info((x, x_full)) #[B, C, w,N]

        elif self.feature_split == 'none':
            # No any other joints info
            x_out = x_full

        elif self.feature_split == 'others':
            x_other, x_out = self._get_partial_input(x_full)
            # Get additional info from partial joints
            m = self.get_part_info((x_other, x_group))
        else:
            raise KeyError('Invalid split strategies, please check srnet_args!')
        # Re-organize inner sequence of features for preparation of a convolution
        x_off = self._reorganize_sequence(x_out, dtype)
        # Since calculate mean value of input will not change temporal dimension, and x_off has changed the temporal dimension. Change m coorrespondingly.
        if self.mean_func and not self.ups_mean:
            if self.recombine == 'concat':
                m1 = []
                for i in range(len(self.out_seq)):
                    m1.append(self._reorganize_sequence(m[i], dtype))
            elif self.recombine == 'add' or 'multiply':
                m1 = self._reorganize_sequence(m, dtype)
            else:
                raise KeyError('Invalid combination operator, please check srnet_args!')
            m = m1

        # Since use the full input with root joint in each group
        if len(self.out_chs) == 2:
            new_group = [[[0, 16]],[[16, 36]]]
        elif len(self.out_chs) == 1:
            new_group = [[[0, 34]]]
        elif len(self.out_chs) == 3:
            new_group = [[[0, 14]],[[14,24]],[[24,38]]]
        elif len(self.out_chs) == 5:
            new_group =[[[0,8]], [[8,16]], [[16,26]], [[26,34]], [[34,42]]]
        else:
            raise KeyError('Invalid group number!')

        # Do [operations] to aggregate information in each group
        if self.recombine == 'concat':
            if x_off.shape[1]<50:
                x_ = self.op_cat(x_off, m, new_group)
            elif x_off.shape[1] == 1024:
                x_ = self.op_cat(x_off, m, self.groups)
        elif self.recombine == 'add':
            x_ = self.op_add(x_off, m)
        elif self.recombine == 'multiply':
            x_ = self.op_mul(x_off, m)
        else: # zero function
            x_ = x_off

        # Fully connection in each group
        if self.recombine == 'concat':
            final_out = self._group_conv(x_, self.concat_group)
        else:
            if x_off.shape[1] >1000:
                final_out = self._group_conv(x_, self.groups)
            else:
                final_out = self._group_conv(x_, new_group)
        return final_out

    def op_add(self, x_out, m):
        return (x_out + m)
    def op_mul(self, x_out, m):
        return (x_out * m)

    def op_cat(self, x_out, m, groups):
        """
        Usage: Concat by input joints ratio of the first layer, always keep the ratio = N_i : 1; N_i is joint number in the ith group.
        :return: Concat with other info and adjust the channel size
        """
        cat_m = []
        for i,group in enumerate(groups):
            indexes = group
            xs = []
            for index in indexes:
                xs.append(x_out[:,index[0]:index[1]])
            x_cat = torch.cat(xs,dim=1)
            cat_m.append(torch.cat([x_cat, m[i]], dim=1))
        return torch.cat(cat_m, dim=1)

    def _keep_ratio(self, inc_num, fix_seq, index, added_dim, by_ratio):
        """
        For concat by a certain ratio, you can change [joint_dim] to give various concat ratios.
        :param inc_num: input channel number of a group. type:torch.Tensor
        :param fix_seq: output index sequence of 1st layer, knowing the groups number.
        :return: a concatenated input channel number
        """
        ori_size = len(fix_seq[index])
        # add [x,y] dimension
        if by_ratio: # Add the dimension by ratio. e.g. 20%, 40%...
            out_num = int(inc_num * (ori_size + added_dim) / ori_size)
        else: # Add the dimension by discrete values
            out_num = added_dim + inc_num
        concat_size = out_num - inc_num
        return out_num, concat_size

    def _get_global_input(self, x):
        return x

    def _get_partial_input(self, x):
        """
        Usage: Get inputs as Group representation
        :param x: all 2d joints inputs, x.shape=[B, 34, T]
        :param out_seq: output index sequence of each layer
        :return: 1. x_self: Each group inputs; type: list
                2. x_other: Out of the group values; type: list
        """
        x_other = []
        x_self = []
        in_dim = []
        v = 0
        for i,group in enumerate(self.groups):
            indexes = group
            xs = []
            for index in indexes:
                xs.append(x[:,index[0]:index[1]])
            x_cat = torch.cat(xs,dim=1)
            x_self.append(x_cat)
            x_other.append(torch.cat([x[:,0:v],x[:,(v+x_cat.shape[1]):]],dim=1))
            in_dim.append(x_other[i].shape[1])
            v += x_cat.shape[1]
        return x_other, x

    def _split_fc(self, x, dtype):
        """
        Usage: Split channels into groups
        :param x: Input features
        :return: x1: each group features. type: list
                 x_cat: concatenate each group features. type:torch.Tensor
        """
        x1 = []
        for i,group in enumerate(self.groups):
            indexes = group
            xs = []
            for index in indexes:
                if len(index) == 1:
                    xs.append(x[:, index[0]:index[0]+1].type(dtype))
                else:
                    xs.append(x[:,index[0]:index[1]].type(dtype))
            x1.append(torch.cat(xs,dim=1)) #Each group features
        x_cat = torch.cat(x1, dim=1)
        return x1, x_cat

    def _group_conv(self, x, groups):
        """
        Usage: fully connection in a group
        :param x: features
        :param groups: depend on concat or not of different input size
        :return: final outputs after group conv.
        """
        outs = []
        ks = self.kernel_size
        for i, group in enumerate(groups):
            indexes = group
            xs = []
            for index in indexes:
                if len(index) == 1:
                    xs.append(x[:, index[0]:index[0]+1])
                else:
                    xs.append(x[:,index[0]:index[1]])
            x1 = torch.cat(xs,dim=1)
            x_out = self._reshape_x_offset(x1, ks)
            outs.append(self.group_conv[i](x_out))
        return torch.cat(outs, dim=1)

    def _reorganize_sequence(self, x, dtype):
        N = self.kernel_size
        # (b, N, w)
        p = self._get_p(x, dtype)
        # (b, w', N)
        p = p.contiguous().permute(0, 2, 1)
        p = torch.clamp(p[..., 0:N], 0, x.size(2) - 1)  # add
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_t = torch.clamp(q_lt[..., 0:N], 0, x.size(2) - 1).long()  # (b,w,N)
        q_b = torch.clamp(q_rb[..., 0:N], 0, x.size(2)).long()

        # bilinear kernel (b, w, N)
        g_t = 1 + (q_t[..., :N].type_as(p) - p[..., :N])
        g_b = 1 - (q_b[..., :N].type_as(p) - p[..., :N])
        # This operation is for p'value >= 4.0, and make sure g_t+g_b=1
        q_b = torch.clamp(q_rb[..., 0:N], 0, x.size(2) - 1).long()

        # (b, c, w, N)
        x_q_t = self._get_x_q(x, q_t, N)
        x_q_b = self._get_x_q(x, q_b, N)

        # (b, c, w, N) # Keep the same value of channels
        x_offset = g_t.unsqueeze(dim=1) * x_q_t + g_b.unsqueeze(dim=1) * x_q_b

        return x_offset

    def _get_p_n(self, N, dtype):
        if self.dilation > 1:
            p_n_x, p_n_y = np.meshgrid(0, range(0, (N - 1) * self.dilation + 1, self.dilation), indexing='ij')
        elif self.kernel_size > 1:
            p_n_x, p_n_y = np.meshgrid(0, range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                       indexing='ij')
        elif self.kernel_size == 1:
            p_n_y = np.array([-1.])
        # (N, 1)
        p_n = p_n_y.flatten()  # [ -1 0 1]
        p_n = np.reshape(p_n, (1, N, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(self, w, N, dtype):
        s = self.stride
        p = self.padding
        if self.dilation > 1:
            p_0_x, p_0_y = np.meshgrid(0, range(0, w - (N - 1) * self.dilation, 1), indexing='ij')
        elif self.kernel_size > 1:
            p_0_x, p_0_y = np.meshgrid(1, range((self.kernel_size - 1) // 2, w - (self.kernel_size - 1) // 2, s),
                                       indexing='ij')
        elif self.kernel_size == 1:
            p_0_x, p_0_y = np.meshgrid(1, range(1, w - (self.kernel_size - 1) // 2 + 1 + 2 * p, s), indexing='ij')

        p_0_y = p_0_y.flatten().reshape(1, 1, -1).repeat(N, axis=1)  # (1,N, (w-ks+2d)/stride))
        p_0 = Variable(torch.from_numpy(p_0_y).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, x, dtype):
        b, N, w = x.size(0), self.kernel_size, x.size(2)
        # (1,N,1)
        p_n = self._get_p_n(N, dtype)
        # (1, N, (w-ks+2d)/stride))
        p_0 = self._get_p_0(self, w, N, dtype)
        # Get final p
        p = p_0 + p_n
        p = p.repeat(b, 1, 1)

        return p

    def _get_x_q(self, x, q, N):
        b, w, _ = q.size()
        # padded_w = x.size(2)
        c = x.size(1)
        # (b, w, N)
        index = q[..., :N]
        # (b, c, w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1).contiguous().view(b, c, -1)
        # this size is exactly (b,c,w*N)
        x_offset = x.gather(dim=-1, index=index)
        x_offset = x_offset.contiguous().view(b, c, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, w, N = x_offset.size()
        x_offset = x_offset.contiguous().view(b, c, w * ks)

        return x_offset
