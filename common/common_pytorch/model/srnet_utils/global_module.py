import torch
from torch import nn
import math

class Generate_Info(nn.Module):
    def __init__(self, inc, outc, out_seq, kernel_size=3, padding=0, dilation=1, stride=1, modulation=False,
                 group_modulation=False, split_modulation=False, channelwise = False, recombine = 'multiply', repeat_concat=False,
                 mean_dim=None, global_info=False, bias=None):
        super(Generate_Info, self).__init__()
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
        self.cat_num = mean_dim #List
        # Experiment Setting
        self.modulation = modulation
        self.group_modulation = group_modulation
        self.split_modulation = split_modulation
        self.channelwise = channelwise
        self.global_info = global_info

        # Operators of combination info.
        self.recombine = recombine

        self.repeat_concat = repeat_concat
        in_group_accmulate = [0]
        self.groups = []
        for i in range(self.in_group_num):
            in_group_accmulate.append(sum(self.in_channel_group[:i + 1]))
        for index,i in enumerate(out_seq):
            indexes = (list(map(lambda x: [in_group_accmulate[x], in_group_accmulate[x + 1]], i)))
            self.groups.append(indexes)

        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        if self.global_info == 'all':
            m = self._forward(x)
        else:
            m = self._part_forward(x)
        return m

class With_all_joints(Generate_Info):
    def __init__(self, *args, **kwargs):
        super(With_all_joints, self).__init__(*args, **kwargs)
        """
        Usage: Inputs are all joints information, which can be used to generate three different learnable values:
                1.Global modulation; 2.Group-wise modulation. 3.Channel-wise modulation. 
                For each of the above format, it can be combined with group convolution by [Addition] and [Multiply].
        """
        if self.modulation:
            print('Use overall global-joint modulation for the all groups')
            self.m_conv = nn.Conv1d(self.in_channel, out_channels=self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

        if self.group_modulation:
            print('Use [group-wise] modulation for each group')
            group_mo = []
            for index, i in enumerate(self.out_seq):
                in_ch = sum(map(lambda x: self.in_channel_group[x], i))
                group_mo.append(nn.Conv1d(in_ch, self.kernel_size, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                nn.init.constant_(group_mo[index].weight, 0)
                group_mo[index].register_backward_hook(self._set_lr)
            self.group_mo = nn.ModuleList(group_mo)

        if self.split_modulation:
            m_conv = []
            print('Use Split [global-joint] modulation for each group')
            for i in range(len(self.out_seq)):
                if self.recombine == 'concat':
                    if self.repeat_concat:
                        print('Use [Repeated values for concat]')
                        m_conv.append(nn.Conv1d(self.in_channel, out_channels=self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                    else:
                        print('Use [different values for concat]')
                        m_conv.append(nn.Conv1d(self.in_channel, out_channels=self.cat_num[i]*self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                else:
                    m_conv.append(nn.Conv1d(self.in_channel, out_channels=self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                nn.init.constant_(m_conv[i].weight, 0)
                m_conv[i].register_backward_hook(self._set_lr)
            self.m_conv = nn.ModuleList(m_conv)

        in_ch_sum = 0
        for index, i in enumerate(self.out_seq):
            in_ch_sum += sum(map(lambda x:self.in_channel_group[x],i))
        if self.channelwise:
            print('Use overall global-joint channelwise modulation for the all groups')
            self.m_conv = nn.Conv1d(self.in_channel, out_channels=in_ch_sum*self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    def _forward(self, x):
        x_ori, x_full = x[0], x[1]
        if self.recombine == 'add':
            m = self._get_learnable_shift(x_ori, x_full)
        elif self.recombine == 'multiply':
            m = self._get_learnable_scale(x_ori, x_full)
        elif self.recombine == 'concat':
            m = self._get_concat(x_ori, self.cat_num)
        else:
            print('Error: There is no gate! Please choose [addition] or [multiply] operator')
        return m

    def _get_concat(self, x, cat_num):
        if self.split_modulation:
            # Use all joint info to generate some dimension for each group
            m = []
            for i in range(self.out_group_num):
                m1 = self.relu(self.m_conv[i](x))
                m1 = reshape_with_kernel(m1, self.kernel_size)
                if self.repeat_concat:
                    m1 = torch.cat([m1 for _ in range(cat_num[i])], dim=1)
                m.append(m1)
        return m

    def _get_learnable_scale(self, x, x_full):
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
            m = m.contiguous().permute(0, 2, 1).unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_full.size(1))], dim=1)

        elif self.channelwise:
            m1 = torch.sigmoid(self.m_conv(x))
            m = reshape_with_kernel(m1, self.kernel_size)
        else:
            m_out = []
            for i, group in enumerate(self.groups):
                indexes = group
                xs = []
                for index in indexes:
                    xs.append(x[:, index[0]:index[1]])
                x_out = torch.cat(xs, dim=1)
                if self.split_modulation:
                    m1 = torch.sigmoid(self.m_conv[i](x))
                elif self.group_modulation:
                    m1 = torch.sigmoid(self.group_mo[i](x_out))
                m1 = m1.contiguous().permute(0, 2, 1).unsqueeze(dim=1)
                m1 = torch.cat([m1 for _ in range(x_out.size(1))], dim=1)
                m_out.append(m1)
            m = torch.cat(m_out, dim=1)
        return m

    def _get_learnable_shift(self, x, x_full):
        if self.modulation:
            m = self.relu(self.m_conv(x))
            m = m.contiguous().permute(0, 2, 1).unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_full.size(1))], dim=1)

        elif self.channelwise:
            m1 = self.relu(self.m_conv(x))
            m = reshape_with_kernel(m1, self.kernel_size)
        else:
            m_out = []
            for i, group in enumerate(self.groups):
                indexes = group
                xs = []
                for index in indexes:
                    xs.append(x[:, index[0]:index[1]])
                x_out = torch.cat(xs, dim=1)
                if self.split_modulation:
                    m1 = self.relu(self.m_conv[i](x))
                elif self.group_modulation:
                    m1 = self.relu(self.group_mo[i](x_out))
                m1 = m1.contiguous().permute(0,2,1).unsqueeze(dim=1)
                m1 = torch.cat([m1 for _ in range(x_out.size(1))], dim=1)
                m_out.append(m1)
            m = torch.cat(m_out, dim=1)
        return m

class With_other_joints(Generate_Info):
    def __init__(self, inc, outc, out_seq, kernel_size, padding, dilation, stride, split_modulation,
                 recombine, repeat_concat, in_c, mean_func, mean_dim, ups_mean):
        """
        Usage: Inputs from [Out of the group] Joints to get their information that can divided into two formats:
                1.Learnable local modulation. 2. Manual function,e.g. Mean opertation
                For each of the above format, it can be combined with group convolution by [Addition], [Multiply] and [Concat].
        :param args: from parents args
        :param kwargs: from parents kwargs
        :param in_channel: different group has different in channel number.[Sum up: Other joints channel number]
        :param mean_dim: The output number of the mean_function. e.g. 1, group number or satisfing the ratio for addition/multipy/concat
        """
        super(With_other_joints, self).__init__(inc, outc, out_seq, kernel_size, padding, dilation, stride, split_modulation, recombine, repeat_concat)

        self.split_modulation = split_modulation
        self.mean_func = mean_func
        self.cat_num = mean_dim

        self.in_channel = in_c
        self.ups_mean =ups_mean
        # Experiment operators
        self.recombine = recombine
        self.repeat_concat = repeat_concat
        if self.recombine == 'multiply':
            layers_bn = []
            for i in range(len(self.out_seq)):
                layers_bn.append(nn.BatchNorm1d(self.kernel_size, momentum=0.1))
            self.layers_bn = nn.ModuleList(layers_bn)

        if self.split_modulation:
            m_conv = []
            print('Use Split [Other-joint] modulation for each group')
            for i in range(len(self.out_seq)):
                if self.recombine == 'concat':
                    if self.repeat_concat:
                        print('Use [Repeated values for concat]')
                        m_conv.append(nn.Conv1d(self.in_channel[i], out_channels=self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                    else:
                        print('Use [Different values for concat]')
                        m_conv.append(nn.Conv1d(self.in_channel[i], out_channels=self.cat_num[i]*self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                else:
                    m_conv.append(nn.Conv1d(self.in_channel[i], out_channels=self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                nn.init.constant_(m_conv[i].weight, 0)
                m_conv[i].register_backward_hook(self._set_lr)
            self.m_conv = nn.ModuleList(m_conv)
        if self.mean_func:
            m_conv = []
            print('Use Split [Other-joint] manual mean values for each group')
            for i in range(len(self.out_seq)):
                if self.recombine == 'concat':
                    if self.ups_mean:
                        print('Use [upsampling mean value] for concat')
                        m_conv.append(nn.Conv1d(1, out_channels=self.cat_num[i]*self.kernel_size, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride))
                        nn.init.constant_(m_conv[i].weight, 0)
                        m_conv[i].register_backward_hook(self._set_lr)
            self.m_conv = nn.ModuleList(m_conv)

    def _part_forward(self, x):
        x_other, x_self = x[0],x[1]
        assert len(x_other) == len(self.out_seq) == len(x_self)
        if self.recombine == 'add':
            m = self._get_learnable_shift(x_other, x_self)
        elif self.recombine == 'multiply':
            m = self._get_learnable_scale(x_other, x_self)
        elif self.recombine == 'concat':
            m = self._get_value(x_other, x_self, self.cat_num)
        else:
            print('Error: There is no gate! Please choose [addition] or [multiply] or [concat] operator')
        return m

    def _get_learnable_shift(self, x, x_self):
        if self.split_modulation:
            m_out = []
            for i in range(self.out_group_num):
                m1 = self.relu(self.m_conv[i](x[i]))
                m1 = m1.contiguous().permute(0, 2, 1).unsqueeze(dim=1)
                m1 = torch.cat([m1 for _ in range(x_self[i].size(1))], dim=1)
                m_out.append(m1)
        elif self.mean_func:
            m_out = self._mean_func(x, self.cat_num, x_self)
        m = torch.cat(m_out, dim=1)
        return m

    def _get_learnable_scale(self, x, x_self):
        if self.split_modulation:
            m_out = []
            for i in range(self.out_group_num):
                m1 = torch.sigmoid(self.layers_bn[i](self.m_conv[i](x[i])))
                m1 = m1.contiguous().permute(0, 2, 1).unsqueeze(dim=1)
                m1 = torch.cat([m1 for _ in range(x_self[i].size(1))], dim=1)
                m_out.append(m1)
        elif self.mean_func:
            m_out = self._mean_func(x, self.cat_num, x_self)
        m = torch.cat(m_out, dim=1)
        return m

    def _get_value(self, x, x_self, cat_num):
        # Get by learnable way
        if self.split_modulation:
            m = []
            for i in range(self.out_group_num):
                m1 = self.relu(self.m_conv[i](x[i]))
                m1 = reshape_with_kernel(m1, self.kernel_size)
                if self.repeat_concat:
                    m1 = torch.cat([m1 for _ in range(cat_num[i])], dim=1)
                m.append(m1)
        elif self.mean_func:
            m = self._mean_func(x, self.cat_num, x_self)
        return m

    def _mean_func(self, x, cat_num, x_self):
        """
        Get mean value of each group from all other joints
        :param x: a list with [other joints] of the group
        :param cat_num: the repeat channel size
        :param x_self: a list with [itself joints] of the group
        :return: the processed mean value
        """
        out_mean = []
        for i, x_g in enumerate(x):
            m_mean = torch.mean(x_g, dim=1, keepdim=True)
            if self.ups_mean: # Upsample to get more variable values from mean value
                m1 = self.m_conv[i](m_mean)
                m1 = reshape_with_kernel(m1, self.kernel_size)
            elif self.repeat_concat:
                m1 = torch.cat([m_mean for _ in range(cat_num[i])], dim=1)
            elif self.recombine == 'add' or 'multiply':
                m1 = torch.cat([m_mean for _ in range(x_self[i].size(1))], dim=1)
            out_mean.append(m1)
        return out_mean

def reshape_with_kernel(input, kernel_size):
    B, C, K = input.shape[0], int(input.shape[1] / kernel_size), input.shape[2]
    m1 = input.unsqueeze(dim=-1)
    m = m1.view(B, C, K, kernel_size)
    return m

