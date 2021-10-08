import torch
import torch.optim as optim
import numpy as np
import os

from common.common_pytorch.loss.loss_family import *
from common.dataset.post_process.process3d import post_process3d
from tensorboardX import SummaryWriter
from common.arguments.basic_args import parse_args
args = parse_args()
tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.model_name))

def train(train_generator, model_pos_train, dataset, optimizer, epoch, norm, i_train=0, bone_length_term=False):
    N = 0

    epoch_loss_3d_train_l1 = 0
    epoch_loss_left_right = 0
    epoch_loss_3d_train_l2 = 0
    model_pos_train.train()

    # Regular supervised scenario
    i = 0

    for use_params, batch_3d, batch_2d in train_generator.next_epoch():
        if norm != 'base':
            normalize_param = use_params['normalization_params']
            cam_intri = use_params['intrinsic']
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))  # torch.Size([1024,1,17,3])
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))  # torch.Size([1024,27,17,2])
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
            if norm != 'base':
                cam = torch.from_numpy(cam_intri.astype('float32')).cuda()  # torch.Size([1024,9])
                normalize_param = torch.from_numpy(normalize_param.astype('float32')).cuda()

        optimizer.zero_grad()
        # Train model
        predicted_3d_pos = model_pos_train(inputs_2d)
        if norm == 'base':
            inputs_3d[:, :, 0] = 0
        # Calculate L1 Loss

        loss_3d_pos_l1 = L1_loss(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_train_l1 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos_l1.item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        # denorm 3d pose for proposed normalization
        if norm != 'base':
            predicted_3d_pos, inputs_3d = post_process3d(predicted_3d_pos, inputs_3d, cam, normalize_param, norm)
        # Calculate L2 error with denormed 3d pose in meters unit.
        loss_l2 = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_train_l2 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_l2.item()

        print('each joint l1 loss', loss_3d_pos_l1.item( ) *1000 ,'Total epoch l1 loss average', epoch_loss_3d_train_l1/ N* 1000)

        # Bone length term to enforce kinematic constraints
        if bone_length_term: #not used by default
            if epoch > 100:
                left = [4, 5, 6, 11, 12, 13]
                right = [1, 2, 3, 14, 15, 16]
                bone_lengths_lift = []
                bone_lengths_right = []
                each_bone_error = []
                left_right_error = 0
                # error = [0.001, 0.0018, 0.0008, 0.0019, 0.0043, 0.0011]
                for i in left:
                    dists_l = predicted_3d_pos[:, :, i, :] - predicted_3d_pos[:, :, dataset.skeleton().parents()[i], :]
                    bone_lengths_lift.append(torch.mean(torch.norm(dists_l, dim=-1), dim=1))
                for i in right:
                    dists_r = predicted_3d_pos[:, :, i, :] - predicted_3d_pos[:, :, dataset.skeleton().parents()[i], :]
                    bone_lengths_right.append(torch.mean(torch.norm(dists_r, dim=-1), dim=1))
                for i in range(len(left)):
                    left_right_error += torch.abs(
                        torch.abs(bone_lengths_right[i] - bone_lengths_lift[i]))
                    each_bone_error.append(torch.mean(torch.abs(bone_lengths_right[i] - bone_lengths_lift[i])))
                    # print('each bone error', each_bone_error[-1] * 1000)
                left_right_err_mean = torch.mean(left_right_error)

                epoch_loss_left_right += inputs_3d.shape[0] * inputs_3d.shape[1] * left_right_err_mean.item()
                print('Each epoch left right error average', (epoch_loss_left_right / N) * 1000)
            else:
                left_right_err_mean = 0
        else:
            left_right_err_mean = 0
        i_train += 1
        tf_writer.add_scalar('loss/training part', (epoch_loss_3d_train_l1 / N) * 1000, i_train)

        # loss_total = loss_3d_pos_l1 + 0.1*left_right_err_mean
        loss_total = loss_3d_pos_l1
        loss_total.backward()
        optimizer.step()
    return epoch_loss_3d_train_l2/N*1000, i_train
