# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim as optim
import os
import sys
import errno
import random
import numpy as np

from common.arguments.basic_args import parse_args

args = parse_args()

from common.dataset.pre_process.utils import fetch
from common.dataset.pre_process.hm36 import load_data, prepare_dataset, load_2d_data, prepare_2d_data, normalization, random_rotate, load_hard_test
from common.dataset.pre_process.get_3dpw import load_3dpw
from common.dataset.pre_process.get_mpi_inf import load_mpi_test
from common.transformation.aug_rotate import rotate

from common.common_pytorch.loss.loss_family import *

from common.dataset.data_generators import ChunkedGenerator, UnchunkedGenerator
from common.common_pytorch.experiment.train import train
from common.common_pytorch.experiment.inference import infer_function, evaluate, eval_mpi_test, eval_hard_test, val
from common.common_pytorch.experiment.tools import print_result, save_model, check_rootfolder, count_params
from common.visualization.show_video import render_video, plot_log

from time import time, asctime, localtime


print(args)

print('Now time is:', asctime(localtime(time())))

torch.manual_seed(args.rand_seed)  # reproducible
np.random.seed(args.rand_seed)
torch.cuda.manual_seed_all(args.rand_seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True


try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# Load & prepare data, you can modify the path.
dataset_root = 'data/'

print('Loading dataset...')
dataset = load_data(dataset_root, args.dataset, args.keypoints)

print('Preparing data...')
prepare_dataset(dataset)
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
print('Loading 2D detections...')
keypoints, keypoints_metadata, kps_left, kps_right = load_2d_data(dataset_root, args.dataset, args.keypoints)
print('Preparing 2D data...')
prepare_2d_data(keypoints, dataset)

# subjects
subjects= args.subjects_full.split(',')
subjects_train = args.subjects_train.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
    action_test = args.test_action.split(',')

else:
    subjects_test = [args.viz_subject]
    action_test = [args.viz_action]
    print('For render visualization, use:',subjects_test,action_test)

# actions
action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)
all_action = args.all_action.split(',')
action_train = args.train_action.split(',')

# cameras
cam_test = args.cam_test
cam_train = args.cam_train

# Use random rotation
if args.train_rotation:
    random_rotate(dataset, keypoints, subjects_train, action_filter, cam_train)
    normalization(dataset, keypoints, subjects_test, action_filter, cam_test, args.norm)
else: # Normalize all data
    normalization(dataset, keypoints, subjects, action_filter, None, args.norm)


# Use 3dpw training data
if args.three_dpw:
    print('Loading 3dpw data:')
    poses_valid, poses_valid_2d, cameras_valid = load_3dpw('test', args.norm)

else:
    # Fetch camera/ pose 2d/ pose 3d input data
    if args.use_action_split:
        cameras_valid, poses_valid, poses_valid_2d = fetch(subjects, keypoints, dataset, args.downsample, all_action,
                                                           cam_test)
        print('Action Protocol: Use those subjects for test: ', subjects)
        print('Action for test are: ', action_test)
    # use subject protocol
    else:
        cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, keypoints, dataset, args.downsample,
                                                           action_filter, cam_test)
        print('Standard Protocol: Use those subjects for test: ', subjects_test)

filter_widths = [int(x) for x in args.architecture.split(',')]

# Choose a model
if args.model == 'srnet':
    from common.common_pytorch.model.srnet import TemporalModel, TemporalModelOptimized1f
#    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
#                                     dataset.skeleton().num_joints(),
#                                     filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
#                                     channels=args.channels)
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                               dataset.skeleton().num_joints(),
                                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                               channels=args.channels)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths = filter_widths, causal = args.causal, dropout = args.dropout, channels = args.channels, dense = args.dense)
    #model_pos = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
    #                         filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)

else:
    from common.common_pytorch.model.fc_baseline import TemporalModel, TemporalModelOptimized1f
    #model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
    #                                 dataset.skeleton().num_joints(),
    #                                 filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
    #                                 channels=args.channels)
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                               dataset.skeleton().num_joints(),
                                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                               channels=args.channels)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths = filter_widths, causal = args.causal, dropout = args.dropout, channels = args.channels, dense = args.dense)
    #model_pos = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
    #                         filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)


receptive_field = model_pos.receptive_field()

for k, v in model_pos_train.state_dict().items():
    print(k,v.shape)

print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos_train.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))

check_rootfolder()
print('Parameters of need to grad is:',count_params(model_pos_train) / 1000000.0)


if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    # print('checkpoint',checkpoint['model_pos'])
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    for k,v in model_pos_train.state_dict().items():
        print(k,v.shape)

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                   pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                   kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                   joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

if not args.evaluate:

    if args.three_dpw:
        print('Loading 3dpw data:')
        poses_train, poses_train_2d, train_cam_in = load_3dpw('train', args.norm)
        poses_train_eval, poses_train_eval_2d, cameras_train_eval = load_3dpw('validation', args.norm)

    else:
        if args.use_action_split:
            train_cam_in, poses_train, poses_train_2d = fetch(subjects, keypoints, dataset, args.downsample,
                                                              action_train, cam_train)
            print('Action Protocol: Use those subjects for training: ', subjects)
            print('Use those actions for training: ', action_train)
        else:
            train_cam_in, poses_train, poses_train_2d = fetch(subjects_train, keypoints, dataset, args.downsample,
                                                              action_filter, cam_train)
            print('Standard Protocol: Use those subjects for training with the whole actions: ', subjects_train)

    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0

    train_generator = ChunkedGenerator(args.batch_size // args.stride, train_cam_in, poses_train, poses_train_2d,
                                       args.stride, pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right, train_rotation=args.train_rotation)
    if args.three_dpw:
        train_generator_eval = ChunkedGenerator(args.batch_size // args.stride, cameras_train_eval, poses_train_eval, poses_train_eval_2d,args.stride,
                                                pad=pad, causal_shift=causal_shift, augment=args.data_augmentation, kps_left=kps_left, kps_right=kps_right,
                                                joints_left=joints_left,joints_right=joints_right, train_rotation=False)
    else:
        train_generator_eval = ChunkedGenerator(args.batch_size // args.stride, train_cam_in, poses_train, poses_train_2d, args.stride,
                                                pad=pad, causal_shift=causal_shift, augment=args.data_augmentation, kps_left=kps_left, kps_right=kps_right,
                                                joints_left=joints_left, joints_right=joints_right, train_rotation=False)
    print('INFO: Training on {} frames'.format(train_generator.num_frames()),'Validation {} frames:'.format(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']
        #lr = 5e-4

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    i_train = 0
    i_eval = 0
    Best_error = 999
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train_l2, i_train = train(train_generator, model_pos_train, dataset, optimizer, epoch, args.norm, i_train, args.bone_length_term)
        losses_3d_train.append(epoch_loss_3d_train_l2)
        print('train L2 mean per joint error:', losses_3d_train[-1])

        # End-of-epoch evaluation
        if not args.no_eval:
            model_pos.load_state_dict(model_pos_train.state_dict())

            # Evaluate on test set
            print('Evaluating...')
            if args.use_action_split:
                print('Test on the Action Protocol:')
                test_3d_error = infer_function(subjects, keypoints, dataset, args.norm, action_test, model_pos, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, cam_test)
                #losses_3d_valid.append(1) #you can skip test for each epoch by using it

            else:
                print('Test on Standard Subject Protocol:')
                e1, e2, e3, ev, em, ea = evaluate(test_generator, model_pos, joints_left, joints_right, args.norm, action=None, return_predictions=False)
                losses_3d_valid.append(e1)

            # Validation:
            eval_3d_error = val(train_generator_eval, model_pos, args.norm)
            losses_3d_train_eval.append(eval_3d_error)

        elapsed = (time() - start_time) / 60

        print_result(epoch, elapsed, lr, losses_3d_train, losses_3d_train_eval, losses_3d_valid)

        # Decay learning rate exponentially
        lr_decay = args.lr_decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        initial_momentum = 0.1
        final_momentum = 0.001
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if Best_error > losses_3d_valid[-1]:
            Best_error = losses_3d_valid[-1]
            Best_model = True
        else:
            Best_model = False     
        out_error = save_model(losses_3d_train, losses_3d_train_eval, losses_3d_valid, train_generator, optimizer, model_pos_train,
                   epoch, lr, Best_model=Best_model)


        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            plot_log(losses_3d_train,losses_3d_train_eval,losses_3d_valid,epoch,args.checkpoint)


if args.render:
    print('Rendering video outputs..')
    render_video(keypoints, dataset, model_pos, keypoints_metadata, pad, causal_shift, kps_left, kps_right, joints_left, joints_right)

else:
    print('Final Evaluating...')
    if args.use_action_split:
        print('Final Evaluating on Cross-Action Protocol:', subjects, all_action)
        infer_function(subjects, keypoints, dataset, args.norm, all_action, model_pos, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, cam_test)
    elif args.use_hard_test:
        print('Final Evaluating by choosing hard/rare pose in the test set:')
        file_path = 'data/unfrequent_0.2_test_gt.npz'
        pose_3d, pose_2d = load_hard_test(file_path, eval_num=2200)
        eval_hard_test(pose_3d, pose_2d, model_pos, args.norm, pad, causal_shift, kps_left, kps_right, joints_left, joints_right)
    elif args.use_mpi_test:
        print('Final Evaluating on MPI-INF-3DHP for cross dataset validation:')
        # You can choose seq from [0, 6] with six scenes, and seq=7 [change file_path='mpi_inf_3dhp_skip_prob.npz' of the whole valid testset.
        seq = 0
        #file_path = 'data/data/mpi_inf_3dhp_seq_prob.npz'
        file_path = 'data/data/mpi_inf_3dhp_small.npz'
        pose_3d, pose_2d, info = load_mpi_test(file_path, seq, args.norm)
        eval_mpi_test([pose_3d], [pose_2d], [info], model_pos, args.norm, pad, causal_shift, kps_left, kps_right, joints_left, joints_right)
    elif args.three_dpw:
        # load 3dpw data:
        print('evaluating on 3dpw...')
        evaluate(test_generator, model_pos, joints_left, joints_right, args.norm, action=None, return_predictions=False)
    else:
        print('Final Evaluating on Subject Protocol:', subjects_test, action_filter)
        infer_function(subjects_test, keypoints, dataset, args.norm, action_filter, model_pos, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, cam_test)

    print('Finish!')
