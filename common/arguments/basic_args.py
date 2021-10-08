# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training SRNet script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME',
                        help='2D detections to use', choices=['gt','cpn_ft_h36m_dbb'])
    parser.add_argument('--rand-seed', default=4321, type=int, metavar='N', help='random seeds')    
    ### Protocol settings
    # Differ from subjects (people), e.g. standard protocol 1 (mpjpe) & 2 (pa-mpjpe)#S5,S6,S7,S8
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST',
                        help='test subjects separated by comma')
    parser.add_argument('--subjects-full', default='S1,S5,S6,S7,S8,S9,S11', type=str, metavar='LIST',
                        help='All subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')

    # Differ from actions, e.g. cross-action validation protocol (one action for training, others for test)
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--use-action-split', default=False, help='Train one some actions, test on others')
    parser.add_argument('--train-action', default='Discussion', type=str, metavar='LIST',
                        help='action name for training')
    parser.add_argument('--test-action',
                        default='Greeting,Sitting,SittingDown,WalkTogether,Phoning,Posing,WalkDog,Walking,Purchases,Waiting,Directions,Smoking,Photo,Eating',
                        type=str, metavar='LIST', help='action name for test')
    parser.add_argument('--all-action',
                        default='Greeting,Sitting,SittingDown,WalkTogether,Phoning,Posing,WalkDog,Walking,Purchases,Waiting,Directions,Smoking,Photo,Eating,Discussion',
                        type=str, metavar='LIST', help='action name for test')
    parser.add_argument('--action_unlabeled', default='', type=str, metavar='LIST', help='action name for training')

    # Differ from camera settings, e.g. cross-camera validation
    parser.add_argument('--cam-test', default='', type=list, metavar='LIST',
                        help='test camera viewpoint, If None, use all cameras; If [5], choose four of them randomly',
                        choices=[0, 1, 2, 3, 5])
    parser.add_argument('--cam-train', default='', type=list, metavar='LIST',
                        help='train camera viewpoint,If None, use all cameras; If [5], choose four of them randomly',
                        choices=[0, 1, 2, 3, 5])
    
    #### The data to test:
    parser.add_argument('--three-dpw', default=False, help='Cross train/test on 3DPW testset')
    parser.add_argument('--use-hard-test', default=False,
                        help='For evaluation setting, using rarest N% test set in S9/S11')
    parser.add_argument('--use-mpi-test', default=False, help='Cross test on MPI-INF-3DHP')

    #### Data normalization
    parser.add_argument('--norm', choices=['base', 'proj', 'weak_proj', 'lcn'], type=str, help='way of data normalization', default='base')

    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory to store models')
    parser.add_argument('-bc', '--best-checkpoint', default='best_checkpoint', type=str, metavar='PATH',
                        help='best checkpoint directory to store the best models')
    parser.add_argument('--checkpoint-frequency', default=1, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-ft', '--finetune', default='', type=str, metavar='FILENAME',
                        help='checkpoint to finetune (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model arguments
    parser.add_argument('--model', default='srnet', type=str,  choices=['srnet', 'fc'],
                        help='the name of models which you train')
    parser.add_argument('-mn', '--model-name', default='sr_h36m_gt2d', type=str,
                        help='the name of models which you want to save')
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=60, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-tb', '--test-batch-size', default=100, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-arc', '--architecture', default='1,1,1', type=str, metavar='LAYERS',
                        help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N',
                        help='number of channels in convolution layers')

    parser.add_argument('-drop', '--dropout', default=0, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR',
                        help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('--conf', default=0, type=int, metavar='N',help='confidence score number')
    #### basic model settings, Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true',
                        help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true',
                        help='disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true',
                        help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')
    parser.add_argument('--root-log', default='log', type=str)

    parser.add_argument('--train-rotation', default=False,
                        help='Use random Y-axis rotation for training stage, please close train-flip augmentation!')
    parser.add_argument('--repeat-num', default=1, type=int, metavar='N', help='number of repeat rotation')

    # Temporal Pose settings
    parser.add_argument('--use-same-3d-input', default=False, help='input frame number is equal to output frame number')

    #### For smooth loss:
    parser.add_argument('--threshold', default=0.0004, type=float, metavar='LR',
                        help='The threshold of smooth loss to control the loss functions')
    parser.add_argument('--mi', default=0.1, type=float, metavar='LR', help='The pow of smooth loss')

    parser.add_argument('--scale', default=0.001, type=float, metavar='LR', help='')
    parser.add_argument('--rnum', default=0, type=int, metavar='LR', help='')

    # Render function Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')

    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    ### SRNet arguments
    ### split features
    parser.add_argument('-mo', '--modulation', default=False,
                        help='Use modulation module for temporal mask self-attention multiply the whole channel [all joint inputs]')
    parser.add_argument('--group-modulation', default=False,
                        help='Use modulation module for multiply each group as local attention [group-wise joint inputs]')
    parser.add_argument('--split-modulation', default=True,
                        help='Use modulation module multiply each group as global attention [except local joint inputs]')
    parser.add_argument('--channelwise', default=False,
                        help='Use modulation module multiply each group with channel-wise attention [all joint inputs]')
    ### recombine feature source

    parser.add_argument('--split', choices=['all', 'others', 'none'], type=str,
                        help='way of feature split', default='others')

    ### recombine operators
    parser.add_argument('--recombine', choices=['multiply', 'add', 'concat'], type=str,
                        help='way of low-dimension global features and local feature recombination', default='multiply')
    parser.add_argument('--mean-func', default=False, help='Use mean function [other joint inputs]')
    parser.add_argument('--repeat-concat', default=False,
                        help='Use [repeat number] concatenate for fusion group feature and other joint features, if True, --concat must be True')

    parser.add_argument('--ups-mean', default=False, help='Use flexible mean function [other joint inputs]')

    # Group number
    parser.add_argument('--group', type=int, default=5, metavar='N', help='Guide the group strategies',choices=[1,2,3,5])

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args
