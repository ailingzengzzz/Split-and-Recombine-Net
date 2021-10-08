# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np

from common.arguments.basic_args import parse_args
from common.transformation.aug_rotate import rotate, process_2d, process_3d

args = parse_args()


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    use-params -- list with two dicts in it: 1.normalization params; 2.camera intrinics
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses with absolute value in the camera coodinate system, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints with pixel value in the image coordinate, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, batch_size, use_params, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=args.rand_seed,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, train_rotation=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert use_params is None or len(use_params) == len(poses_2d)

        if train_rotation:
            lens = len(poses_2d)
            pose2d_rot = []
            pose3d_rot = []
            print('use rotation augmentation')
            for i in range(lens):
                p_2d, p_3d = rotate(poses_2d[i][:, np.newaxis], poses_3d[i][:, np.newaxis],
                                    np.repeat(use_params[i]['intrinsic'][np.newaxis, :], poses_3d[i].shape[0], axis=0),
                                    args.repeat_num)
                pose2d_rot.append(p_2d.squeeze())
                pose3d_rot.append(p_3d.squeeze())
            poses_2d = pose2d_rot.copy()
            poses_3d = pose3d_rot.copy()
        else:  # For eval
            if poses_2d[0].shape[-2] == 18:
                pose2d = []
                pose3d = []
                print('do normalization in generator')

                for i in range(len(poses_2d)):
                    pose2d.append(process_2d(poses_2d[i][:, np.newaxis]).squeeze())
                    pose3d.append(process_3d(poses_3d[i][:, np.newaxis]).squeeze())
                poses_2d = pose2d.copy()
                poses_3d = pose3d.copy()

        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_2d[i].shape[0]
            # change poses_2d[i].shape[0] to poses_3d[i].shape[0], since CPN 2d detector have more data than 3d.
            n_chunks = (poses_3d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_3d[i].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)
        # Initialize buffers
        if use_params is not None:
            self.batch_cam = np.empty((batch_size, use_params[0]['intrinsic'].shape[-1]))
            if 'normalization_params' in use_params[0]:
                self.norm_params = np.empty(
                    (batch_size, chunk_length, chunk_length, use_params[0]['normalization_params'].shape[-1]))
        if poses_3d is not None:
            if args.use_same_3d_input:
                self.batch_3d = np.empty(
                    (batch_size, chunk_length + 2 * pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
            else:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.use_params = use_params
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    # Chunk 2d pose
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                                  'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]
                    if flip:
                        # Flip 2D joints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                                                                              self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        if args.use_same_3d_input:
                            start_3d = start_2d
                            end_3d = end_2d
                        # Chunk 3d pose
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d],
                                                      ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = self.batch_3d[i, :,
                                                                                        self.joints_right + self.joints_left]
                    # Cameras
                    if self.use_params is not None:
                        self.batch_cam[i] = self.use_params[seq_i]['intrinsic']
                        # if flip:
                        #     # Flip horizontal distortion coefficients
                        #     self.batch_cam[i, 2] *= -1
                        #     self.batch_cam[i, 7] *= -1
                        if 'normalization_params' in self.use_params[seq_i]:
                            # Chunk normalization parameters
                            seq_params = self.use_params[seq_i]['normalization_params']  # [T,1,5]
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, seq_params.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d
                            if pad_left_3d != 0 or pad_right_3d != 0:
                                self.norm_params[i] = np.pad(seq_params[low_3d:high_3d],
                                                             ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                            else:
                                self.norm_params[i] = seq_params[low_3d:high_3d]  # [1,5]
                            if flip:
                                # Flip 3D joints
                                self.norm_params[i, :, :, 2] = 2 * self.batch_cam[
                                    i, np.newaxis, np.newaxis, 2] - self.norm_params[i, :, :, 2]  # Flip offset_x
                    use_params = {}
                    if self.use_params is not None:
                        if 'normalization_params' in self.use_params[0]:
                            use_params['normalization_params'] = self.norm_params[:len(chunks)]
                        use_params['intrinsic'] = self.batch_cam[:len(chunks)]
                        self.use_param = use_params

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.use_params is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.use_params is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.use_param, None, self.batch_2d[:len(chunks)]
                else:
                    yield self.use_param, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False


class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    use-params -- list with two dicts in it: 1.normalization params; 2.camera intrinics
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses with absolute value in the camera coordinate system, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints with pixel value in the image coordinate, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, use_params, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert use_params is None or len(use_params) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if use_params is None else use_params
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam['intrinsic'], axis=0)  # [1,9]
            if seq_cam:
                if 'normalization_params' in seq_cam:
                    batch_norm_param = None if seq_cam is None else np.expand_dims(seq_cam['normalization_params'],
                                                                                   axis=0)  # [1,T,1,5]
            if args.use_same_3d_input:
                batch_2d = np.expand_dims(seq_2d, axis=0)
            else:
                batch_2d = np.expand_dims(
                    np.pad(seq_2d, ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                           'edge'), axis=0)
            # batch_3d = None if seq_3d is None else np.expand_dims(np.pad(seq_3d,((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),'edge'), axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)

            if self.augment:
                # Append flipped version
                if seq_cam:
                    if batch_cam is not None:
                        batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    if 'normalization_params' in seq_cam:
                        batch_norm_param = np.concatenate((batch_norm_param, batch_norm_param), axis=0)
                        batch_norm_param[1, :, :, 2] = 2 * batch_cam[0, 2] - batch_norm_param[0, :, :, 2]  # offset_x

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :,
                                                                           self.joints_right + self.joints_left]
                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            use_params = {}
            if seq_cam:
                if 'normalization_params' in seq_cam:
                    use_params['normalization_params'] = batch_norm_param
                use_params['intrinsic'] = batch_cam

            yield use_params, batch_3d, batch_2d
