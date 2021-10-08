import numpy as np

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
#import h5py

mpii_metadata = {
    'layout_name': 'mpii',
    'num_joints': 16,
    'keypoints_symmetry': [
        [3, 4, 5, 13, 14, 15],
        [0, 1, 2, 10, 11, 12],
    ]
}

coco_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}

h36m_metadata = {
    'layout_name': 'h36m',
    'num_joints': 17,
    'keypoints_symmetry': [
        [4, 5, 6, 11, 12, 13],
        [1, 2, 3, 14, 15, 16],
    ]
}

humaneva15_metadata = {
    'layout_name': 'humaneva15',
    'num_joints': 15,
    'keypoints_symmetry': [
        [2, 3, 4, 8, 9, 10],
        [5, 6, 7, 11, 12, 13]
    ]
}

humaneva20_metadata = {
    'layout_name': 'humaneva20',
    'num_joints': 20,
    'keypoints_symmetry': [
        [3, 4, 5, 6, 11, 12, 13, 14],
        [7, 8, 9, 10, 15, 16, 17, 18]
    ]
}


def suggest_metadata(name):
    names = []
    for metadata in [mpii_metadata, coco_metadata, h36m_metadata, humaneva15_metadata, humaneva20_metadata]:
        if metadata['layout_name'] in name:
            return metadata
        names.append(metadata['layout_name'])
    raise KeyError('Cannot infer keypoint layout from name "{}". Tried {}.'.format(name, names))


def import_detectron_poses(path):
    # Latin1 encoding because Detectron runs on Python 2.7
    data = np.load(path, encoding='latin1')
    kp = data['keypoints']
    bb = data['boxes']
    results = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0:
            assert i > 0
            # Use last pose in case of detection failure
            results.append(results[-1])
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        keypoints = kp[i][1][best_match].T.copy()
        results.append(keypoints)
    results = np.array(results)
    return results[:, :, 4:6]  # Soft-argmax
    # return results[:, :, [0, 1, 3]] # Argmax + score


def import_cpn_poses(path):
    data = np.load(path)
    kp = data['keypoints']
    return kp[:, :, :2]


def import_sh_poses(path):
    with h5py.File(path) as hf:
        positions = hf['poses'].value
    return positions.astype('float32')


def suggest_pose_importer(name):
    if 'detectron' in name:
        return import_detectron_poses
    if 'cpn' in name:
        return import_cpn_poses
    if 'sh' in name:
        return import_sh_poses
    raise KeyError('Cannot infer keypoint format from name "{}". Tried detectron, cpn, sh.'.format(name))


def fetch(subjects, keypoints, dataset, downsample, action_filter=None, cam_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            action_split = action.split(' ')[0]
            if action_filter is not None:
                found = False
                # distinguish the actions:'Sitting' and 'SittingDown'
                for act in action_filter:
                    act = act.split(' ')[0]
                    if action_split == act:
                        found = True
                        break
                if not found:
                    continue
            poses_2d = keypoints[subject][action]
            index = np.random.randint(0,4)
            if cam_filter==[5]: #random camera index
                out_poses_2d.append(poses_2d[index])
                print('choose a camera index for each action:', index)

            elif cam_filter:
                for j in cam_filter: # Select by some camera viewpoints
                    out_poses_2d.append(poses_2d[j])

            else:
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                if cam_filter==[5]:
                    cam = cams[index]
                    if 'intrinsic' in cam:
                        use_params = {}
                        use_params['intrinsic'] = cam['intrinsic']
                        if 'normalization_params' in dataset[subject][action]:
                            use_params['normalization_params'] = \
                                dataset[subject][action]['normalization_params'][index]
                        out_camera_params.append(use_params)
                elif cam_filter:
                    for j in cam_filter:
                        for i, cam in enumerate(cams):
                            if j == i:
                                if 'intrinsic' in cam:
                                    use_params = {}
                                    use_params['intrinsic'] = cam['intrinsic']
                                    if 'normalization_params' in dataset[subject][action]:
                                        use_params['normalization_params'] = \
                                        dataset[subject][action]['normalization_params'][i]
                                    out_camera_params.append(use_params)
                else:
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            use_params = {}
                            use_params['intrinsic'] = cam['intrinsic']
                            if 'normalization_params' in dataset[subject][action]:
                                use_params['normalization_params'] = dataset[subject][action]['normalization_params'][i]
                            out_camera_params.append(use_params)

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                if cam_filter==[5]:
                    out_poses_3d.append(poses_3d[index])
                elif cam_filter :
                    for j in cam_filter:
                        out_poses_3d.append(poses_3d[j])
                else:
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d
