import numpy as np
from common.transformation.cam_utils import *
from common.dataset.pre_process.norm_data import norm_to_pixel
from common.dataset.h36m_dataset import Human36mDataset

def load_data(dataset_root, dataset_name, kpt_name):
    dataset_path = dataset_root + 'data_3d_' + dataset_name + '.npz'
    if dataset_name == 'h36m':
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')
    return dataset


# dataset is modified.
def prepare_dataset(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d


def load_2d_data(dataset_root, dataset_name, kpt_name):
    keypoints = np.load(dataset_root + 'data_2d_' + dataset_name + '_' + kpt_name + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    keypoints = keypoints['positions_2d'].item()
    use_smooth_2d = False
    if use_smooth_2d:
        print('use smooth 2d pose:')
        smooth_2d = np.load('common/dataset/pre_process/smooth_cpn_ft_81_all.npz', allow_pickle=True)
        keypoints = smooth_2d['positions_2d'].item()

    return keypoints, keypoints_metadata, kps_left, kps_right

def load_hard_test(file_path, eval_num):
    # Used for load hard test set in our evaluation
    hard_pose = np.load(file_path, allow_pickle=True)
    pose_3d = hard_pose['pose_3d'] # Have normalized, type:list
    pose_2d = hard_pose['pose_2d'] # Have normalized; type:list
    if len(pose_3d)==1:
        num = pose_3d[0].shape[0]
        size = num // eval_num
        t_3d = []
        t_2d = []
        t = 0
        for j in range(size):
            t_3d.append(pose_3d[0][t:t + eval_num])
            t_2d.append(pose_2d[0][t:t + eval_num])
            t += eval_num
        t_3d.append(pose_3d[0][t:])
        t_2d.append(pose_2d[0][t:])
        return t_3d, t_2d
    else:
        return pose_3d, pose_2d

# keypoints are midified. dataset remains.
def prepare_2d_data(keypoints, dataset):
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

def random_rotate(dataset, keypoints, subjects, action_filter=None, cam_filter=None):
    print('Random rotate 3d pose around Y axis, output rotated 2d and 3d poses ')
    for subject in subjects:
        for action in dataset[subject].keys():
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
            cams = dataset.cameras()[subject]
            poses_3d = dataset[subject][action]['positions_3d']
            poses_2d = keypoints[subject][action]
            assert len(poses_3d) == len(cams), 'Camera count mismatch'
            assert len(cams) == len(poses_2d), 'Camera count mismatch'

            if cam_filter:
                for i in cam_filter: # Select by some camera viewpoints
                    dataset[subject][action]['positions_3d'][i] = poses_3d[i]
                    w, h = np.repeat(np.array(cams[i]['res_w'])[np.newaxis,np.newaxis,np.newaxis], poses_2d[i].shape[0], axis=0), \
                           np.repeat(np.array(cams[i]['res_h'])[np.newaxis,np.newaxis,np.newaxis], poses_2d[i].shape[0], axis=0)
                    wh = np.concatenate((w,h),axis=-1)
                    keypoints[subject][action][i] = np.concatenate((poses_2d[i],wh),axis=1)
            else:
                for i in range(len(poses_3d)):
                    dataset[subject][action]['positions_3d'][i] = poses_3d[i]
                    w, h = np.repeat(np.array(cams[i]['res_w'])[np.newaxis,np.newaxis,np.newaxis], poses_2d[i].shape[0], axis=0), \
                           np.repeat(np.array(cams[i]['res_h'])[np.newaxis,np.newaxis,np.newaxis], poses_2d[i].shape[0], axis=0)
                    wh = np.concatenate((w,h),axis=-1)
                    keypoints[subject][action][i] = np.concatenate((poses_2d[i],wh),axis=1)

def normalization(dataset, keypoints, subjects, action_filter, cam_filter, norm):
    print('Start to normalize input 2d and 3d pose: ')
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
            cams = dataset.cameras()[subject]
            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            assert len(cams) == len(poses_2d), 'Camera count mismatch'
            norm_params = []
            if cam_filter:
                for i in cam_filter: # Select by some camera viewpoints
                    if norm == 'base':
                        # Remove global offset, but keep trajectory in first position
                        poses_3d[i][:, 1:] -= poses_3d[i][:, :1]
                        normed_pose_3d = poses_3d[i]
                        normed_pose_2d = normalize_screen_coordinates(poses_2d[i][..., :2], w=cams[i]['res_w'], h=cams[i]['res_h'])

                    else:
                        normed_pose_3d, normed_pose_2d, pixel_ratio, rescale_ratio, offset_2d, abs_root_Z = norm_to_pixel(
                            poses_3d[i], poses_2d[i], cams[i]['intrinsic'], norm)
                        norm_params.append(np.concatenate((pixel_ratio, rescale_ratio, offset_2d, abs_root_Z), axis=-1))  # [T, 1, 5], len()==4
                    keypoints[subject][action][i] = normed_pose_2d
                    dataset[subject][action]['positions_3d'][i] = normed_pose_3d
                if norm_params:
                    dataset[subject][action]['normalization_params'] = norm_params
            else:
                for i in range(len(poses_2d)):
                    if norm == 'base':
                        # Remove global offset, but keep trajectory in first position
                        poses_3d[i][:, 1:] -= poses_3d[i][:, :1]
                        normed_pose_3d = poses_3d[i]
                        normed_pose_2d = normalize_screen_coordinates(poses_2d[i][..., :2], w=cams[i]['res_w'], h=cams[i]['res_h'])

                    else:
                        normed_pose_3d, normed_pose_2d,  pixel_ratio, rescale_ratio, offset_2d, abs_root_Z = norm_to_pixel(poses_3d[i], poses_2d[i], cams[i]['intrinsic'], norm)
                        norm_params.append(np.concatenate((pixel_ratio, rescale_ratio, offset_2d, abs_root_Z), axis=-1))  # [T, 1, 5], len()==4
                    keypoints[subject][action][i] = normed_pose_2d
                    dataset[subject][action]['positions_3d'][i] = normed_pose_3d
                if norm_params:
                    dataset[subject][action]['normalization_params'] = norm_params

