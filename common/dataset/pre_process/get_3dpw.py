import sys
sys.path.append('.')

import os
import cv2
import numpy as np
import pickle as pkl
import os.path as osp

import matplotlib
matplotlib.use('Agg')

from common.dataset.pre_process.kpt_index import get_perm_idxs
from common.dataset.pre_process.norm_data import norm_to_pixel

from common.transformation.cam_utils import normalize_screen_coordinates
from common.visualization.plot_pose3d import plot17j
from common.visualization.plot_pose2d import ColorStyle, color1, link_pairs1, point_color1

def get_3dpw(part):
    # part can be :'train', 'validation', 'test'

    folder = '/data/ailing/Video3d/data/3dpw/'
    NUM_JOINTS = 24
    VIS_THRESH = 0.3
    MIN_KP = 6

    sequences = [x.split('.')[0] for x in os.listdir(osp.join(folder, 'sequenceFiles',part))]
    print('action sequence:',sequences,len(sequences))
    imgs_path = []
    pose_3d = []
    pose_2d = []
    cam_ex = []
    cam_intri = []

    # start to process 3dpw raw data
    for i, seq in enumerate(sequences):
        print('sub sequence index:',i)
        data_file = osp.join(folder, 'sequenceFiles', part, seq + '.pkl')
        data = pkl.load(open(data_file, 'rb'), encoding='latin1')
        img_dir = osp.join(folder, 'imageFiles', seq)

        num_people = len(data['poses'])
        num_frames = len(data['img_frame_ids'])
        print('open action file:',data_file,img_dir,'has number people:',num_people,'with frame number:',num_frames)

        assert (data['poses2d'][0].shape[0] == num_frames)

        for p_id in range(num_people):
            print('person number:',p_id)
            j3d = data['jointPositions'][p_id].reshape(-1, 24,3)
            j2d = data['poses2d'][p_id].transpose(0,2,1)
            cam_in = data['cam_intrinsics'] #[3,3]
            cam_pose = data['cam_poses'] #[T, 4, 4] all people in a image will share the same

            campose_valid = data['campose_valid'][p_id] #[T,]
            print('invalid frames:',np.where(campose_valid==0),'valid frame number:',np.count_nonzero(campose_valid))
            new_j2d = np.zeros((j2d.shape[0],17,3))
            new_j3d = np.zeros((j3d.shape[0],17,3))

            # process 2d 3dpw keypoints into hm36 style
            perm_idxs = get_perm_idxs('3dpw', 'h36m')
            j2d = j2d[:, perm_idxs]
            new_j2d[:, 0] = (j2d[:,0] + j2d[:,3])/2
            new_j2d[:,1:7] = j2d[:,0:6]
            # new_j2d[:,4:7] = j2d[:,0:3]
            # new_j2d[:,1:4] = j2d[:,3:6]

            new_j2d[:,8] = (j2d[:,7]+j2d[:,10])/2 #neck
            new_j2d[:,7] = 0.7*new_j2d[:,0]+0.3*new_j2d[:,8]
            new_j2d[:,9] = j2d[:,6]
            new_j2d[:, 10] = 2*j2d[:, 6] - new_j2d[:,9]
            new_j2d[:,11:14] = j2d[:,10:13]
            new_j2d[:,14:17] = j2d[:,7:10]

            new_j2d[:, :, 2] = new_j2d[:, :, 2] > 0.3  # set the visibility flags

            # process 3d 3dpw_smpl joints into hm36 style
            perm_idxs = get_perm_idxs('smpl', 'h36m')
            j3d = j3d[:, perm_idxs]
            new_j3d[:,10] = 2*j3d[:, 9] - j3d[:,8]
            new_j3d[:,:10] = j3d[:,:10]
            new_j3d[:,11:] = j3d[:,10:]
            new_j3d[:,7] = 0.7*new_j3d[:,0] + 0.3*new_j3d[:,8] #update lower spine position
            #print('new pose 2d/3d shape:',new_j2d.shape, new_j3d.shape)

            # get camere params.
            cam_rt = cam_pose[:,0:3, 0:3]
            cam_t = cam_pose[:,0:3, 3:4]
            cam_pose3d = np.zeros_like(new_j3d) # get 3d pose under camere coordination system
            for j in range(len(new_j3d)):
                for k, kpt in enumerate(new_j3d[0]):
                    cam_pose3d[j,k][:,np.newaxis] = np.dot(cam_rt[j], new_j3d[j,k][:,np.newaxis])+cam_t[j]

            #cam_pose3d[:,8]=(cam_pose3d[:,11]+cam_pose3d[:,14])/2
            cam_pose3d[:,0]=(cam_pose3d[:,1]+cam_pose3d[:,4])/2

            cam_f = np.array([cam_in[0, 0], cam_in[1, 1]])
            cam_c = cam_in[0:2, 2]
            h = int(2 * cam_c[1])
            w = int(2 * cam_c[0])

            # verify cam_pose is right:
            XX = cam_pose3d[:, :,:2] / cam_pose3d[:,:, 2:]
            if np.array(XX).any() > 1 or np.array(XX).any() < -1:
                print(np.array(XX).any() > 1 or np.array(XX).any() < -1)
                print('Attention for this pose!!!')
            pose_2 = cam_f * XX + cam_c


            show_2d = False
            show_3d = False
            for index in range(0, len(pose_2)):
                #index = 350
                img_path = os.path.join(img_dir + '/image_%05d.jpg' % index)
                text = "Root 3d: ({:04.2f},{:04.2f},{:04.2f})m".format(cam_pose3d[index, 0, 0], cam_pose3d[index, 0, 1],
                                                                         cam_pose3d[index, 0, 2])
                print(text, 'seq', seq, 'person_id', p_id, 'index', index)
                if show_2d:
                    colorstyle = ColorStyle(color1, link_pairs1, point_color1)
                    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
                    img = cv2.imread(img_path)

                    kps = pose_2 # projected 2d pose
                    kps_gt = new_j2d #given 2d pose
                    for j, c in enumerate(connections):
                        start = kps[index,c[0]]
                        end = kps[index,c[1]]
                        cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), colorstyle.line_color[j], 3)
                        cv2.circle(img, (int(kps[index,j,0]), int(kps[index,j,1])), 4, colorstyle.ring_color[j], 2)

                        start_gt = kps_gt[index, c[0]]
                        end_gt = kps_gt[index, c[1]]
                        cv2.line(img, (int(start_gt[0]), int(start_gt[1])), (int(end_gt[0]), int(end_gt[1])), (255, 0, 0), 3)
                        cv2.circle(img, (int(kps_gt[index, j, 0]), int(kps_gt[index, j, 1])), 3, (255, 100, 0), 2)
                    text = "Root 3d: ({:04.2f}, {:04.2f}, {:04.2f})m".format(cam_pose3d[index,0,0],cam_pose3d[index,0,1],cam_pose3d[index,0,2])
                    print(part, text, 'seq',seq, 'person_id',p_id, 'index',index)
                    # cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('3DPW Example', img)

                    # cv2.imwrite('data/3dpw/validation/{}_{}_{:05d}.jpg'.format(seq, p_id, index), img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            if show_3d:
                plot17j(np.concatenate((new_j3d[345:349], cam_pose3d[345:349]),axis=0), None,'a','a')


            # Filter out keypoints
            indices_to_use = np.where((j2d[:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0] # you can change the VIS_THRESH to get pose_2d with different quality
            print('selected indexes:',indices_to_use)
            print('selected valid frame number:',len(indices_to_use))

            #norm pose 3d use zero-root
            #cam_pose_norm = cam_pose3d-cam_pose3d[:,:1]
            #pose_2_norm  = normalize_screen_coordinates(pose_2, w, h)
            #pose_2_norm[indices_to_use] = normalize_screen_coordinates(new_j2d[indices_to_use,:,:2], w, h)
            #pose_3d.append(cam_pose_norm)
            #pose_2d.append(pose_2_norm)

            if indices_to_use.any():
                pose_2 = pose_2[indices_to_use]
                cam_pose3d = cam_pose3d[indices_to_use]
                print('final pose shape:',pose_2.shape, cam_pose3d.shape)
            cam_int = np.zeros((9))
            cam_int[:2] = cam_f
            cam_int[2:4] = cam_c

            pose_2d.append(pose_2)
            pose_3d.append(cam_pose3d)
            cam_intri.append(cam_int)


    print('total length:',len(pose_3d))
    file_name = 'data/3dpw_{}'.format(part)
    np.savez_compressed(file_name, pose_3d=pose_3d, pose_2d=pose_2d,intrinsic=cam_intri)
    print('Saved as:', file_name)
    print('Done')


def load_3dpw(part, norm):
    data_3dpw_test = np.load('data/3dpw_{}_valid.npz'.format(part), allow_pickle=True)
    poses_valid = data_3dpw_test['pose_3d']
    poses_valid_2d = data_3dpw_test['pose_2d']
    valid_cam_in = data_3dpw_test['intrinsic']
    # normalize
    norm_val = []
    cameras_valid = []
    for i in range(len(poses_valid)):
        if norm == 'base':
            poses_valid[i][:, 1:] -= poses_valid[i][:, :1]
            normed_pose_3d = poses_valid[i]
            c_x, c_y = valid_cam_in[i][2], valid_cam_in[i][3]
            img_w = int(2 * c_x)
            img_h = int(2 * c_y)
            normed_pose_2d = normalize_screen_coordinates(poses_valid_2d[i][..., :2], w=img_w, h=img_h)
            cameras_valid = None
        else:
            normed_pose_3d, normed_pose_2d, pixel_ratio, rescale_ratio, offset_2d, abs_root_Z = norm_to_pixel(poses_valid[i],
                                                                                                              poses_valid_2d[i],
                                                                                                              valid_cam_in[i],
                                                                                                              norm)
            norm_val.append(np.concatenate((pixel_ratio, rescale_ratio, offset_2d, abs_root_Z), axis=-1))  # [T, 1, 5], len()==4
            use_params = {}
            use_params['intrinsic'] = valid_cam_in[i]
            use_params['normalization_params'] = norm_val[i]
            cameras_valid.append(use_params)
        poses_valid_2d[i] = normed_pose_2d
        poses_valid[i] = normed_pose_3d
    return poses_valid, poses_valid_2d, cameras_valid