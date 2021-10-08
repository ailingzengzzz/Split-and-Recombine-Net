import numpy as np
from common.dataset.pre_process.norm_data import norm_to_pixel
from common.transformation.cam_utils import normalize_screen_coordinates

def load_mpi_test(file_path, seq, norm):
    """
    Usage: Load a section once
    :param dataset_root: root path
    :param section: There are six sequences in this (seq=0,1,2,3,4,5). And 2935 poses in a unique set(seq==7).
    If you want to evaluate by scene setting, you can use the sequencewise evaluation
    to convert to these numbers by doing
    #1:Studio with Green Screen (TS1*603 + TS2 *540)/ (603+540)
    #2:Studio without Green Screen (TS3*505+TS4*553)/(505+553)
    #3:Outdoor (TS5*276+TS6*452)/(276+452)
    :return: Normalized 2d/3d pose, normalization params and camera intrinics. All types: List
    """
    info = np.load(file_path, allow_pickle=True)
    if seq in range(0,6):
        pose_3d = info['pose3d_univ'][seq]
        pose_2d = info['pose2d'][seq]
        if seq in [0, 1, 2, 3]:
            img_w, img_h = 2048, 2048
            cam_intri = np.array([1500.0686135995716, 1500.6590966853348, 1017.3794860438494, 1043.062824876024, 1,1,1,1,1])
        elif seq in [4, 5]:
            img_w, img_h = 1920, 1080
            cam_intri = np.array([1683.482559482185, 1671.927242063379, 939.9278168524228, 560.2072491988034, 1,1,1,1,1])

    elif seq == 7:
        pose_3d = info['pose3d_univ'][0]
        pose_2d = info['pose2d'][0]
        img_w, img_h = 2048, 2048
        cam_intri = np.array([1504.1479043534127, 1556.86936732066, 991.7469587022122, 872.994958045596, 1, 1, 1, 1, 1])
    params = {}
    if norm == 'base':
        # Remove global offset, but keep trajectory in first position
        pose_3d[:, 1:] -= pose_3d[:, :1]
        normed_pose_3d = pose_3d/1000
        normed_pose_2d = normalize_screen_coordinates(pose_2d[..., :2], w=img_w, h=img_h)
        params['intrinsic'] = cam_intri
    else:
        normed_pose_3d, normed_pose_2d, pixel_ratio, rescale_ratio, offset_2d, abs_root_Z = norm_to_pixel(pose_3d/1000, pose_2d, cam_intri, norm)
        norm_params=np.concatenate((pixel_ratio, rescale_ratio, offset_2d, abs_root_Z), axis=-1)  # [T, 1, 5], len()==4
        params['intrinsic'] = cam_intri
        params['normalization_params'] = norm_params
    return normed_pose_3d, normed_pose_2d, params