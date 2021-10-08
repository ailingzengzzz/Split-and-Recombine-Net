import numpy as np
import torch

from common.transformation.cam_utils import project_to_2d_linear, normalize_screen_coordinates
from common.common_pytorch.utils import wrap


def week_perspective_scale(camera_params, depth):
    fx = camera_params[..., 0:1]
    return fx / depth

def change_to_mm(input):
    return input*1000

def change_to_m(input):
    return input/1000

def norm_to_pixel(pose_3d, pose_2d, camera, norm):
    pose_3d = change_to_mm(pose_3d)  # change m into mm
    pose3d_pixel, pixel_ratio = norm_to_pixel_s1(pose_3d, camera, norm)
    normed_3d, normed_2d, rescale_ratio, offset_2d, abs_root_Z = norm_to_pixel_s2(pose3d_pixel, pose_3d[:, 0:1], pose_2d, camera)
    if norm =='lcn':
        c_x, c_y = camera[2], camera[3]
        img_w = int(2 * c_x)
        img_h = int(2* c_y)
        normed_3d = normalize_screen_coordinates(pose3d_pixel,img_w,img_h)
        normed_2d = normalize_screen_coordinates(pose_2d,img_w,img_h)
    return normed_3d, normed_2d, pixel_ratio, rescale_ratio, offset_2d, abs_root_Z


def norm_to_pixel_s1(pose_3d, camera, norm):
    """
    pose_3d: 3d joints with absolute location in the camera coordinate system (meters)
    pose_3d.shape = [T, K, N], e.g. [1500, 17, 3]
    pose_2d: 2d joints with pixel location in the images coordinate system (pixels)
    pose_3d.shape = [T, K, M], e.g. [1500, 17, 2]
    return: normed_3d: root joint contain relative [x,y] offset and absolute depth of root Z. others joints are normed 3d joints in pixel unit
            normed_2d: zero-center root with resize into a fixed bbox
    """
    # stage1: linear project 3d X,Y to pixel unit, corresponding scale Z to keep the same 3d scale
    pose3d_root_Z = pose_3d[:, 0:1, 2:3].copy()

    camera = np.repeat(camera[np.newaxis, :], pose3d_root_Z.shape[0], axis=0)
    if norm == 'lcn':
        ratio1 = week_perspective_scale(camera[:,np.newaxis], pose3d_root_Z)+1 #[T,1,1] project depth as the same scale with XY
    else:
        ratio1 = week_perspective_scale(camera[:,np.newaxis], pose3d_root_Z) #[T,1,1] project depth as the same scale with XY

    pose3d_pixel = np.zeros_like(pose_3d)
    if norm == 'weak_proj':
        pose3d_root = np.repeat(pose3d_root_Z, 17, axis=-2)  # (T,17,1) # For weak perspective projection
        pose3d_pixel[..., :2] = pose_3d[..., :2]/pose3d_root * camera[:, np.newaxis, :2] + camera[:, np.newaxis, 2:4]
    else:
        pose3d_pixel[..., :2] = wrap(project_to_2d_linear, pose_3d.copy(), camera)  # Keep all depth from each joints, projected 2d xy are more precise.
    pose3d_relative_depth = minus_root(pose_3d[..., 2:3])  # Make root depth=0
    pose3d_stage1_depth = pose3d_relative_depth * ratio1 # Root_depth=0 [2000,17,1]
    pose3d_pixel[..., 2:3] = pose3d_stage1_depth.copy()
    return pose3d_pixel, ratio1

def norm_to_pixel_s2(pose3d_pixel, root_joint, pose_2d, camera, bbox_scale=2):
    # stage2: Resize 2d and 3d pixel position into one fixed bbox_scale
    pose3d_root_Z = root_joint[:, :, 2:3].copy()
    tl_3d_joint, br_3d_joint = make_3d_bbox(root_joint)

    camera = np.repeat(camera[np.newaxis, :], root_joint.shape[0], axis=0)
    tl2d = wrap(project_to_2d_linear, tl_3d_joint, camera)  # Use weak perspective
    br2d = wrap(project_to_2d_linear, br_3d_joint, camera)  # Use weak perspective
    bbox_2d = np.concatenate((tl2d.squeeze(), br2d.squeeze()), axis=-1)

    diff_bbox_2d = bbox_2d[..., 2:3] - bbox_2d[..., 0:1]  # (x_br - x_tl)
    ratio2 = bbox_scale / diff_bbox_2d  # ratio2.all() == (bbox_scale/(ratio2 * rectange_3d_size).all())

    # Get normed 3d joints
    pixel_xy_root = pose3d_pixel[:, 0:1, 0:2]  # [T,1,2]
    reshape_3d = pose3d_pixel * ratio2[:, np.newaxis, :]
    normed_3d = minus_root(reshape_3d)

    # Get normed 2d joints
    reshape_2d = pose_2d * ratio2[:, :, np.newaxis]
    normed_2d = minus_root(reshape_2d)
    return normed_3d, normed_2d, ratio2[:, np.newaxis], pixel_xy_root, pose3d_root_Z

def make_3d_bbox(pose_3d_root, rectangle_3d_size=2000):
    tl_joint = pose_3d_root.copy()
    tl_joint[..., :2] -= rectangle_3d_size/2 # 1000mm
    br_joint = pose_3d_root.copy()
    br_joint[..., :2] += rectangle_3d_size/2 # 1000mm
    return tl_joint, br_joint

def minus_root(pose):
    # Assume pose.shape = [T, K ,N]
    pose_root = pose[:,:1]
    relative_pose = pose - pose_root
    return relative_pose

def get_ratio(abs_root_3d, camera):
    # abs_root_3d.shape = [B, T, 1, 1]
    # camera.shape = [B, 9] & [2, 9]
    bbox_scale = 1
    rectangle_3d_size = 2000
    camera = camera.unsqueeze(dim=1).unsqueeze(dim=1)
    fx, fy = camera[:,:,:,0:1], camera[:,:,:,1:2]
    pixel_depth_ratio = fx / abs_root_3d
    rescale_bbox = bbox_scale / pixel_depth_ratio
    rescale_bbox_ratio = rescale_bbox / rectangle_3d_size
    return rescale_bbox_ratio, pixel_depth_ratio

