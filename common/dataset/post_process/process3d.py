import torch
from common.transformation.cam_utils import image_coordinates, reprojection

# For inverse inference to get final 3d XYZ
def get_final_3d_coord(pos_3d_out, abs_root_3d, relaive_root_3d, camera, rescale_bbox_ratio, pixel_depth_ratio,norm):
    if norm=='lcn': # Another way to process pixel XY -> normalize xy by image height and width. Same way with LCN
        img_w, img_h = int(camera[0,2]*2), int(camera[0,3]*2)
        pixel_pose_3d = image_coordinates(pos_3d_out, img_w, img_h)
        pos_3d_stage3 = torch.zeros_like(pos_3d_out)
        pos_3d_stage3[...,2:3] = pixel_pose_3d[...,2:3]/pixel_depth_ratio
        pos_3d_stage3[...,:2] = pixel_pose_3d[...,:2]

    else:
        pose_relative = torch.zeros_like(pos_3d_out)
        pose_relative[..., :2] = relaive_root_3d
        pos_3d_stage1 = pos_3d_out / rescale_bbox_ratio  # To recover xyz bbox scale
        pos_3d_stage2 = pos_3d_stage1 + pose_relative  # (2000,1,17,3) #To recover xy first.

        pos_3d_stage3 = torch.zeros_like(pos_3d_out)
        pos_3d_stage3[:, :, :, 2:3] = pos_3d_stage2[:, :, :, 2:3] / pixel_depth_ratio
        pos_3d_stage3[..., :2] = pos_3d_stage2[..., :2].clone()

    abs_depth_z = pos_3d_stage3[..., 2:3].clone()
    abs_depth = abs_depth_z + abs_root_3d
    # Reprojection to get 3d X,Y
    reproject_3d = reprojection(pos_3d_stage3, abs_depth, camera)
    final_3d = reproject_3d - reproject_3d[:, :, :1]
    return final_3d/1000 #Use meters

def post_process3d(predicted_3d, inputs3d, cam, normalize_param, norm):
    inputs_3d_depth = normalize_param[..., 4:5]
    inputs_3d_relative_xy = normalize_param[..., 2:4]
    rescale_bbox_ratio, pixel_depth_ratio = normalize_param[..., 1:2], normalize_param[..., 0:1]
    predicted_3d_pos = get_final_3d_coord(predicted_3d, inputs_3d_depth, inputs_3d_relative_xy, cam, rescale_bbox_ratio,
                                          pixel_depth_ratio,norm)
    inputs_3d = get_final_3d_coord(inputs3d, inputs_3d_depth, inputs_3d_relative_xy, cam, rescale_bbox_ratio,
                                   pixel_depth_ratio,norm)
    return predicted_3d_pos, inputs_3d
