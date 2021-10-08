import numpy as np
import torch

def kpt_to_bone_vector(pose_3d, parent_index=None):
    if parent_index is not None:
        hm36_parent = parent_index

    else:
        hm36_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15] #by body kinematic connections
    #print('random parent index:',hm36_parent)
    bone = []
    for i in range(1, len(hm36_parent)):
        bone_3d = pose_3d[:, :, i] - pose_3d[:,:,hm36_parent[i]]
        bone.append(bone_3d.unsqueeze(dim=-2))
    bone_out = torch.cat(bone, dim=-2)
    return bone_out

def two_order_bone_vector(bone_3d, parent_index=None):
    if parent_index is not None:
        hm36_parent = parent_index
    else:
        hm36_parent = [-1, 0, 1, 0, 3, 4, 0, 6, 7, 8, 7, 10, 11, 7, 13, 14] #by body kinematic connections, same to calculate angles
    #print('random parent index:',hm36_parent)
    bone = []
    for i in range(1, len(hm36_parent)):
        bone_3d_2 = bone_3d[:, :, i] - bone_3d[:,:,hm36_parent[i]]
        bone.append(bone_3d_2.unsqueeze(dim=-2))
    bone_out = torch.cat(bone, dim=-2)
    return bone_out