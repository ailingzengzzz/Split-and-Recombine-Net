import torch
import numpy as np

from common.common_pytorch.loss.loss_family import p_mpjpe
from common.visualization.plot_pose3d import plot17j

def gather_3d_metrics(expected, actual):
    """

    :param expected: Predicted pose
    :param actual: Ground Truth
    :return: evaluation results
    """
    unaligned_pck = pck(actual, expected)
    unaligned_auc = auc(actual, expected)
    expect_np = expected.cpu().numpy().reshape(-1, expected.shape[-2], expected.shape[-1])
    actual_np = actual.cpu().numpy().reshape(-1, expected.shape[-2], expected.shape[-1])
    aligned_mpjpe, aligned = p_mpjpe(expect_np, actual_np)
    #plot17j(np.concatenate((actual_np[100:104],expect_np[100:104],aligned[0,100:104].cpu().numpy()),axis=0),'aa','aa')
    aligned_pck = pck(aligned, actual)
    aligned_auc = auc(aligned, actual)
    return dict(
        pck=unaligned_pck,
        auc=unaligned_auc,
        aligned_mpjpe=aligned_mpjpe,
        aligned_pck=aligned_pck,
        aligned_auc=aligned_auc,
    )

def pck(actual, expected,threshold=150):
    dists = torch.norm((actual - expected), dim=len(actual.shape)-1)
    error = (dists < threshold).double().mean().item()
    return error

def auc(actual, expected):
    # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = torch.linspace(0, 150, 31).tolist()
    pck_values = torch.DoubleTensor(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pck_values[i] = pck(actual, expected, threshold=threshold)
    return pck_values.mean().item()

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

def cal_bone_sym(predicted_3d_pos):
    # calculate bone length symmetry
    left = [4,5,6,11,12,13]
    right = [1,2,3,14,15,16]
    hm36_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

    bone_lengths_lift = []
    bone_lengths_right = []
    each_bone_error = []
    left_right_error = 0
    for i in left:
        dists_l = predicted_3d_pos[:, :, i, :] - predicted_3d_pos[:, :, hm36_parent[i], :]
        bone_lengths_lift.append(torch.mean(torch.norm(dists_l, dim=-1), dim=1))
    for i in right:
        dists_r = predicted_3d_pos[:, :, i, :] - predicted_3d_pos[:, :, hm36_parent[i], :]
        bone_lengths_right.append(torch.mean(torch.norm(dists_r, dim=-1), dim=1))
    for i in range(len(left)):
        left_right_error += torch.abs(bone_lengths_right[i] - bone_lengths_lift[i])
        each_bone_error.append(torch.mean(torch.abs(bone_lengths_right[i] - bone_lengths_lift[i])))
    txt1 = 'Bone symmetric error (right-left): Hip {0}mm, Upper Leg {1}mm, Lower Leg {2}mm, '\
          'Shoulder {3}mm, Upper elbow {4}mm, Lower elbow {5}mm'.format(each_bone_error[0]*1000,each_bone_error[1]*1000,each_bone_error[2]*1000,
                                                                        each_bone_error[3]*1000,each_bone_error[4]*1000,each_bone_error[5]*1000)
    print(txt1)
    left_right_err_mean = torch.mean(left_right_error/6)
    print('all parts mean symmetric error: ',left_right_err_mean.item()*1000)
    return torch.tensor(each_bone_error)

def angle_np(v1, v2, acute=False):
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
    #angle = angle[:, :, np.newaxis]
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

def angle_torch(v1, v2, torch_pi, acute=False):
    # v1 is your firsr 3d vector, v.shape: [B, T, 1, 3]
    # v2 is your second 3d vector
    v1_len = torch.norm(v1, dim=-1)
    v2_len = torch.norm(v2, dim=-1)
    angle = torch.mean(torch.acos(torch.sum(v1.mul(v2), dim=-1) / (v1_len * v2_len+torch.Tensor([1e-8]).cuda()))) #shape: [B, T. 1]
    if (acute == True):
        return angle
    else:
        return 2 * torch_pi - angle

def cal_bone_angle(bone_3d, indexes=None):
    torch_pi = torch.acos(torch.zeros(1)).item() * 2
    bone_parent_index = [-1, 0, 1, 0, 3, 4, 0, 6, 7, 8, 7, 10, 11, 7, 13, 14]
    bone_angle = []
    text = []
    # init_vector = [0,-1,0] #relative to root joint

    if indexes:
        # calculate specific pair joint angle
        for index in indexes:
            bone_child = bone_3d[index]
            bone_parent = bone_3d[bone_parent_index[index]]
            bone_angle.append(180 * angle_torch(bone_child, bone_parent, torch_pi, acute=True)/torch.pi)
            text.append('The bone angle between child bone {} and parent bone {} is :{}'.format(index, bone_parent_index[index], bone_angle[-1]))
        print(text)

    else:
        # calculate each pair joint angle
        for index in range(1, 16):
            bone_child = bone_3d[:, :, index]
            bone_parent = bone_3d[:, :, bone_parent_index[index]]
            #angle1 = 180*angle(bone_child.squeeze().cpu().detach().numpy()[0], bone_parent.squeeze().cpu().detach().numpy()[0], acute=True)/np.pi
            joint_angle = 180 * angle_torch(bone_child.squeeze(), bone_parent.squeeze(), torch_pi, acute=True) / torch_pi
            bone_angle.append(joint_angle.unsqueeze(dim=-1))
            text.append('The bone angle between child bone {} and parent bone {} is :{}'.format(index, bone_parent_index[index],bone_angle[-1]))

        body_angle = torch.cat(bone_angle, dim=-1)
        print(text)
    return body_angle

def cal_bone_length_np(pose_3d):
    hm36_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    hm36_num = 17
    e4 = []
    for i in range(1, hm36_num):
        bone_3d = pose_3d[:, :, i] - pose_3d[:,:,hm36_parent[i]]
        e4.append(np.mean(np.mean(np.linalg.norm(bone_3d, axis=-1)*100,0),0))
        print('std of each bone length',np.std(np.mean(np.linalg.norm(bone_3d, axis=-1)*100,0),0),'cm')
    kpt_txt = 'Bone length of RHip is {0}cm,RUleg is {1}cm, RLleg is {2}cm, Lhip is {3}cm,  LUleg is {4}cm, LLleg is {5}cm, ' \
              'Lspine is {6}cm, Uspine is {7}cm, Neck is {8}cm, Head is {9}cm, Lshoulder is {10}cm, LUelbow is {11}cm, LLelbow is {12}cm, '\
                'Rshoudle is {13}cm, RUelbow is {14}cm, RLelbow is {15}cm:'.format(e4[0], e4[1], e4[2], e4[3],
                                                                                            e4[4], e4[5],
                                                                                            e4[6],
                                                                                            e4[7], e4[8],
                                                                                            e4[9], e4[10],
                                                                                            e4[11], e4[12],
                                                                                            e4[13],
                                                                                            e4[14], e4[15])
    print(kpt_txt)
    return e4