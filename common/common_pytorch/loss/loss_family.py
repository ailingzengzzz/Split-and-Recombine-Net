# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import numpy as np
import math


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    #l2_error = torch.mean(torch.norm((predicted - target), dim=len(target.shape) - 1), -1).squeeze()
    #print('each joint error:', torch.norm((predicted - target), dim=len(target.shape) - 1))
    #index = np.where(l2_error.cpu().detach().numpy() > 0.3)  # mean body l2 distance larger than 300mm
    #value = l2_error[l2_error > 0.3]
    #print('Index of mean body l2 distance larger than 300mm', index, value)
    return torch.mean(torch.norm((predicted - target), dim=len(target.shape) - 1))


def mpjae(predicted, target):
    """
    Mean per-joint angle error (3d bone vector angle error between gt and predicted one)
    """
    assert predicted.shape == target.shape  # [B,T, K]
    joint_error = torch.mean(torch.abs(predicted - target).cuda(), dim=0)  # Calculate each joint angle
    print('each bone angle error:', joint_error)
    return torch.mean(joint_error)


# def weighted_mpjpe(predicted, target):
# take each joint with a weight

def mpjpe_smooth(predicted, target, threshold, mi, L1):
    """
    Referred in triangulation 3d pose paper
    """
    assert predicted.shape == target.shape
    if L1:
        diff_norm = torch.abs((predicted - target), dim=len(target.shape) - 1)
        diff = diff_norm.clone()
    else:  # MSE
        diff = (predicted - target) ** 2
    diff[diff > threshold] = torch.pow(diff[diff > threshold], mi) * (threshold ** (1 - mi))
    loss = torch.mean(diff)
    return loss


def L1_loss(predicted, target):
    assert predicted.shape == target.shape
    abs_error = torch.mean(torch.mean(torch.abs(predicted - target).cuda(), dim=-2), dim=0)
    error = torch.mean(abs_error)
    return error


def kpt_mpjpe(predicted, target):
    #   Mean per-joint position error for each keypoint(i.e. mean Euclidean distance)
    #   This function is just for evaluate!! input shape is (1,t,17,3)
    assert predicted.shape == target.shape
    kpt_error = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    kpt_xyz = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 2), dim=1)
    print('X,Y,Z error of T input frames:', kpt_xyz / np.sqrt(17))
    kpt_17 = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=0), dim=0)
    return kpt_17


def kpt_test(predicted, target):
    #   Mean per-joint position error for each keypoint(i.e. mean Euclidean distance)
    #   This function is just for evaluate!! input shape is (2,t,17,3)
    assert predicted.shape == target.shape
    print('The frame number is', target.size())
    kpt_xyz = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 2), dim=0), dim=0)
    kpt_17 = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=0), dim=0)
    return kpt_xyz, kpt_17

def class_accuracy(predicted, target, confidence, threshold=0.04):
    # confidence.shape = [B,T,1,1]
    confidence = torch.mean(torch.mean(torch.mean(confidence, -1), -1), 0)
    sig = nn.Sigmoid()

    diff = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=0), dim=-1)
    class0 = (diff<=threshold).cpu()
    conf = (sig(confidence)>0.5).cpu()
    correct = (conf == class0).sum()
    print('ooo',diff.shape,correct, predicted.shape[1],correct/predicted.shape[1])
    return correct

    

def Uncertain_CE(predicted, target, confidence, threshold=0.04, L1_loss=True):
    # confidence.shape = [B,T,1,1]
    confidence = torch.mean(torch.mean(torch.mean(confidence, -1), -1), -1)
    above_thre = torch.zeros_like(confidence).cuda()
    below_thre = torch.zeros_like(confidence).cuda()
    # class0 = torch.zeros((len(confidence), 2)).cuda()
    class0 = torch.zeros_like(confidence).cuda()
    if L1_loss:
        diff = torch.mean(torch.mean(torch.mean(torch.abs(predicted - target), dim=1), dim=-1), dim=-1) #[B]
    else:
        diff = torch.mean(torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=1), dim=-1) #[B]
    threshold = torch.mean(diff)
    above_thre = diff * (diff>threshold).float()
    below_thre = diff * (diff<=threshold).float()
    # print('ccc',above_thre,'ppp',below_thre)
    # class0[:, 0] = (diff>threshold).long().cuda()
    # class0[:, 1] = (diff<=threshold).long().cuda()
    class0 = (diff<=threshold).float().cuda()
    sig = nn.Sigmoid()
    a1 = sig(confidence)
    # a1 = 0.5
    a2 = 1 - a1

    weight = 0.1
    # pos_weight = torch.sum(diff>threshold)/torch.sum(diff<=threshold)
    # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    BCE = nn.BCEWithLogitsLoss()
    item1 = torch.mean(a1 * below_thre)
    item2 = torch.mean(a2 * above_thre)
    item3 = torch.mean(weight*BCE(confidence, class0))
    print('xxx',item3,item2,item1)
    loss = item1 + item2 + item3
    return loss




class L1GaussianRegressionNewFlow(nn.Module):
    ''' L1 Joint Gaussian Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(L1GaussianRegressionNewFlow, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def weighted_l2_loss(self, pred, gt, weight):
        diff = (pred - gt) ** 2
        diff = diff * weight
        return diff.sum() / (weight.sum() + 1e-9)

    def _generate_activation(self, gt_coords, pred_coords):
        sigma = 2 * 2 / 64
        # (B, K, 1, 2)
        gt_coords = gt_coords.permute(0,2,1,3)
        # (B, 1, K, 2)
        pred_coords = pred_coords

        diff = torch.sum((gt_coords - pred_coords) ** 2, dim=-1)
        activation = torch.exp(-(diff / (2 * (sigma ** 2))))

        return activation

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma

        gt_uv = labels
        weight = 1
        #gaussian = weight * torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)
        gaussian = torch.abs(gt_uv - pred_jts)
        residual = True
        if residual:
            weight1 = 1
            nf_loss = weight1 * nf_loss + gaussian

        if self.size_average > 0:
            regression_loss = nf_loss.sum() / len(nf_loss)
            #regression_loss = torch.mean(nf_loss)#todo

        else:
            regression_loss = nf_loss.sum()

        loss = regression_loss

        return loss


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape  # (3071, 17, 3)
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    # Remove scale
    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))
    # print('target',normX,'predice',normY)
    X0 /= (normX + 1e-8)
    if normY.any() == 0:
        normY = normY + 1e-8

    Y0 /= (normY + 1e-8)
    # Optimum rotation matrix of Y0
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Standarized Distance between X0 and a*Y0*R+c
    d = 1 - tr ** 2

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R)
    trans_aligned = predicted_aligned + t
    error = np.mean(np.linalg.norm(trans_aligned - target, axis=len(target.shape) - 1))
    # Return MPJPE
    return error, torch.from_numpy(trans_aligned).unsqueeze(dim=0).cuda()


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape  # [1, 1703, 17, 3]
    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    out = torch.mean(torch.norm((scale * predicted - target), dim=len(target.shape) - 1))
    return out


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))
