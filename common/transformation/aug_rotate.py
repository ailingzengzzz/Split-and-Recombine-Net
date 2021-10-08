#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from common.transformation.cam_utils import *
from common.arguments.basic_args import parse_args
from common.visualization.plot_pose3d import plot17j

args = parse_args()
np.random.seed(4321)

def rotate(batch_2d_in, batch_3d_in, cam, repeat_num):
    pose_3d = []
    pose_2d = []
    for i in range(repeat_num):
        batch_3d, batch_2d = axis_rotation(batch_3d_in*1000, cam)
        pose_3d.append(batch_3d)
        pose_2d.append(batch_2d)
    rotate_3d = np.concatenate(pose_3d, axis=0)
    batch_3d_out = np.concatenate((batch_3d_in, rotate_3d), axis=0)
    batch_3d_relative = batch_3d_out-batch_3d_out[:,:,:1]

    w, h = batch_2d_in[:,:,17:18,0:1], batch_2d_in[:,:,17:18,1:2]
    batch_2d_in_norm = process_2d(batch_2d_in)

    pose_out = []
    for i in range(repeat_num):
        batch_2d_norm = norm_pixel(pose_2d[i], w, h)
        pose_out.append(batch_2d_norm)
    batch_2d_out = np.concatenate((batch_2d_in_norm, np.concatenate(pose_out,axis=0)), axis=0)
    return batch_2d_out, batch_3d_relative

def axis_rotation(batch_3d,cam):
    # Input batch 3d pose is presented by the Relative value in pose model coordination. Root=[0,0,0]
    batch_root = batch_3d[:,:,:1].copy()
    batch_size = batch_3d.shape[0]
    batch_pose = batch_3d - batch_root
    theta = np.random.uniform(-np.pi, np.pi, batch_size).astype('f') # Y axis - roll
    beta = np.random.uniform(-np.pi/5, np.pi/5, batch_size).astype('f') # X axis - pitch
    alpha = np.random.uniform(-np.pi/5, np.pi/5, batch_size).astype('f') #Z axis - yaw

    cos_theta = np.cos(theta)[:, None,None,None]
    sin_theta = np.sin(theta)[:, None,None,None]

    cos_beta = np.cos(beta)[:, None,None,None]
    sin_beta = np.sin(beta)[:, None,None,None]

    cos_alpha = np.cos(alpha)[:, None,None,None]
    sin_alpha = np.sin(alpha)[:, None,None,None]

    X = batch_pose[...,0:1]
    Y = batch_pose[...,1:2]
    Z = batch_pose[...,2:3]

    # rotate around Y axis
    new_x = X * cos_theta + Z * sin_theta
    new_y = Y
    new_z = - X * sin_theta + Z * cos_theta

    # rotate around X axis
    new_x = new_x
    new_y = new_y * cos_beta - new_z * sin_beta
    new_z = new_y * sin_beta + new_z * cos_beta

    # rotate around Z axis
    new_x = new_x * cos_alpha - new_y *sin_alpha
    new_y = new_x * sin_alpha + new_y * cos_alpha
    new_z = new_z

    rotated_pose = np.concatenate((new_x,new_y,new_z),axis=-1)
    rotated_abs_3d = rotated_pose + batch_root
    rotated_2d = wrap(project_to_2d, rotated_abs_3d, cam)
    rotated_3d = rotated_abs_3d / 1000.0 #change unit from mm to m
    return rotated_3d, rotated_2d

def process_3d(pose_3d_in):
    pose_3d_out = pose_3d_in - pose_3d_in[:, :, :1]
    return pose_3d_out

def process_2d(pose_2d_in):
    pose_2d_joint = pose_2d_in[:,:,:17]
    w, h = pose_2d_in[:,:,17:18,0:1], pose_2d_in[:,:,17:18,1:2]
    pose_2d_in_norm = norm_pixel(pose_2d_joint, w, h)
    return pose_2d_in_norm

def norm_pixel(pose_2d, w, h):
    X = pose_2d[...,0:1]
    Y = pose_2d[...,1:2]
    norm_X = X/w * 2 - 1
    w_abs = np.abs(w)
    norm_Y = Y/w_abs *2 - h/w_abs #The flip influences the w value.
    norm_2d = np.concatenate((norm_X,norm_Y),axis=-1)
    return norm_2d
