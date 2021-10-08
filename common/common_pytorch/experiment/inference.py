import torch
import numpy as np
from time import time

from common.arguments.basic_args import parse_args
from common.transformation.cam_utils import *
from common.dataset.pre_process.utils import fetch

from common.dataset.post_process.process3d import post_process3d
from common.visualization.plot_pose3d import plot17j
from common.common_pytorch.loss.loss_family import *
from common.dataset.data_generators import ChunkedGenerator, UnchunkedGenerator
from common.common_pytorch.experiment.eval_metrics import gather_3d_metrics, cal_bone_sym, cal_bone_angle, kpt_to_bone_vector
import json
args = parse_args()
def infer_function(subjects_tests, keypoints, dataset, norm, action_test, model_pos, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, test_cam):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []
    errors_em = []
    errors_ea = []
    if action_test == None:
        print(' Input action is action_filter, so evaluate on every action seperately')
        action_test = args.all_action.split(',')
    for action_key in action_test:
        print('Start evaluate on action:', action_key)
        cam_act, poses_act, poses_2d_act = fetch(subjects_tests, keypoints, dataset, args.downsample, [action_key], test_cam)  # len(pose_act)=batch_size, type:list

        gen = UnchunkedGenerator(cam_act, poses_act, poses_2d_act,
                                 pad=pad, causal_shift=causal_shift,
                                 augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)

        e1, e2, e3, ev, em, ea = evaluate(gen, model_pos, joints_left, joints_right, norm, action_key, False)

        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)
        errors_em.append(em)
        errors_ea.append(ea)
    print('Mean protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 2), 'mm')
    print('Mean protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 2), 'mm')
    print('Mean protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 2), 'mm')
    print('Mean velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
    # print('Mean bone sym error        action-wise average:', round(np.mean(errors_em), 2), 'mm')
    print('Mean angle absolute Error  action-wise average:', round(np.mean(errors_ea), 2), 'degree')
    return np.mean(errors_p1)

def eval_hard_test(pose_3d, pose_2d, model_pos, norm, pad, causal_shift, kps_left, kps_right, joints_left, joints_right):
    gen = UnchunkedGenerator(None, pose_3d, pose_2d,
                             pad=pad, causal_shift=causal_shift,
                             augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)

    evaluate(gen, model_pos, joints_left, joints_right, norm)

def eval_mpi_test(pose_3d, pose_2d, info, model_pos, norm, pad, causal_shift, kps_left, kps_right, joints_left, joints_right):
    gen = UnchunkedGenerator(info, pose_3d, pose_2d,
                             pad=pad, causal_shift=causal_shift,
                             augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)

    evaluate(gen, model_pos, joints_left, joints_right, norm)

def eval_lcn(pose_3d, pose_2d, model_pos, norm, pad, causal_shift, kps_left, kps_right, joints_left, joints_right):
    pred_3d = []
    gen = UnchunkedGenerator(None, pose_3d, pose_2d,
                             pad=pad, causal_shift=causal_shift,
                             augment=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                             joints_right=joints_right)
    out_pose = np.concatenate(evaluate(gen, model_pos, joints_left, joints_right, norm, return_predictions=False)[0],axis=0)
    pred_3d.append(out_pose)
    pose = np.concatenate(pred_3d, axis=0)
    return pose


# Evaluate
def evaluate(test_generator, model_pos, joints_left, joints_right, norm, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_kpt_3d_pos = 0
    epoch_sym_error = 0
    epoch_angle_loss = 0

    mean_pck = 0
    mean_apck = 0
    mean_auc = 0
    mean_aauc = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        i = 0
        pred_3d = []
        time_now = time()
        predict_std1 = 0
        predict_std2 = 0
        for use_params, batch, batch_2d in test_generator.next_epoch():
            i+=1
            print('this the {} action in the test set.'.format(i))
            if norm != 'base':
                normalize_param = use_params['normalization_params']
                cam = use_params['intrinsic']

            inputs_2d = torch.from_numpy(batch_2d.astype('float32')) #shape:(2,t,17,2),if data augment
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                if norm != 'base':
                    cam = torch.from_numpy(cam.astype('float32')).cuda()  # torch.Size([2,9])
                    normalize_param = torch.from_numpy(normalize_param.astype('float32')).cuda() #[2, T, 1, 5]
            # Positional model
            predicted_3d_pos = model_pos(inputs_2d) #shape:(2,t,17,3)


            if norm != 'base':
                predicted_3d_pos, inputs_3d = post_process3d(predicted_3d_pos, inputs_3d, cam, normalize_param, norm)
            elif norm == 'base':
                inputs_3d[:, :, 0] = 0

            
            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                # joints left[4, 5, 6, 11, 12, 13]
                # joints right[1, 2, 3, 14, 15, 16]
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            sym_bone = cal_bone_sym(predicted_3d_pos)
            epoch_sym_error += inputs_3d.shape[0] * inputs_3d.shape[1] * sym_bone

            bone_gt = kpt_to_bone_vector(inputs_3d)
            bone_angle_gt = cal_bone_angle(bone_gt, indexes=None)
            bone_pred = kpt_to_bone_vector(predicted_3d_pos)
            bone_angle_pred = cal_bone_angle(bone_pred, indexes=None)
            angle_loss = mpjae(bone_angle_pred, bone_angle_gt)
            epoch_angle_loss += inputs_3d.shape[0] * inputs_3d.shape[1] * angle_loss.item()
            #
            if args.three_dpw:
                #only evaluate 14 joints
                predicted_3d_pos = torch.cat([predicted_3d_pos[:,:,:7],predicted_3d_pos[:,:,8:9],predicted_3d_pos[:,:,11:17]],-2)
                inputs_3d = torch.cat([inputs_3d[:,:,:7],inputs_3d[:,:,8:9],inputs_3d[:,:,11:17]],-2)
            # Calculate mean per joint position error in meters.
            gather_error = gather_3d_metrics(predicted_3d_pos*1000, inputs_3d*1000)

            print(gather_error)
            mean_pck += inputs_3d.shape[0] * inputs_3d.shape[1] * gather_error['pck']
            mean_apck += inputs_3d.shape[0] * inputs_3d.shape[1] * gather_error['aligned_pck']
            mean_auc += inputs_3d.shape[0] * inputs_3d.shape[1] * gather_error['auc']
            mean_aauc += inputs_3d.shape[0] * inputs_3d.shape[1] * gather_error['aligned_auc']
            pred_3d.append(predicted_3d_pos.squeeze(0).cpu().numpy())

            error = mpjpe(predicted_3d_pos, inputs_3d)
            print('each subaction MPJPE:',error.item()*1000)
            error_kpt = kpt_mpjpe(predicted_3d_pos, inputs_3d)
            print('error kpt of T input frames:',(error_kpt)*1000)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
            print('each subaction 3d scale error (NMPJPE):',n_mpjpe(predicted_3d_pos,inputs_3d).item()*1000)

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            print('mean pck/aligned_pck/auc/aligned_auc:',mean_pck/N, mean_apck/N, mean_auc/N, mean_aauc/N)

            print('mean MPJPE:',(epoch_loss_3d_pos/N)*1000)
            print('Average angle L1 error:', epoch_angle_loss/N)
            epoch_kpt_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1]*error_kpt
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)[0]
            print('each 3d procrustes error:',p_mpjpe(predicted_3d_pos, inputs)[0]*1000)
            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
            print('each 3d mean velocity error',mean_velocity_error(predicted_3d_pos, inputs)*1000)
    if args.rnum:
        print('final STD',predict_std1/i, predict_std2/i, torch.mean(predict_std1)/i,torch.mean(predict_std2)/i)

    if action is None:
        print('------Not evaluate by actions------')
    else:
        print('----' + action + '----')
        print('Have i epoch',i)
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    e4 = (epoch_kpt_3d_pos / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    each_bone_error = (epoch_sym_error/N)*1000
    each_angle_error = (epoch_angle_loss/N)
#    error_cat = np.concatenate(pred_3d)
#    np.save('fc_predicted_3dpose_{}.npy'.format(action), error_cat)
    print('Test time augmentation:', test_generator.augment_enabled())
    print(action, 'Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    if args.three_dpw:
        print(action, 'Error(MPJPE) of Hip(root) is {0}mm, Rhip is {1}mm, Rknee is {2}mm, Rfoot is {3}mm, Lhip is {4}mm, Lknee is {5}mm, '
              'Lfoot is {6}mm, Thorax is {7}mm, Lshoulder is {8}mm, Lelbow is {9}mm, '
              'Lwrist is {10}mm, Rshoulder is {11}mm, Relbow is {12}mm, Rwrist is {13}mm:'.format(e4[0],e4[1],e4[2],e4[3],e4[4],e4[5],e4[6],
                                                                                                  e4[7],e4[8],e4[9],e4[10],e4[11],e4[12],e4[13]))

    else:
        print(action, 'Error(MPJPE) of Hip(root) is {0}mm, Rhip is {1}mm, Rknee is {2}mm, Rfoot is {3}mm, Lhip is {4}mm, Lknee is {5}mm, '
              'Lfoot is {6}mm, Spine is {7}mm, Thorax is {8}mm, Neck is {9}mm, Head is {10}mm, Lshoulder is {11}mm, Lelbow is {12}mm, '
              'Lwrist is {13}mm, Rshoulder is {14}mm, Relbow is {15}mm, Rwrist is {16}mm:'.format(e4[0],e4[1],e4[2],e4[3],e4[4],e4[5],e4[6],
                                                                                                  e4[7],e4[8],e4[9],e4[10],e4[11],e4[12],e4[13],
                                                                                                  e4[14],e4[15],e4[16]))
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('Average test time for each frame is: ',(time()-time_now)/N)
    print(action, 'Mean Action bone error (right-left): Hip {0}mm, Upper Leg {1}mm, Lower Leg {2}mm, '\
          'Shoulder {3}mm, Upper elbow {4}mm, Lower elbow {5}mm'.format(each_bone_error[0],each_bone_error[1],each_bone_error[2],
                                                                        each_bone_error[3],each_bone_error[4],each_bone_error[5]))
    print('3d Angle absolute error:',each_angle_error, 'degree')
    print('----------')
    return e1,e2,e3,ev, each_bone_error, each_angle_error


def val(trainval_generator, model_pos, norm):
    with torch.no_grad():
        model_pos.eval()
        epoch_loss_3d_train_eval = 0
        N = 0
        for use_params, batch, batch_2d in trainval_generator.next_epoch():
            if batch_2d.shape[1] == 0:
                # This can only happen when downsampling the dataset
                continue
            if norm != 'base':
                normalize_param = use_params['normalization_params']
                cam = use_params['intrinsic']
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                if norm != 'base':
                    cam = torch.from_numpy(cam.astype('float32')).cuda()  # torch.Size([1024,9])
                    normalize_param = torch.from_numpy(normalize_param.astype('float32')).cuda()
            # Compute 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            # Denorm 3d pose, from pixel unit to meter unit.
            if norm != 'base':
                predicted_3d_pos, inputs_3d = post_process3d(predicted_3d_pos, inputs_3d, cam, normalize_param, norm)

            else:
                inputs_3d[:, :, 0] = 0

            # Calculate L2 error here
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

    return epoch_loss_3d_train_eval/N*1000




