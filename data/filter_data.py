import numpy as np
import torch
from run_utility.prepare_dataset import load_data, prepare_dataset, load_2d_data, prepare_2d_data, normalization
from common.arguments import parse_args
from common.plot_pose3d import plot17j
import os
args = parse_args()

cal_mean = False
cal_distance = True
dataset_root = '../../Video3d/data/'

print('Loading dataset...')
dataset = load_data(dataset_root, args.dataset, args.keypoints)

print('Preparing data...')
prepare_dataset(dataset)
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

print('Loading 2D detections...')
keypoints, keypoints_metadata, kps_left, kps_right = load_2d_data(dataset_root, args.dataset, args.keypoints)

print('Preparing 2D data...')
prepare_2d_data(keypoints, dataset)

# Normalize all data
normalization(dataset, keypoints, args.baseline_normalize,args.weak_norm)

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')
subject_full = args.subjects_full.split(',')

action_test = args.test_action.split(',')
action_train = args.train_action.split(',')
all_action = args.all_action.split(',')


print('Start to fetch training 3d pose: ')

from time import time
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data(subjects, action_filter):
    train_pose_3d = []
    label = []
    j = np.zeros(1)
    for subject in subjects:
        #print('subject',subject)
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
            print('training action is',subject, action)
            # poses_2d = keypoints[subject][action]
            poses_3d = dataset[subject][action]['positions_3d']
            #out = []
            for i in range(len(poses_3d)):
                    # Remove global offset, but keep trajectory in first position
                    poses_3d[i][:,0] = 0
                    #plot17j(poses_3d[i][500:510],'a','b')
                    train_pose_3d.append(poses_3d[i])
                    # la = np.arange(0,poses_3d)
                    # label.append(la)
                    j += poses_3d[i].shape[0]
                    #pose = poses_3d[0]
    full_pose = np.concatenate(train_pose_3d, axis=0)
    #full_label = np.concatenate(label, axis=0)
    N = full_pose.shape[0]
    print('Total number:',N)
    part_pose = full_pose[::10]
    print('After filter, training data:',part_pose.shape)

    return part_pose


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / len(label)),
        #          fontdict={'weight': 'bold', 'size': 12})
        plt.scatter(data[i, 0], data[i, 1], 20, plt.cm.Set1(label[i] / len(label)))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def tsne():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(np.array(result), np.array(label),
                         't-SNE embedding of the S9/S11 test set (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)

def get_index(subjects, ra):
    """
    Usage: sort input poses by the distance to [mean pose] from train data
           sorted from large to small
    :param subjects: e.g. Test set
    :return: Reversed Index in the Test set
    """
    train_pose_3d = []
    for subject in subjects:
        #print('subject',subject)
        for action in dataset[subject].keys():
            #print('action',action)
            # poses_2d = keypoints[subject][action]
            poses_3d = dataset[subject][action]['positions_3d']
            #out = []
            for i in range(len(poses_3d)):
                    # Remove global offset, but keep trajectory in first position
                    poses_3d[i] -= poses_3d[i][:, :1]
                    if cal_mean:
                        mean_3d_1 = np.mean(poses_3d[i], axis=0)
                    elif cal_distance:
                        ext_mean_pose = np.repeat(mean_pose[np.newaxis, :, :], poses_3d[i].shape[0], axis=0)
                        assert ext_mean_pose.shape == poses_3d[i].shape
                        pose_dis = np.linalg.norm((ext_mean_pose - poses_3d[i]), axis=-1)
                        pose_dis_mean = np.mean(pose_dis, axis=-1)
                    #out.append(pose_dis_mean)
                    train_pose_3d.append(pose_dis_mean)
            #plot17j(out, subject, action, show_animation=False)

    full_pose = np.concatenate(train_pose_3d, axis=0)
    # Sorted from large to small distance
    sorted_index = np.argsort(-full_pose)
    full_pose.tolist()
    #sorted_dis = sorted(full_pose, reverse=True)
    #print('From large to small value:',sorted_dis)
    print('index',sorted_index)
    num = len(full_pose)
    print('Total pose:',num)
    ratio = ra
    pick_num = int(ratio*num)
    print('Picked number:',pick_num)
    pick_index = sorted_index[:pick_num]
    np.set_printoptions(threshold=np.inf)
    #print(pick_index)
    rerank = sorted(pick_index)
    print('rerank',len(rerank))
    return rerank

def find_mean(train_pose_3d):
    print('----------------Finish fetching training data-------------')
    N = 0
    sum_3d = np.zeros([17,3])
    # Calculte mean pose
    for i in range(len(train_pose_3d)):
        # pose.shape = [T,17,3]
        N += train_pose_3d[i].shape[0]
        print('N is :',N)
        train_pose_3d[i][:, 0] = 0
        mean_3d = np.mean(train_pose_3d[i], axis = 0)
        print('mm',mean_3d.shape,mean_3d)
        sum_3d += mean_3d * train_pose_3d[i].shape[0]

    mean1 = sum_3d / N
    #plot17j([mean], 'mean', 'mean')
    print('Training set mean 3d pose is:',np.mean(mean1,axis=-1))

def split_data(index):
    """
    Partition index into a list, make one more dimension
    :param index: a so long list
    :return out: splited index, type: List
    """
    out = []
    j = 0
    for i in index:
        if i < len(index)-1:
            if index[i+1] - index[i]>5:
                print('Split index into smaller groups:',j,i)
                out.append(index[j:(i+1)])
                j = i+1
        elif i==len(index)-1:
            out.append(index[j:])
    print('Split group:',len(out))
    return out

def svd(X):
    # Data matrix X, X doesn't need to be 0-centered
    #n, m = X.shape
    # Compute full SVD
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False,  # It's not necessary to compute the full matrix of U or V
                                 compute_uv=True)
    # Transform X with SVD components
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd


def cal_mean_var(action_filter):
    #    get_data()
    train_pose_3d = []
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
            #print(action)
            poses_3d = dataset[subject][action]['positions_3d']
            out = []
            for i in range(len(poses_3d)):
                poses_3d[i][:,0] = 0
                dis = np.linalg.norm(poses_3d[i], axis=-1)
                train_pose_3d.append(dis)
                # NUm = poses_3d[i].shape[0]
                # in_var = np.var((poses_3d[i]),axis=0)
                # joint_var = np.mean(in_var, axis=-1)
                # train_pose_3d.append(joint_var)
    full_pose = np.concatenate(train_pose_3d, axis=0)
    print(full_pose.shape)
    var_train = np.var(full_pose, axis=0)
    mean_train = np.mean(full_pose, axis=0)
    print(mean_train)
    print(var_train)

    #mean = find_mean(train_pose_3d)
    # dis = np.linalg.norm((test_mean-train_mean),axis=-1)
    # out = np.mean(dis,axis=0)
    # print(dis, out)

def pick_mean_var():
    action_all= args.all_action.split(',')
    for action in action_all:
        print('This action:',action)
        cal_mean_var([action])

def un_frequent(subjects, ra):
    train_pose = get_data(subjects_train,)
    train_pose = torch.from_numpy(train_pose)
    print(train_pose.shape)
    train_pose_3d = []
    for subject in subjects:
        for action in dataset[subject].keys():
            print('action',action)
        #     # poses_2d = keypoints[subject][action]
            poses_3d = dataset[subject][action]['positions_3d']
        #     #out = []
            for i in range(len(poses_3d)):
                    # Remove global offset, but keep trajectory in first position
                    poses_3d[i][:,0] = 0
                    for k in range(poses_3d[i].shape[0]):
                        time_now = time()
                        single_3d = poses_3d[i][k]
                        single_ext = torch.from_numpy(single_3d).unsqueeze(dim=0)
                        dis = torch.norm((single_ext.cuda()-train_pose.cuda()), dim=-1)
                        dis_ = torch.mean(dis)
                        dis_ = dis_.cpu().numpy()
                        #dis = np.mean(np.linalg.norm((train_pose-single_ext),axis=-1))
                        train_pose_3d.append(dis_)
                        #print(train_pose_3d)
                        print('Each frame cost:',time()-time_now)
                            # la = np.repeat(j[np.newaxis, :], poses_3d[i].shape[0], axis=0)
                            # label.append(la)
                        # j += 1
                    #pose = poses_3d[0]
    #test_distance = torch.cat(train_pose_3d, dim=0)
    test_distance = np.array(train_pose_3d)
    print('test',len(test_distance),test_distance,type(test_distance))
    print('Saving First...')
    file_name = 'data/unfrequent_test_distance'
    np.savez_compressed(file_name, pose_3d=test_distance)
    print('Done.')
    # full_label = np.concatenate(label, axis=0)
    # Sorted from large to small distance
    sorted_index = np.argsort(-test_distance)
    test_distance.tolist()
    #sorted_dis = sorted(full_pose, reverse=True)
    #print('From large to small value:',sorted_dis)
    #print('index',sorted_index)
    num = len(test_distance)
    #print('Total pose:',num)
    ratio = ra
    pick_num = int(ratio*num)
    print('Picked number:',pick_num)
    pick_index = sorted_index[:pick_num]
    np.set_printoptions(threshold=np.inf)
    rerank = sorted(pick_index)
    print('rerank number',len(rerank),rerank)
    return rerank


def get_distance(ra):
    input_dis = np.load('data/unfrequent_test_distance.npz', allow_pickle=True)
    test_distance = input_dis['pose_3d']
    print(test_distance.shape)
    sorted_index = np.argsort(-test_distance)
    test_distance.tolist()
    #sorted_dis = sorted(full_pose, reverse=True)
    #print('From large to small value:',sorted_dis)
    #print('index',sorted_index)
    num = len(test_distance)
    #print('Total pose:',num)
    ratio = ra
    pick_num = int(ratio*num)
    print('Picked number:',pick_num)
    pick_index = sorted_index[:pick_num]
    #np.set_printoptions(threshold=np.inf)
    #rerank = sorted(pick_index)
    print('rerank number',pick_index)
    return pick_index

def final_filter(subjects):
    full_out = []
    full_out_2d = []
    ratio = 1
    for subject in subjects:
        for action in dataset[subject].keys():
            poses_3d = dataset[subject][action]['positions_3d']
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_3d)):
                full_out.append(poses_3d[i])
                full_out_2d.append(poses_2d[i])
    full_pose = np.concatenate(full_out, axis=0)
    full_pose_2d = np.concatenate(full_out_2d, axis=0)
    #sorted_index = get_index(subjects, ra=ratio)
    #sorted_index = un_frequent(subjects, ra=ratio)
    sorted_index = get_distance(ratio)
    #split_index = split_data(sorted_index)
    sorted_pose = []
    sorted_pose_2d = []
    #for i, v in enumerate(split_index):
        #print('Each test group length:',len(v))
    sorted_pose.append(full_pose[sorted_index])
    sorted_pose_2d.append(full_pose_2d[sorted_index])
    #plot17j(sorted_pose[400:420], 'dis','large')

    print('Saving...')
    file_name = 'data/unfrequent_{}_test_gt'.format(ratio)
    np.savez_compressed(file_name, pose_3d=sorted_pose, pose_2d=sorted_pose_2d)
    print('Done.')

def sort_K(dist, K):
    # Sort from small to large distance
    dist = dist.cpu().numpy()
    sorted_index = np.argsort(dist)

    pick_index = sorted_index[:K]
    pick_dis = dist[pick_index]
    mean_dis =np.mean(pick_dis)
    return mean_dis
    
def pose_similar(dist, sigma):
    theta = torch.Tensor([sigma]).cuda()
    ps = torch.exp(-dist/theta)
    ps_mean = torch.mean(ps, dim=0, keepdim=True)
    #print(ps_mean.shape,ps_mean)
    return ps_mean

def filter_Kmin(get_KNN, subjects, action_filter, sig):
    # Filter K nearest pose of training data as the distance (represent the similarity)
    train_pose = get_data(subjects_train, all_action)
    train_pose = torch.from_numpy(train_pose)
    print(train_pose.shape)
    train_pose_3d = []
    part1_dis = []
    part2_dis = []
    part3_dis = []
    part4_dis = []
    part5_dis = []
    minu = []
    train_action = action_train
    for subject in subjects:
        # print('subject',subject)
        for action in keypoints[subject].keys():
            time_now = time()
            n = 0
            mean_min = 0
            mean_pose = 0
            mean_part = 0
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
            #print('test action is', subject, action)
            poses_3d = dataset[subject][action]['positions_3d']
            for i in range(len(poses_3d)):
                # Remove global offset, but keep trajectory in first position
                poses_3d[i][:,0] = 0
                for k in range(poses_3d[i].shape[0]):
                    if k%5==0:
                        n += 1
                        single_3d = poses_3d[i][k]
                        single_ext = torch.from_numpy(single_3d).unsqueeze(dim=0)
                        # For all joints mean distance:
                        pose_dis1 = torch.norm((single_ext.cuda()-train_pose.cuda()), dim=-1) #[N, 17]
                        # First calculate the joint similarity by Gaussian distribution normalize the joint distance.
                        pose_dis = pose_similar(pose_dis1, sigma=sig)

                        all_dis = torch.mean(pose_dis, dim=-1) #[N,]
                        # For part joints mean distance:
                        part1 = torch.mean(pose_dis[:, 0:4], dim=-1)
                        part2 = torch.mean(pose_dis[:, 4:7], dim=-1)
                        part3 = torch.mean(pose_dis[:, 7:11], dim=-1)
                        part4 = torch.mean(pose_dis[:, 11:14], dim=-1)
                        part5 = torch.mean(pose_dis[:, 14:17], dim=-1)
                        # Get the K Nearest mean distance
                        if get_KNN:
                            k_ = 5
                            pose_min = sort_K(all_dis, K=k_)
                            part1_min = sort_K(part1, K=k_)
                            part2_min = sort_K(part2, K=k_)
                            part3_min = sort_K(part3, K=k_)
                            part4_min = sort_K(part4, K=k_)
                            part5_min = sort_K(part5, K=k_)

                        # Get Mean pose similarity
                        # else:
                        #     pose_min = pose_similar(all_dis, sigma)
                        #     part1_min = pose_similar(part1, sigma)
                        #     part2_min = pose_similar(part2, sigma)
                        #     part3_min = pose_similar(part3, sigma)
                        #     part4_min = pose_similar(part4, sigma)
                        #     part5_min = pose_similar(part5, sigma)
                        #
                        # minus = pose_min - (4*part1_min+3*part2_min+4*part3_min+3*part4_min+3*part5_min)/17
                        # mean_min += minus
                        # mean_pose += pose_min
                        # mean_part += (part1_min + part2_min + part3_min + part4_min +part5_min)/5
                        # print(part1_min ,part2_min ,part3_min ,part4_min,part5_min,'mean',mean_part/n)
                        # train_pose_3d.append(pose_min)
                        # part1_dis.append(part1_min)
                        # part2_dis.append(part2_min)
                        # part3_dis.append(part3_min)
                        # part4_dis.append(part4_min)
                        # part5_dis.append(part5_min)
                        # minu.append(minus)
                        train_pose_3d.append(np.array(all_dis.cpu()))
                        part1_dis.append(np.array(part1.cpu()))
                        part2_dis.append(np.array(part2.cpu()))
                        part3_dis.append(np.array(part3.cpu()))
                        part4_dis.append(np.array(part4.cpu()))
                        part5_dis.append(np.array(part5.cpu()))
            print('N is',n)
            # print('Subject Action',subject, action, 'Each mean whole pose distance:',mean_pose/n*100, 'mean part',mean_part/n*100, 'mean more',mean_min/n*100)
            print('One calculation cost:',time()-time_now)
    test_distance = np.array(train_pose_3d)
    p1_dist = np.array(part1_dis)
    p2_dist = np.array(part2_dis)
    p3_dist = np.array(part3_dis)
    p4_dist = np.array(part4_dis)
    p5_dist = np.array(part5_dis)
    # minu_all = np.array(minu)
    print('test', len(test_distance))
    print('Saving First...')
    # file_name = 'data/K{}_test_5f_{}_distance'.format(k_,train_action)
    # np.savez_compressed(file_name, pose_3d_dist=test_distance, part1_dist=p1_dist, part2_dist=p2_dist,
    #                     part3_dist=p3_dist, part4_dist=p4_dist, part5_dist=p5_dist)
    file_name = 'data/S{}_rare_testset'.format(sig)
    np.savez_compressed(file_name, pose_3d_dist=test_distance, part1_dist=p1_dist, part2_dist=p2_dist,
                        part3_dist=p3_dist, part4_dist=p4_dist, part5_dist=p5_dist)
    print('Saved as:',file_name)
    print('Done.')

def fetch_data(k, sig):
    input_dis = np.load('data/S{}_rare_testset.npz'.format(sig), allow_pickle=True)
    test_distance = input_dis['pose_3d_dist']
    p1 = input_dis['part1_dist']
    p2 = input_dis['part2_dist']
    p3 = input_dis['part3_dist']
    p4 = input_dis['part4_dist']
    p5 = input_dis['part5_dist']
    return test_distance, p1, p2, p3, p4, p5

def sort_testset(test_distance, ra):
    # Sort from large to small distance in the test set
    sorted_index = np.argsort(test_distance[:,0])
    num = test_distance.shape[0]
    ratio = ra
    pick_num = int(ratio*num)
    print('Picked number:',pick_num)
    pick_index = sorted_index[:pick_num]
    print('rerank number',pick_index)
    return pick_index

def final_KN(subjects,sig):
    full_out = []
    full_out_2d = []
    ratio = 1
    k_ = 1
    filter_Kmin(False, subjects_test, all_action, sig)
    for subject in subjects:
        for action in dataset[subject].keys():
            print(action)
            poses_3d = dataset[subject][action]['positions_3d']
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_3d)):
                full_out.append(poses_3d[i][::5])
                full_out_2d.append(poses_2d[i][::5])
    full_pose = np.concatenate(full_out, axis=0)
    full_pose_2d = np.concatenate(full_out_2d, axis=0)
    print(full_pose_2d.shape)
    all_pose_dist, p1_dist,p2_dist,p3_dist,p4_dist,p5_dist = fetch_data(k_, sig)
    sorted_index = sort_testset(all_pose_dist, ra=ratio)

    sorted_pose = []
    sorted_pose_2d = []

    sorted_pose.append(full_pose[sorted_index])
    sorted_pose_2d.append(full_pose_2d[sorted_index])
    #plot17j(sorted_pose[400:420], 'dis','large')

    print('Saving...')
    file_name = 'data/whole_body_S{}_f5_{}_gt'.format(sig, ratio)
    np.savez_compressed(file_name, pose_3d=sorted_pose, pose_2d=sorted_pose_2d)
    print('Saved in:',file_name)

def cal_dist(subjects, action_filter):
    f = open('log/filter_direct.log')
    file = f.readlines()
    k = []
    g = []
    for subject in subjects:
        # print('subject',subject)
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
            for i, line in enumerate(file):
                if action in line:
                    word = line.split(' ')
                    if action == word[-1] or action == word[-2]:
                        print('action filter',subject,action,i)
                        k.append(i)
                        g.append([subject,action])
    print(len(k))
    sort_in = np.argsort(np.array(k))
    sort_i, sort_sub, sort_act = np.array(k)[sort_in], np.array(g)[sort_in][0], np.array(g)[sort_in][1]
    print(sort_i, sort_sub, sort_act)

if __name__ == '__main__':
    #final_filter(subjects_test)
    #filter_Kmin(False, subjects_test, all_action)
    final_KN(subjects_test, sig=0.2)
    #cal_dist(subjects_test,action_test)
