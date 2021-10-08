# This code is used for plot the training/test errors with each epoch for each model.
# I have made these four picture in one picture

import matplotlib
import numpy as np
import matplotlib.pyplot as plt



# input_1 = open('log/1031_cpn_l1.log','r')
# input_2 = open('log/1031_cpn_lr_l1.log','r')
# input_3 = open('log/1031_cpn_lr_l2.log','r')
# input_4 = open('log/1031_cpn_lr_l1_243.log','r')
# input_5 = open('log/1031_hg_l2.log','r')
# input_6 = open('log/1031_hg_l1.log','r')
# input_7 = open('log/1031_hg_lr_l1.log','r')
# input_8 = open('log/1031_hg_lr_l2.log','r')
# input_9 = open('log/1101_cpn_l2_243.log','r')
# input_10 = open('log/1101_cpn_l1_lr002_243.log','r')

input_1 = open('log/r_100_gp5_lr1e-3.log','r')
input_2 = open('log/r_15_gp5.log','r')
# input_3 = open('log/test_3dpw_gp3_t243.log','r')
# input_4 = open('log/test_3dpw_fc2.log','r')
#input_4 = open('log/test_3dpw_t3_v3.log','r')

# input_5 = open('log/1019_ori_33333_gt_gp2.log','r')
# input_6 = open('log/1101_mask_l2_ori_243.log','r')
# input_7 = open('log/1101_hg_l2_ori_243.log','r')
# input_8 = open('log/1101_mask_l1_lr002_243.log','r')
# input_9 = open('log/1101_cpn_l2_243.log','r')
# input_10 = open('log/1101_cpn_l1_lr002_243.log','r')

Training_result_1 = []
Test_result_1 = []
Training_result_2 = []
Test_result_2 = []
Training_result_3 = []
Test_result_3 = []
Training_result_4 = []
Test_result_4 = []
Training_result_5 = []
Test_result_5 = []
Training_result_6 = []
Test_result_6 = []
Training_result_7 = []
Test_result_7 = []
Training_result_8 = []
Test_result_8 = []
Training_result_9 = []
Test_result_9 = []
Test_result_10 = []
Test_result_11 = []
Test_result_12 = []
Test_result_13 = []
Test_result_14 = []
Test_result_15 = []
Test_result_16 = []
Test_result_17 = []
Test_result_18 = []
Test_details = []
i = []
j = 0
for line in input_1:
    if '3d_valid' in line:
        word = line.split(' ')
        Test_result_1.append(float(word[-5][:5]))
    # if 'aligned_pck' in line:
    #     word = line.split(' ')
        Test_result_2.append(float(word[-1][:5]))
    # if 'previous 9' in line:
    #     word = line.split(' ')
    #     Test_result_1.append(float(word[6][7:12]))
    # if 'after 8' in line:
    #     word = line.split(' ')
    #     Test_result_2.append(float(word[6][7:12]))
    # if 'Hip(root)' in line:
    #     word = line.split(' ')
    #     Test_result_3.append(float(word[7][0:4]))
    #     Test_result_4.append(float(word[10][0:4]))
    #     Test_result_5.append(float(word[13][0:4]))
    #     Test_result_6.append(float(word[16][0:4]))
    #     Test_result_7.append(float(word[19][0:4]))
    #     Test_result_8.append(float(word[22][0:4]))
    #     Test_result_9.append(float(word[25][0:5]))
    #     Test_result_10.append(float(word[28][0:4]))
    #     Test_result_11.append(float(word[31][0:4]))
    #     Test_result_12.append(float(word[34][0:4]))
    #     Test_result_13.append(float(word[37][0:4]))
    #     Test_result_14.append(float(word[40][0:4]))
    #     Test_result_15.append(float(word[43][0:3]))
    #     Test_result_16.append(float(word[46][0:4]))
    #     Test_result_17.append(float(word[49][0:4]))
#     #     Test_result_18.append(float(word[52][0:4]))
for line in input_2:
    # if '3d_train' in line:
    #     word = line.split(' ')
    #     Test_result_2.append(float(word[-1]))
    if '3d_valid' in line:
        word = line.split(' ')
        Test_result_3.append(float(word[-5][:5]))
    # if 'aligned_pck' in line:
    #     word = line.split(' ')
        Test_result_4.append(float(word[-1][:5]))
#
# for line in input_3:
#     # if '3d_train' in line:
#     #     word = line.split(' ')
#     #     Test_result_2.append(float(word[-1]))
#     if 'mean pck' in line:
#         word = line.split(' ')
#         Test_result_5.append(float(word[-2][:5]))
#     # if 'aligned_pck' in line:
#     #     word = line.split(' ')
#         Test_result_6.append(float(word[-1][:5]))
# #
# #
# for line in input_4:
#     if 'mean pck' in line:
#         word = line.split(' ')
#         Test_result_7.append(float(word[-2][:5]))
#     # if 'aligned_pck' in line:
#     #     word = line.split(' ')
#         Test_result_8.append(float(word[-1][:5]))
    # if '3d_train' in line:
    #     word = line.split(' ')
    #     Test_result_4.append(float(word[-1]))
#
# for line in input_5:
#     if '3d_train' in line:
#         word = line.split(' ')
#         Test_result_5.append(float(word[-1]))
#
# for line in input_6:
#     if '3d_train' in line:
#         word = line.split(' ')
#         Test_result_6.append(float(word[-1]))
# #
# for line in input_7:
#     if '3d_train' in line:
#     #     Training_result_7.append(float(line[36:46]))
#     # # elif '3d_valid' in line:
#         word = line.split(' ')
#         Test_result_7.append(float(word[-1]))
#
# for line in input_8:
#     if '3d_train' in line:
#         word = line.split(' ')
#         Test_result_8.append(float(word[-1]))
# #
# for line in input_9:
#     if '3d_train' in line:
#         word = line.split(' ')
#         Test_result_9.append(float(word[-1]))
#
# for line in input_10:
#     if '3d_train' in line:
#         word = line.split(' ')
#         Test_result_10.append(float(word[-1]))
#

fig = plt.figure()
plt.title('Train and Test on 3dpw testset(24 scenes with 37 people) Model')
# ax_1 = plt.subplot(311)
# ax_1.set_title("matte input with 4 channels")
#plt.plot(Training_result_1,'r-x', label = 'Training error of origin_256_384/mm')
plt.plot(Test_result_1,'r-^', label = 'Monocular setup-gp5-Training L1 loss - finetune 100epoch by LR=1e-4')
# plt.plot(Training_result_2,'b-x', label = 'Training error of origin_256_1024/mm')
plt.plot(Test_result_2,'r-x', label = 'Monocular setup-gp5-Test error/mm')
# # plt.plot(Training_result_3,'g-x', label = 'Training error of origin_1024_1024/mm')
plt.plot(Test_result_3,'g-^', label = 'Monocular setup-gp5-Training L1 loss, finetune 15epoch by LR=5e-4')
plt.plot(Test_result_4,'g-x', label = 'Temporal setup-gp5-Test error/mm')
# plt.plot(Test_result_5,'b-^', label = 'Temporal setup-gp3-Test error of AUC/mm')
# # # # #plt.plot(Training_result_5,'m-x', label = 'Training error of dcn_1024_1024/mm')
# # plt.plot(Test_result_4,'b-^', label = 'Test error of ori gp1/mm')
# plt.plot(Test_result_6,'b-x', label = 'Temporal setup-gp3-Test error of aligned AUC/mm')
# plt.plot(Test_result_7,'k-^', label = 'Monocular setup-Video3d-Test error of AUC/mm')
# plt.plot(Test_result_8,'k-x', label = 'Monocular setup-Video3d-Test error of aligned AUC/mm')
# plt.plot(Test_result_9,'tab:orange', label = 'Test error of Lfoot/mm')
# plt.plot(Test_result_10,'tab:blue', label = 'Test error of Spine')
#
# plt.plot(Test_result_11,'y-x', label = 'Training error of Thorax/mm')
# plt.plot(Test_result_12,'k-x', label = 'Training error of Neck/mm')
# plt.plot(Test_result_13,'m-x', label = 'Training error of Head/mm')
# plt.plot(Test_result_14,'r-x', label = 'Training error of Lshoulder/mm')
# plt.plot(Test_result_15,'b-x', label = 'Training error of Lelbow/mm')
# plt.plot(Test_result_16,'g-x', label = 'Training error of Lwrist/mm')
# plt.plot(Test_result_15,'c-x', label = 'Training error of Rshoulder/mm')
# plt.plot(Test_result_16,'y-^', label = 'Training error of Relbow/mm')
# plt.plot(Test_result_17,'tab:green', label = 'Training error of Rwrist/mm')

# #plt.plot(Training_result_7,'k-x', label = 'Training error of refine_dcn_1024/mm')

plt.legend()
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('MPMJE/mm')
# my_x_ticks = np.arange(0,20,1)
# my_y_ticks = np.arange(0,70,10)
# plt.xticks(my_x_ticks)git
# plt.yticks(my_y_ticks)
#
# ax_2 = plt.subplot(312)
# ax_2.set_title('rgb2matte input with 4 channels')
# plt.plot(Training_result_2,'r-x', label = 'Training error/mm')
# plt.plot(Test_result_2,'b-^', label = 'Test error/mm')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('MPMJE/mm')
# my_x_ticks = np.arange(0,20,1)
# my_y_ticks = np.arange(0,70,10)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
#
# ax_3 = plt.subplot(313)
# ax_3.set_title('rgb input with 12 channels')
# plt.plot(Training_result_3,'r-x', label = 'Training error/mm')
# plt.plot(Test_result_3,'b-^', label = 'Test error/mm')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('MPMJE/mm')
# my_x_ticks = np.arange(0,20,1)
# my_y_ticks = np.arange(0,70,10)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)

# ax_4 = plt.subplot(224)
# ax_4.set_title('p2c4_frame5')
# plt.plot(Training_result_4,'r-x', label = 'Training error/mm')
# plt.plot(Test_result_4,'b-^', label = 'Test error/mm')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('MPMJE/mm')
# my_x_ticks = np.arange(0,20,1)
# my_y_ticks = np.arange(0,120,10)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
plt.tight_layout()
plt.show()