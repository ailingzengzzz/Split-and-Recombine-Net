import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# This code is used for plot the detailed test errors with each frame for each model.
# I have made these four picture in one picture

input_1 = open('log/test_lcn.log','r')
# input_2 = open('log/eval_gp5_mul.log','r')
# input_3 = open('log/eval_gp5_add.log','r')
# input_4 = open('log/eval_fc_l8_bone0.1.log','r')
# input_5 = open('log/eval_gp5_mul_bone1.log','r')
# input_6 = open('log/eval_gp5_mul_bone0.11.log','r')
# input_7 = open('log/eval_gp5_mul_bone0.01.log','r')
# input_8 = open('log/eval_gp5_add_bone0.01.log','r')
# input_9 = open('log/1210_gp1_test.log','r')
# input_10 = open('log/1210_gp2_test.log','r')
# input_5 = open('log/1210_gp10_test.log','r')
# input_6 = open('log/1210_gp3_test.log','r')
# input_7 = open('log/1210_gp4_test.log','r')
# input_8 = open('log/1210_gp5_test.log','r')
#input_8 = open('log/1013_eva_epoch_20_all_in_bn_nbias_gp2.log','r')
#Training_result means the accuracy of each frame in testset.
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
Test_result_9 = []
Test_result_10 = []
Test_result_11 = []
Test_result_12 = []
Test_result_13 = []
Test_result_14 = []
Test_result_15 = []
Test_result_16 = []

# lines_3 = input_3.readlines()
# lines_last_3 = lines_3[-16:-1]
# lines_last_3.append(lines_3[-1])
# for v in lines_last_3:
#     v = v.split()
#     Test_result_3.append(float(v[-1]))
#

def average(*args):
    l = len(args)
    sum = 0
    if l==0:
        return 0.0
    i = 0
    while i < l:
        sum += args[i]
        i += 1
    return sum*1.0/l

# Group=2,1rd Group: average(float(word[4][:8]),float(word[7][:8]),float(word[10][:8]),float(word[13][:8]),float(word[16][:8]),float(word[19][:8]),float(word[22][:8]),float(word[25][:8])) float(word[28][:8]),
# Group=2,2rd Group:average(float(word[28][:8]),float(word[31][:8]),float(word[34][:8]),float(word[37][:8]),float(word[40][:8]),float(word[43][:8]),float(word[46][:8]),float(word[49][:8]),float(word[52][:8])))

# Group=3; 1rd Group: average(float(word[4][:8]),float(word[7][:8]),float(word[10][:8]),float(word[13][:8]),float(word[16][:8]),float(word[19][:8]),float(word[22][:8])))
# Group=3; 2rd Group: average(float(word[25][:8]),float(word[28][:8]),float(word[31][:8]),float(word[34][:8])))
# Group=3; 3rd Group: average(float(word[37][:8]),float(word[40][:8]),float(word[43][:8]),float(word[46][:8]),float(word[49][:8]),float(word[52][:8])))
a=14
for line in input_1:
    if 'Bone length' in line:
        word = line.split(' ')
        Test_result_1.append(float(word[5][:5]))
        Test_result_2.append(float(word[7][:5]))
        Test_result_3.append(float(word[10][:5]))
        Test_result_4.append(float(word[13][:5]))
        Test_result_5.append(float(word[17][:5]))
        Test_result_6.append(float(word[20][:5]))
        Test_result_7.append(float(word[23][:5]))
        Test_result_8.append(float(word[26][:5]))
        Test_result_9.append(float(word[29][:5]))
        Test_result_10.append(float(word[32][:5]))
        Test_result_11.append(float(word[35][:5]))
        Test_result_12.append(float(word[38][:5]))
        Test_result_13.append(float(word[41][:5]))
        Test_result_14.append(float(word[44][:5]))
        Test_result_15.append(float(word[47][:5]))
        Test_result_16.append(float(word[50][:5]))

# calculate mean/variance
cal = Test_result_1
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_2
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_3
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_4
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_5
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_6
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_7
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_8
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_9
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_10
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_11
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_12
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_13
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_14
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_15
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)
cal = Test_result_16
part_mean = np.mean(cal)
part_var = np.std(cal)
print(part_mean,part_var)


# for line in input_2:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_2.append(float(word[7+a][:6]))
#
# for line in input_3:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_3.append(float(word[7+a][:6]))
#
# for line in input_4:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_4.append(float(word[7+a][:6]))
#
# for line in input_5:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_5.append(float(word[7+a][:6]))
# for line in input_6:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_6.append(float(word[7+a][:6]))
#
# for line in input_7:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_7.append(float(word[7+a][:6]))
# for line in input_8:
#     if 'Action bone' in line:
#         word = line.split(' ')
#         Test_result_8.append(float(word[7+a][:6]))
# #
# for line in input_9:
#     if 'Hip(root)' in line:
#         word = line.split(' ')
#         Test_result_9.append(average(float(word[4][:8]),float(word[7][:8]),float(word[10][:8]),float(word[13][:8]),float(word[16][:8]),float(word[19][:8]),float(word[22][:8]),float(word[25][:8])))
#
# for line in input_10:
#     if 'Hip(root)' in line:
#         word = line.split(' ')
#         Test_result_10.append(average(float(word[4][:8]),float(word[7][:8]),float(word[10][:8]),float(word[13][:8]),float(word[16][:8]),float(word[19][:8]),float(word[22][:8]),float(word[25][:8]),float(word[28][:8])))
fig = plt.figure()
plt.title("all subjects data in human3.6M")
plt.plot(Test_result_1,'r-x', label = 'RHip')
plt.plot(Test_result_2,'b-x', label = 'URLeg')
plt.plot(Test_result_3,'g-x', label = 'LRLeg')
plt.plot(Test_result_4,'y-x', label = 'Lhip')
plt.plot(Test_result_5,'m-x', label = 'ULLeg')
plt.plot(Test_result_6,'c-x', label = 'LLleg')
plt.plot(Test_result_7,'k-x', label = 'Lspine')
plt.plot(Test_result_8,'tab:pink', label = 'Uspine')
plt.plot(Test_result_9,'tab:blue', label = 'Neck')
plt.plot(Test_result_10,'tab:green', label = 'Head')
plt.plot(Test_result_11,'c-^', label = 'Rshoulder')
plt.plot(Test_result_12,'k-^', label = 'URelbow')
plt.plot(Test_result_13,'r-^', label = 'LRelbow')
plt.plot(Test_result_14,'b-^', label = 'Lshoulder')
plt.plot(Test_result_15,'g-^', label = 'ULelbow')
plt.plot(Test_result_16,'y-^', label = 'LLelbow')


plt.legend()
plt.grid(True)
plt.xlabel('each subaction-person')
plt.ylabel('Bone length/cm')
plt.tight_layout()
# my_x_ticks = np.arange(0,2181,10)
# #my_y_ticks = np.arange(0,300,10)
# plt.xticks(my_x_ticks)
# #plt.yticks(my_y_ticks)
#
# ax_2 = plt.subplot(212)
# ax_2.set_title('Test error for each action')
# plt.plot(Test_result_1,'r-x', label = 'p2c12_f4_nfeat128_ep11')
# plt.plot(Test_result_2,'b-x', label = 'p2c16_f4_nfeat128_ep20')
# plt.plot(Test_result_3,'g-x', label = 'p2c12_f5_nfeat128_ep18')
# plt.plot(Test_result_4,'y-x', label = 'p2c4_test_ep17')
# plt.plot(Test_result_5,'m-x', label = 'p2c16_f4_nfeat128_ep15_best')
# plt.plot(Test_result_6,'c-x', label = 'p2c4_f4_nfeat 256_ep19_best')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('MPMJE/mm')
# scale_ls = range(16)
# #index_ls = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown','Smoking','Waiting','WalkDog','WalkTogether','Walking','Mean error']
# index_ls = ['Greeting','Sitting','SittingDown','WalkTogether','Phoning','Posing','WalkDog','Walking','Purchases','Waiting','Directions','Smoking','Photo','Eating','Discussion','Average']
# plt.xticks(scale_ls,index_ls)

plt.show()
