import torch

def get_input(group):
    if group == 2:
        print('Now group is:', group)
        conv_seq = [range(0, 16), [0, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]
        final_outc = 55
    elif group == 3:
        print('Now group is:', group)
        conv_seq = [range(0, 14), [0, 1, 14, 15, 16, 17, 18, 19, 20, 21],
                         [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]
        final_outc = 58
    elif group == 5:
        print('Now group is:', group)
        conv_seq = [range(0, 8), [0, 1, 8, 9, 10, 11, 12, 13], [0, 1, 14, 15, 16, 17, 18, 19, 20, 21],
                         [0, 1, 22, 23, 24, 25, 26, 27], [0, 1, 28, 29, 30, 31, 32, 33]]
        final_outc = 64
    elif group == 1:
        print('Now group is:', group)
        conv_seq = [range(0, 34)]
        final_outc = 51
    else:
        raise KeyError('Invalid group number!')

    return conv_seq, final_outc

# #
def shrink_output(x):
    num_joints_out = x.shape[-1]
    pose_dim = 3 # means [X,Y,Z]: three values
    if num_joints_out == 1:
        x = x[:, :, :pose_dim]
    elif num_joints_out == 64: #Group = 5
        x = torch.cat([x[:, :, :(4*pose_dim)], x[:, :, (5*pose_dim):(8*pose_dim)], x[:, :, (9*pose_dim):(13*pose_dim)], x[:, :, (14*pose_dim):(17*pose_dim)], x[:, :, (18*pose_dim):(21*pose_dim)]], dim=-1)

    elif num_joints_out == 58: #Group = 3
        x = torch.cat([x[:, :, :(7*pose_dim)], x[:, :, (8*pose_dim):(12*pose_dim)], x[:, :, (13*pose_dim):(19*pose_dim)]], dim=-1)

    elif num_joints_out == 55: #Group = 2
        x = torch.cat([x[:, :, :(8*pose_dim)], x[:, :, (9*pose_dim):(18*pose_dim)]], dim=-1)

    elif num_joints_out == 52: #Group = 1
        x = x[:, :, :(17*pose_dim)]
    else:
        raise KeyError('Invalid outputs!')
    return x