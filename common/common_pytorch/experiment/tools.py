import torch

import os
import sys
sys.path.append("../../..")

from common.arguments.basic_args import parse_args
args = parse_args()

from tensorboardX import SummaryWriter
tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.model_name))

def count_params(model, ):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_rootfolder():
    ####### Create log and model folder #######3
    folders_util = [args.root_log, args.checkpoint,
                    os.path.join(args.root_log, args.model_name),
                    os.path.join(args.checkpoint, args.model_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder: '+folder)
            os.mkdir(folder)

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def print_result(epoch, elapsed, lr, losses_3d_train, losses_3d_train_eval, losses_3d_valid):
    if args.no_eval:
        print('[%d] time %.2f lr %f 3d_train %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1]))
    else:
        output = ('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1],
            losses_3d_train_eval[-1],
            losses_3d_valid[-1]))

        tf_writer.add_scalar('loss/valid', losses_3d_train_eval[-1], epoch + 1)
        tf_writer.add_scalar('loss/test', losses_3d_valid[-1], epoch + 1)
    tf_writer.add_scalar('lr', lr, epoch + 1)
    tf_writer.add_scalar('loss/train', losses_3d_train[-1], epoch + 1)
    print(output)

def save_model(losses_3d_train, losses_3d_train_eval, losses_3d_valid, train_generator, optimizer, model_pos_train, epoch, lr, Best_model=False):
    if epoch % args.checkpoint_frequency == 0:
        chk_path = os.path.join(args.checkpoint, 'latest_epoch_{}.bin'.format(args.model_name))
        print('Saving checkpoint to', chk_path)
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'loss 3d train': losses_3d_train[-1],
            'loss 3d eval': losses_3d_train_eval[-1],
            'loss 3d test': losses_3d_valid[-1],
            'random_state': train_generator.random_state(),
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict(),
        }, chk_path)
    if Best_model:
        print('Best model got in epoch', epoch, ' with test error:', losses_3d_valid[-1])
        best_path = os.path.join(args.best_checkpoint, 'model_best' + args.model_name + '.bin')
        print('Saving best checkpoint to', best_path)
        out = 'Best model got in epoch {0} \n with test error: {1} \n Saving best checkpoint to{2}'.format(epoch+1, losses_3d_valid[-1], best_path)
        print(out)
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'loss 3d train': losses_3d_train[-1] * 1000,
            'loss 3d eval': losses_3d_train_eval[-1] * 1000,
            'loss 3d test': losses_3d_valid[-1],
            'random_state': train_generator.random_state(),
            'optimizer': optimizer.state_dict(),
            'model_pos': model_pos_train.state_dict(),
        }, best_path)
