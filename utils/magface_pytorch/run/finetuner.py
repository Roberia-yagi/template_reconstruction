# https://github.com/IrvingMeng/MagFace
#!/usr/bin/env python
import sys
sys.path.append("..")
from collections import OrderedDict
from dataloader import dataloader
import torch.nn as nn
from models import magface
from utils import utils
import numpy as np
from termcolor import cprint
import datetime
from inference.network_inf import builder_inf
import torch.nn.functional as F
import torch
import argparse
import warnings
import time
import pprint
import os


warnings.filterwarnings("ignore")


# parse the args
cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for Magface')
parser.add_argument('--arch', default='iresnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--train_list', default='', type=str,
                    help='')

parser.add_argument("--gpu_idx", type=int, default=0, help="index of cuda devices")

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=85742, type=int,
                    help='The num of last fc layers for using softmax')
parser.add_argument('--pretrained_path', default='/home/akasaka/nas/models/magface_epoch_00025.pth',
                    type=str, help='path to pretrained model')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60, 90], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pth-save-fold', default='tmp', type=str,
                    help='The folder to save pths')
parser.add_argument('--pth-save-epoch', default=1, type=int,
                    help='The epoch to save pth')


# magface parameters
parser.add_argument('--l_a', default=10, type=float,
                    help='lower bound of feature norm')
parser.add_argument('--u_a', default=110, type=float,
                    help='upper bound of feature norm')
parser.add_argument('--l_margin', default=0.45,
                    type=float, help='low bound of margin')
parser.add_argument('--u_margin', default=0.8, type=float,
                    help='the margin slop for m')
parser.add_argument('--lambda_g', default=20, type=float,
                    help='the lambda for function g')
parser.add_argument('--arc-scale', default=64, type=int,
                    help='scale for arcmargin loss')
parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def main(args):
    global device
    global save_path

    # Decide device
    device = f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu"

    # device = get_freer_gpu()

    # check the feasible of the lambda g
    s = 64
    k = (args.u_margin-args.l_margin)/(args.u_a-args.l_a)
    min_lambda = s*k*args.u_a**2*args.l_a**2/(args.u_a**2-args.l_a**2)
    color_lambda = 'red' if args.lambda_g < min_lambda else 'green'
    cprint('min lambda g is {}, currrent lambda is {}'.format(
        min_lambda, args.lambda_g), color_lambda)

    cprint('=> torch version : {}'.format(torch.__version__), 'green')
    ngpus_per_node = torch.cuda.device_count()
    cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')

    # time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # save_path = os.path.join(args.pth_save_fold, time_stamp)
    os.makedirs(args.pth_save_fold, exist_ok=True)

    main_worker(ngpus_per_node, args)

def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    return _state_dict

def load_dict_inf(args, model):
    if os.path.isfile(args.pretrained_path):
        cprint('=> loading pth from {} ...'.format(args.pretrained_path))
        checkpoint = torch.load(args.pretrained_path)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args.resume))
    return model

def main_worker(ngpus_per_node, args):
    global best_acc1

    cprint('=> modeling the network ...', 'green')
    model = magface.builder(args).to(device)
    model = load_dict_inf(args, model)
    for param in model.parameters():
        param.requires_grad = False

    # replace layers for small features
    model.features.fc = nn.Linear(in_features=model.features.fc.in_features, out_features=128, bias=True)
    model.features.features = nn.BatchNorm1d(128, eps=model.features.features.eps, momentum=0.9, affine=True, track_running_stats=True)
    model.fc = magface.MagLinear(in_features=128, out_features=args.last_fc_size)

    # for name, param in model.named_parameters():
    #     cprint(' : layer name and parameter size - {} - {}'.format(name, param.size()), 'green')
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    model.to(device)

    cprint('=> building the oprimizer ...', 'green')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, params_to_update),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    pprint.pprint(optimizer)

    cprint('=> building the dataloader ...', 'green')
    train_loader = dataloader.train_loader(args)

    cprint('=> building the criterion ...', 'green')
    criterion = magface.MagLoss(args.l_a, args.u_a, args.l_margin, args.u_margin)

    global iters
    iters = 0

    cprint('=> starting training engine ...', 'green')
    for epoch in range(args.start_epoch, args.epochs):

        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0:
            state_dict = model.state_dict()

            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, False,
                filename=os.path.join(
                args.pth_save_fold, '{}.pth'.format(
                    str(epoch+1).zfill(5))
            ))
            cprint(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    throughputs = utils.AverageMeter('ThroughPut', ':.2f')

    losses_id = utils.AverageMeter('L_ID', ':.3f')
    losses_mag = utils.AverageMeter('L_mag', ':.6f')
    progress_template = [batch_time, data_time, throughputs, 'images/s',
                         losses, losses_id, losses_mag, 
                         top1, top5, learning_rate]

    progress = utils.ProgressMeter(
        len(train_loader),
        progress_template,
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1

        # Original
        # input = input.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        input = input.to(device)
        target = target.to(device)

        # compute output
        output, x_norm = model(input, target)

        loss_id, loss_g, one_hot = criterion(output, target, x_norm)
        loss = loss_id + args.lambda_g * loss_g

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(args, output[0], target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(args.lambda_g*loss_g.item(), input.size(0))

        # compute gradient and do solver step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        duration = time.time() - end
        batch_time.update(duration)
        end = time.time()
        throughputs.update(args.batch_size / duration)

        if i % args.print_freq == 0:
            progress.display(i)
            debug_info(x_norm, args.l_a, args.u_a,
                           args.l_margin, args.u_margin)

        if args.vis_mag:
            if (i > 10000) and (i % 100 == 0):
                x_norm = x_norm.detach().cpu().numpy()
                cos_theta = torch.masked_select(
                    output[0], one_hot.bool()).detach().cpu().numpy()
                logit = torch.masked_select(
                    F.softmax(output[0]), one_hot.bool()).detach().cpu().numpy()
                np.savez('{}/vis/epoch_{}_iter{}'.format(args.pth_save_fold, epoch, i),
                         x_norm, logit, cos_theta)

def debug_info(x_norm, l_a, u_a, l_margin, u_margin):
    """
    visualize the magnitudes and magins during training.
    Note: modify the function if m(a) is not linear
    """
    mean_ = torch.mean(x_norm).detach().cpu().numpy()
    max_ = torch.max(x_norm).detach().cpu().numpy()
    min_ = torch.min(x_norm).detach().cpu().numpy()
    m_mean_ = (u_margin-l_margin)/(u_a-l_a)*(mean_-l_a) + l_margin
    m_max_ = (u_margin-l_margin)/(u_a-l_a)*(max_-l_a) + l_margin
    m_min_ = (u_margin-l_margin)/(u_a-l_a)*(min_-l_a) + l_margin
    print('  [debug info]: x_norm mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(mean_, min_, max_))
    print('  [debug info]: margin mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(m_mean_, m_min_, m_max_))


if __name__ == '__main__':

    pprint.pprint(vars(args))
    main(args)
