import sys
sys.path.append("..")
from dataloader import dataloader
import torch
import argparse

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
train_loader = dataloader.train_loader(args)
for batch, labels in train_loader:
    print(batch)
    print(type(batch))
    print(f'max value of image is {torch.max(batch)}')
    print(f'min value of image is {torch.min(batch)}')
    print(f'mean of image is {torch.mean(batch)}')
    print(f'var of image is {torch.var(batch)}')
    break