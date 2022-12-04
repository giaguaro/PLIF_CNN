import os
from os.path import exists
import builtins
import shutil
import time
import warnings
import argparse
import math
import numpy as np
import time
from datetime import datetime
import random
from sklearn import metrics
import matplotlib.pylab as plt
import pandas as pd
import pickle
import glob
from sklearn.metrics import *
from scipy.stats import *
from torch.utils.tensorboard import SummaryWriter
import subprocess
import gc
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='cnn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-file', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


class Net(nn.Module):
    # input size - the number of "classes"
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(94, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=0),
            #nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=8, stride=2))
        self.fc0 = nn.Linear(746496,1024)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("in",x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)

        out = out.reshape(out.size(0), -1)

        out = self.fc0(out)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
       # out = self.sigmoid(out)
        #print(out.type())
        return out

class CNNDataLoader(Dataset):

    def __init__(self, data):
        """
        Args:
            input_pickle (string): Directory with to pickle file processed tensor data
            master_file (string): Path to the master csv file with annotations. Column 'kd\ki' has labels.
        """

        self.data = data



    def __len__(self):
           return len(self.data)

    def __getitem__(self, idx):

        grids_path, label_path = self.data[idx]

        with open(label_path,'rb') as f:
            label = pickle.load(f)

        with open(grids_path,'rb') as f:
            grid = pickle.load(f)

        #torch.unsqueeze(grid, dim=0)
        a_grid = grid[0].to_dense().half() if grid.shape==(1, 200, 200, 200, 94) else grid.to_dense().half()
        try: a_label = torch.tensor(-1 * math.log(label[0]))
        except: a_label = torch.tensor(-1 * math.log(label))

        return a_grid, a_label


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # slurm available
    import os
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url." + jobid  + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = Net()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = args.workers # int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay, eps=1e-08)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    
    
    os.chdir("/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/")
    with open("train_data.pkl",'rb') as f:
        train_data=pickle.load(f)
    with open("validate_data.pkl",'rb') as f:
        validate_data=pickle.load(f)
    with open("test_data.pkl",'rb') as f:
        test_data=pickle.load(f)

    random.Random(4).shuffle(validate_data)
    train_data, validate_data =train_data+validate_data[:5000], validate_data[5000:]

## test the model productivity ##
    train_data, validate_data = train_data[:1000], validate_data[:1000]
#################################

    train_dataset = CNNDataLoader(train_data)
    
    validate_dataset = CNNDataLoader(validate_data)
    val_sampler = None

    test_dataset = CNNDataLoader(test_data)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None


    #Initiate the dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=10, pin_memory=True, sampler=train_sampler)
    
    val_loader = DataLoader(validate_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=True)
    
    testing_loader = DataLoader(test_dataset, pin_memory=False, batch_size=2, shuffle=True, num_workers=args.workers)
    

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        print("starting training ..")
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('R2', ':.2ff')
    #progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top,
    #                         prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for step, (inp,target) in enumerate(train_loader):
        print("step: ", step)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.view(target.shape[0], 1)
        inp = inp.view(inp.shape[0],-1,200,200,200)

        if args.gpu is not None:
            inp = inp.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with autocast():
            output = model(inp)
        loss = criterion(output.float(), target.float())

        # measure R2 and record loss
        ytrue = target.detach().cpu().float().data.numpy()
        ypred = output.detach().cpu().float().data.numpy()

        try:
            r2 = r2_score(ytrue, ypred)
            print(r2)
        except:
            print("nan values in ypred")

        losses.update(loss.item(), inp.size(0))
        top.update(r2, inp.size(0))

        #torch.cuda.empty_cache()

        # compute gradient and do ADAM step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(retain_graph=False)
        scaler.step(optimizer)
        scaler.update()

        #torch.cuda.empty_cache()  #decrease about 2GB
        #del loss  # nothing change
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if step % args.print_freq == 0:
        #    progress.print(step)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('R2', ':.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, r2,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (inp, target) in enumerate(val_loader):

            target = target.view(target.shape[0], 1)
            inp = inp.view(inp.shape[0],-1,200,200,200)

            if args.gpu is not None:
                inp = inp.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with autocast():
                output = model(inp)
            loss = criterion(output, target)

            # measure R2 and record loss
            ytrue = target.detach().cpu().float().data.numpy()
            ypred = output.detach().cpu().float().data.numpy()

            print(ytrue, ypred)
            r2 = r2_score(ytrue, ypred)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % args.print_freq == 0:
                progress.print(step)

        # TODO: this should also be done with the ProgressMeter
        print(' * R2 {top.avg:.3f} '
              .format(top=top))

    return top.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def R2(output, target):
    """Computes the R2"""
    with torch.no_grad():
 
        ytrue = target.float().data.numpy()
        ypred = target.detach().cpu().float().data.numpy()        

        print(ytrue_arr, ypred_arr)
        r2 = r2_score(ytrue_arr, ypred_arr) 


        return r2


if __name__ == '__main__':
    main()
