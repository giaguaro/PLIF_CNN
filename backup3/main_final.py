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

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

clip = 5
best_r2 = 0.0

class Net(nn.Module):
    # input size - the number of "classes"
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(628, 157, kernel_size=1, stride=1),
            nn.BatchNorm3d(157),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(157, 157, kernel_size=3, stride=1),
            nn.BatchNorm3d(157),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(157, 157, kernel_size=3, stride=1),
            nn.BatchNorm3d(157),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(157, 628, kernel_size=1, stride=1),
            nn.BatchNorm3d(628),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(628, 1256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(1256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.layer6 = nn.Sequential(
            nn.Conv3d(1256, 2512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(2512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.layer7 = nn.Sequential(
            nn.Conv3d(2512, 5024, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(5024),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.fc0 = nn.Sequential( 
            nn.Linear(40192 ,5024),
            nn.BatchNorm1d(5024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        
#         160768x2 and 40192x5024
        self.fc1 = nn.Sequential(
            nn.Linear(5024, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):

        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        
        out = out.reshape(out.size(0), -1)

        out = self.fc0(out)

        out = self.fc1(out)

        out = self.fc2(out)

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
        a_grid = grid[0].to_dense().half() if grid.shape==(1, 20, 20, 20, 628) else grid.to_dense()
        try: a_label = torch.tensor(round(-1 * math.log(label[0]),1))
        except: a_label = torch.tensor(round(-1 * math.log(label),1))

        return a_grid, a_label
    
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='3DCNN', type=str)
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=112, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=1500, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
    args = parser.parse_args()
    return args
                                         
def main(args):
    global best_r2

    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    model = Net()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    if args.lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)


    ### resume training if necessary ###
    if args.resume:
        if os.path.isfile(args.resume):
            torch.cuda.empty_cache()
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_r2 = checkpoint['best_r2']
            #if args.gpu is not None:
            #    # best_loss may be from a checkpoint from a different GPU
            #    best_loss = torch.Tensor(best_loss).to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    
    ### data ###
    os.chdir("/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/")
    with open("zero_train_zfeats.pkl",'rb') as f:
        train_data=pickle.load(f)
    with open("test_data_2.pkl",'rb') as f:
        test_data=pickle.load(f)

    random.Random(4).shuffle(train_data)
    train_data, validate_data =train_data[:-500], train_data[-500:]

## test the model productivity ##
#    train_data, validate_data = train_data[:1000], train_data[:1000]
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
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(validate_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    testing_loader = DataLoader(test_dataset, pin_memory=False, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    
    cudnn.benchmark = True
    
    val_loss = []

    print("starting training ..")
    torch.cuda.empty_cache()
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        print("epoch: ", epoch)
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        # adjust_learning_rate(optimizer, epoch, args)
         
        print("actually starting training")
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        r2 = 0.0 # dummy
        # evaluate on validation set
        if epoch != 0 and epoch % 5 == 0 and args.rank == 0: # only val and save on master node
            val_epoch_loss,ypred_list,ytrue_list = validate(val_loader, model, criterion, epoch, args)
            val_loss.append(val_epoch_loss)
            r2 = r2_score(ytrue_list, ypred_list) 
                 
            if args.lr_scheduler:
                lr_scheduler(val_epoch_loss)
              
        # remember best r2 and save checkpoint
        is_best =  r2 > best_r2
        best_r2 = max(r2, best_r2)


        if epoch != 0 and epoch % 5 == 0 and args.rank == 0: # only val and save on master node
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.net,
                'state_dict': model.state_dict(),
                'best_r2': best_r2,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

           # validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('R2', ':.2ff')
    r2s = []
    #progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top,
    #                         prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for step, (inp,target) in enumerate(train_loader):
        print("step: ", step)
        current_loss = 0.0
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.view(target.shape[0], 1)
        inp = inp.view(inp.shape[0],-1,20,20,20)

        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        #with autocast():
        output = model(inp)
        loss = criterion(output.float(), target.float())

        # measure R2 and record loss
        ytrue = target.detach().cpu().float().data.numpy()
        ypred = output.detach().cpu().float().data.numpy()

        try:
            r2 = r2_score(ytrue, ypred)
            print("r2= ", r2)
            top.update(r2, inp.size(0))
            r2s.append(r2)
        except:
            print("nan values in ypred")

        losses.update(loss.item(), inp.size(0))

        
        current_loss += loss.item()
        print("optimizing memory and loss", datetime.now()) 
        # compute gradient and do ADAM step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        optimizer.step()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        #scaler.step(optimizer)
        #scaler.update()
        print("loss backward", loss.data.item())
        print("time", datetime.now())

        torch.cuda.empty_cache()  #decrease about 2GB
        # del loss  # nothing change
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print("epoch loss = ", loss.data.item()) 
    writer.add_scalar("Train Loss/Epochs", current_loss, epoch)
    writer.add_scalar("Train R2/Epochs", sum(r2s) / len(r2s), epoch)

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('R2', ':.2f')
    r2s = []
    ytrue_list,ypred_list=[],[]
    progress = ProgressMeter(len(val_loader), batch_time, losses, top,
                             prefix='Test: ')

    torch.cuda.empty_cache()
    # switch to evaluate mode
    model.eval()
    val_running_loss = 0.0
    counter = 0

    with torch.no_grad():
        end = time.time()
        for step, (inp, target) in enumerate(val_loader):

            counter += 1
            current_loss = 0.0
            target = target.view(target.shape[0], 1)
            inp = inp.view(inp.shape[0],-1,20,20,20)

            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
           # with autocast():
            output = model(inp)
            loss = criterion(output, target)

            current_loss += loss.item()
            val_running_loss += loss.item()

            # measure R2 and record loss
            ytrue = target.detach().cpu().float().data.numpy()
            ypred = output.detach().cpu().float().data.numpy()
            ytrue_list.extend(ytrue)
            ypred_list.extend(ypred)

            print(ytrue, ypred)
            r2 = r2_score(ytrue, ypred)
            r2s.append(r2)
            print("evaluation R2 score is: ", r2)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        val_loss = val_running_loss / counter
        writer.add_scalar("Validation Loss/Epochs", current_loss, epoch)
        writer.add_scalar("Validation R2/Epochs", sum(r2s) / len(r2s), epoch)
    return val_loss,ypred_list,ytrue_list

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
        return fmtstr.format(**self.__dict__)


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
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("current lr", lr)
    
        
def R2(output, target):
    """Computes the R2"""
    with torch.no_grad():

        ytrue = target.float().data.numpy()
        ypred = target.detach().cpu().float().data.numpy()

        print(ytrue_arr, ypred_arr)
        r2 = r2_score(ytrue_arr, ypred_arr)


        return r2        
        
if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter()
    main(args)
