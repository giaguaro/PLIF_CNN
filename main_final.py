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
import torch.nn.functional as F
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
# import EarlyStopping
from pytorchtools import EarlyStopping


# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

def CountFrequency(my_list):
 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    for key, value in freq.items():
        print ("% d : % d"%(key, value))


clip = 5
best_acc1 = 0

class Net(nn.Module):
    # input size - the number of "classes"
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(628, 628, kernel_size=1, stride=1),
            nn.BatchNorm3d(628),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(628, 628, kernel_size=1, stride=1),
            nn.BatchNorm3d(628),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(628, 1256, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm3d(1256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.layer4 = nn.Sequential(
            nn.Conv3d(1256, 2512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(2512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.layer5 = nn.Sequential(
            nn.Conv3d(2512, 5024, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(5024),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.layer6 = nn.Sequential(
            nn.Conv3d(5024, 10048, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(10048),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2))
        self.fc0 = nn.Sequential(
            nn.Linear(10048,10048),
            nn.BatchNorm1d(10048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

#         160768x2 and 40192x5024
#        self.fc1 = nn.Sequential(
#            nn.Linear(5024, 2512),
#            nn.BatchNorm1d(2512),
#            nn.ReLU(inplace=True),
#            nn.Dropout(0.5))
#        self.fc2 = nn.Sequential(
#            nn.Linear(2512, 1256),
#            nn.BatchNorm1d(1256),
#            nn.ReLU(inplace=True),
#            nn.Dropout(0.5))
#        self.fc3 = nn.Sequential(
#            nn.Linear(1256, 628),
#            nn.BatchNorm1d(628),
#            nn.ReLU(inplace=True),
#            nn.Dropout(0.5))
        self.fc1 = nn.Linear(10048, 34)

    def forward(self, x):

        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc0(out)

        out = self.fc1(out)

        #out = self.fc2(out)

        #out = self.fc3(out)

        #out = self.fc4(out)
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
        a_grid = grid[0].to_dense() if grid.shape==(1, 20, 20, 20, 628) else grid.to_dense()
        try: a_label = torch.tensor(round(-1 * math.log(label[0])))
        except: a_label = torch.tensor(round(-1 * math.log(label)))

        return a_grid, a_label
    
class LRScheduler():
    """
    Learning rate scheduler. If the validation4loss does not decrease for the 
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
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=2500, type=int, help='number of total epochs to run')
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
    global best_acc1

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
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=1e-4)
    
    if args.lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)


    ### resume training if necessary ###
    if args.resume:
        if os.path.isfile(args.resume):
            torch.cuda.empty_cache()
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            #if args.gpu is not None:
            #    # best_acc1 may be from a checkpoint from a different GPU
            #    best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume)) 
    
    ### data ###
    os.chdir("/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/")
    with open("zero_train_zfeats.pkl",'rb') as f:    #train_data_2.pkl
        train_data=pickle.load(f)
    with open("test_data_2.pkl",'rb') as f:
        test_data=pickle.load(f)

    all_labels=[]
    for i in train_data:
        with open(i[1],'rb') as f2: all_labels.append(round(-1 * math.log(pickle.load(f2))))
    CountFrequency(all_labels)
    num_classes=len(set(all_labels))
    print("num_classes =", num_classes)
    classes_dict={b: a for a, b in enumerate(set(all_labels))}

    df = pd.DataFrame({'target': all_labels})
    df['target'] = df['target'].astype('category')
    class_count_df = df.groupby(all_labels).count()
    class_weight=[]
    def calc_weight(idx,n_1, num_classes):
        n_0 = class_count_df.iloc[idx, 0]

        return (n_1) / (num_classes * n_0)

    for x in range(num_classes):
        class_weight.append(calc_weight(x,class_count_df.iloc[:,0].sum(), num_classes))

    class_weights=torch.FloatTensor(class_weight)
    print(class_weights)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu) # BCEWithLogitsLoss(pos_weight=class_weights).cuda(args.gpu) 
    
    random.Random(4).shuffle(train_data)
    train_data, validate_data =train_data[:-500], train_data[-500:]  #2000

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
#        break
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        # adjust_learning_rate(optimizer, epoch, args)
         
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, num_classes, classes_dict, args)

        best_acc1=0
        # evaluate on validation set
        if epoch != 0 and epoch % 3 == 0 and args.rank == 0: # only val and save on master node
            val_epoch_loss, acc1 = validate(val_loader, model, criterion, epoch, num_classes, classes_dict, args)
            val_loss.append(val_epoch_loss)
                 
            if args.lr_scheduler:
                lr_scheduler(val_epoch_loss)
              
            # remember best acc and save checkpoint
            is_best =  acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)


            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.net,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_epoch_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

           # validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

    print("TESTING MODEL")
    val_epoch_loss, acc1 = validate(testing_loader, model, criterion, epoch, num_classes, classes_dict, args)

def train(train_loader, model, criterion, optimizer, epoch, num_classes, classes_dict, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

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
        target = target.squeeze()
        target_temp = target.tolist()
        #print(target_temp)
        try: target = [classes_dict.get(e, e) for e in target_temp]
        except: target = [classes_dict.get(e, e) for e in [target_temp]]
        target = torch.LongTensor(target)
        target = target.cuda(non_blocking=True)

        # compute output
        #with autocast():
        output = model(inp)
        loss = criterion(output, target)

        print("ytrue ", target.detach().cpu().float().data.numpy())
        print("ypred ", torch.max(output.detach().cpu().data,1))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))
        
        current_loss += loss.item()
        print("optimizing memory and loss", datetime.now()) 
        # compute gradient and do ADAM step
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)
        optimizer.step()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        #scaler.step(optimizer)
        #scaler.update()
        print("time", datetime.now())

        torch.cuda.empty_cache()  #decrease about 2GB
        # del loss  # nothing change
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print("epoch loss = ", loss.data.item()) 
    writer.add_scalar("Train Loss/Epochs", current_loss, epoch)
    writer.add_scalar('Accuracy (top-1)/train', top1.avg, epoch)
    writer.add_scalar('Accuracy (top-5)/train', top5.avg, epoch)

def validate(val_loader, model, criterion, epoch, num_classes, classes_dict, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    ypred=[]
    ytest=[]
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
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
            target = target.squeeze()
            target_temp = target.tolist()
            try: target = [classes_dict.get(e, e) for e in target_temp]
            except: target = [classes_dict.get(e, e) for e in [target_temp]]
            target = torch.LongTensor(target)
            target = target.cuda(non_blocking=True)

            output = model(inp) 
            # compute output
           # with autocast():
            print("ytrue ", target.detach().cpu().float().data.numpy())
            print("ypred ", torch.max(output.detach().cpu().data,1))
#            print("ypred ", output.detach().cpu().float().data.numpy()[1])

 
            loss = criterion(output, target.squeeze())

            current_loss += loss.item()
            val_running_loss += loss.item()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_classes)))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        val_loss = val_running_loss / counter
        writer.add_scalar("Validation Loss/Epochs", current_loss, epoch)
        writer.add_scalar('Accuracy (top-1)/val', top1.avg, epoch)
        writer.add_scalar('Accuracy (top-5)/val', top5.avg, epoch)

    print("validation epoch loss = ", loss.data.item())

    return val_loss,top1.avg

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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("current lr", lr)
    
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score



if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter()
    main(args)
