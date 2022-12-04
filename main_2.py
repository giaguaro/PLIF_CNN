import os
from os.path import exists
import builtins
import argparse
import math
import torch.distributed as dist
import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transform
import random
from sklearn import metrics
import matplotlib.pylab as plt
import pandas as pd
import pickle
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import *
from scipy.stats import *
from torch.utils.tensorboard import SummaryWriter
import subprocess
import gc

# Some utility functions
#*************************************
def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def mydate() :
    return (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# Read/write directory parameters
#*************************************
datadir = 'training_data'
savemodeldir = 'new_model_2'
loadmodelpath = 'model_2/2018-10-30_03-12-21_model_epoch30.pth'

# Pytorch parameters
#*************************************
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
savemodel = True
savemodel_interval = 1  #if 0 (and savemodel=True) will only save model at the end of entire training
loadmodel = False
model_to_save = None

# Training parameters
#*************************************
batch_size = 2
num_epochs = 200
lr = 1e-4
log_interval = 10
random.seed(1234) #for dataset splitting set to None of leave blank if do not need to preserve random order

loss_idx_value= 0.0

train_losses = []
valid_losses = []
list_of_auc = []

# Preprocessing parameters
#*************************************
bins = 48
hrange = 24

def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
		return

	for key in keys:
		newkey = key[len(prefix) :]
		state_dict[newkey] = state_dict.pop(key)

	try:
		metadata = state_dict._metadata
	except AttributeError:
		pass
	else:
		for key in list(metadata.keys()):
			if len(key) == 0:
				continue
			newkey = key[len(prefix) :]
			metadata[newkey] = metadata.pop(key)


class CNN(nn.Module):
    # input size - the number of "classes"
    def __init__(self):
        super(CNN, self).__init__()
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
    
CNN()

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
        a_grid = grid[0].to_dense() if grid.shape==(1, 200, 200, 200, 94) else grid.to_dense()
        try: a_label = torch.tensor(-1 * math.log(label[0]))
        except: a_label = torch.tensor(-1 * math.log(label))
        
        return a_grid, a_label


# def collate_fn(data):
#     """
#        data: is a list of tuples with (example, label, length)
#              where 'example' is a tensor of arbitrary shape
#              and label/length are scalars
#     """
#     _, labels, lengths = zip(*data)
#     max_len = max(lengths)
#     n_ftrs = data[0][0].size(1)
#     features = torch.zeros((len(data), max_len, n_ftrs))
#     labels = torch.tensor(labels)
#     lengths = torch.tensor(lengths)

#     for i in range(len(data)):
#         j, k = data[i][0].size(0), data[i][0].size(1)
#         features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

#     return features.float(), labels.long(), lengths.long()

    
    
    #Make calls to the dataloader
#     for tensor_batch, label_batch in collected_batch:
#         print("Batch of tensors has shape: ", tensor_batch.shape)
#         print("Batch of labels has shape: ", label_batch)

# Define the training cycle (100% teacher forcing for now)
#*************************************
def train(model,epoch, optimizer, criterion, training_loader, train_data, validation_loader, validation_data, model_to_save):
    model.train() #put in training mode
   
    stepss=0
    global loss_idx_value
    loss_idx_value += 0

    for step, (inp,target) in enumerate(training_loader):
        #print("step",datetime.now())
        stepss += step
        current_loss = 0.0

        target = target.float()
        inp,target = inp.cuda(), target.cuda()
        target = target.view(target.shape[0], 1)
        inp = inp.view(inp.shape[0],-1,200,200,200)

        # Forward + Backward + Optimize
        #print("train",datetime.now())
        outputs = model(inp)
        #print("stop_train",datetime.now())
        #print(outputs,target)
        loss = criterion(outputs.float(), target.float())
        #print(loss.data.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("optimized",datetime.now())

#        if step % args.checkpoint_iter == 0:
#                checkpoint_dict = {
#                        "model_state_dict": model_to_save.state_dict(),
#                        "optimizer_state_dict": optimizer.state_dict(),
#                        "loss": loss,
#                        "step": step,
#                        "epoch": epoch
#                }
#                torch.save(checkpoint_dict, args.model_path,_use_new_zipfile_serialization=False)
#                print("checkpoint saved: %s" % args.model_path)

        current_loss += loss.item()
        #writer.add_scalar("Loss/Minibatches", current_loss, loss_idx_value)
        #print("writer",datetime.now())
        loss_idx_value += 1
        if step % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (step + 1, current_loss / 500))
            current_loss = 0.0
        #print("fetching next set of batches",datetime.now())
    print ('{:%Y-%m-%d %H:%M:%S} Epoch [{}/{}], Step [{}/{}] Loss: {:.6f}'.format(
        datetime.now(), epoch+1, num_epochs, step+1, len(train_data)//batch_size, loss.item()))

    #writer.add_scalar("Loss/Epochs", current_loss, epoch)

    train_losses.append(loss.data.item())
    print("train_loss: ", train_losses) 
    print("total steps: ", stepss)

    if epoch != 0 and epoch % 100 == 0:
        if args.rank == 0:
            checkpoint_dict = {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                            "step": step,
                            "epoch": epoch
                    }
            torch.save(checkpoint_dict, args.model_path,_use_new_zipfile_serialization=False)
            print("checkpoint saved: %s" % args.model_path)
                  
            gc.collect()
            torch.cuda.empty_cache()
            evaluate_mse(model, validation_loader, validation_data)

    print("epoch: ", epoch)
              
    if savemodel_interval != 0 and savemodel:
        if (epoch+1) % savemodel_interval == 0:
            torch.save(model.state_dict(),
                       '{}/{:%Y-%m-%d_%H-%M-%S}_model_epoch{}_step{}.pth'.format(savemodeldir,datetime.now(),epoch+1,step+1))
            print('model saved at epoch{} step{}'.format(epoch+1,step+1))

# Initialize the network, optimizer and objective func
#*************************************
if loadmodel: # load checkpoint if needed
    print("Loading existing checkpoint...")
    cnn.load_state_dict(torch.load(loadmodelpath))
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(299,dtype=torch.float,device=device))  #
#criterion = nn.MSELoss().float() #nn.BCEWithLogitsLoss()  ##nn.MSELoss()




def evaluate(model, validation_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inp, target in validation_loader:
            inp, target = inp.cuda(), target.cuda()
            inp = inp.view(inp.shape[0],-1,200,200,200)
            target = target.view(target.shape[0], 1)
            
            outputs = model(inp)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            print(torch.max(target, 1)[1])
            correct += (predicted == torch.max(target, 1)[1]).sum().item()

        print('Accuracy of the model on the validation set: {} %'.format(100 * correct / total))
        
        
def evaluate_mse(model,validation_loader,validation_data):

    ytrue_arr = np.zeros((len(validation_data),1), dtype=np.float32)
    ypred_arr = np.zeros((len(validation_data),1), dtype=np.float32)
    pred_list = []

    model.eval()
    with torch.no_grad():
        out = []
        targets = []
        for step, (inp, target) in enumerate(validation_loader):
            inp = inp.cuda()
            inp = inp.view(inp.shape[0],-1,200,200,200)
            target = target.view(target.shape[0], 1)

            ypred_batch = model(inp)

            ytrue = target.float().data.numpy()
            ypred = ypred_batch.detach().cpu().float().data.numpy()
            ytrue_arr[step*batch_size:step*batch_size+inp.shape[0]] = ytrue
            ypred_arr[step*batch_size:step*batch_size+inp.shape[0]] = ypred

            if args.save_pred:
            	for i in range(inp.shape[0]):
            		pred_list.append([step + i, ytrue[i], ypred[i]])
        

            batch_count = len(validation_data) // batch_size
            print("[%d/%d] evaluating" % (step+1, batch_count))

        ytrue_arr=np.squeeze(ytrue_arr)
        ypred_arr=np.squeeze(ypred_arr)
        print(ytrue_arr, ypred_arr)
        rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
        mae = mean_absolute_error(ytrue_arr, ypred_arr)
        r2 = r2_score(ytrue_arr, ypred_arr)
        pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
        spearman, spval = spearmanr(ytrue_arr, ypred_arr)
        mean = np.mean(ypred_arr)
        std = np.std(ypred_arr)
        print("Evaluation Summary:")
        print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (rmse, mae, r2, pearson, spearman, mean, std))

        if args.save_pred:
	        csv_fpath = "%s_%s_pred.csv" % ("model", args.dataset_test_name)
	        df = pd.DataFrame(pred_list, columns=["cid", "label", "pred"])
	        df.to_csv(csv_fpath, index=False)


#            outputs_numpy = outputs.detach().cpu().numpy()
#            targets_numpy = target.numpy()
#            for i in range(outputs_numpy.shape[0]):
#                out.append(outputs_numpy.item(i))
#                targets.append(targets_numpy.item(i))
#        auc = auc_curve(out,targets)
#        list_of_auc.append(auc)
            
def auc_curve(output,target):
    """Plot a ROC curve"""
    fpr, tpr, _ = metrics.roc_curve(target,  output)
    auc = metrics.roc_auc_score(target, output)
    plt.figure() 
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    plt.savefig('auc.png')
    return(auc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='cnn', type=str)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--checkpoint-iter', type=int, default=500, help='checkpoint save rate')
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
    parser.add_argument("--model-path", default='/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/new_model_2/checkpoint_model_20221010.pth', 
                        help="model checkpoint file path")
    parser.add_argument('--dataset_test_name', default="test_set", type=str,
                        help='give a prefix for test dataset')
    parser.add_argument("--save-pred", default=True, action="store_true", 
                        help="whether to save prediction results in csv")
    args = parser.parse_args()
    return args


def main(args):

    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
        )

    session = tf.compat.v1.Session(config=config)
    
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node", ngpus_per_node)

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"]) 
            args.gpu = args.rank % torch.cuda.device_count()
       # args.rank = int(os.environ["RANK"])
        print("args.rank: " , args.rank)
        print("args.gpu: " , args.gpu)

        print("Initializing PyTorch distributed ...")
        #command = subprocess.Popen(["fuser", "-k", "/dev/nvidia3"], stderr=subprocess.PIPE)
        #command.wait()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("done initializing!")



    #print("rank", args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    model = CNN()


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print([args.gpu])
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
           #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MSELoss().float()
   
    torch.backends.cudnn.benchmark = True

    model_to_save = model.module

    torch.manual_seed(0)
    torch.cuda.set_device(args.gpu)

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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    validate_dataset = CNNDataLoader(validate_data)
    val_sampler = None

    test_dataset = CNNDataLoader(test_data)

    #Initiate the dataloader
    training_loader = DataLoader(train_dataset, pin_memory=False, batch_size=2, num_workers=6,
                                 shuffle=(train_sampler is None), sampler=train_sampler)
    validation_loader = DataLoader(validate_dataset, pin_memory=False, batch_size=2, num_workers=6,
                                   shuffle=(val_sampler is None),sampler=val_sampler)
    testing_loader = DataLoader(test_dataset, pin_memory=False, batch_size=2, shuffle=True, num_workers=6)
    
    
    
    ### main loop ###
       # Train!
    #*************************************
    gc.collect()
    torch.cuda.empty_cache()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
    epoch_start = 0
    if exists(args.model_path):
        checkpoint = model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print("loaded")
        model_state_dict = checkpoint.pop("model_state_dict")
        strip_prefix_if_present(model_state_dict, "module.")
        model_to_save.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print("checkpoint loaded: %s" % args.model_path)

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    output_dir = os.path.dirname(args.model_path)

    print('{:%Y-%m-%d %H:%M:%S} Starting training...'.format(datetime.now()))
    start_time = time.monotonic()

    for epoch in range(epoch_start,num_epochs):

        np.random.seed(epoch)
        random.seed(epoch)
        if args.distributed: 
            training_loader.sampler.set_epoch(epoch)
        train(model, epoch, optimizer, criterion, training_loader, train_data, validation_loader, validate_data, model_to_save)

    elapsed_time = time.monotonic() - start_time
    print('Training time taken:',time_taken(elapsed_time))

    if savemodel_interval == 0 and savemodel:
        torch.save(model.state_dict(), 
           '{}/{:%Y-%m-%d_%H-%M-%S}_model_epoch{}.pth'.format(savemodeldir,datetime.now(),num_epochs))
        print('model saved at epoch{}'.format(num_epochs))


if __name__ == '__main__':

    
    args = parse_args()
    #writer = SummaryWriter()
    main(args)
