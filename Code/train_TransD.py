from torch.utils.tensorboard import SummaryWriter
import os, utils, glob
import sys
import math
# from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
# from data import datasets, trans
import numpy as np
import torch
# from torchvision import transforms
from torch import optim
import torch.nn as nn
from scipy.stats import pearsonr,spearmanr
# from natsort import natsorted
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from options.train_options import get_TransD_config
from data.data_loader import CreateDataLoader
from models.TransD import TransD
from data.data_util import *
from models.TransUnet_Model import ResTransUNet3D





class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=1.4):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        


def main(config):
    
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    
    exp_path = config.root+'/experiment'
    save_dir = config.checkpoints_dir
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)

    sys.stdout = Logger(exp_path + save_dir)
    lr = config.lr # learning rate
    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch
    cont_training = config.use_checkpoint #if continue training
    '''
    If continue from previous training
    '''
    if cont_training:
        print('Using checkpoint: ', config.checkpoint)
        model = TransD(config)
        model.to(config.device)
    else:
        model = TransD(config)
        model.to(config.device)
    updated_lr = lr

    '''
    Initialize training
    '''
    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
    config.type="train"
    config.datatype="Discriminator"
    data_loader_train = CreateDataLoader(config)
    dataset_train = data_loader_train.load_data()
    dataset_size_train = len(data_loader_train)
    
    config.type="validation"
    data_loader_valid = CreateDataLoader(config)
    dataset_valid = data_loader_valid.load_data()
    dataset_size_valid = len(data_loader_valid)
    print('#training images = %d' % dataset_size_train)
    
    data_loader_validation = CreateDataLoader(get_TransD_config(type='validation'))
    dataset_validation = data_loader_validation.load_data()
    dataset_size_validation = len(data_loader_validation)
    print('#training images = %d' % dataset_size_validation)
    print('Data Loaded!')

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    
    # PATH="/content/drive/MyDrive/Chenghao_CycleICTD/model/experiment/saved_model_CycleTransMorph/CheckPoint_231_model_SSIM.pt"
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_start = checkpoint['epoch']+1
    # loss = checkpoint['loss']


    criterion=nn.CrossEntropyLoss()
    img_minus=True
    stop_num = 10
    early_stop = stop_num
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = os.path.join(exp_path + save_dir)
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Current Epoch', 'Train Total Loss','Train Acc','Val Total Loss','Val Acc'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf
    ################################################################################################################################
    ########################################################### Training ###########################################################
    ################################################################################################################################
    for epoch in range(epoch_start, max_epoch):
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        epoch_train_acc = []
        step = 0
        for (label,volume) in tqdm(dataset_train):
        # for step in tqdm(range(config.train_iteration)):#config.train_iteration
            step += 1
            step_start_time = time.time()
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # (label,volume) = next(iter(dataset_train.__iter__()))
            prediction = model(volume.to(config.device))
            loss = criterion(prediction,label.to(config.device))
            print("prediction",prediction)
            print("lanel",label)
            epoch_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (prediction.argmax(dim=-1) == label.to(config.device)).float().mean()
            if step%1 == 0:
                grad_norms = [torch.norm(p.grad) for p in model.parameters()]
                print(f"Gradient norms at iteration {step}: {grad_norms}")
                print('steps {0:d} - training loss {1:.4f}- acc {2:.4f}'.format(step,loss.item(),acc))
                
            # get compute time
            epoch_train_acc.append(acc)
            epoch_step_time.append(time.time() - step_start_time)
            epoch_total_loss.append(loss.item())
        train_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        train_acc = sum(epoch_train_acc) / len(epoch_train_acc)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{max_epoch:03d} ] loss = {train_loss:.5f},acc = {train_acc:.5f}")

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
        with torch.no_grad():
            model.eval()
            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []
            epoch_train_acc = []
            for (label,volume) in tqdm(dataset_valid):
            # for step in tqdm(range(config.validation_iteration)):

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                # (label,volume) = next(iter(dataset_train.__iter__()))
                prediction = model(volume.to(config.device))
                loss = criterion(prediction,label.to(config.device))
                acc = (prediction.argmax(dim=-1) == label.to(config.device)).float().mean()
                epoch_loss.append(loss)
                epoch_total_loss.append(loss.item())
                epoch_train_acc.append(acc)
                epoch_step_time.append(time.time() - step_start_time)

            valid_loss = sum(epoch_total_loss) / len(epoch_total_loss)
            valid_acc = sum(epoch_train_acc) / len(epoch_train_acc)
            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {valid_loss:.5f},acc = {valid_acc:.5f}")
            # print(f"param_loss = {valid_param_loss:.5f}")
            # print(f"img_loss = {valid_img_loss:.5f}")


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch + 1), '%.15f' % train_loss,'%.15f' % train_acc, '%.15f' % valid_loss,'%.15f' % valid_acc])
            f.close()
            

        #############################################################################
        # Save best model and early stop
        #############################################################################

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            model.save(os.path.join(LossPath, 'Best_model.pt'))
            early_stop = stop_num
        else:
            early_stop = early_stop - 1
        if epoch % 1 == 0:    ### update
            model.save(os.path.join(LossPath, 'Best_{}_model.pt'.format(epoch)))  
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, os.path.join(LossPath, 'CheckPoint_{}_model.pt'.format(epoch)))
        if early_stop and epoch < max_epoch - 1:  
            continue  
        else:
          print('early stop at'+str(epoch)+'epoch')
          break  
        model.save(os.path.join(LossPath, 'model.pt'))  
