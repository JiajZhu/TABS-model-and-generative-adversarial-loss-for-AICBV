from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
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
from models.TransG import TransG,ResAttU_Net3D
from models.TransUnet_Model import ResTransUNet3D
from data.data_util import *
from robust_loss_pytorch.adaptive import AdaptiveLossFunction,AdaptiveImageLossFunction
from data.dataset import get_loader_3D
# from torchmetrics import PeakSignalNoiseRatio



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
    
    exp_path = config.root+'/experiment/Generator_TABS'
    save_dir = "/test"
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)

    sys.stdout = Logger(exp_path + save_dir)
    lr = config.lr # learning rate
    epoch_start = 0
    # max_epoch = config.max_epoch #max traning epoch
    max_epoch=5
    cont_training = config.use_checkpoint #if continue training
    '''
    If continue from previous training
    '''
    if cont_training:
        print('Using checkpoint: ', config.checkpoint)
        # model = TransG(config)
        model = ResTransUNet3D().to(config.device)
        # model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
        model.to(config.device)
    else:
        # model = TransG(config)
        model = ResTransUNet3D().to(config.device)
        # model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
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

    test_loader_3D = get_loader_3D(pre_folder = config.test_pre_folder,
							   post_folder = config.test_post_folder,
							   brain_mask_folder = config.test_brain_mask_folder,
							   batch_size = 1,
							   num_workers = config.num_workers,
							   shuffle = False,
							   mode = 'test',
							   augment=False)

    print('Data Loaded!')



    ##checkpoint for generator
    # model_path="/content/drive/MyDrive/SwinGan/experiment/GAN_Unet/transformer_D/CheckPoint_4_model_G.pt"
    # model.load_state_dict(torch.load(model_path,map_location='cuda:0'))

    model_path="/content/drive/MyDrive/SwinGan/experiment/GAN_TABS_patch/CheckPoint_38_model_G.pt"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion=AdaptiveLossFunction(num_dims = 1, float_dtype = np.float32, alpha_init = 2, alpha_hi = 3.5, device = config.device)
    img_minus=True
    stop_num = 10
    early_stop = stop_num
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = "/content/drive/MyDrive/SwinGan/experiment/GAN_TABS_patch/test"
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/gan.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Current Epoch', 'Test Total Loss','Test PCC','Test SCC'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
   
##################################################################################################################################
########################################################### Testing ###########################################################
##################################################################################################################################
    with torch.no_grad():
        step=0
        model.eval()
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        scc_list=[]
        pcc_list=[]
        for pre_img,post_img,mask in tqdm(test_loader_3D):
            pre_img = pre_img[:,None,:,:,:]
            post_img = post_img[:,None,:,:,:]
            mask = mask[:,None,:,:,:]
            post_img[mask!=1]=0
            pre_img[mask!=1]=0
            step += 1
            step_start_time = time.time()
            model.train()
            # target = (post_img-pre_img)[0] #target is the CBV map only
            target = (post_img-pre_img)
            target = target.to(config.device)
            prediction = model(pre_img.to(config.device))
            prediction_vector = prediction.view(prediction.size(0), -1)
            target_vector = target.view(target.size(0),-1).to(config.device)
            mask_vector = mask.view(mask.size(0), -1).to(config.device)
            loss = torch.mean(criterion.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
            epoch_loss.append(loss)
            prediction[mask!=1]=0
            pcc,_=pearsonr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
            scc,_=spearmanr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
            epoch_step_time.append(time.time() - step_start_time)
            epoch_total_loss.append(loss.item())
            pcc_list.append(pcc)
            scc_list.append(scc)
            print('steps {0:d} - loss {1:.4f}- pcc {2:.4f}- scc {3:.4f}'.format(step,loss.item(),pcc,scc))
        
        test_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        test_pcc = sum(pcc_list) / len(pcc_list)
        test_scc = sum(scc_list) / len(scc_list)
        


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/gan.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%.15f' % test_loss, '%.15f' % test_pcc, '%.15f' % test_scc])
            f.close()
            

