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
from models.TransG import TransG,ResAttU_Net3D
from models.TransUnet_Model import ResTransUNet3D
from models.TransD import TransD
from data.data_util import *
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from evaluation import compute_MAE, compute_MSE, compute_PSNR, compute_NCC, compute_SR, compute_PR
# from data.dataset import get_loader_3D
from data.data_loader_3_patch import get_loader_3D
from models.patch.TABS_Model import TABS
def get_metrics(cur_prediction,coordinate,pre_image,post_image, brain_mask_image,patch_size,device):
  (x,y,z)=coordinate
  curr_target = post_image[x:x+patch_size,y:y+patch_size,z:z+patch_size] - pre_image[x:x+patch_size,y:y+patch_size,z:z+patch_size]
  curr_mask = brain_mask_image[x:x+patch_size,y:y+patch_size,z:z+patch_size]
  cur_prediction_vector = cur_prediction.contiguous().view(1,-1).to(device)
  curr_target_vector = curr_target.contiguous().view(1,-1).to(device)
  curr_brain_mask_vector = curr_mask.contiguous().view(1,-1).to(device)
  # print()
  # curr_brain_mask_vector = curr_mask.view(1, -1)
  cur_prediction_vector_array = cur_prediction_vector[curr_brain_mask_vector == 1].cpu().detach().numpy()
  curr_target_vector_array = curr_target_vector[curr_brain_mask_vector == 1].cpu().detach().numpy()

  if len(cur_prediction_vector_array) > 0:
    mse=compute_MSE(cur_prediction_vector_array, curr_target_vector_array)
    pr,_=compute_PR(cur_prediction_vector_array, curr_target_vector_array)
    sr,_=compute_SR(cur_prediction_vector_array, curr_target_vector_array)
  else:
    mse=None
    pr=None
    sr=None
  return mse,pr,sr

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
        
discriminator_loss_weight = 1
img_loss_weight = 1

def main(config):
    
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    
    exp_path = config.root+'/experiment/'
    save_dir = "GAN_TABS_patch"
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)

    sys.stdout = Logger(exp_path + save_dir)
    
    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch
    cont_training = config.use_checkpoint #if continue training
    '''
    If continue from previous training
    '''
    if cont_training:
        print('Using checkpoint: ', config.checkpoint)
        netG = TABS(img_dim=96).to(config.device)
        # netG = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel).to(config.device)
        # netG = TransG(config).to(config.device)
        netD = TransD(config).to(config.device)
    else:
        netG = TABS(img_dim=96).to(config.device)
        # netG = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel).to(config.device)
        # netG = TransG(config).to(config.device)
        netD = TransD(config).to(config.device)
  

    '''
    Initialize training
    '''
    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
    test_loader_3D = get_loader_3D(pre_folder = config.test_pre_folder,
							   post_folder = config.test_post_folder,
							   brain_mask_folder = config.test_brain_mask_folder,
							   batch_size = config.batchsize,
							   num_workers = config.num_workers,
							   shuffle = True,
							   mode = 'validation',
							   augment=False)

    # optimizerD = optim.Adam(netD.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizerG = optim.SGD(netG.parameters(), lr=config.TABS_lr, momentum=config.TABS_momentum)
    optimizerD = optim.SGD(netD.parameters(), lr=config.Discriminator_lr, momentum=config.Discriminator_momentu)

    # model_type="GAN"
    model_type="TABS"

    if model_type=="TABS":
      GPATH="/content/drive/MyDrive/SwinGan/experiment/models/TABS_Patch/TABS-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl"
      netG.load_state_dict(torch.load(GPATH,map_location='cuda:0'))
    elif model_type=="GAN":
      GPATH="/content/drive/MyDrive/SwinGan/experiment/GAN_TABS_patch/attempt1/CheckPoint_39_model_G0.pt"
      checkpointG = torch.load(GPATH)
      netG.load_state_dict(checkpointG['model_state_dict'])
      


    # loss = checkpoint['loss']
    criterion_img = AdaptiveLossFunction(num_dims = 1, float_dtype = np.float32, alpha_init = 2, alpha_hi = 3.5, device = config.device) 
    softmax = nn.Softmax()
    img_minus=True
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = os.path.join(exp_path + save_dir)
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/GAN.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['model type','img loss','PCC','SCC','MSE','PCC2','SCC2','MSE2'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf
    

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
    with torch.no_grad():

        epoch_img_loss = []
        scc_list=[]
        pcc_list=[]
        mse_list=[]
        scc2_list=[]
        pcc2_list=[]
        mse2_list=[]
        step=0
        for pre_img,post_img,mask in tqdm(test_loader_3D):
            step+=1
            step_start_time = time.time()
            netG.eval()
            pre_img = pre_img[:,None,:,:,:]
            post_img = post_img[:,None,:,:,:]
            mask = mask[:,None,:,:,:]
            # post_img[mask!=1]=0
            # pre_img[mask!=1]=0
            prediction = netG(pre_img.to(config.device))
            prediction[mask!=1] = 0
            # print("prediction.shape",prediction.shape)
            fake = pre_img.to(config.device) + prediction
            target = post_img-pre_img #target is the CBV map only
            target = target.to(config.device)
            prediction_vector = prediction.view(prediction.size(0), -1)
            target_vector = target.view(target.size(0),-1).to(config.device)
            mask_vector = mask.view(mask.size(0), -1).to(config.device)
            img_loss = torch.mean(criterion_img.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
            
            predict_img = (prediction + pre_img.to(config.device)).clone().detach().cpu().numpy()
            predict_cbv = prediction.clone().detach().cpu().numpy()
            gt_img = post_img.clone().detach().cpu().numpy()
            gt_cbv = (post_img-pre_img).clone().detach().cpu().numpy()
            slice_index = predict_img.shape[2]//2
            plt.subplot(1,5,1)
            plt.imshow(predict_img[0,0,slice_index,:,:], cmap='gray')
            plt.axis('off')
            plt.subplot(1,5,2)
            plt.imshow(gt_img[0,0,slice_index,:,:], cmap='gray')
            plt.axis('off')
            plt.subplot(1,5,3)
            plt.imshow(predict_cbv[0,0,slice_index,:,:], cmap='gray')
            plt.axis('off')
            plt.subplot(1,5,4)
            plt.imshow(gt_cbv[0,0,slice_index,:,:], cmap='gray')
            plt.axis('off')
            plt.subplot(1,5,5)
            plt.imshow(np.abs(predict_cbv[0,0,slice_index,:,:]-gt_cbv[0,0,slice_index,:,:]), cmap='seismic')
            plt.axis('off')
            plt.show()
            if len(prediction_vector[mask_vector == 1]) != 0:
              pcc,_=pearsonr(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
              scc,_=spearmanr(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
              mse=compute_MSE(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
              mse2,pcc2,scc2=get_metrics(prediction_vector,[0,0,0],pre_img[0][0],post_img[0][0], mask[0][0],96,config.device)
              # mse2=0
              # pcc2,_=compute_PR(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
              # scc2,_=compute_SR(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
              pcc_list.append(pcc)
              scc_list.append(scc)
              mse_list.append(mse)
              pcc2_list.append(pcc2)
              scc2_list.append(scc2)
              mse2_list.append(mse2)              
            epoch_img_loss.append(img_loss.item())

            print('steps {0:d} - pcc {1:.4f} - scc {2:.4f} -mse {3:.4f}- pcc2 {4:.4f} - scc2 {5:.4f} -mse2 {6:.4f}'.format(step,pcc,scc,mse,pcc2,scc2,mse2))

        img_loss = sum(epoch_img_loss) / len(epoch_img_loss)
        pcc = sum(pcc_list) / len(pcc_list)
        scc = sum(scc_list) / len(scc_list)
        mse = sum(mse_list) / len(mse_list)
        pcc2 = sum(pcc2_list) / len(pcc2_list)
        scc2 = sum(scc2_list) / len(scc2_list)
        mse2 = sum(mse2_list) / len(mse2_list)
        # Print the information.
    


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/GAN.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%s' % model_type,'%.15f' % img_loss,'%.15f' % pcc,'%.15f' % scc,'%.15f' % mse,'%.15f' % pcc2,'%.15f' % scc2,'%.15f' % mse2])
            f.close()
            
 