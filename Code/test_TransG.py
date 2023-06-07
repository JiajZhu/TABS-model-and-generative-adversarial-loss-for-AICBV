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
from robust_loss_pytorch import AdaptiveLossFunction
from data.dataset import get_loader_3D
import pytorch_msssim.pytorch_msssim as pytorch_msssim_real
from evaluation import compute_MAE, compute_MSE, compute_PSNR, compute_NCC, compute_SR, compute_PR
# from torchmetrics import PeakSignalNoiseRatio
import losses
import nibabel as nib
from models.patch.TABS_Model import TABS
def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


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
        
def psnr_3d(image1, image2):
    image1 = imgnorm(image1)
    image2 = imgnorm(image2)
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(image1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm
def main(config):
    
    print('##############################################################')
    print("Configuration:")
    print(config)
    print('##############################################################')
    
    exp_path = config.root+'/experiment/Generator_Unet'
    save_dir = "/tuning"
    if not os.path.exists(exp_path + save_dir):
        os.makedirs(exp_path + save_dir)

    sys.stdout = Logger(exp_path + save_dir)
    lr = config.Trans_lr # learning rate
    epoch_start = 0
    max_epoch = config.max_epoch #max traning epoch
    cont_training = config.use_checkpoint #if continue training
    '''
    If continue from previous training
    '''
    if cont_training:
        print('Using checkpoint: ', config.checkpoint)
        # model = TransG(config)
        # model = ResTransUNet3D().to(config.device)
        model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
        # model = TABS(img_dim=192)
        model.to(config.device)
    else:
        # model = TransG(config)
        # model = ResTransUNet3D().to(config.device)
        model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
        # model = TABS(img_dim=192)
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
    # config.datatype="Generator"
    # data_loader_train = CreateDataLoader(config)
    # dataset_train = data_loader_train.load_data()
    # dataset_size_train = len(data_loader_train)
    train_loader_3D = get_loader_3D(pre_folder = config.train_pre_folder,
						  post_folder = config.train_post_folder,
						  brain_mask_folder = config.train_brain_mask_folder,
						  batch_size = config.batchsize,
						  num_workers = config.num_workers,
						  shuffle = True,
						  mode = 'train',
						  augment=False)

    validation_loader_3D = get_loader_3D(pre_folder = config.validation_pre_folder,
							   post_folder = config.validation_post_folder,
							   brain_mask_folder = config.validation_brain_mask_folder,
							   batch_size = config.batchsize,
							   num_workers = config.num_workers,
							   shuffle = False,
							   mode = 'test',
							   augment=False)
    config.type="test"
    # data_loader_valid = CreateDataLoader(config)
    # dataset_valid = data_loader_valid.load_data()
    # dataset_size_valid = len(data_loader_valid)
    # print('#training images = %d' % dataset_size_train)
    
    # data_loader_validation = CreateDataLoader(get_TransD_config(type='validation'))
    # dataset_validation = data_loader_validation.load_data()
    # dataset_size_validation = len(data_loader_validation)
    # print('#training images = %d' % dataset_size_validation)
    print('Data Loaded!')

    optimizer = optim.Adam(model.parameters(), lr=config.Trans_lr, weight_decay=0, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=config.Trans_lr)

    GPATH="/content/ResAttU_Net3D-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-epoch54.pkl"
    model.load_state_dict(torch.load(GPATH,map_location='cuda:0'))
    # PATH="/content/drive/MyDrive/SwinGan/experiment/TransG/CheckPoint_82_model.pt"
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    SSIM=losses.SSIM3D()
    img_minus=True
    stop_num = 10
    early_stop = stop_num
    ##############################################################################
    # Learning curve
    ##############################################################################
    LossPath = "/content/drive/MyDrive/SwinGan/experiment/TransG"
    print('LossPath',LossPath)
    with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['PCC','SCC','SSIM','PSNR'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
    with torch.no_grad():
        step=0
        model.eval()
        epoch_loss = []
        epoch_step_time = []
        scc_list=[]
        pcc_list=[]
        ssim_list=[]
        psnr_list=[]

        for pre_img,post_img,mask in tqdm(validation_loader_3D):
            pre_img = pre_img[:,None,:,:,:]
            post_img = post_img[:,None,:,:,:]
            mask = mask[:,None,:,:,:]
            # post_img[mask!=1]=0
            # pre_img[mask!=1]=0
            step += 1
            step_start_time = time.time()
            model.train()
            # target = (post_img-pre_img)[0] #target is the CBV map only
            target = (post_img-pre_img)
            target = target.to(config.device)
            print("pre_shape",pre_img.shape)
            prediction = model(pre_img.to(config.device))
            Prediction_vector = prediction.contiguous().view(1,-1).cuda(config.device)
            Target_vector = target.contiguous().view(1,-1).cuda(config.device)
            # mask_vector = brain_mask.contiguous().view(1,-1).cuda(args.device)
            brain_mask_vector = mask.contiguous().view(1,-1).cuda(config.device)
            # print()

            # brain_mask_vector = brain_mask.view(brain_mask.size(0), -1)
            Prediction_vector_array = Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy()
            Target_vector_array = Target_vector[brain_mask_vector == 1].cpu().detach().numpy()
            # print("target.shape",target.shape)
            # print("prediction.shape",prediction.shape))
            # prediction[mask!=1]=0
            # pcc,_=pearsonr(Prediction_vector_array,Target_vector_array)
            # scc,_=spearmanr(Prediction_vector_array,Target_vector_array)
            pcc, pr_p_val = compute_PR(Prediction_vector_array, Target_vector_array)
            scc, sr_p_val =  compute_SR(Prediction_vector_array, Target_vector_array)
            ssim = pytorch_msssim_real.ssim(torch.mul(prediction.squeeze(0), mask.squeeze(0).type(torch.cuda.FloatTensor)), torch.mul(target.squeeze(0), mask.squeeze(0).type(torch.cuda.FloatTensor))).item()
            psnr = compute_PSNR(Prediction_vector_array, Target_vector_array)
            epoch_step_time.append(time.time() - step_start_time)
            pcc_list.append(pcc)
            scc_list.append(scc)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            print("pcc {},scc {}, ssim{}, psnr{}".format(pcc,scc,ssim,psnr))
        test_pcc = sum(pcc_list) / len(pcc_list)
        test_scc = sum(scc_list) / len(scc_list)
        test_ssim = sum(ssim_list) / len(ssim_list)
        test_psnr = sum(psnr_list) / len(psnr_list)
        # Print the information.
        x = nib.load("/content/drive/MyDrive/SwinGan/Dataset/post_train/debug/59002_1__post_normalized.nii.gz")
        prediction[mask!=1]=0
        save_img(prediction.squeeze(0).squeeze(0).cpu().detach().numpy(),"/content/drive/MyDrive/SwinGan/experiment/results/RAU/plot.nii.gz",x.header,x.affine)
        # print(f"param_loss = {valid_param_loss:.5f}")
        # print(f"img_loss = {valid_img_loss:.5f}")


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%.15f' % test_pcc, '%.15f' % test_scc, '%.15f' % test_ssim,\
            '%.15f' % test_psnr])
            f.close()
            

        #############################################################################
        # Save best model and early stop
        #############################################################################

       