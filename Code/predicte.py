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
import nibabel as nib
from options.train_options import get_TransD_config
from data.data_loader import CreateDataLoader
from models.TransG import TransG,ResAttU_Net3D
from data.data_util import *
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from data.dataset import get_loader_3D
from options.train_options import get_TransD_config
from models.TransUnet_Model import ResTransUNet3D

# from torchmetrics import PeakSignalNoiseRatio
def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    # percentile = np.percentile(img, 99.9)
    # norm = img/percentile
    # max_norm=np.max(norm)
    return norm

config = get_TransD_config()
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
              shuffle = True,
              mode = 'validation',
              augment=False)
model=ResTransUNet3D()
# model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
model.to(config.device)
# model_path="/content/drive/MyDrive/SwinGan/experiment/ResAttU_Net3D-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl"
model_path="/content/drive/MyDrive/SwinGan/experiment/models/RAU_Patch/ResAttU_Net3D-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl"
model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
predict_train_folder="/content/drive/MyDrive/SwinGan/Dataset/predictions/train"
predict_valid_folder="/content/drive/MyDrive/SwinGan/Dataset/predictions/validation"

for pre_img,post_img,mask,pre_path in tqdm(train_loader_3D):
    pre_img=pre_img.to(config.device)
    post_img=post_img.to(config.device)
    pre_img = pre_img[:,None,:,:,:]
    post_img = post_img[:,None,:,:,:]
    mask_index =mask[:,None,:,:,:]
    post_img[mask_index!=1]=0
    pre_img[mask_index!=1]=0
    prediction = model(pre_img.to(config.device))
    prediction[mask_index!=1]=0
    generated_cbv = pre_img+prediction
    generated_cbv = imgnorm(generated_cbv.squeeze(0).squeeze(0).detach().cpu().numpy())
    pre_path=pre_path[0]

    pre_img=nib.load(pre_path)
    pre_header=pre_img.header
    pre_affine=pre_img.affine
    predictnib=nib.nifti1.Nifti1Image(generated_cbv, affine=pre_affine, header=pre_header)
    predict_name=pre_path.split("/")[-1]
    predict_name_components=predict_name.split("__")
    predict_name_components[-1] = "predicted_masked_normalized.nii.gz"
    predict_name="__".join(predict_name_components)
    nib.save(predictnib, "{}/{}".format(predict_train_folder,predict_name))


    real_cbv=imgnorm(post_img.squeeze(0).squeeze(0).detach().cpu().numpy())
    realnib=nib.nifti1.Nifti1Image(real_cbv, affine=pre_affine, header=pre_header)
    real_name=pre_path.split("/")[-1]
    real_name_components=real_name.split("__")
    real_name_components[-1] = "real_masked_normalized.nii.gz"
    real_name="__".join(real_name_components)
    nib.save(realnib, "{}/{}".format(predict_train_folder,real_name))
for pre_img,post_img,mask,pre_path in tqdm(validation_loader_3D):
    pre_img=pre_img.to(config.device)
    post_img=post_img.to(config.device)
    pre_img = pre_img[:,None,:,:,:]
    post_img = post_img[:,None,:,:,:]
    mask_index =mask[:,None,:,:,:]
    post_img[mask_index!=1]=0
    pre_img[mask_index!=1]=0
    prediction = model(pre_img.to(config.device))
    prediction[mask_index!=1]=0
    generated_cbv = pre_img+prediction
    generated_cbv = imgnorm(generated_cbv.squeeze(0).squeeze(0).detach().cpu().numpy())

    pre_path=pre_path[0]
    pre_img=nib.load(pre_path)
    pre_header=pre_img.header
    pre_affine=pre_img.affine

    predictnib=nib.nifti1.Nifti1Image(generated_cbv, affine=pre_affine, header=pre_header)
    predict_name=pre_path.split("/")[-1]
    predict_name_components=predict_name.split("__")
    predict_name_components[-1] = "predicted_masked_normalized.nii.gz"
    predict_name="__".join(predict_name_components)
    nib.save(predictnib, "{}/{}".format(predict_valid_folder,predict_name))

    real_cbv=imgnorm(post_img.squeeze(0).squeeze(0).detach().cpu().numpy())
    realnib=nib.nifti1.Nifti1Image(real_cbv, affine=pre_affine, header=pre_header)
    real_name=pre_path.split("/")[-1]
    real_name_components=real_name.split("__")
    real_name_components[-1] = "real_masked_normalized.nii.gz"
    real_name="__".join(real_name_components)
    nib.save(realnib, "{}/{}".format(predict_valid_folder,real_name))
# import glob
# import torchio as tio
# import nibabel as nib
# train_path="/content/drive/MyDrive/SwinGan/Dataset/predictions/train"
# valid_path="/content/drive/MyDrive/SwinGan/Dataset/predictions/validation"
# real_folder=sorted(glob.glob(train_path + '/*real_masked_normalized.nii.gz'))
# fake_folder=sorted(glob.glob(train_path + '/*predicted_masked_normalized.nii.gz'))
# for i in range(10):
#   # real=tio.ScalarImage(real_folder[i])
#   # real.plot()
#   real=nib.load(real_folder[i])
#   fake=nib.load(fake_folder[i])
#   real=real.get_fdata().astype(np.float32)
#   fake=fake.get_fdata().astype(np.float32)
#   # real=tio.ScalarImage(real_folder[i])
#   # fake=tio.ScalarImage(fake_folder[i])
#   res=np.abs(real-fake)
#   plt.subplot(1,3,1)
#   plt.imshow(real[80,:,:], cmap='gray')
#   plt.subplot(1,3,2)
#   plt.imshow(fake[80,:,:], cmap='gray')
#   plt.subplot(1,3,3)
#   plt.imshow(res[80,:,:], cmap='seismic')
#   plt.show()