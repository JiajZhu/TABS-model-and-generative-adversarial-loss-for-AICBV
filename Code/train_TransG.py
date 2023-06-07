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
        model.to(config.device)
    else:
        # model = TransG(config)
        # model = ResTransUNet3D().to(config.device)
        model = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel)
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
							   shuffle = True,
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


    PATH="/content/drive/MyDrive/SwinGan/experiment/TransG/CheckPoint_82_model.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']+1
    criterion=AdaptiveLossFunction(num_dims = 1, float_dtype = np.float32, alpha_init = 2, alpha_hi = 3.5, device = config.device)
    # SSIM=losses.SSIM3D()
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
        wr.writerow(['Current Epoch', 'Train Total Loss','Train PCC','Train SCC','Val Total Loss','Val PCC','Val SCC'])   
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
        scc_list=[]
        pcc_list=[]
        step = 0
        for pre_img,post_img,mask in tqdm(train_loader_3D):
        # for step in tqdm(range(config.train_iteration)):#config.train_iteration
            pre_img = pre_img[:,None,:,:,:]
            post_img = post_img[:,None,:,:,:]
            mask =mask[:,None,:,:,:]
            post_img[mask!=1]=0
            pre_img[mask!=1]=0
            step += 1
            step_start_time = time.time()
            model.train()
            # target = (post_img-pre_img)[0] #target is the CBV map only
            target = (post_img-pre_img)
            target = target.to(config.device)
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # prediction = model(pre_img.to(config.device))[0]
            prediction = model(pre_img.to(config.device))
            prediction_vector = prediction.view(prediction.size(0), -1)
            target_vector = target.view(target.size(0),-1).to(config.device)
            mask_vector = mask.view(mask.size(0), -1).to(config.device)
            # print("target.shape",target.shape)
            # print("prediction.shape",prediction.shape)
            # loss = torch.mean(criterion.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
            loss = torch.mean(criterion.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
            # loss = criterion(prediction,target.to(config.device))
            epoch_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            prediction[mask!=1]=0
            pcc,_=pearsonr(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
            scc,_=spearmanr(prediction_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy(),target_vector[mask_vector == 1].clone().detach().flatten().cpu().numpy())
            if step%1 == 0:
                # grad_norms = [torch.norm(p.grad) for p in model.parameters()]
                # print(f"Gradient norms at iteration {step}: {grad_norms}")
                print('steps {0:d} - training loss {1:.4f}'.format(step,loss.item()))
                
                # predict_img = (prediction + pre_img.to(config.device)).clone().detach().cpu().numpy()

                # predict_cbv = prediction.clone().detach().cpu().numpy()
                # gt_img = post_img
                # gt_cbv = (post_img-pre_img).detach().cpu().numpy()
                # slice_index = predict_img.shape[2]//2
                # plt.subplot(1,5,1)
                # plt.imshow(predict_img[0,0,slice_index,:,:], cmap='gray')
                # plt.axis('off')
                # plt.subplot(1,5,2)
                # plt.imshow(gt_img[0,0,slice_index,:,:], cmap='gray')
                # plt.axis('off')
                # plt.subplot(1,5,3)
                # plt.imshow(predict_cbv[0,0,slice_index,:,:], cmap='gray')
                # plt.axis('off')
                # plt.subplot(1,5,4)
                # plt.imshow(gt_cbv[0,0,slice_index,:,:], cmap='gray')
                # plt.axis('off')
                # plt.subplot(1,5,5)
                # plt.imshow(np.abs(predict_cbv[0,0,slice_index,:,:]-gt_cbv[0,0,slice_index,:,:]), cmap='seismic')
                # plt.axis('off')
                # plt.show()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
            epoch_total_loss.append(loss.item())
            pcc_list.append(pcc)
            scc_list.append(scc)
        train_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        train_pcc = sum(pcc_list) / len(pcc_list)
        train_scc = sum(scc_list) / len(scc_list)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{max_epoch:03d} ] loss = {train_loss:.5f}")

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
        with torch.no_grad():
            step=0
            model.eval()
            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []
            scc_list=[]
            pcc_list=[]
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
                adjust_learning_rate(optimizer, epoch, max_epoch, lr)
                # prediction = model(pre_img.to(config.device))[0]
                prediction = model(pre_img.to(config.device))
                prediction_vector = prediction.view(prediction.size(0), -1)
                target_vector = target.view(target.size(0),-1).to(config.device)
                mask_vector = mask.view(mask.size(0), -1).to(config.device)
                # print("target.shape",target.shape)
                # print("prediction.shape",prediction.shape)
                loss = torch.mean(criterion.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
                # loss = torch.mean(criterion.lossfun((prediction_vector[mask_vector == 1] - prediction_vector[mask_vector == 1])[:,None]))
                # loss = criterion(prediction,target.to(config.device))
                epoch_loss.append(loss)
                prediction[mask!=1]=0
                pcc,_=pearsonr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
                scc,_=spearmanr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
                epoch_step_time.append(time.time() - step_start_time)
                epoch_total_loss.append(loss.item())
                pcc_list.append(pcc)
                scc_list.append(scc)
            valid_loss = sum(epoch_total_loss) / len(epoch_total_loss)
            valid_pcc = sum(pcc_list) / len(pcc_list)
            valid_scc = sum(scc_list) / len(scc_list)
            
            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {valid_loss:.5f}")
            # print(f"param_loss = {valid_param_loss:.5f}")
            # print(f"img_loss = {valid_img_loss:.5f}")


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch + 1), '%.15f' % train_loss, '%.15f' % train_pcc, '%.15f' % train_scc,\
            '%.15f' % valid_loss, '%.15f' % valid_pcc, '%.15f' % valid_scc])
            f.close()
            

        #############################################################################
        # Save best model and early stop
        #############################################################################

        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(LossPath, 'Best_model.pt'))
            early_stop = stop_num
        else:
            early_stop = early_stop - 1
        if epoch % 1==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(LossPath, 'CheckPoint_{}_model.pt'.format(epoch)))

        # if early_stop and epoch < max_epoch - 1:  
        #     continue  
        # else:
        #   print('early stop at'+str(epoch)+'epoch')
        #   break  
        # model.save(os.path.join(LossPath, 'model.pt'))  
