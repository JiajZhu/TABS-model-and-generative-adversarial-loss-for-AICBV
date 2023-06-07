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
from data.dataset import get_loader_3D




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
    
    exp_path = config.root+'/experiment/GAN_Unet'
    save_dir = "/transformer_D"
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
        netG = ResTransUNet3D().to(config.device)
        # netG = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel).to(config.device)
        # netG = TransG(config).to(config.device)
        netD = TransD(config).to(config.device)
    else:
        netG = ResTransUNet3D().to(config.device)
        # netG = ResAttU_Net3D(UnetLayer = config.UnetLayer, img_ch = config.img_ch, output_ch = 1, first_layer_numKernel = config.first_layer_numKernel).to(config.device)
        # netG = TransG(config).to(config.device)
        netD = TransD(config).to(config.device)
    updated_lr = lr

    '''
    Initialize training
    '''
    ##############################################################################
    # Initialize dataloader
    ##############################################################################
    print("loading data")
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


    # optimizerD = optim.Adam(netD.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizerD = optim.SGD(netD.parameters(), lr=config.Unet_lr, momentum=config.Unet_momentum)
    optimizerG = optim.Adam(netG.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    # DPATH="/content/drive/MyDrive/SwinGan/experiment/GAN_Unet/transformer_D/CheckPoint_4_model_D.pt"
    # checkpointD = torch.load(DPATH)
    # optimizerD.load_state_dict(checkpointD['optimizer_state_dict'])
    # epoch_start = checkpointD['epoch']+1
    # netD.load_state_dict(checkpointD['model_state_dict'])
    GPATH="/content/drive/MyDrive/SwinGan/experiment/models/TABS_Patch/TABS-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl"
    # checkpointG = torch.load(GPATH)
    # netG.load_state_dict(checkpointG['model_state_dict'])
    netG.load_state_dict(torch.load(GPATH,map_location='cuda:0'))


    # loss = checkpoint['loss']
    criterionD = nn.CrossEntropyLoss()
    criterion_img = AdaptiveLossFunction(num_dims = 1, float_dtype = np.float32, alpha_init = 2, alpha_hi = 3.5, device = config.device) 
    softmax = nn.Softmax()
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
        wr.writerow(['Current Epoch','Train errD','Train errG','Train img loss','Train PCC','Train SCC',\
        'Valid errD','Valid errG','Valid img loss','Valid PCC','Valid SCC'])   
        f.close()
    
    print('Start training ...')
    torch.cuda.empty_cache()
    best_loss = np.inf
    ################################################################################################################################
    ########################################################### Training ###########################################################
    ################################################################################################################################
    for epoch in range(epoch_start, max_epoch):
        epoch_errD = []
        epoch_errG = []
        epoch_img_loss = []
        scc_list=[]
        pcc_list=[]
        step = 0
        real_label = 1
        fake_label = 0
        for pre_img,post_img,mask in tqdm(train_loader_3D):
        # for step in tqdm(range(config.train_iteration)):#config.train_iteration
            pre_img = pre_img[:,None,:,:,:]
            post_img = post_img[:,None,:,:,:]
            mask = mask[:,None,:,:,:]
            post_img[mask!=1]=0
            pre_img[mask!=1]=0
            step += 1
            step_start_time = time.time()
            netD.train()
            netG.train()
            # adjust_learning_rate(optimizerD, epoch, max_epoch, lr)
            # adjust_learning_rate(optimizerG, epoch, max_epoch, lr)
            #########################
            #1) update netD
            ########################
            ##train with all real
            netD.zero_grad()
            

            real = post_img.to(config.device)
            b_size  = real.size(0)
            label = torch.full((b_size,), real_label,  device=config.device)
            output = netD(real)
            errD_real = criterionD(output, label)
            errD_real.backward()
            # D_x = softmax(output)[:,1].mean().item()
            #train with all generated
            prediction = netG(pre_img.to(config.device))
            prediction[mask!=1] = 0
            fake = pre_img.to(config.device) + prediction
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterionD(output, label)
            errD_fake.backward()
            # D_G_z1 = softmax(output)[:,1].mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            #########################
            #1) update netG
            ########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            target = post_img-pre_img #target is the CBV map only
            target = target.to(config.device)

            prediction_vector = prediction.view(prediction.size(0), -1)
            target_vector = target.view(target.size(0),-1).to(config.device)
            mask_vector = mask.view(mask.size(0), -1).to(config.device)
            img_loss = torch.mean(criterion_img.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
            errG = discriminator_loss_weight*criterionD(output, label) + img_loss_weight*img_loss
            errG.backward()
            # D_G_z2 = softmax(output)[:,1].mean().item()
            optimizerG.step()
            pcc,_=pearsonr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
            scc,_=spearmanr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
            if step%1 == 0:
                # grad_norms_D = [torch.norm(p.grad) for p in netD.parameters()]
                # grad_norms_G = [torch.norm(p.grad) for p in netG.parameters()]
                # print(f"Gradient norms for D at iteration {step}: {grad_norms_D}")
                # print(f"Gradient norms for G at iteration {step}: {grad_norms_G}")
                print('steps {0:d} - errD {1:.4f} - errG {2:.4f} -img loss {3:.4f}'.format(step,errD.item(),errG.item(),img_loss))
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
            pcc_list.append(pcc)
            scc_list.append(scc)
            epoch_errD.append(errD.item())
            epoch_errG.append(errG.item())
            epoch_img_loss.append(img_loss.item())
        train_errD = sum(epoch_errD) / len(epoch_errD)
        train_errG = sum(epoch_errG) / len(epoch_errG)
        train_img_loss = sum(epoch_img_loss) / len(epoch_img_loss)
        train_pcc = sum(pcc_list) / len(pcc_list)
        train_scc = sum(scc_list) / len(scc_list)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{max_epoch:03d} ] errD = {train_errD:.5f},errG = {train_errG:.5f},img loss = {train_img_loss:.5f}")

##################################################################################################################################
########################################################### Validation ###########################################################
##################################################################################################################################
        with torch.no_grad():
            epoch_errD = []
            epoch_errG = []
            epoch_img_loss = []
            scc_list=[]
            pcc_list=[]
            step=0
            for pre_img,post_img,mask in tqdm(validation_loader_3D):
                step+=1
                step_start_time = time.time()
                netD.eval()
                netG.eval()
                pre_img = pre_img[:,None,:,:,:]
                post_img = post_img[:,None,:,:,:]
                mask = mask[:,None,:,:,:]
                post_img[mask!=1]=0
                pre_img[mask!=1]=0
                #########################
                #1) update netD
                ########################
                ##train with all real
                # real = post_img.to(config.device)
                # b_size  = real.size(0)
                # label = torch.full((b_size,), real_label, dtype=torch.float, device=config.device)
                # output = netD(real).view(-1)
                # errD_real = criterion(output, label)
                # D_x = output.mean().item()

                #train with all generated
                prediction = netG(pre_img.to(config.device))
                prediction[mask!=1] = 0
                fake = pre_img.to(config.device) + prediction
                # label.fill_(fake_label)
                # output = netD(fake.detach()).view(-1)
                errD_fake = criterionD(output, label)
                # D_G_z1 = output.mean().item()
                # errD = errD_real + errD_fake

                #########################
                #1) update netG
                ########################


                label.fill_(real_label)
                output = netD(fake)
                target = post_img-pre_img #target is the CBV map only
                target = target.to(config.device)

                prediction_vector = prediction.view(prediction.size(0), -1)
                target_vector = target.view(target.size(0),-1).to(config.device)
                mask_vector = mask.view(mask.size(0), -1).to(config.device)
                img_loss = torch.mean(criterion_img.lossfun((prediction_vector[mask_vector == 1] - target_vector[mask_vector == 1])[:,None]))
                errG = discriminator_loss_weight*criterionD(output, label) + img_loss_weight*img_loss
                D_G_z2 = softmax(output)[:,1].mean().item()  
                pcc,_=pearsonr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
                scc,_=spearmanr(prediction.clone().detach().flatten().cpu().numpy(),target.clone().detach().flatten().cpu().numpy())
                epoch_errD.append(errD.item())
                epoch_errG.append(errG.item())
                epoch_img_loss.append(img_loss.item())
                pcc_list.append(pcc)
                scc_list.append(scc)
                print('steps {0:d} - errD {1:.4f} - errG {2:.4f} -img loss {3:.4f}'.format(step,errD.item(),errG.item(),img_loss))
            valid_errD = sum(epoch_errD) / len(epoch_errD)
            valid_errG = sum(epoch_errG) / len(epoch_errG)
            valid_img_loss = sum(epoch_img_loss) / len(epoch_img_loss)
            valid_pcc = sum(pcc_list) / len(pcc_list)
            valid_scc = sum(scc_list) / len(scc_list)
            # Print the information.
            print(f"[ Validation | {epoch + 1:03d}/{max_epoch:03d} ] errD = {valid_errD:.5f},errG = {valid_errG:.5f},img loss = {valid_img_loss:.5f}")


        ##############################################################################
        # Learning curve
        ##############################################################################
        with open(os.path.join(LossPath + '/learning_curve.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['%d' %(epoch + 1), '%.15f' % train_errD, '%.15f' % train_errG, '%.15f' % train_img_loss,'%.15f' % train_pcc,'%.15f' % train_scc,\
            '%.15f' % valid_errD, '%.15f' % valid_errG, '%.15f' % valid_img_loss,'%.15f' % valid_pcc,'%.15f' % valid_scc])
            f.close()
            
        # wr.writerow(['Current Epoch', 'Train errD','Train errG','Train img loss','Train PCC','Train SCC',\
        # 'Valid errD','Valid errG','Valid img loss','Valid PCC','Valid SCC'])
        #############################################################################
        # Save best model and early stop
        #############################################################################

        if valid_errG < best_loss:
            best_epoch = epoch
            best_loss = valid_errG
            # netD.save(os.path.join(LossPath, 'Best_model_D.pt'))
            # netG.save(os.path.join(LossPath, 'Best_model_G.pt'))
            early_stop = stop_num
        else:
            early_stop = early_stop - 1
        if epoch % 1 == 0:    ### update
            # netD.save(os.path.join(LossPath, 'Best_{}_model_D.pt'.format(epoch)))  
            # netG.save(os.path.join(LossPath, 'Best_{}_model_G.pt'.format(epoch)))  
            torch.save({
            'epoch': epoch,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'loss': errD,
            }, os.path.join(LossPath, 'CheckPoint_{}_model_D.pt'.format(epoch)))

            torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'loss': errG,
            }, os.path.join(LossPath, 'CheckPoint_{}_model_G.pt'.format(epoch)))
        if early_stop and epoch < max_epoch - 1:  
            continue  
        else:
          print('early stop at'+str(epoch)+'epoch')
          break  
        # netD.save(os.path.join(LossPath, 'model_D.pt'))  
        # netG.save(os.path.join(LossPath, 'model_G.pt'))  