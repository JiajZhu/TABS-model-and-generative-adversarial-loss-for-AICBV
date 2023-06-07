import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from torch import Tensor
# from network import  ResAttU_Net3D
from scipy.stats import pearsonr,spearmanr
import os
import nibabel as nib
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import img_as_float
import argparse
import pickle
from robust_loss_pytorch import AdaptiveLossFunction
import pytorch_msssim.pytorch_msssim as pytorch_msssim_real
from evaluation import compute_MAE, compute_MSE, compute_PSNR, compute_NCC, compute_SR, compute_PR
from models.patch.TABS_Model import TABS
from data.data_loader_3_patch import get_loader_3D
from misc import printProgressBar
import time

parser = argparse.ArgumentParser()
#/content/drive/MyDrive/SwinGan/Dataset/pre_test/preprocessed/
#/content/drive/MyDrive/SwinGan/Dataset/post_test/preprocessed/
#/content/drive/MyDrive/SwinGan/Dataset/brain_mask_test/
parser.add_argument('--test_pre_folder', default='/content/drive/MyDrive/SwinGan/Dataset/pre_test/preprocessed/', type=str)
parser.add_argument('--test_post_folder', default='/content/drive/MyDrive/SwinGan/Dataset/post_test/preprocessed/', type=str)
parser.add_argument('--pretrained_path', default='/content/drive/MyDrive/SwinGan/experiment/models/TABS_Patch/TABS-brain_mask-SGD-0.0010-CVPR_Adaptive_loss-1-best.pkl', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--current_prediction_path', default='/content/drive/MyDrive/SwinGan/experiment/results/TABS', type=str)
parser.add_argument('--masks_folder', default='/content/drive/MyDrive/SwinGan/Dataset/brain_mask_test/', type=str)
parser.add_argument('--save_prediction_nifti', default=True, type=bool)


args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def sequential_patch(image: NDArray, patch_size, step):

    (H, W, D) = image.shape  # (144,208,208)
    patch_total_num = (1+len(range(0, D - patch_size, step))) * (1+len(range(0, W - patch_size, step))) * (1+len(range(0, H - patch_size, step)))
    count=0
    coordinate_list = []
    patch_mat = np.float32(np.zeros((patch_total_num,patch_size,patch_size,patch_size)))
    #image_zeropadding = patch/2
    for z in range(0, D - patch_size, step):
        for y in range(0, W- patch_size, step):
            for x in range(0, H - patch_size, step):

                patch = image[x : x + patch_size, y : y + patch_size, z : z + patch_size]
                patch_mat[count,:,:,:]=patch

                coordinate = (x, y, z)
                coordinate_list.append(coordinate)
                count=count+1
                del patch

    return patch_mat, coordinate_list, patch_total_num

def reconstruction(patch_mat,coordinate_list,image_shape,patch_size):
    (H, W, D) = image_shape
    map = np.zeros((H, W, D))
    repeat = np.zeros((H, W, D))
    patch_mat[np.isnan(patch_mat)] = 0

    for index in range(len(coordinate_list)):
        prep = np.squeeze(patch_mat[index,:,:,:])
        (x,y,z)=coordinate_list[index]
        map[x : x + patch_size, y : y + patch_size, z : z + patch_size] = map[x : x + patch_size, y : y + patch_size, z : z + patch_size] + prep
        repeat[x : x + patch_size, y : y + patch_size, z : z + patch_size] = repeat[x : x + patch_size, y : y + patch_size, z : z + patch_size] + 1
    map = map/repeat
    return map

def zero_pad(scan):
    post_padding_dimension = 240
    pad_val = int((post_padding_dimension - scan.shape[1])/2)
    padded = np.pad(scan,((pad_val,pad_val),(pad_val,pad_val), (0,0)), mode='constant', constant_values=0)
    padded = padded[:,:,0:155]

    return padded

def get_metrics(cur_prediction,coordinate,pre_image,post_image, brain_mask_image,patch_size):
  (x,y,z)=coordinate
  curr_target = post_image[x:x+patch_size,y:y+patch_size,z:z+patch_size] - pre_image[x:x+patch_size,y:y+patch_size,z:z+patch_size]
  curr_mask = brain_mask_image[x:x+patch_size,y:y+patch_size,z:z+patch_size]
  cur_prediction_vector = cur_prediction.contiguous().view(1,-1).cuda(args.device)
  curr_target_vector = curr_target.contiguous().view(1,-1).cuda(args.device)
  curr_brain_mask_vector = curr_mask.contiguous().view(1,-1).cuda(args.device)
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
if __name__ == '__main__':
    start_time = time.time()
    print(96//16)
    print(start_time)

    test_loader_3D = get_loader_3D(pre_folder = args.test_pre_folder,
             post_folder = args.test_post_folder,
             brain_mask_folder = args.masks_folder,
             tumor_mask_folder = args.masks_folder,
             batch_size = 1,
             num_workers = 0,
             shuffle = False, mode = 'test', augment=False)

    unet = TABS(img_dim=96)
    unet.load_state_dict(torch.load(args.pretrained_path))
    # PATH="/content/drive/MyDrive/SwinGan/experiment/GAN_TABS_patch/CheckPoint_4_model_G.pt"
    # checkpoint = torch.load(PATH)
    # unet.load_state_dict(checkpoint['model_state_dict'])
    print('loaded best model')
    unet = unet.cuda(args.device)

    # Mke the directory for the current prediction.
    if not os.path.exists(args.current_prediction_path):
        os.makedirs(args.current_prediction_path)

    unet.train(False)
    unet.eval()
    length = 0
    pr_length = 0
    sr_length = 0
    test_epoch_loss_sum = 0

    # Evaluation metrics.
    per_person_psnr = 0
    mean_absolute_error = 0
    mean_squared_error = 0
    peak_SNR = 0
    structural_similarity = 0
    multi_scale_structural_similarity = 0
    normalized_cross_correlation = 0
    pr = 0
    sr = 0

    test_pre_paths = list(np.sort(glob(args.test_pre_folder + '*.nii.gz')))
    test_post_paths = list(np.sort(glob(args.test_post_folder + '*.nii.gz')))
    mse_list=[]
    pcc_list=[]
    scc_list=[]
    index_list=[]
    for batch, (pre_image_norms, coordinate_list, pre_image, post_image, brain_mask_image) in enumerate(test_loader_3D):
        patch_size=96
        pre_image_norms=pre_image_norms.to(device)
        coordinate_list=coordinate_list.to(device)
        pre_image=pre_image.to(device)
        post_image=post_image.to(device)
        brain_mask_image=brain_mask_image.to(device)
        corresponding_pre_scan_path = test_pre_paths[batch]
        file_name_with_extension = corresponding_pre_scan_path.split('/')[-1]
        filename = file_name_with_extension[:-11]
        if args.save_prediction_nifti == True:
            corresponding_pre_nifti = nib.load(corresponding_pre_scan_path)
            current_scan_affine = corresponding_pre_nifti.affine
            current_scan_header = corresponding_pre_nifti.header
            del corresponding_pre_nifti

        patch_total_num = pre_image_norms.shape[1]
        prediction_mat=np.float32(np.zeros((patch_total_num,96,96,96)))
        # print("patch_total_num",patch_total_num)
        for i in range(0,patch_total_num):
            # pre_image = pre_images[:,i]
            pre_image_norm = pre_image_norms[:,i]
            # print("pre_image_norm.shape",pre_image_norm.shape)
            # post_image = post_images[:,i]
            # tumor_mask = tumor_masks[:,i]

            inp_image = pre_image_norm.to(args.device)
            # post_image = post_image.to(args.device)

            # Create a 5th dimension, as needed by 3D convolution
            inp_image = inp_image[:,np.newaxis,:,:,:]
            # pre_image_norm = pre_image_norm[:,np.newaxis,:,:,:]
            # post_image = post_image[:,np.newaxis,:,:,:]

            # Use the network to make predictions with pre_image as input.
            cur_prediction = unet(inp_image)[0]
            # print("cur_prediction.shape",cur_prediction.shape)
            
            mse,pcc,scc=get_metrics(cur_prediction[0],coordinate_list[0][i],pre_image[0],post_image[0], brain_mask_image[0],patch_size)
            if mse != None:
              mse_list.append(mse)
              pcc_list.append(pcc)
              scc_list.append(scc)
              index_list.append(i)
            cur_prediction = cur_prediction.detach().cpu().numpy()
            prediction_mat[i,:,:,:] = cur_prediction[0]
            del cur_prediction
            # break
        coordinate_list = coordinate_list.cpu().detach().numpy()
        Prediction=reconstruction(prediction_mat,coordinate_list[0],(192,192,192),patch_size)
        print("Prediction.shape",Prediction.shape)
        Prediction[np.isnan(Prediction)] = 0
        brain_mask = brain_mask_image.to(args.device)
        print("brain_mask.shape",brain_mask.shape)
        Target = (post_image - pre_image)[0]
        print("Target.shape",Target.shape)
        # Target = (Target - torch.min(Target)) / (torch.max(Target) - torch.min(Target))

        pre_image_comp = pre_image[0]

        Target = Target.cuda(args.device)

        Prediction = torch.tensor(Prediction).cuda(args.device)

        # print()
        Target = Target.type(torch.cuda.FloatTensor)
        Prediction = Prediction.type(torch.cuda.FloatTensor)

        # Prediction_vector = Prediction.view(Prediction.size(0), -1)
        # Target_vector = Target.view(Target.size(0), -1)
        # pre_vector = pre_image_comp.view(pre_image_comp.size(0), -1)


        # print(Target.shape)
        # print(pre_image_comp.shape)
        Prediction_vector = Prediction.contiguous().view(1,-1).cuda(args.device)
        Target_vector = Target.contiguous().view(1,-1).cuda(args.device)
        pre_vector = pre_image_comp.contiguous().view(1,-1).cuda(args.device)
        # mask_vector = brain_mask.contiguous().view(1,-1).cuda(args.device)
        brain_mask_vector = brain_mask.contiguous().view(1,-1).cuda(args.device)
        # print()

        # brain_mask_vector = brain_mask.view(brain_mask.size(0), -1)
        Prediction_vector_array = Prediction_vector[brain_mask_vector == 1].cpu().detach().numpy()
        Target_vector_array = Target_vector[brain_mask_vector == 1].cpu().detach().numpy()
        structural_similarity += pytorch_msssim_real.ssim(torch.mul(Prediction, brain_mask.type(torch.cuda.FloatTensor)), torch.mul(Target, brain_mask.type(torch.cuda.FloatTensor))).item()
        multi_scale_structural_similarity += pytorch_msssim_real.msssim(torch.mul(Prediction, brain_mask.type(torch.cuda.FloatTensor)), torch.mul(Target, brain_mask.type(torch.cuda.FloatTensor))).item()
        # else:
        #     Prediction_vector_array = Prediction_vector.cpu().detach().numpy()
        #     Target_vector_array = Target_vector.cpu().detach().numpy()
        #     structural_similarity += pytorch_msssim.ssim(Prediction, Target).item()
            # multi_scale_structural_similarity += pytorch_msssim.msssim(Prediction, Target).item()



        current_psnr = compute_PSNR(Prediction_vector_array, Target_vector_array)
        mean_absolute_error += compute_MAE(Prediction_vector_array, Target_vector_array)
        mean_squared_error += compute_MSE(Prediction_vector_array, Target_vector_array)
        peak_SNR += current_psnr
        normalized_cross_correlation += compute_NCC(Prediction_vector_array, Target_vector_array)
        per_person_psnr += current_psnr

        length += 1

        pr_val, pr_p_val = compute_PR(Prediction_vector_array, Target_vector_array)
        sr_val, sr_p_val =  compute_SR(Prediction_vector_array, Target_vector_array)
        if sr_p_val < 0.05:
            sr += sr_val
            sr_length += 1
        if pr_p_val < 0.05:
            pr += pr_val
            pr_length += 1

        if batch != 0:
            printProgressBar(batch, len(test_loader_3D))


        # Add the prediction images to the prediction scan if appropriate if we would like to have nifti output.
        if args.save_prediction_nifti == True:
            Prediction = np.squeeze(Prediction).cpu().detach().numpy()
            brain_mask = np.squeeze(brain_mask).cpu().detach().numpy()
            Target = np.squeeze(Target.cpu().detach().numpy())
            # print(Prediction.shape)
            # print(brain_mask.shape)
            # print(Target.shape)

            # brain_mask = zero_pad(brain_mask.transpose((1, -1, 0)))
            # Target = zero_pad(Target.transpose((1, -1, 0)))
            # current_prediction_scan = zero_pad(Prediction.transpose((1, -1, 0)))
            brain_mask = brain_mask.transpose((1, -1, 0))
            Target = Target.transpose((1, -1, 0))
            current_prediction_scan = Prediction.transpose((1, -1, 0))
            # print(current_prediction_scan.shape)
            # print(brain_mask.shape)
            # print(Target.shape)


            # Apply brain mask to target and prediction scans such that artifacts outside the brain region are removed
            # Upon saving the image
            current_prediction_scan = current_prediction_scan * brain_mask
            current_prediction_scan_nifti = nib.Nifti1Image(current_prediction_scan, current_scan_affine, current_scan_header)
            nib.save(current_prediction_scan_nifti, args.current_prediction_path + filename + '_modelprediction.nii.gz')

            # Save the normalized Gado-uptake ground truth if we ask for that.
            if 1 == 1:
                Target = Target * brain_mask
                corresponding_gado_uptake_GT_nifti_normalized = nib.Nifti1Image(Target, current_scan_affine, current_scan_header)
                nib.save(corresponding_gado_uptake_GT_nifti_normalized, args.current_prediction_path + filename + '_gado-uptake.nii.gz')
                # Save the normalized and cross-subject standardized GT.
                del corresponding_pre_scan_path, corresponding_gado_uptake_GT_nifti_normalized
            del current_prediction_scan, current_prediction_scan_nifti, current_scan_affine, current_scan_header
            # Empty cache to free up memory at the end of each batch.
            torch.cuda.empty_cache()

        # del Prediction_vector, Target_vector, pre_vector, mask_vector, batch, pre_image_norms, coordinate_list, pre_image, post_image, brain_mask_image, pre_image_norm, inp_image, prediction_mat, Prediction, brain_mask, Target, pre_image_comp, brain_mask_vector, Prediction_vector_array, Target_vector_array
        del Prediction_vector, Target_vector, pre_vector, batch, pre_image_norms, coordinate_list, pre_image, post_image, brain_mask_image, pre_image_norm, inp_image, prediction_mat, Prediction, brain_mask, Target, pre_image_comp, brain_mask_vector, Prediction_vector_array, Target_vector_array
        # break
    # break

    test_epoch_loss = test_epoch_loss_sum / length

    mean_absolute_error = mean_absolute_error / length
    mean_squared_error = mean_squared_error / length
    peak_SNR = peak_SNR / length
    structural_similarity = structural_similarity / length
    multi_scale_structural_similarity = multi_scale_structural_similarity / length
    normalized_cross_correlation = normalized_cross_correlation / length
    pr = pr / pr_length
    sr = sr / sr_length
    mse_patch = sum(mse_list)/len(mse_list)
    pcc_patch = sum(pcc_list)/len(pcc_list)
    scc_patch = sum(scc_list)/len(scc_list)
    # print('Model type: %s, Special Note: %s, Optimizer: %s, Initial learning rate: %.4f, Loss function: %s, Batch size: %d, Best or last: %s, Test Loss: %.6f' \
    #         % (self.model_type, self.special_note, self.optimizer_choice, self.initial_lr, self.loss_function_name, self.batch_size, which_unet, test_epoch_loss))
    # print('MAE: %.6f, MSE: %.6f, PSNR: %.6f, SSIM: %.6f, MSSSIM: %.6f, NCC: %.6f' \
    # 		% (mean_absolute_error, mean_squared_error, peak_SNR, structural_similarity, multi_scale_structural_similarity, normalized_cross_correlation))
    print('MAE: %.6f, MSE: %.6f, PSNR: %.6f, SSIM: %.6f, MS_SSIM: %.6f, NCC: %.6f' \
            % (mean_absolute_error, mean_squared_error, peak_SNR, structural_similarity, multi_scale_structural_similarity, normalized_cross_correlation))

    print(f'PR: {pr} ', f'SR: {sr}')

    print('time taken: {}'.format(time.time() - start_time))

    log_filename = '/content/drive/MyDrive/SwinGan/experiment/GAN_TABS_patch/test/results.txt'
    with open(log_filename, 'a') as log_file:

        # Write a timestamp and a message to the log file
        message = 'MAE: %.6f, MSE: %.6f, PSNR: %.6f, SSIM: %.6f, MS_SSIM: %.6f, NCC: %.6f' \
                % (mean_absolute_error, mean_squared_error, peak_SNR, structural_similarity, multi_scale_structural_similarity, normalized_cross_correlation)
        log_file.write(message)
        log_file.write('\n')
        message2 = f'PR: {pr} ' + f'SR: {sr}'+ f'mse_patch: {mse_patch} ' + f'pcc_patch: {pcc_patch}' + f'scc_patch: {scc_patch} '
        log_file.write(message2)

    with open('patch_lists.pkl', 'wb') as f:
        pickle.dump((mse_list,pcc_list,scc_list,index_list), f)