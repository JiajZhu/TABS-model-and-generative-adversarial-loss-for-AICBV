import os
import random
from glob import glob
from random import shuffle, getrandbits, randrange
import numpy as np
import nibabel as nib
import torch
import csv
from scipy import ndimage
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from sklearn.feature_extraction import image as sklearn_image
from PIL import Image
from matplotlib import pyplot as plt

import torchio
from torchio import AFFINE, DATA, TYPE
from torchio.transforms import RandomAffine, RandomElasticDeformation, RandomNoise, RandomBiasField, RandomMotion, RandomGhosting, RandomFlip, RandomBlur
from torchio.transforms.interpolation import Interpolation
from torchvision.transforms import ToTensor

def center_crop_or_pad(input_scan, desired_dimension):
    input_dimension = input_scan.shape
    #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

    x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
    y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
    z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
    x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
    y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
    z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

    x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
    y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
    z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
    x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
    y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
    z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]

    output_scan = np.zeros(desired_dimension).astype(np.double)  

    output_scan[x_lowerbound_target : x_upperbound_target, \
                y_lowerbound_target : y_upperbound_target, \
                z_lowerbound_target : z_upperbound_target] = \
    input_scan[x_lowerbound_input: x_upperbound_input, \
                y_lowerbound_input: y_upperbound_input, \
                z_lowerbound_input: z_upperbound_input]

    return output_scan
class NiftiDataset(data.Dataset):
    def __init__(self, pre_folder, post_folder, mode, brain_mask_folder, tumor_mask_folder, augment):
        """Initializes nifti file paths and preprocessing module."""
        # Define the directories where the pre-gado and post-gado nifti scans are stored
        self.pre_folder = pre_folder
        self.post_folder = post_folder
        self.augment = augment

        if self.augment:
            print('Data will be augmented')
        else:
            print('Data will NOT be augmented.')

        print(self.pre_folder)
        print(self.post_folder)
        # Grab all the files in these directories.
        self.pre_paths = list(np.sort(glob(pre_folder + '*.nii.gz')))
        self.post_paths = list(np.sort(glob(post_folder + '*.nii.gz')))
        self.patch_size = [96,96,96]


        # Define the current mode of operation: train, validation or test.
        self.mode = mode

        # Report the number of files in each of the pre-gado, post-gado, and brain mask directories.
        print('Pre {} nifti file count: {}'.format(self.mode, len(self.pre_paths)))
        print('Post {} nifti file count: {}'.format(self.mode, len(self.post_paths)))

        # May use in the future, currently leave it not a required input to the Dataset class
        if brain_mask_folder is not None:
            self.brain_mask_folder = brain_mask_folder

        if tumor_mask_folder is not None:
            tumor_mask_folder = None
            self.tumor_mask_folder = tumor_mask_folder

        # Record how many slices each nifti scan contain in a list.
        # Store depth of each image file in a list by taking last dimension
        self.slices_by_scan = []
        for current_scan_path in self.pre_paths:
            self.slices_by_scan.append(nib.load(current_scan_path).shape[-1]);

    # Center crop the images, keeping the original depth
    # def center_crop(self, img, dim=160):
    def center_crop(self, img, dim=192):
        y, x, z = img.shape
        start_x = x//2 - dim//2
        start_y = y//2 - dim//2
        new_img = img[start_y:start_y+dim, start_x:start_x+dim, :]
        return img[start_y:start_y+dim, start_x:start_x+dim, :]

    # The assumption is that we would apply two classes of augmentations, namely 1) spatial transform
    # and 2) intensity transform, to the incoming data. For spatial transforms, exactly the same
    # transforms shall be applied to all images. For intensity transforms, the same transforms are
    # only applied to pre and post images but not to the masks. The probability for each transform
    # to occur is described in detail below.
    # For medical images, I recommend nearest neighbor interpolation. Linear can also be an acceptable choice. -

    def augment_data(self, pre_img, post_img, brain_mask, tumor_mask=None):
        # No transform, RandomAffine, RandomElasticDeformation, Random Flip
        spatial_augmentation_type = np.random.choice(4, 1, p = [0.1, 0.4, 0.25, 0.25]).item()
        # No transform, RandomNoise, RandomBiasField, RandomMotion, RandomGhosting
        # For now, only introduce random gaussian noise
        intensity_augmentation_type = np.random.choice(4, 1, p = [0.05, 0.48, 0.36, 0.11]).item()

        if spatial_augmentation_type == 1:
            spatial_transform = RandomAffine(seed = 202003, scales = (0.5, 2), degrees = (-5, 5), image_interpolation = Interpolation.NEAREST)
        elif spatial_augmentation_type == 2:
            spatial_transform = RandomElasticDeformation(seed = 202003, image_interpolation = Interpolation.NEAREST, proportion_to_augment = 1, deformation_std= 50)
        elif spatial_augmentation_type == 3:
            axis = np.random.randint(3)
            spatial_transform = RandomFlip(seed = 202003, flip_probability = 1, axes=axis)

        if intensity_augmentation_type == 1:
            intensity_transform = RandomNoise(seed = 202003, std_range = [0.025, 0.025])
        elif intensity_augmentation_type == 2:
            intensity_transform = RandomBlur(seed = 202003, std_range = [0.025, 0.025])
        elif intensity_augmentation_type == 3:
            intensity_transform = RandomGhosting(seed = 202003, proportion_to_augment = 1, num_ghosts = (1, 3))

        subject = {'pre_img': {DATA: torch.from_numpy(pre_img.get_fdata().astype(np.float32)).unsqueeze(0), AFFINE: pre_img.affine, TYPE: torchio.INTENSITY},
                    'post_img': {DATA: torch.from_numpy(post_img.get_fdata().astype(np.float32)).unsqueeze(0), AFFINE: post_img.affine, TYPE: torchio.INTENSITY},
                    'brain_mask': {DATA: torch.from_numpy(brain_mask.get_fdata().astype(np.float32)).unsqueeze(0), AFFINE: brain_mask.affine, TYPE: torchio.LABEL}}

        # Apply spatial transform.
        # spatial_transformed_subject = spatial_transform(subject) if spatial_augmentation_type > 0 else subject
        transformed_subject = spatial_transform(subject) if spatial_augmentation_type > 0 else subject
        transformed_subject = intensity_transform(transformed_subject) if intensity_augmentation_type > 0 else transformed_subject

        pre_img = np.squeeze(transformed_subject['pre_img']['data'].numpy())
        post_img = np.squeeze(transformed_subject['post_img']['data'].numpy())
        brain_mask = np.squeeze(transformed_subject['brain_mask']['data'].numpy())

        return pre_img.astype(np.float32), post_img.astype(np.float32), brain_mask.astype(np.float32)

    # Images have a depth of 155, add 5 layers to make them 192
    def zero_pad_depth(self, input_img):
        padding = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)
        padding = padding[:,:,np.newaxis]
        # for tabs
        
        # padding1 = np.repeat(padding, 18, axis=2)
        # padding2 = np.repeat(padding, 19, axis=2)
        # return np.dstack([padding1,input_img, padding2])
        # padding = np.repeat(padding, 5, axis=2)
        return np.dstack([padding,input_img])
    def sequential_patch(self, image, patch_size, step):
        (H, W, D) = image.shape  # (144,208,208)
        # patch_total_num = (1+len(range(0, D - patch_size, step))) * (1+len(range(0, W - patch_size, step))) * (1+len(range(0, H - patch_size, step)))
        patch_total_num = (len(range(0, D - patch_size, step))) * (len(range(0, W - patch_size, step))) * (len(range(0, H - patch_size, step)))
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
        return patch_mat,coordinate_list, patch_total_num

    # Get the appropriate slice in image given index and path to folder
    def get_scan(self, folder, filename_with_extension_components):
        filename_with_extension = '_'.join(filename_with_extension_components)
        mask_path = folder + filename_with_extension
        nifti_scan = nib.load(mask_path)
        #nifti_scan = nib.load(mask_path).get_fdata().astype(np.float32)
        return nifti_scan

    # vish added
    def intensity_normalize(self, pre, post):
        # scan = ToTensor()(scan)
        min_pre = np.min(pre)
        max_pre = np.max(pre)
        pre_norm = (pre - min_pre) / (max_pre - min_pre)
        post_norm = (post - min_pre) / (max_pre - min_pre)
        # scan = (2*(scan/np.max(scan))-1)
        return pre_norm, post_norm

    # This corresponds to the index position of a particular slice
    def __getitem__(self, index):
        if self.mode != 'test':
            #index = index//50
            index = index//5
        # pre_path = self.pre_paths[index]
        pre_path= self.pre_paths[index]
#         print(pre_path)
        if 'BraTS' in pre_path:
            pre_path = self.pre_paths[index]
            pre_nifti_scan = nib.load(pre_path)

            filename_with_extension = pre_path.split('/')[-1]
            filename_with_extension_components = filename_with_extension.split('_t1reg')
            filename_with_extension_components[-1] = 't1cereg_sc.nii.gz'
            filename_with_extension_post = '_'.join(filename_with_extension_components)
            post_path = self.post_folder + filename_with_extension_post
            post_nifti_scan = nib.load(post_path)

            # Get brain mask
            filename_with_extension_components[-1] = 't1regmask.nii.gz'
            brain_mask_nifti_image = self.get_scan(self.brain_mask_folder, filename_with_extension_components)

        else:
            pre_nifti_scan = nib.load(pre_path)
            filename_with_extension = pre_path.split('/')[-1]
            filename_with_extension_components = filename_with_extension.split('__')
            filename_with_extension_components[-1] = 'post_normalized.nii.gz'
            filename_with_extension_post = '__'.join(filename_with_extension_components)
            post_path = self.post_folder + filename_with_extension_post
            post_nifti_scan = nib.load(post_path)
            filename_with_extension_components[-1] = 'brain_mask.nii.gz'
            filename_with_extension_brain_mask = '_'.join(filename_with_extension_components)
            brain_mask_path = self.brain_mask_folder + filename_with_extension_brain_mask
            brain_mask_nifti_image = nib.load(brain_mask_path)

        if self.augment:
            pre_nifti_scan, post_nifti_scan, brain_mask_nifti_image = self.augment_data(pre_nifti_scan, post_nifti_scan, brain_mask_nifti_image)
        else:
            pre_nifti_scan = pre_nifti_scan.get_fdata().astype(np.float32)
            post_nifti_scan = post_nifti_scan.get_fdata().astype(np.float32)
            brain_mask_nifti_image = brain_mask_nifti_image.get_fdata().astype(np.float32)
        # Center crop as per network input requirements
        # pre_nifti_image  = self.center_crop(pre_nifti_scan)
        # post_nifti_image  = self.center_crop(post_nifti_scan)
        # brain_mask_nifti_image = self.center_crop(brain_mask_nifti_image)

        image_dim=(192,192,192)
        pre_nifti_image = center_crop_or_pad(pre_nifti_scan,image_dim)
        post_nifti_image = center_crop_or_pad(post_nifti_scan,image_dim)
        brain_mask_nifti_image = center_crop_or_pad(brain_mask_nifti_image,image_dim)
        # #Add depth to the image
        # pre_nifti_image = self.zero_pad_depth(pre_nifti_image)
        # post_nifti_image = self.zero_pad_depth(post_nifti_image)
        # brain_mask_nifti_image = self.zero_pad_depth(brain_mask_nifti_image)

        # Transpose to make depth the first dimension
        brain_mask_nifti_image = brain_mask_nifti_image.transpose((-1,0,1))
        post_nifti_image = post_nifti_image.transpose((-1,0,1))
        pre_nifti_image = pre_nifti_image.transpose((-1,0,1))

        output_shape = pre_nifti_image.shape
        patch_size = self.patch_size
        mask = brain_mask_nifti_image
        pre_np = pre_nifti_image
        post_np = post_nifti_image

        if self.mode != 'test':
            
            switch=random.uniform(0,1)
            if switch >=0.2:
                for i in range(1000):
                    ind0 = random.sample(range(round(patch_size[0]/2), output_shape[0] - round(patch_size[0]/2)), 1)[0]
                    ind1 = random.sample(range(round(patch_size[1]/2), output_shape[1] - round(patch_size[1]/2)), 1)[0]
                    ind2 = random.sample(range(round(patch_size[2]/2), output_shape[2] - round(patch_size[2]/2)), 1)[0]
                    if mask[ind0,ind1,ind2]==1:
                          break
            else:
                for i in range(1000):
                    ind0 = random.sample(range(round(patch_size[0]/2), output_shape[0] - round(patch_size[0]/2)), 1)[0]
                    ind1 = random.sample(range(round(patch_size[1]/2), output_shape[1] - round(patch_size[1]/2)), 1)[0]
                    ind2 = random.sample(range(round(patch_size[2]/2), output_shape[2] - round(patch_size[2]/2)), 1)[0]
                    if mask[ind0,ind1,ind2]==0:
                        break
            # ind0 = round(patch_size[0]/2)
            # ind1 = output_shape[1] - round(patch_size[1]/2)
            # ind2 = round(patch_size[2]/2)
            mask_patch=mask[ind0 - round(patch_size[0]/2) : ind0 + round(patch_size[0]/2), ind1-round(patch_size[1]/2) : ind1 + round(patch_size[1]/2),
                            ind2- round(patch_size[2]/2) : ind2 + round(patch_size[2]/2)]
            pre_patch = pre_np[ind0 - int(patch_size[0]/2) : ind0 + int(patch_size[0]/2), ind1-int(patch_size[1]/2) : ind1 + int(patch_size[1]/2),
                            ind2- int(patch_size[2]/2) : ind2 + int(patch_size[2]/2)]
            post_patch = post_np[ind0 - int(patch_size[0]/2) : ind0 + int(patch_size[0]/2), ind1-int(patch_size[1]/2) : ind1 + int(patch_size[1]/2),
                            ind2- int(patch_size[2]/2) : ind2 + int(patch_size[2]/2)]

            return pre_patch, post_patch, mask_patch

        else:

            patch_size=96
            # pre_patches, post_patches,mask_patches,coordinate_list, _ =  self.sequential_patch(pre_np, patch_size, step=63)
            pre_patches,coordinate_list, _ =  self.sequential_patch(pre_np,patch_size,step=8)
            coordinate_list = np.array(coordinate_list)
            # print("right before test return",pre_patches.shape,coordinate_list.shape,pre_nifti_image.shape,post_nifti_image.shape,brain_mask_nifti_image.shape)
            return  pre_patches, coordinate_list, pre_nifti_image, post_nifti_image, brain_mask_nifti_image
            # return  coordinate_list, pre_nifti_image, post_nifti_image, brain_mask_nifti_image
    def __len__(self):
        """Returns the total number of nifti images."""
        # return len(self.slices_by_scan)
        if self.mode != 'test':
          if self.mode == 'validation':
            return len(self.pre_paths)*1
          elif self.mode == 'train':
            return len(self.pre_paths)*2
        else:
            # return len(self.pre_paths)
            return 20
def get_loader_3D(pre_folder, post_folder, batch_size, num_workers = 1, mode = 'train', shuffle = True, brain_mask_folder=None, augment=False, tumor_mask_folder=None):
    """Builds and returns Dataloader."""
    dataset = NiftiDataset(pre_folder = pre_folder, post_folder = post_folder, mode = mode, brain_mask_folder=brain_mask_folder, augment=augment, tumor_mask_folder=tumor_mask_folder)
    data_loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory=False)
    return data_loader
