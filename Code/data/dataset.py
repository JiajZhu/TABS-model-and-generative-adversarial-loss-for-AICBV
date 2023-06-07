### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### This script is modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from .base_dataset import BaseDataset
from data.data_util import *
import torch
# import nibabel as nib
import numpy as np
import nibabel as nib
from torch.utils import data
from glob import glob
# import random

def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    # percentile = np.percentile(img, 99.9)
    # norm = img/percentile
    # max_norm=np.max(norm)
    return norm

class DiscriminatorDataset(BaseDataset):
  def initialize(self,opt):
    self.opt = opt
    self.dataroot=opt.dataroot
    dir_volume = opt.volume_train if opt.type=="train" else opt.volume_validation
    self.dir_volume = os.path.join(opt.dataroot, dir_volume)
    self.file_paths=sorted(make_dataset(self.dir_volume, opt.extension))
    self.img_size = opt.img_size
  def __getitem__(self,index):
    filename = self.file_paths[index]
    suffix = filename.split("__")[-1]
    if suffix == "real_masked_normalized.nii.gz":
      label = 1 #real
    elif suffix == "predicted_masked_normalized.nii.gz":
      label = 0 #fake
    else:
      print("unkown suffix， could not determine label")
    img_unresized = nib.load(filename)
    img_data = img_unresized.get_fdata()
    volume = center_crop_or_pad(img_data,self.img_size) # 3D image
    volume = imgnorm(volume)
    volume = torch.tensor(volume)
    volume=volume[None,:,:,:].type(torch.FloatTensor)
    return label,volume
    
  def __len__(self):
    return len(self.file_paths)

# class GeneratorDataset(BaseDataset):
#   def initialize(self,opt):
#     self.opt = opt
#     self.dataroot=opt.dataroot
#     if opt.type=="train":
#       self.pre_folder= opt.pre_train
#       self.post_folder= opt.post_train
#       self.mask_folder= opt.mask_train
#     elif opt.type=="validation":
#       self.pre_folder= opt.pre_validation
#       self.post_folder= opt.post_validation
#       self.mask_folder= opt.mask_validation
#     elif opt.type=="test": 
#       self.pre_folder= opt.pre_test
#       self.post_folder= opt.post_test
#       self.mask_folder= opt.mask_test      

#     self.pre_paths=sorted(make_dataset(self.pre_folder, opt.extension))
#     self.post_paths=sorted(make_dataset(self.post_folder, opt.extension))
#     self.mask_paths=sorted(make_dataset(self.mask_folder, opt.extension))
#     assert len(self.pre_paths) == len(self.post_paths),"Unequal file nums between pre and post files"
#     assert len(self.post_paths) == len(self.mask_paths),"Unequal file nums between post and mask files"

#     self.img_size = opt.img_size




#   def __getitem__(self,index):
#     pre_filename = self.pre_paths[index]

#     filename_with_extension = pre_filename.split('/')[-1]
#     filename_with_extension_components = filename_with_extension.split('__')
#     filename_with_extension_components[-1] = 'post_normalized.nii.gz'
#     filename_with_extension_post = '__'.join(filename_with_extension_components)
#     post_filename = self.post_folder +  filename_with_extension_post
#     filename_with_extension_components[-1] = 'brain_mask.nii.gz'
#     filename_with_extension_brain_mask = '_'.join(filename_with_extension_components)
#     mask_filename = self.mask_folder + filename_with_extension_brain_mask
#     pre_img = center_crop_or_pad(nib.load(pre_filename).get_fdata(),self.img_size)
#     post_img = center_crop_or_pad(nib.load(post_filename).get_fdata(),self.img_size)
#     mask = center_crop_or_pad(nib.load(mask_filename).get_fdata(),self.img_size)
#     pre_img = imgnorm(pre_img)
#     post_img = imgnorm(post_img)
#     # print('pre_img',pre_img)
#     # print("size of pre_img",pre_img.shape)
#     pre_img = torch.tensor(pre_img)[None,:,:,:].type(torch.FloatTensor)
#     post_img = torch.tensor(post_img)[None,:,:,:].type(torch.FloatTensor)
#     mask = torch.tensor(mask)[None,:,:,:]
#     return pre_img,post_img,mask
    
#   def __len__(self):
#     return len(self.pre_paths)
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

    # Images have a depth of 155, add 5 layers to make them 160
    def zero_pad_depth(self, input_img):
        padding = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)
        padding = padding[:,:,np.newaxis]
        # for tabs
        padding = np.repeat(padding, 5, axis=2)
        return np.dstack([padding,input_img])

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
        pre_path = self.pre_paths[index]
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
#         print(pre_path)
#         print(post_path)
#         print(brain_mask_path)

        if self.augment:
            pre_nifti_scan, post_nifti_scan, brain_mask_nifti_image = self.augment_data(pre_nifti_scan, post_nifti_scan, brain_mask_nifti_image)
        else:
            pre_nifti_scan = pre_nifti_scan.get_fdata().astype(np.float32)
            post_nifti_scan = post_nifti_scan.get_fdata().astype(np.float32)
            brain_mask_nifti_image = brain_mask_nifti_image.get_fdata().astype(np.float32)
            # print(brain_mask_nifti_image.shape)
            # print(pre_nifti_scan.shape)
            # print(post_nifti_scan.shape)


        # Center crop as per network input requirements
        # print(pre_nifti_scan.shape)
        pre_nifti_image  = self.center_crop(pre_nifti_scan)
        post_nifti_image  = self.center_crop(post_nifti_scan)
        brain_mask_nifti_image = self.center_crop(brain_mask_nifti_image)

        # Add depth to the image
        pre_nifti_image = self.zero_pad_depth(pre_nifti_image)
        post_nifti_image = self.zero_pad_depth(post_nifti_image)
        brain_mask_nifti_image = self.zero_pad_depth(brain_mask_nifti_image)

    # Intensity normalization
    # pre_nifti_image , post_nifti_image = self.intensity_normalize(pre_nifti_image,post_nifti_image)



        # Transpose to make depth the first dimension
        brain_mask_nifti_image = brain_mask_nifti_image.transpose((-1,0,1))
        post_nifti_image = post_nifti_image.transpose((-1,0,1))
        pre_nifti_image = pre_nifti_image.transpose((-1,0,1))
        # print(brain_mask_nifti_image.shape)
        # print(pre_nifti_image_normalized.shape)
        # print(post_nifti_image_normalized.shape)

        # vish added: intensity normalize -1 to 1
        if 'GBM' not in pre_path:
            pre_nifti_image, post_nifti_image = self.intensity_normalize(pre_nifti_image, post_nifti_image)
            # print(brain_mask_nifti_image.shape)
            # print(pre_nifti_image_normalized.shape)
            # print(post_nifti_image_normalized.shape)

        if hasattr(self, 'tumor_mask_folder'):
#             filename_with_extension_components[-1] = 'seg_regmask.nii.gz'
#             tumor_mask_nifit_image = self.get_scan(self.tumor_mask_folder, filename_with_extension_components).get_fdata().astype(np.float32)
#             tumor_mask_nifit_image = self.center_crop(tumor_mask_nifit_image)
#             tumor_mask_nifit_image = self.zero_pad_depth(tumor_mask_nifit_image)
#             tumor_mask_nifit_image = tumor_mask_nifit_image.transpose((-1, 0, 1))
            tumor_mask_nifit_image = brain_mask_nifti_image
            # print(brain_mask_nifti_image.shape)
            # print(pre_nifti_image_normalized.shape)
            pre_nifti_image = np.pad(pre_nifti_image, [(16, 16), (0,0), (0,0)])
            post_nifti_image = np.pad(post_nifti_image, [(16, 16), (0,0), (0,0)])
            brain_mask_nifti_image = np.pad(brain_mask_nifti_image, [(16, 16), (0,0), (0,0)])
            # print(brain_mask_nifti_image.shape)

            return pre_nifti_image, post_nifti_image, brain_mask_nifti_image, tumor_mask_nifit_image
        else:
            # print(brain_mask_nifti_image.shape)
            # print(post_nifti_image_normalized.shape)
            # print(pre_nifti_image.shape)

            # uncomment these two lines line if not using TABS
            pre_nifti_image = np.pad(pre_nifti_image, [(16, 16), (0,0), (0,0)])
            post_nifti_image = np.pad(post_nifti_image, [(16, 16), (0,0), (0,0)])
            brain_mask_nifti_image = np.pad(brain_mask_nifti_image, [(16, 16), (0,0), (0,0)])
            # print(pre_nifti_image.shape)
            return pre_nifti_image, post_nifti_image, brain_mask_nifti_image

    def __len__(self):
        """Returns the total number of nifti images."""
        # return len(self.slices_by_scan)
        # return len(self.pre_paths)
        return 1
def get_loader_3D(pre_folder, post_folder, batch_size, num_workers = 1, mode = 'train', shuffle = True, brain_mask_folder=None, augment=False, tumor_mask_folder=None):
    """Builds and returns Dataloader."""
    dataset = NiftiDataset(pre_folder = pre_folder, post_folder = post_folder, mode = mode, brain_mask_folder=brain_mask_folder, augment=augment, tumor_mask_folder=tumor_mask_folder)
    data_loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory=True)
    return data_loader
