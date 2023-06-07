import ml_collections
import torch.nn as nn
import torch
import numpy as np

'''
********************************************************
                   Swin Gan
********************************************************
'''
def get_TransD_config(type='train'):
    #####################################################
    # Training information
    #####################################################
    config = ml_collections.ConfigDict()
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.train=True
    config.pin_memory=True
    config.train_iteration = 2
    config.validation_iteration = 1
    config.max_epoch = 100
    config.root = "/content/drive/MyDrive/SwinGan"
    config.Discriminator_dataroot = config.root + "/Datasets/predictions"
    #####################################################
    # Model configurations
    #####################################################
    ###SwinTransformer
    config.patch_size = 4
    config.in_chans = 1
    config.reg_head_chan = 16
    config.depths = [2, 2, 6, 2]
    config.depths_decoder = [2, 2, 6, 2]
    config.num_heads=[3, 6, 12, 24]
    config.mlp_ratio = 4.
    config.qkv_bias = True
    config.qk_scale = None
    config.drop_rate = 0.5
    config.attn_drop_rate = 0.
    config.drop_path_rate = 0.1
    config.norm_layer = nn.LayerNorm
    config.ape = False
    config.patch_norm = True
    config.int_steps= 7
    config.int_downsize = 2
    config.final_upsample = 'expand_first'
    ###ResAttU_Net3D
    config.UnetLayer = 6   
    config.img_ch = 1
    config.first_layer_numKernel = 8
    config.Unet_momentum = float(0.9)
    config.Unet_lr = 0.001
    config.Trans_lr = 1e-6
    #####################################################
    # Data loader configurations
    #####################################################
    config.datatype="Discriminator"
    config.train_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre_train/preprocessed/"
    config.train_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post_train/preprocessed/"
    config.train_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/brain_mask_train/"
    # config.validation_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre_validation/preprocessed/"
    # config.validation_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post_validation/preprocessed/"
    # config.validation_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/brain_mask_validation/"
    # config.test_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre_test/preprocessed/"
    # config.test_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post_test/preprocessed/"
    # config.test_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/brain_mask_test/"

    # config.validation_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre"
    # config.validation_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post"
    # config.validation_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/mask"
    # config.train_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre_train/debug/"
    # config.train_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post_train/debug/"
    # config.train_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/brain_mask_train/debug/"
    config.validation_pre_folder = "/content/drive/MyDrive/SwinGan/Dataset/pre_test/preprocessed/"
    config.validation_post_folder = "/content/drive/MyDrive/SwinGan/Dataset/post_test/preprocessed/"
    config.validation_brain_mask_folder = "/content/drive/MyDrive/SwinGan/Dataset/brain_mask_test/"

    config.img_size = (192, 192, 192)
    config.window_size = (6, 6, 6)
    config.embed_dim = 48
    config.dataroot = '/content/drive/MyDrive/SwinGan/Dataset/predictions'
    config.volume_train = 'train/'
    config.volume_validation = 'validation/'
    config.resolution = 1.0
    config.checkpoints_dir = '/TransD'
    config.use_checkpoint = False
    # config.checkpoint = '/learning_curve.pt'
    config.pad_shape = config.img_size
    config.add_feat_axis = True
    config.resize_factor = 1.
    # config.add_batch_axis = False
    
    config.Augmentation = False
    config.random_blur = True
    #####################################################
    # Train configurations
    #####################################################
    config.nThreads = 0
    config.gpu_ids = 0
    config.lr = 1e-5



    config.type = type
    if type == 'train':
        config.batchsize = 1
        config.num_workers = 1
    elif type == 'validation':
        config.batchsize = 1
        config.num_workers = 1
    elif type == 'test':
        config.batchsize = 1
        config.num_workers = 1
    config.extension = '.nii.gz'
    return config

