import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
# import models.configs_TransMorph as configs
from models.modelio import LoadableModel, store_config_args
class Attention_block_3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace = True)
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class resconv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):
        
        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x

class up_conv_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )
        self.ch_in = ch_in
        self.ch_out = ch_out
    def forward(self,x):
        x = self.up(x)
        return x
class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
class Unet_D(LoadableModel):
    @store_config_args
    def __init__(self, UnetLayer, img_ch = 1, output_ch = 1, first_layer_numKernel = 64):
        super(Unet_D, self).__init__()

        self.UnetLayer = UnetLayer
        self.Maxpool = nn.MaxPool3d(kernel_size = 2, stride = 2)

        self.Conv1 = resconv_block_3D(ch_in = img_ch, ch_out = first_layer_numKernel)
        self.Conv2 = resconv_block_3D(ch_in = first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Conv3 = resconv_block_3D(ch_in = 2 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Conv4 = resconv_block_3D(ch_in = 4 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
        self.Conv5 = resconv_block_3D(ch_in = 8 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)
        self.Conv6 = resconv_block_3D(ch_in = 16 * first_layer_numKernel, ch_out = 32 * first_layer_numKernel)

        self.Up6 = up_conv_3D(ch_in = 32 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)
        self.Att6 = Attention_block_3D(F_g = 16 * first_layer_numKernel, F_l = 16 * first_layer_numKernel, F_int = 8 * first_layer_numKernel)
        self.Up_conv6 = resconv_block_3D(ch_in = 32 * first_layer_numKernel, ch_out = 16 * first_layer_numKernel)

        self.Up5 = up_conv_3D(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)
        self.Att5 = Attention_block_3D(F_g = 8 * first_layer_numKernel, F_l = 8 * first_layer_numKernel, F_int = 4 * first_layer_numKernel)
        self.Up_conv5 = resconv_block_3D(ch_in = 16 * first_layer_numKernel, ch_out = 8 * first_layer_numKernel)

        self.Up4 = up_conv_3D(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        self.Att4 = Attention_block_3D(F_g = 4 * first_layer_numKernel, F_l = 4* first_layer_numKernel, F_int = 2 * first_layer_numKernel)
        self.Up_conv4 = resconv_block_3D(ch_in = 8 * first_layer_numKernel, ch_out = 4 * first_layer_numKernel)
        
        self.Up3 = up_conv_3D(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        self.Att3 = Attention_block_3D(F_g = 2 * first_layer_numKernel, F_l = 2 * first_layer_numKernel, F_int = first_layer_numKernel)
        self.Up_conv3 = resconv_block_3D(ch_in = 4 * first_layer_numKernel, ch_out = 2 * first_layer_numKernel)
        
        self.Up2 = up_conv_3D(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)
        self.Att2 = Attention_block_3D(F_g = first_layer_numKernel, F_l = first_layer_numKernel, F_int = int(first_layer_numKernel / 2))
        self.Up_conv2 = resconv_block_3D(ch_in = 2 * first_layer_numKernel, ch_out = first_layer_numKernel)

        self.Conv_1x1 = nn.Conv3d(first_layer_numKernel, output_ch, kernel_size = 1, stride = 1, padding = 0)
        self.mlp = Mlp(in_features=32*first_layer_numKernel*3*3*3, hidden_features=2048, out_features=2)
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        # d3 = self.Up3(x3)
        if self.UnetLayer > 3:    
            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)

            # d4 = self.Up4(x4)

            if self.UnetLayer > 4:
                x5 = self.Maxpool(x4)
                x5 = self.Conv5(x5)

                # d5 = self.Up5(x5)

                if self.UnetLayer > 5:
                    x6 = self.Maxpool(x5)
                    x6 = self.Conv6(x6)
        x = x = torch.flatten(x6,start_dim=1)
        x = self.mlp(x)
 

        return x


    



