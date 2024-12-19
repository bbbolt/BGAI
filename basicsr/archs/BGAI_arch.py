from audioop import bias

from einops import rearrange

"""
不进行修改
"""
# import thop
# from ptflops import get_model_complexity_info

"""
加上大号dense连接
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.init import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.modules.utils import _pair

def window_partition(x, window_size):
    """
     将feature map按照window_size划分成一个个没有重叠的window
     Args:
         x: (B, H, W, C)
         window_size (int): window size(M)

     Returns:
         windows: (num_windows*B, window_size, window_size, C)
     """
    B, H, W, C = x.shape
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H_p, W_p, C = x.shape
    x = x.view(B, H_p // window_size, window_size, W_p // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    pad_true = bool(pad_b + pad_r + pad_l + pad_t)
    return x, pad_true, H_p, W_p


def window_reverse(windows, window_size, H, W):
    """
        将一个个window还原成一个feature map
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size(M)
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
    """
    # print("H:", H)
    # print("W:", W)
    # print("window shape", windows.shape)

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwc1 = Dconv_for_MLP(hidden_features, hidden_features, 3, 'same')
        # self.dwc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,groups=hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.dwc1(self.fc1(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.drop(x)
        return x

class Diagomal_DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dir=-1):
        super(Diagomal_DWConv, self).__init__()
        self.shift_conv1 = nn.Conv2d(in_channels * kernel_size, out_channels, (1, 1), groups=in_channels)
        self.dir = dir
        self.ks = kernel_size
        self.pad = kernel_size // 2

    def forward(self, input):  # B,C,H,W
        B, C, H, W = input.shape
        d_base = W + self.pad * 2 - self.ks + 1  # distance_base = W+2*P-k+1
        out = torch._C._nn.im2col(input, (1, self.ks), _pair(1), (0, self.pad), _pair(1))  # B k*C HW
        if self.dir == -1:
            for i in range(self.ks):
                if i != self.pad:
                    out[:, i::self.ks, :] = torch.roll(out[:, i::self.ks, :], (i - self.pad) * d_base)  # B k*C HW
        else:
            for i in range(self.ks):
                if i != self.pad:
                    out[:, i::self.ks, :] = torch.roll(out[:, i::self.ks, :], (self.pad - i) * d_base)  # B k*C HW
        out = self.shift_conv1(out.view(B, -1, H, W))
        return out

class Dconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.dim = in_dim // 4
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        # self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
        #                          groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (15, 1), padding=(7,0),
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (1, 15), padding=(0,7),
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=kernel_size//2, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=kernel_size//2, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
        self.conv1x1_2 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.act = nn.GELU()

    def forward(self, input, flag=False):
        out = self.conv1(input)
        y1=out[:, :self.dim, :, :]
        y2=self.conv2_2(out[:, self.dim:self.dim * 2, :, :])
        y3= self.conv2_3(out[:, self.dim * 2:self.dim * 3, :, :])
        y4=self.conv2_4(out[:, self.dim * 3:, :, :])
        out = torch.cat([y1,y2,y3,y4], dim=1)+input
        out = self.conv1x1_2(self.act(out))
        return out

class Dconv_for_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.dim = in_dim // 4
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=kernel_size//2,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (5, 3), padding=(2,1),
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (3, 5), padding=(1,2),
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=kernel_size//2, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=kernel_size//2, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
    def forward(self, input):
        out = input
        y1 =self.conv2_1(out[:, :self.dim, :, :])
        y2 = self.conv2_2(out[:, self.dim:self.dim * 2, :, :])
        y3 = self.conv2_3(out[:, self.dim * 2:self.dim * 3, :, :])
        y4 = self.conv2_4(out[:, self.dim * 3:, :, :])
        out = torch.cat([y1, y2, y3, y4], dim=1)
        return out


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return img


class Conv_Gelu_Res(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding='same'):
        super().__init__()
        self.conv1 = Dconv(in_dim, out_dim, kernel_size, padding)
        # self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input) + input
        return out


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=4, dim_out=None, num_heads=1, attn_drop=0., proj_drop=0.,
                 qk_scale=None, shift_size=0):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx==1:
            W_sp, H_sp = self.resolution, self.split_size
        elif idx==2:
            W_sp, H_sp = self.resolution//2, self.resolution//2
        else:
            W_sp, H_sp = self.resolution, self.resolution
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.lepe_v = nn.Conv2d(dim, dim, 5,1,2,groups=dim)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x, attn_input):
        """
        x: B L C
        """
        B, C, H, W = x.shape
        v = self.proj_v(x)
        lepe = self.lepe_v(v)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.H_sp//2, -self.W_sp//2), dims=(-2, -1))
            v = torch.roll(v, shifts=(-self.H_sp//2, -self.W_sp//2), dims=(-2, -1))
            lepe = torch.roll(lepe, shifts=(-self.H_sp//2, -self.W_sp//2), dims=(-2, -1))

        else:
            x = x

        q = self.proj_q(x)
        k = x
        ### Img2Window
        q = self.im2cswin(q+lepe)
        k = self.im2cswin(k+lepe)
        # lepe = self.get_lepe(v)
        v = self.im2cswin(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        if attn_input is not None:
            attn = (nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)+self.temperature*attn_input)# //2?

        else:
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W) # B C H' W'
        # shift 还原
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.H_sp//2, self.W_sp//2), dims=(-2, -1))
            lepe = torch.roll(lepe, shifts=(self.H_sp//2, self.W_sp//2), dims=(-2, -1))
        else:
            x = x
        return x+lepe, attn

class Gated_Module(nn.Module):
    def __init__(self, dim1,dim2):
        super().__init__()
        self.gated_reset = nn.Conv2d(dim1+dim2, dim2, 1)
        self.gated_update = nn.Conv2d(dim1+dim2, dim1, 1)
        self.gated_fusion = nn.Conv2d(dim1+dim2, dim1, 1)
        self.extract = nn.Conv2d(dim2, dim1, 1)

    def forward(self, x, h):
        r = torch.sigmoid(self.gated_reset(torch.cat([x, h], 1)))
        z = torch.sigmoid(self.gated_update(torch.cat([x, h], 1)))
        h_hat = torch.tanh(self.gated_fusion(torch.cat([x, h * r], 1)))
        out = (1. - z) * self.extract(h) + z * h_hat
        return out

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super(SwinTransformerBlock, self).__init__()
        self._dim = dim//5
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ration = mlp_ratio
        "shift_size must in 0-window_size"
        assert 0 <= self.shift_size < self.window_size
        self.attn1 = LePEAttention(self._dim*2, window_size, 3, shift_size=shift_size)  #16*16
        self.attn2 = LePEAttention(self._dim, window_size, 0, shift_size=shift_size)  #16*4
        self.attn3 = LePEAttention(self._dim, window_size, 1, shift_size=shift_size)  #4*16
        self.attn4 = LePEAttention(self._dim, window_size, 2, shift_size=shift_size)  #8*8
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(self._dim*5, dim)
        self.proj1_router = nn.Conv2d(dim, self._dim*2, 1)
        self.proj1 = nn.Conv2d(dim, self._dim*3, 1)
        self.gate_layer1 = Gated_Module(self._dim, self._dim*2)
        self.gate_layer2 = Gated_Module(self._dim, self._dim*2)
        self.gate_layer3 = Gated_Module(self._dim, self._dim*2)

    def forward(self, x, attns_shifts_input):  # x: B,C,H,W

        B, C, H, W = x.shape # x: B,C,H,W
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        shift_x_norm = self.norm1(x).permute(0,3,1,2).contiguous()
        x_router = self.proj1_router(shift_x_norm)
        x_window = self.proj1(shift_x_norm)
        x_slices = torch.chunk(x_window,3,dim=1)

        (attn1, attn2, attn3, attn4) = attns_shifts_input

        x1, attn1 = self.attn1(x_router, attn1)
        x2, attn2 = self.attn2(self.gate_layer1(x_slices[0], x1), attn2)
        x3, attn3 = self.attn3(self.gate_layer2(x_slices[1]+x2, x1), attn3)
        x4, attn4 = self.attn4(self.gate_layer3(x_slices[2]+x3, x1), attn4)
        attened_x = torch.cat([x1, x2, x3, x4], dim=1)
        del x1, x2,x3,x4

        x = self.proj(attened_x.permute(0, 2, 3, 1).contiguous())

        if pad_r or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            shortcut = shortcut[:, :H, :W, :].contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # x: B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, (attn1, attn2, attn3, attn4)  # x: B,C,H,W


class Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads):
        super().__init__()

        swin_body = []
        self.window_size = windows_size
        for i in range(depth):
            if i % 2:
                shift_size = windows_size // 2
            else:
                shift_size = 0
            self.shift_size = shift_size
            swin_body.append(SwinTransformerBlock(c_dim, num_heads, window_size=windows_size, shift_size=shift_size,
                                                  mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                                                  act_layer=nn.GELU))
        self.swin_body = nn.Sequential(*swin_body)

    def creat_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size) * self.window_size)
        Wp = int(np.ceil(W / self.window_size) * self.window_size)

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_s = (slice(0, -self.window_size),
               slice(-self.window_size, -self.shift_size),
               slice(-self.shift_size, None))
        w_s = (slice(0, -self.window_size),
               slice(-self.window_size, -self.shift_size),
               slice(-self.shift_size, None))
        c = 0
        for h in h_s:
            for w in w_s:
                img_mask[:, h, w, :] = c
                c += 1
        mask_window = window_partition(img_mask, self.window_size)[0]  # [nW, Mh, Mw, 1]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)
        mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, x, attns_shifts_input):
        src = x
        _, _, H, W, = x.shape
        attns_shifts_out = []
        for idx, body in enumerate(self.swin_body):
            src, attns = body(src, attns_shifts_input[idx])
            attns_shifts_out.append(attns)

        info_mix = src

        return info_mix, attns_shifts_out


class Res_Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads):
        super().__init__()
        modules_body = []
        modules_body.append(Conv_Gelu_Res(c_dim, c_dim, 3, padding='same'))
        modules_body.extend([Spatial_Attn(c_dim, depth, windows_size, num_heads)])
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, attns_shifts_input):
        res = self.body[0](x)
        res, attns_shifts_out = self.body[1](res, attns_shifts_input)
        res += x
        return res, attns_shifts_out


def Pixelshuffle_Block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (kernel_size, kernel_size), (stride, stride),
                     padding='same')
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class BasicLayer(nn.Module):
    def __init__(self, c_dim, reduction, RC_depth, RS_depth, depth, windows_size, num_heads):
        super(BasicLayer, self).__init__()

        self.body_1 = []
        self.body_1.extend([Res_Spatial_Attn(c_dim, depth, windows_size, num_heads) for _ in range(RS_depth)])
        self.body_1.append(Dconv(c_dim, c_dim, 3, padding='same'))
        self.res_spatial_attn = nn.Sequential(*self.body_1)

        self.conv1x1_2 = nn.Conv2d(c_dim, c_dim // 2, 1)
        self.conv1x1_3 = nn.Conv2d(int(c_dim * 1.5), c_dim, 1)

    def forward(self, x, attns_shifts_input):
        short_cut = x
        res2 = self.conv1x1_2(x)
        x, attns_shifts_input = self.res_spatial_attn[0](x, attns_shifts_input)
        # x, attns_shifts_input = self.res_spatial_attn[1](x, attns_shifts_input)
        x = self.res_spatial_attn[1](x)
        out_B = self.conv1x1_3(torch.cat([res2, x], dim=1))
        out_lr = out_B + short_cut
        return out_lr, attns_shifts_input


@ARCH_REGISTRY.register()
class BGAI(nn.Module):
    def __init__(self, rgb_mean=[0.4488, 0.4371, 0.4040], upscale_factor=3, c_dim=52, reduction=16, Bsc_depth=4, RS_depth=1, RC_depth=0, depth=2,
                 windows_size=16, num_heads=4):
        super(BGAI, self).__init__()
        self.body = []
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_shallow = nn.Conv2d(3, c_dim, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.body.extend([BasicLayer(c_dim, reduction, RC_depth, RS_depth, depth, windows_size, num_heads) for _ in
                          range(Bsc_depth)])
        self.conv_before_upsample = nn.Sequential(Dconv(c_dim, c_dim, 3, padding='same'))
        self.upsample = nn.Sequential(Pixelshuffle_Block(c_dim, 3, upscale_factor=upscale_factor, kernel_size=3))
        self.bsc_layer = nn.Sequential(*self.body)
        self.c = nn.Conv2d(Bsc_depth * c_dim, c_dim, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        out_fea = self.conv_shallow(x)

        attns_shifts = [[None]*4,]*2
        x1, attns_shifts = self.bsc_layer[0](out_fea, attns_shifts)
        x2, attns_shifts = self.bsc_layer[1](x1, attns_shifts)
        x3, attns_shifts = self.bsc_layer[2](x2, attns_shifts)
        x4, attns_shifts = self.bsc_layer[3](x3, attns_shifts)
        out_B = self.c(torch.cat([x1, x2, x3, x4], dim=1))
        out_lr = self.conv_before_upsample(out_B) + out_fea

        output = self.upsample(out_lr) + self.mean

        return output


if __name__ == '__main__':

    model = BGAI(rgb_mean=[0.4488, 0.4371, 0.4040], upscale_factor=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, 256, 256))
    model.cuda()
    out = model(x.cuda())
    print(out.shape)
