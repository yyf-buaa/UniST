import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np



import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super().__init__()
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        kernel_size = [t_patch_size, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_in', 
                    nonlinearity='leaky_relu'
                )
        
        # 可学习的无效 token 嵌入
        self.invalid_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.invalid_token, std=0.02)  # 用小的标准差初始化

    def forward(self, x):
        # 保存原始输入用于检测全-1的 patch
        original_input = x  # [B, C, T, H, W]
        
        # 进行卷积嵌入
        x = self.tokenConv(x)  # [N, d_model, T', H', W']
        x = x.flatten(3)       # [N, d_model, T', H'*W']
        x = torch.einsum("ncts->ntsc", x)  # [N, T', H'*W', d_model]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, seq_len, d_model]
        
        # 生成掩码：检测原始输入中每个 patch 是否全为 -1
        mask = self._detect_all_neg_one_patches(original_input)  # [N, seq_len]
        # 将全-1的 patch 替换为可学习的无效 token
        invalid_tokens = self.invalid_token.expand(1, mask.sum(), -1).squeeze()
        x[mask] = invalid_tokens
        
        return x  # 返回嵌入向量和掩码（可选）

    def patchify(self, imgs):
        """
        imgs: (N, 2, T, H, W)
        x: (N, L, patch_size**2 *2)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_size
        u = self.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 2, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 2))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def _detect_all_neg_one_patches(self, x):
        patches = self.patchify(x)  # [N, seq_len, patch_size**2 * C]
        return (patches == -1).all(dim=-1)  # [N, seq_len]

class SpatialPatchEmb(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super(SpatialPatchEmb, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.reshape(B, C, H*W//self.patch_size**2).permute(0,2,1)
        return x



class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, t_patch_size=1, hour_size=48, weekday_size=7, month_size=12, day_size=31, quarter_size=4):
        super(TemporalEmbedding, self).__init__()
        
        # 定义各个时间特征的 embedding 层
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.quarter_embed = nn.Embedding(quarter_size, d_model)
        
        # 时间卷积层
        self.timeconv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=t_patch_size, stride=t_patch_size)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, num_features)，其中 num_features 包含 [weekday, hour, month, day, year, quarter]
        :return: 经过时间嵌入和卷积后的时间嵌入张量
        """
        batch_size, seq_len, _ = x.shape
        
        # 对每个时间特征进行嵌入
        weekday_x = self.weekday_embed(x[:, :, 0])  # 星期几
        month_x = self.month_embed(x[:, :, 1])      # 月份
        day_x = self.day_embed(x[:, :, 2])          # 天
        quarter_x = self.quarter_embed(x[:, :, 4])  # 季度
        
        # 合并所有时间嵌入（沿最后一个维度求和）
        time_emb = weekday_x + month_x + day_x + quarter_x
        # 使用卷积层进一步处理时间嵌入
        time_emb = self.timeconv(time_emb.transpose(1, 2)).transpose(1, 2)
        
        return time_emb

class DataEmbedding(nn.Module):
    def __init__(self, c_in, feat_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7):
        super(DataEmbedding, self).__init__()
        self.args = args
        # self.feat_embedding = TokenEmbedding(c_in=feat_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, t_patch_size = args.t_patch_size,  patch_size=args.patch_size)
        self.temporal_embedding = TemporalEmbedding(t_patch_size = args.t_patch_size, d_model=d_model, hour_size  = size1, weekday_size = size2) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, is_time=1):
        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        N, T, C, H, W = x.shape
        TokenEmb = self.value_embedding(x)
        #FeatEmb = self.feat_embedding(x_feat)
        TimeEmb = self.temporal_embedding(x_mark)
        assert TokenEmb.shape[1] == TimeEmb.shape[1] * H // self.args.patch_size * W // self.args.patch_size
        TimeEmb = torch.repeat_interleave(TimeEmb, TokenEmb.shape[1]//TimeEmb.shape[1], dim=1)
        assert TokenEmb.shape == TimeEmb.shape
        if is_time==1:
            x = TokenEmb + TimeEmb
        else:
            x = TokenEmb
        return self.dropout(x), TimeEmb


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size1, grid_size2, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim]
    """
    # res = torch.FloatTensor(res).to(device)
    res = res.to(device)
    grid_h = torch.arange(grid_size1, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size2, dtype=torch.float32, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    
    pos_embed = pos_embed.reshape(h * w, embed_dim).numpy()
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, res):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = pos * res
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
