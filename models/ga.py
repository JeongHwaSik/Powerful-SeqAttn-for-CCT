import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., expansion=4, fap=False):
        super().__init__()
        self.num_heads = num_heads
        self.expansion = expansion
        head_dim = dim // num_heads
        head_dim = head_dim // expansion
        self.scale = head_dim ** -0.5
        self.fap = fap

        self.q = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.k = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.v = nn.Linear(dim, dim//expansion, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//expansion, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // self.expansion
        if self.fap:
            N = N - 5
            q = self.q(x[:, 5:]).unsqueeze(1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x[:, 0:5]).reshape(B, 5, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale
            v = self.v(x[:, 0:5]).reshape(B, 5, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        else:
            q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            q = q * self.scale
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.fap:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

class GroupConvMlp(nn.Module):
    """
    MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, drop=0., groups=1):
        super().__init__()
        self.groups = groups

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True, groups=groups)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True, groups=groups)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = channel_shuffle(x, self.groups)
        x = self.fc2(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        return x

class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=GroupConvMlp, mlp_block_groups=2, init_values=1e-4, fap=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, fap=fap)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim//1, hidden_features=mlp_hidden_dim//1, act_layer=act_layer,
                             out_features=dim//1, drop=drop, groups=mlp_block_groups)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls

def channel_shuffle(x, group):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(batchsize, num_channels, height, width)

    return x

class GramianAttnHead(nn.Module):

    def __init__(self, curr_dim, num_classes, branches=8, img_size=224, gram_dim=192):
        super(GramianAttnHead, self).__init__()
        self.curr_dim = curr_dim
        self.gram_dim = gram_dim
        self.num_classes = num_classes
        self.branches = branches

        self.gram_contraction = nn.ModuleList()
        self.gram_embedding = nn.ModuleList()
        self.gram_attention = nn.ModuleList()
        self.fc = nn.ModuleList()

        # # TODO:
        # self.fc_in = nn.ModuleList()
        # self.fc_out = nn.ModuleList()

        for i in range(branches):
            # Gram Contraction
            self.gram_contraction.append(
                nn.Sequential(
                    Rearrange('b (h w) c -> b c h w', h=img_size//2, w=img_size//2), # fixme: Hard coded for CIFAR-10/100
                    nn.Conv2d(curr_dim, gram_dim, kernel_size=1, stride=1, padding=0, bias=True, groups=8),
                    nn.BatchNorm2d(gram_dim)
                )
            )

            # Gram Embedding
            self.gram_embedding.append(
                nn.Sequential(
                    nn.Conv2d(((gram_dim + 1) * gram_dim // 2), curr_dim, kernel_size=1, stride=1, padding=0, bias=True, groups=8),
                    nn.BatchNorm2d(curr_dim)
                )
            )

            # Gram Attention
            self.gram_attention.append(
                LayerScaleBlockClassAttn(dim=curr_dim, num_heads=8)
            )

            # # TODO
            # self.fc_in.append(
            #     nn.Linear(curr_dim, 1)
            # )
            # self.fc_out.append(
            #     nn.Linear(2*curr_dim, curr_dim)
            # )

            # FC
            self.fc.append(
                nn.Linear(curr_dim, num_classes)
            )

        self.gram_idx = self.upper_tria_gram(gram_dim)

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0) # beta
            nn.init.constant_(m.weight, 1.0) # gamma


    def get_gram(self, xbp, gram_idx):
        """
        From Gram Matrix, get [CLS] Token

        xbp: (B, gram_dim, H, W)
        gram_idx: return value of upper_tria_gram func.
        """
        B, gram_dim, _, _ = xbp.size()

        xbp = xbp / xbp.size()[2]
        xbp = torch.reshape(xbp, (
        xbp.size()[0], xbp.size()[1], xbp.size()[2] * xbp.size()[3]))  # (B, gram_dim, H, W) -> (B, gram_dim, HW)
        xbp = torch.bmm(xbp, torch.transpose(xbp, 1, 2)) / (
        xbp.size()[2])  # (B, gram_dim, HW) bmm (B, HW, gram_dim) -> (B, gram_dim, gram_dim)
        xbp = torch.reshape(xbp, (B, gram_dim ** 2))  # (B, gram_dim, gram_dim) -> (B, gram_dim^2)
        xbp = xbp[:, gram_idx]  # use upper-half of gram matrix: (B, (gram_dim(gram_dim+1)/2)) = (B, k)

        xbp = torch.nn.functional.normalize(xbp)
        xbp = xbp.float()
        xbp = torch.reshape(xbp, (xbp.size()[0], xbp.size()[1], 1, 1))  # (B, k, 1, 1)
        return xbp

    def upper_tria_gram(self, gram_dim):
        """
        Get elements of upper triangle of Gram Matrix
        """
        gram_idx = np.zeros(((gram_dim + 1) * gram_dim // 2))
        count = 0
        for i in range(gram_dim):
            for j in range(gram_dim):
                if j >= i:
                    gram_idx[count] = (i * gram_dim) + j
                    count += 1
        return gram_idx


    def forward(self, x): # x: (B, N=HW, D=curr_dim)

        x_out = []
        for k in range(self.branches):
            # 1. Gram Contraction: (B, N, curr_dim) -> (B, curr_dim, H, W) -> (B, gram_dim, H, W)
            gram = self.gram_contraction[k](x)

            # 2. get_gram: From Gram Matrix, get [CLS] token
            B, gram_dim, _, _ = gram.size()
            gram = self.get_gram(gram, self.gram_idx) # (B, gram_dim, H, W) -> (B, k, 1, 1)

            # 3. Gram Embedding
            gram = self.gram_embedding[k](gram) # (B, k, 1, 1) -> (B, curr_dim, 1, 1)
            gram = gram.view(gram.size()[0], gram.size()[1], -1) # (B, curr_dim, 1, 1) -> (B, curr_dim, 1)
            gram = gram.permute(0, 2, 1) # (B, curr_dim, 1) -> (B, 1, curr_dim): [CLS] token

            # # TODO:
            # out = nn.Softmax(dim=-1)(self.fc_in[k](x).transpose(-2, -1)) # (B, 1, N)
            # out = out.bmm(x) # (B, 1, curr_dim)
            # gram = torch.cat((gram, out), dim=-1) # (B, 1, 2*curr_dim)
            # gram = self.fc_out[k](gram) # (B, 1, 2*curr_dim) -> # (B, 1, curr_dim)


            # 4. Class Attention
            gram = self.gram_attention[k](x, gram) # x: (B, N, curr_dim), gram: (B, 1, curr_dim) -> out: (B, 1, curr_dim)
            gram = gram.view(gram.size()[0], -1) # (B, 1, curr_dim) -> (B, curr_dim)

            # 5. FC layer
            gram = self.fc[k](gram) # (B, curr_dim) -> (B, num_classes)

            # Aggregate all outputs
            x_out.append(gram)

        return x_out

