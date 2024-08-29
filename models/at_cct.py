import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from timm import create_model
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F

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

class AttnHead(nn.Module):

    def __init__(self, curr_dim, num_classes, branches=8, img_size=224):
        super(AttnHead, self).__init__()
        self.curr_dim = curr_dim
        self.num_classes = num_classes
        self.branches = branches

        # ToDo: Options: 1FC, 2FC, GAP, FC-GAP
        # self.fc = nn.Linear(int((img_size*img_size)/4), 1) # fixme: Hard coded for CIFAR-10/100
        self.fc = nn.Sequential(
            nn.Linear(int((img_size * img_size) / 4), 128),
            nn.AdaptiveAvgPool1d(1) # nn.Linear(128, 1)
        )
        # self.gap = nn.AdaptiveAvgPool1d(1)

        self.attention = nn.ModuleList()
        self.final_fc = nn.ModuleList()


        for i in range(branches):

            # Gram Attention
            self.attention.append(
                LayerScaleBlockClassAttn(dim=curr_dim, num_heads=8)
            )

            # FC
            self.final_fc.append(
                nn.Linear(curr_dim, num_classes)
            )

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


    def forward(self, x): # x: (B, N=HW, D=curr_dim)
        x_out = []

        cls = self.fc(x.permute(0, 2, 1)).transpose(-1, -2)  # (B, N, D) -> (B, D, N) -> (B, D, 1) -> (B, 1, D)
        # cls = self.gap(x.permute(0, 2, 1)).transpose(-1, -2)  # (B, N, D) -> (B, D, N) -> (B, D, 1) -> (B, 1, D)
        for k in range(self.branches):
            # Class Attention
            out = self.attention[k](x, cls) # x: (B, N, D), out: (B, 1, D) -> out: (B, 1, D)
            out = out.view(out.size()[0], -1) # (B, 1, D) -> (B, D)

            # FC layer
            out = self.final_fc[k](out) # (B, curr_dim) -> (B, num_classes)

            # Aggregate all outputs
            x_out.append(out)
        return x_out



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 'shape' to work with diff dim tensors, not just 2D ConvNets (Transformer: (B, 1, 1), Conv: (B, 1, 1, 1))
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize: 0 or 1
    output = x.div(keep_prob) * random_tensor # (B, N, D) * (B, 1, 1) : drop some batches

    return output


class DropPath(nn.Module):
    """
    Drop Path also called Stochastic Depth
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.proj = Linear(dim, dim)

        self.attn_drop = Dropout(attention_dropout)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.activation = F.gelu

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src):
        src = self.linear1(src)
        src = self.activation(src)
        src = self.dropout1(src)
        src = self.linear2(src)
        src = self.dropout2(src)

        return src


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.attn = Attention(dim=d_model, num_heads=nhead,
                              attention_dropout=attention_dropout,
                              projection_dropout=dropout)
        self.mlp = MLP(d_model, dim_feedforward, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()


    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.attn(self.norm1(src)))
        src = src + self.drop_path(self.mlp(self.norm2(src)))
        return src


# ViT with Gramian Attention Classifier
class TransformerClassifier(Module):
    def __init__(self,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 attn_head=True,
                 branches=8,
                 img_size=224,
        ):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.attn_head = attn_head
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not attn_head:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
            self.num_tokens = 1
        else:
            self.no_gram_attn_head = AttnHead(
                curr_dim=embedding_dim,
                num_classes=num_classes,
                branches=branches,
                img_size=img_size
            )

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim), requires_grad=True) # (1, N, D)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False) # (N, D)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.attn_head:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.attn_head:
            x = self.no_gram_attn_head(x) # list of (B, num_classes)
        else:
            x = x[:, 0]
            x = self.fc(x) # (B, num_classes)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


# First Convolution Layer for tokenizing input images
class Tokenizer(nn.Module):
    def __init__(
            self,
            kernel_size, stride, padding,
            pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
            n_conv_layers=1,
            n_input_channels=3,
            n_output_channels=64,
            in_planes=64,
            activation=None,
            max_pool=True,
            conv_bias=False
        ):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class AT_CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 attn_head=True,
                 branches=8,
                 *args,
                 **kwargs
        ):
        super(AT_CCT, self).__init__()

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels,
                height=img_size,
                width=img_size
            ),
            embedding_dim=embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            attn_head=attn_head,
            branches=branches,
            img_size=img_size
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _at_cct(
         arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs
    ):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = AT_CCT(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        attn_head=True,
        *args,
        **kwargs
    )

    return model


def at_cct_7(arch, pretrained, progress, *args, **kwargs):
    return _at_cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, *args, **kwargs)


@register_model
def at1_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return at_cct_7(
        'at1_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch = 1,
        *args,
        **kwargs
    )

@register_model
def at2_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return at_cct_7(
        'at2_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch = 2,
        *args,
        **kwargs
    )

@register_model
def at4_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return at_cct_7(
        'at4_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch = 4,
        *args,
        **kwargs
    )

@register_model
def at8_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10, *args, **kwargs):
    return at_cct_7(
        'at8_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch = 8,
        *args,
        **kwargs
    )

if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    model = create_model('at8_cct_7_3x1_32', num_classes=100)
    y = model(x)



