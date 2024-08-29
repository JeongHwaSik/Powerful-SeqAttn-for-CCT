import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm import create_model
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

class SimGramHead(nn.Module):

    def __init__(self, curr_dim, num_classes, branches=8, img_size=224, gram_dim=192):
        super(SimGramHead, self).__init__()
        self.curr_dim = curr_dim
        self.gram_dim = gram_dim
        self.num_classes = num_classes
        self.branches = branches

        self.gram_contraction = nn.ModuleList()
        self.gram_embedding = nn.ModuleList()
        self.fc = nn.ModuleList()


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
            gram = gram.view(gram.size()[0], -1) # (B, 1, curr_dim) -> (B, curr_dim)

            # 4. FC layer
            gram = self.fc[k](gram) # (B, curr_dim) -> (B, num_classes)

            # Aggregate all outputs
            x_out.append(gram)

        return x_out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
                x.ndim - 1)  # 'shape' to work with diff dim tensors, not just 2D ConvNets (Transformer: (B, 1, 1), Conv: (B, 1, 1, 1))
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize: 0 or 1
    output = x.div(keep_prob) * random_tensor  # (B, N, D) * (B, 1, 1) : drop some batches

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
                 gramian=True,
                 branches=8,
                 img_size=224,
                 gram_dim=192
                 ):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.gramian = gramian
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not gramian:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
            self.num_tokens = 1
        else:
            self.simgram = SimGramHead(
                curr_dim=embedding_dim,
                num_classes=num_classes,
                branches=branches,
                img_size=img_size,
                gram_dim=gram_dim
            )

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)  # (1, N, D)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)  # (N, D)
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

        if not self.gramian:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.gramian:
            x = self.simgram(x)  # (B, num_classes)
        else:
            x = x[:, 0]
            x = self.fc(x)  # (B, num_classes)
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


class SG_CCT(nn.Module):
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
                 gramian=True,
                 branches=8,
                 gram_dim=192,
                 *args,
                 **kwargs
                 ):
        super(SG_CCT, self).__init__()

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
            gramian=gramian,
            branches=branches,
            img_size=img_size,
            gram_dim=gram_dim
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _sg_cct(
        arch, pretrained, progress,
        num_layers, num_heads, mlp_ratio, embedding_dim,
        kernel_size=3, stride=None, padding=None,
        positional_embedding='learnable',
        *args, **kwargs
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = SG_CCT(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        gram_dim = 64,
        *args, **kwargs
    )

    return model


def sg_cct_7(arch, pretrained, progress, *args, **kwargs):
    return _sg_cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, gramian=True, *args, **kwargs)


@register_model
def sg8_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                     *args, **kwargs):
    return sg_cct_7(
        'sg8_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch=8,
        *args,
        **kwargs
    )

@register_model
def sg4_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                     *args, **kwargs):
    return sg_cct_7(
        'sg4_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch=4,
        *args,
        **kwargs
    )

@register_model
def sg2_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                     *args, **kwargs):
    return sg_cct_7(
        'sg2_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch=2,
        *args,
        **kwargs
    )

@register_model
def sg1_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                     *args, **kwargs):
    return sg_cct_7(
        'sg2_cct_7_3x1_32',
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        branch=1,
        *args,
        **kwargs
    )


if __name__ == '__main__':
    x = torch.rand(2, 3, 32, 32)
    model = create_model('sg8_cct_7_3x1_32', num_classes=100)
    y = model(x)



