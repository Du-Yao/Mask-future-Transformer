import torch
import torch.nn as nn
import torchvision
# import timm
from .transformer import Transformer
# from .order_transformer import OrderTransformer
from einops import repeat
# from configs import model_config

# from thop import profile
# from torchstat import stat


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, out_channels),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.net(x)


class ISLR(nn.Module):
    def __init__(self, num_class, dim, depth, heads, dim_head, mlp_dim, dropout = 0., pool='gap'):
        super(ISLR, self).__init__()
        # self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.temporal = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.temporal = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp = MLP(2048, dim, mlp_dim)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_class)
        )
        self.fc2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_class)
        )
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pool = pool

    def forward(self, video ):
        bs, c, t, h, w = video.shape
        x = video.transpose(1, 2).contiguous().reshape(-1, c, h, w)
        x = self.backbone(x)
        x = x.reshape(bs, t, -1)
        x = self.mlp(x)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = bs)
        # x = torch.cat((cls_tokens, x), dim=1)

        y = self.temporal(x)
        x = y.mean(dim=1) if self.pool == 'mean' else y[:, -1]
        return self.fc1(x), self.fc2(y)


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 224, 224)
    model = ISLR(**model_config)
    # logits = model(x)
    # print(logits.shape)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ', total_num, '   Trainable: ', trainable_num)

    flops, params = profile(model, inputs=(x,))
    print('Flops:{}'.format(flops))
    print('parameters:{}'.format(params))
    stat(model, (3, 32, 224, 224))