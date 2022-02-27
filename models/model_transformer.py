import torch
import torch.nn as nn
# import torchvision
import timm
from .order_transformer import OrderTransformer
from .transformer import Transformer
from einops import repeat, rearrange
from .configs import model_config, vit_config
# from vit import ViT

# from thop import profile
# from torchstat import stat


class ISLR(nn.Module):
    def __init__(self, num_class, dim, depth, heads, dim_head, mlp_dim, dropout = 0., pool='gap'):
        super(ISLR, self).__init__()
        self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.backbone.reset_classifier(0)
        self.temporal = OrderTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.temporal = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
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
        x = rearrange(x, '(b t) c -> b t c', b = bs)

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = bs)
        # x = torch.cat((cls_tokens, x), dim=1)

        y = self.temporal(x)
        #TODO: if Order Transformer, return -1, else return 0
        # x = y.mean(dim=1) if self.pool == 'mean' else y[:, 0]
        # return self.fc1(x), self.fc2(y[:, 1:])
        x = y.mean(dim=1) if self.pool == 'mean' else y[:, -1]
        return self.fc1(x), self.fc2(y)


if __name__ == '__main__':
    x = torch.randn(3, 3, 32, 224, 224)
    model = ISLR(**model_config)
    logits = model(x)
    print(logits.shape)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ', total_num, '   Trainable: ', trainable_num)
    exit()

    flops, params = profile(model, inputs=(x,))
    print('Flops:{}'.format(flops))
    print('parameters:{}'.format(params))
    stat(model, (3, 32, 224, 224))