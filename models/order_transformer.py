import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
import math
# from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PositionalEncoding(nn.Module):

    def __init__(self, size: int = 0, max_len: int = 500):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class OrderTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ]))
        # TODO:
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, OrderAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.add_pos_embedding = PositionalEncoding(size = dim)
    def forward(self, x):
        # x = self.add_pos_embedding(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def order_attention_mask(x_len):
    bs = len(x_len)
    t = max(x_len)
    mask = torch.zeros(bs, t, t)
    for x, length in enumerate(x_len):
        for i in range(length):
            for j in range(i + 1):
                mask[x, i, j] = 1.
    return mask.unsqueeze(1)


class OrderAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # TODO: Add order mask
        mask = self._order_masks(n).cuda()
        dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def _order_masks(self, t):
        mask = torch.zeros(t, t)
        for i in range(t):
            for j in range(i + 1):
                mask[i, j] = 1.
        return mask.unsqueeze(0).unsqueeze(0)


if __name__ == '__main__':
    # pos_emb = PositionalEncoding(size=768)
    # x = torch.randn(5, 64, 768)
    # x_add_pos = pos_emb(x)
    # print(x_add_pos.shape)
    x_len = torch.LongTensor([3, 6, 4])
    # mask = get_mask(x_len)
    # print(mask)
    # attn_mask = get_attention_mask(mask)
    # print(attn_mask)
    # mask = order_attention_mask(x_len)
    # for i in range(3):
    #     print(mask[i, 0, :])
    # exit()
    
    order_attention = OrderAttention(512, 8, 64, 0.1)
    order_transformer = OrderTransformer(512, 2, 8, 64, 1024, 0.1)
    x = torch.randn(1, 3, 512)
    out = order_transformer(x)
    print(out.shape)