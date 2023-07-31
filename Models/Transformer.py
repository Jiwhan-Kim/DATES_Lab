import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size

        # convolution 이용
        self.projection = nn.Conv2d(in_channels, emb_size, patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size)) # class token
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size)) # positional embedding

    def forward(self, x):
        b = x.shape[0] # batch size. x: b, 3(channel), 32, 32
        x = self.projection(x) # x: b, 3, 32, 32 -> b, 128(embedded size), 32/4, 32/4
        x = x.view(x.shape[0], x.shape[2]*x.shape[3], x.shape[1]) # rearrange tensor : b, 128, 8, 8 -> b, 64, 128
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1) # b, 64, 128 -> b, 65, 128
        # add position embedding to prejected patches
        x += self.positions
        return x

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=128, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.layernorm = nn.LayerNorm(emb_size)
        self.attention_drop = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(emb_size, emb_size)
        

    def forward(self, x):
        residual = x # x: b, 65, 128
        x = self.layernorm(x)
        # split keys, queries and values in num_heads
        queries = self.queries(x)
        queries = queries.view(queries.shape[0], self.num_heads, queries.shape[1], queries.shape[2]//self.num_heads) # b, 65, 128 -> b, 8(head), 65, 16
        keys = self.keys(x)
        keys = keys.view(keys.shape[0], self.num_heads, keys.shape[1], keys.shape[2]//self.num_heads) # b, 65, 128 -> b, 8(head), 65, 16
        values = self.values(x)
        values = values.view(values.shape[0], self.num_heads, values.shape[1], values.shape[2]//self.num_heads) # b, 65, 128 -> b, 8(head), 65, 16

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # Q*K^T, energy: b, 8, 65, 65
        scaling = self.emb_size ** (1/2)
        attention = torch.softmax(energy, dim=-1) / scaling
        attention = self.attention_drop(attention)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', attention, values) # att*V, out: b, 8, 65, 16
        out = out.view(out.shape[0], out.shape[2], out.shape[1]*out.shape[3]) # b, 8, 65, 16 -> b, 65, 128
        out = self.projection(out)
        out = out + residual
        return out

class FeedForwardNetwork(nn.Sequential):
    def __init__(self, emb_size=128, expansion=4, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.layernorm = nn.LayerNorm(emb_size)
        self.linear1 = nn.Linear(emb_size, expansion * emb_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(expansion * emb_size, emb_size)
    def forward(self, x):
        residual = x # x: b, 65, 128
        x = self.layernorm(x)
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual
        return out

# Define the ViT architecture
class Transformer(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32, depth=6, n_classes=10):
        super(Transformer, self).__init__()
        self.pe = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.mha = MultiHeadAttention(emb_size, num_heads=8, dropout_rate=0.1)
        self.ffn = FeedForwardNetwork(emb_size, expansion=4, dropout_rate=0.1)
        self.reduce_mean = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
        self.depth = depth
    
    def forward(self, x):
        x = self.pe(x)
        for i in range(self.depth):
            x = self.mha(x)
            x = self.ffn(x)
        # x: b, 65, 128
        # Reduce the sequence length to 1 by taking the average of all tokens
        x = self.reduce_mean(x.permute(0, 2, 1)).squeeze(2) # b, 65, 128 -> b, 128, 65 -> b, 128
        # Apply Layer Normalization
        x = self.layer_norm(x) 
        # Linear layer for classification
        output = self.linear(x) # x: b, 128 -> b, 10
        return output