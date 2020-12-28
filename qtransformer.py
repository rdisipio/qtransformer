import torch
import torch.nn as nn
import torch.nn.functional as F


# see also:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/
# https://github.com/pbloem/former
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


class TokenEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embed(x)# * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int=4,
        dropout: float=0.1,
        mask=None,
        use_bias=False,
        ):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads != 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def separate_heads(self, x):
        '''
        split into N heads
        from (batch_size, seq_len, embed_dim)
        to   (batch_size, seq_len, num_heads, projection_dim)
        then transpose to make mat mult straightforward
        '''
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def attention(self, query, key, value, mask=None, dropout=None):
        '''
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        attn = torch.matmul(scores, value)
        return attn, scores

    def forward(self, x, mask):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        K = self.separate_heads(K)
        Q = self.separate_heads(Q)
        V = self.separate_heads(V)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_head: int,
        ff_dim: int,
        dropout: float=0.1,
        mask=None,
        ):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_head, mask=mask)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.dropout1(self.att(x))
        out1 = self.norm1(x + attn_output)
        ffn_output = self.dropout2(self.ffn(out1))
        out2 = self.norm2(out1 + ffn_output)
    

class TextClassifier(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        vocab_size: int,
        dropout=0.1,
        ):
        super(Classifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = TokenEmbedding(embed_dim, vocab_size)
        self.pos_embedding = PositionalEncoding(embed_dim)

        transformer_blocks = [
            TransformerBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*transformer_blocks)
        self.class_logits = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        tokens = self.token_embedding(x)
        batch_size, seq_len, embed_dim = x.shape()
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)  # global average pooling, works in 1D
        x = self.dropout(x)
        x = self.class_logits(x)
        return F.log_softmax(x, dim=1)

