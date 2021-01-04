import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pennylane as qml


# see also:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/
# https://github.com/pbloem/former
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


class MultiHeadAttentionBase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttentionBase, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        self.k_linear = None
        self.q_linear = None
        self.v_linear = None
        self.combine_heads = None
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def separate_heads(self, x):
        '''
        split into N heads
        from (batch_size, seq_len, embed_dim)
        to   (batch_size, seq_len, num_heads, embed_dim)
        then transpose (1,2) to (batch_size, num_heads, seq_len, embed_dim)
        to make mat mult straightforward for each head
        '''
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def attention(self, query, key, value, mask=None, dropout=None):
        '''
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))V
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # see also: https://tensorchiefs.github.io/dlday2018/tutorial/einsum.html
        #scores = torch.einsum('bijh, bkjh -> bikh', query, key) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        attn = torch.matmul(scores, value)
        return attn, scores
    
    def downstream(self, query, key, value, batch_size, mask=None):
        Q = self.separate_heads(query)
        K = self.separate_heads(key)
        V = self.separate_heads(value)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.combine_heads(concat)
        return output

    def forward(self, x, mask=None):
        raise NotImplementedError("Base class does not execute forward function.")


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttentionClassical, self).__init__(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask, use_bias=use_bias)

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = self.k_linear(x)
        Q = self.q_linear(x)
        V = self.v_linear(x)

        return self.downstream(Q, K, V, batch_size, mask)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout=0.1,
                 mask=None,
                 use_bias=False,
                 n_qubits: int = 4,
                 n_qlayers: int = 1,
                 q_device="default.qubit"):
        super(MultiHeadAttentionQuantum, self).__init__(embed_dim, num_heads, dropout=dropout, mask=mask, use_bias=use_bias)
        
        # todo: add intermediate layer to "dress" quantum circuit
        assert n_qubits == embed_dim, "Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.q_device = q_device
        self.dev = qml.device(self.q_device, wires=self.n_qubits)

        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {self.n_qubits})")

        self.k_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.q_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.v_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.combine_heads = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        K = [self.k_linear(x[:, t, :]) for t in range(seq_len)]
        Q = [self.q_linear(x[:, t, :]) for t in range(seq_len)]
        V = [self.v_linear(x[:, t, :]) for t in range(seq_len)]

        K = torch.Tensor(pad_sequence(K))
        Q = torch.Tensor(pad_sequence(Q))
        V = torch.Tensor(pad_sequence(V))

        return self.downstream(Q, K, V, batch_size, mask)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForwardBase, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, ffn_dim)
        self.linear_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError("Base class does not implement forward function")


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForwardClassical, self).__init__(embed_dim, ffn_dim, dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim, n_qubits, n_qlayers=1, dropout=0.1, q_device="default.qubit"):
        super(FeedForwardQuantum, self).__init__(embed_dim, ffn_dim=n_qubits, dropout=dropout)

        self.n_qubits = n_qubits
        self.dev = qml.device(q_device, wires=self.n_qubits)

        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.vqc = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.linear_1(x)
        X = [self.vqc(x[:, t, :]) for t in range(seq_len)]
        x = torch.Tensor(pad_sequence(X))
        # dropout?
        x = self.linear_2(x)
        return x


class TransformerBlockBase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_head: int,
                 ff_dim: int,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlockBase, self).__init__()
        self.attn = None
        self.ffn = None
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)

        return x


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlockClassical, self).__init__(embed_dim, num_heads, ff_dim, dropout, mask)
        self.attn = MultiHeadAttentionClassical(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, mask=mask)
        self.ffn = FeedForwardClassical(embed_dim, ff_dim)


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlockQuantum, self).__init__(embed_dim, num_heads, ff_dim, dropout, mask)
        
        self.n_qubits_transformer = n_qubits_transformer
        self.n_qubits_ffn = n_qubits_ffn
        self.n_qlayers = n_qlayers

        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                              num_heads,
                                              n_qubits=n_qubits_transformer,
                                              n_qlayers=n_qlayers,
                                              dropout=dropout,
                                              mask=mask)
        self.ffn = FeedForwardQuantum(embed_dim, n_qubits_ffn, n_qlayers)


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)  # .cuda()
        return x


class TextClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
                 ff_dim: int = 32,
                 n_qubits: int = 0,
                 n_qlayers: int = 1,
                 dropout=0.1):
        super(TextClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        print(f"++ There will be {num_blocks} transformer blocks")
        if n_qubits > 0:
            if n_qubits > 0:
                print(f"++ Transformer will use {n_qubits} qubits and {n_qlayers} q layers")

            transformer_blocks = [
                TransformerBlockQuantum(embed_dim, num_heads,
                                        ff_dim,
                                        n_qubits_transformer=n_qubits,
                                        n_qubits_ffn=n_qubits//2,
                                        n_qlayers=n_qlayers) for _ in range(num_blocks)
                ]
        else:
            transformer_blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)
            ]

        self.transformers = nn.Sequential(*transformer_blocks)
        if self.num_classes > 2:
            self.class_logits = nn.Linear(embed_dim, num_classes)
        else:
            self.class_logits = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        # batch_size, seq_len, embed_dim = x.size()
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)  # global average pooling, works in 1D
        x = self.dropout(x)
        # x = self.class_logits(x)
        # return F.log_softmax(x, dim=1)
        return self.class_logits(x)
        
