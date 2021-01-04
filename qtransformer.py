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


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 n_qubits: int = 0,
                 n_qlayers: int = 1,
                 q_device="default.qubit",
                 mask=None,
                 use_bias=False):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

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
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # projection dimensions
        if self.n_qubits > 0:
            self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
            self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
            self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
            self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        else:
            self.k_linear = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
            self.q_linear = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
            self.v_linear = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
            self.combine_heads = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
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

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        # NB: we're using x as q, k and v, but may be three different tensors
        if self.n_qubits == 0:
            K = self.k_linear(x)
            Q = self.q_linear(x)
            V = self.v_linear(x)
        else:
            K = []
            Q = []
            V = []
            for t in range(seq_len):
                # get features from the t-th element in seq, for all entries in the batch
                x_t = x[:, t, :]
            
                K_t = self.k_linear(x_t)
                Q_t = self.q_linear(x_t)
                V_t = self.v_linear(x_t)

                K.append(torch.Tensor(K_t))
                Q.append(torch.Tensor(Q_t))
                V.append(torch.Tensor(V_t))
            K = torch.Tensor(pad_sequence(K))
            Q = torch.Tensor(pad_sequence(Q))
            V = torch.Tensor(pad_sequence(V))

        K = self.separate_heads(K)
        Q = self.separate_heads(Q)
        V = self.separate_heads(V)

        x, self.attn_weights = self.attention(Q, K, V, mask, dropout=self.dropout)

        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.combine_heads(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim, n_qubits, n_qlayers=1, dropout=0.1, q_device="default.qubit"):
        super(FeedForwardQuantum, self).__init__()

        self.n_qubits = n_qubits
        self.dev = qml.device(q_device, wires=self.n_qubits)

        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.linear_1 = nn.Linear(embed_dim, n_qubits)
        self.vqc = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
        self.linear_2 = nn.Linear(n_qubits, embed_dim)
        # dropout?
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.vqc(x)
        # dropout?
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_head: int,
                 ff_dim: int,
                 n_qubits: int = 0,
                 n_qlayers: int = 1,
                 dropout: float = 0.1,
                 mask=None):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim,
                                       num_head,
                                       n_qubits=n_qubits,
                                       n_qlayers=n_qlayers,
                                       mask=mask)
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers

        if self.n_qubits == 0:
            self.ffn = FeedForward(embed_dim, ff_dim)
        else:
            self.ffn = FeedForwardQuantum(embed_dim, n_qubits, n_qlayers)
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

        transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, n_qubits, n_qlayers) for _ in range(num_blocks)
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
        
