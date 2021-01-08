import tensorflow as tf
import numpy as np


# see also: https://www.tensorflow.org/tutorials/text/transformer


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttentionBase(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
         Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionClassical, self).__init__(d_model, num_heads)
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class TransformerBlockBase(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockBase, self).__init__()
        self.mha = None
        self.ffn = None

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockClassical, self).__init__(d_model, num_heads, dff, dropout_rate)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)


class EncoderLayerBase(tf.keras.layers.Layer):
    def __init__(self, 
                num_layers, 
                d_model, 
                num_heads, 
                dff, 
                vocab_size,
                maximum_position_encoding, 
                dropout_rate=0.1):
        super(EncoderLayerBase, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = None
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayerClassical(EncoderLayerBase):
    def __init__(self, 
                num_layers, 
                d_model, 
                num_heads, 
                dff, 
                vocab_size,
                maximum_position_encoding, 
                dropout_rate=0.1):
        super(EncoderLayerClassical, self).__init__(num_layers, d_model, num_heads, dff, vocab_size, maximum_position_encoding, dropout_rate)
        
        self.enc_layers = [TransformerBlockClassical(d_model, num_heads, dff, dropout_rate) 
                        for _ in range(num_layers)]


class TextClassifierTF(tf.keras.Model):
    def __init__(self, 
                num_layers, 
                d_model, 
                num_heads, 
                dff, 
                vocab_size, 
                num_classes, 
                maximum_position_encoding: int=10000, 
                dropout_rate=0.1):
        super(TextClassifierTF, self).__init__()
        self.encoder = EncoderLayerClassical(num_layers, d_model, num_heads, dff, 
                            vocab_size, maximum_position_encoding, dropout_rate)
        
        if num_classes < 2:
            raise RuntimeError("Number of classes must be at least 2")
        elif num_classes == 2:
            self.final_layer = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        else:
            self.final_layer = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax)
    
    def call(self, x, training):
        encoded_output = self.encoder(x, training)  # (batch_size, inp_seq_len, d_model)
        pooled_output = encoded_output[:,0,:]
        final_output = self.final_layer(pooled_output)  # (batch_size, tar_seq_len, num_classes)

        return final_output