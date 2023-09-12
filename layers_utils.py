# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import math

class PosWiseFF(tf.keras.layers.Layer):
    """
    Feed-Forward Network (FFN): Position-Wise (Dense layers applied to the last dimension)
    - The first dense layer initially projects the last dimension of the input to
    a higher dimension with a certain expansion ratio
    - The second dense layer projects it back to the initial last dimension

    Args:
    - d_model [int]: embedding dimension
    - d_ff [int]: number of hidden neurons for the first dense layer (expansion ratio)
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout

    """

    def __init__(self, d_model, d_ff, atv_fun, dropout_rate, **kwargs):
        super(PosWiseFF, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(units=self.d_ff, activation=self.atv_fun)
        self.dense_2 = tf.keras.layers.Dense(units=self.d_model, activation=self.atv_fun)
        self.dropout_layer_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        """

        Args:
        - x: attention outputs

        Shape:
        - Inputs:
        - x: (B,L,E) where B is the batch size, L is the sequence length, E is the embedding dimension
        - Outputs:
        - x: (B,L,E) where B is the batch size, L is the input sequence length, E is the embedding dimension

        """

        x = self.dense_1(x)
        x = self.dropout_layer_1(x)
        x = self.dense_2(x)
        x = self.dropout_layer_2(x)

        return x

    def get_config(self):
        config = super(PosWiseFF, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate})
        return config


class attn_pad_mask(tf.keras.layers.Layer):
    """
    Attention Padding Mask Layer: Creates the Padding mask for the attention weights

    """

    def __init__(self, **kwargs):
        super(attn_pad_mask, self).__init__(**kwargs)

        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x, 0), dtype=tf.float32))
        self.reshape_layer = tf.keras.layers.Reshape([1, 1, -1])

    def call(self, x):
        """

        Args:
        - x: input sequences

        Shape:
        - Inputs:
        - x: (B,L) where B is the batch size, L is the sequence length
        - Outputs:
        - x: (B,1,1,L) where B is the batch size, L is the input sequence length

        """

        x = self.lambda_layer(self.reshape_layer(x))

        return x


def add_reg_token(x, voc_size):
    """
    Rp and Rs Tokens Function: adds the Rp or the Rs token to the input sequences

    Args:
    - x: inputs sequences
    - voc_size [int]: number of unique tokens

    Shape:
    - Inputs:
    - x: (B,L) where B is the batch size, L is the sequence length
    - Outputs:
    - x: (B,1+L) where B is the batch size, L is the input sequence length

    """

    reg_token = tf.convert_to_tensor(voc_size + 1, dtype=tf.int32)
    broadcast_shape = tf.where([True, False], tf.shape(x), [0, 1])
    reg_token = tf.broadcast_to(reg_token, broadcast_shape)

    return tf.concat([reg_token, x], axis=1)

class PosLayer(tf.keras.layers.Layer):
    """
    Embedding Layer: generates a learned embedding to every token with a fixed size

    Args:
    - voc_size [int]: number of unique tokens
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout
    - positional_enc [boolean]: positional encoding option: if true adds a positional embedding
    to the output of the embedding layer

    """

    def __init__(self, max_len, d_model, dropout_rate, **kwargs):
        super(PosLayer, self).__init__(**kwargs)

        self.max_len = max_len
        self.d_model = d_model
        self.dropout_rate = dropout_rate


    def build(self, input_shape):
        self.pos_enc_layer = tf.keras.layers.Embedding(self.max_len, self.d_model)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs):
        """
        Shape:
        - Inputs:
        - inputs: (B,L,E): where B is the batch size, L is the sequence length,
                        E is the embedding dimension

        - Outputs:
        - output_tensor: (B,L,E): where B is the batch size, L is the sequence length,
                        E is the embedding dimension

        """

        tgt_tensor = tf.range(self.max_len)

        output_tensor = inputs + self.pos_enc_layer(tgt_tensor)
        output_tensor = self.dropout_layer(output_tensor)
        return output_tensor

    def get_config(self):
        config = super(PosLayer, self).get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate})

        return config

class BertPooler(tf.keras.layers.Layer):
    """
    Position-Wise Pooling Block to map the aggregated representation of the SMILES Strings
    to the last dimension of the protein tokens (embedding/representation size)

    Args:
    - dense_size [int]: number of hidden units of the projection dense layer
    - atv_fun: dense layer activation function
    - dense_opt [int]: 1 - project the last dimension of the aggregated representation

    """

    def __init__(self, dense_size, atv_fun, dense_opt, drop_rate, **kwargs):
        super(BertPooler, self).__init__(**kwargs)

        self.dense_size = dense_size
        self.atv_fun = atv_fun
        self.dense_opt = dense_opt
        self.drop_rate = drop_rate

    def build(self, input_shape):
        if self.dense_opt:
            self.dense_layer = tf.keras.layers.Dense(units=self.dense_size, activation=self.atv_fun)
            self.drop_layer = tf.keras.layers.Dropout(self.drop_rate)

        self.reshape_layer = tf.keras.layers.Reshape((1, -1))
        self.lambda_layer = tf.keras.layers.Lambda((lambda x: tf.gather(x, 0, axis=1)))
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, hidden_inputs):
        """

        Args:
        - hidden_inputs: SMILES Transformer-Encoder Outputs

        Shape:
        - Inputs:
        - hidden_inputs: (B,L,E): where B is the batch size, L is the SMILES sequence length,
                        E is the embedding dimension

        - Outputs:
        - cls_token: (B,1,E_Proj):  where B is the batch size,
                                    E_Proj is the projected dimension (protein token representation dimension)

        """

        cls_token = self.lambda_layer(hidden_inputs)
        cls_token = self.reshape_layer(cls_token)

        if bool(self.dense_opt):
            # cls_token = self.drop_layer(cls_token)
            cls_token = self.layernorm(self.dense_layer(cls_token))
        return cls_token

    def get_config(self):
        config = super(BertPooler, self).get_config()
        config.update({
            'dense_size': self.dense_size,
            'atv_fun': self.atv_fun,
            'dense_opt': self.dense_opt})

        return config

def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


def opt_config(opt):
    opt_fn = None
    if opt[0] == 'radam':
        opt_fn = tfa.optimizers.RectifiedAdam(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                              beta_2=float(opt[3]), epsilon=float(opt[4]),
                                              weight_decay=float(opt[5]), total_steps=int(opt[6]),
                                              warmup_proportion=float(opt[7]),
                                              min_lr=float(opt[8]))
    elif opt[0] == 'adam':
        opt_fn = tf.keras.optimizers.Adam(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                          beta_2=float(opt[3]), epsilon=float(opt[4]))

    elif opt[0] == 'adamw':
        opt_fn = tfa.optimizers.AdamW(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                      beta_2=float(opt[3]), epsilon=float(opt[4]),
                                      weight_decay=float(opt[5]))

    elif opt[0] == 'lamb':
        opt_fn = tfa.optimizers.LAMB(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                     beta_2=float(opt[3]), epsilon=float(opt[4]))

    return opt_fn

# Activation Configuration Function
def af_config(activation_fn):
    if activation_fn == 'gelu':
        activation_fn = gelu

    elif activation_fn == 'tanh':
        activation_fn = tf.math.tanh

    else:
        activation_fn = activation_fn

    return activation_fn