import tensorflow as tf
from transformer_encoder import *
from embedding_layer import *
import time
import os
import itertools
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import math

def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
    

def bert_mlm(seq_len, dict_len, full_attn, return_intermediate,
             d_model, dropout_rate, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dim_k, parameter_sharing):


    if enc_atv_fun == 'gelu':
        enc_atv_fun = gelu

    inputs = tf.keras.Input(shape=seq_len+1, dtype=tf.int64, name='input_layer')
    pad_mask = attn_pad_mask()(inputs)
    input_embedding = tf.keras.layers.Embedding(dict_len + 3, d_model, name='emb_layer')(inputs)
    char_embedding = PosLayer(seq_len + 1, d_model, dropout_rate, name='pos_layer')(input_embedding)

    trans_encoder,_ = Encoder(d_model, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dropout_rate, dim_k, parameter_sharing,
                             full_attn, return_intermediate, name='trans_encoder')(char_embedding, pad_mask)


    outputs = tf.keras.layers.Dense(units=dict_len+3, name='mlm_cls')(trans_encoder)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def load_bert_mlm(model, checkpoint_path):
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-03, beta_1=0.9,
                                               beta_2=0.999, epsilon=1e-08,
                                               weight_decay=1e-04, total_steps=512500,
                                               warmup_proportion=0.01, min_lr=1e-05)
                                               
    ckpt_obj = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    ckpt_obj.restore(latest).expect_partial()
    model = ckpt_obj.model
    new_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('trans_encoder').output)
    new_model._name = 'smiles_transformer'
    return new_model





















