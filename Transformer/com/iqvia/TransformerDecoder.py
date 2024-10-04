import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from anyio import value
from keras.backend import dtype
from keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization
from oauthlib.uri_validate import query
from tensorflow import newaxis

from com.iqvia.Embeddings import Embeddings
from com.iqvia.TransformerEncoder import TransformerEncoder


class TransformerDecoder(keras.layers.Layer):
    def __init__(self, num_heads, dense_dim, emb_dim):
        super(TransformerDecoder,self).__init__()
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.emb_dim = emb_dim

        self.attention1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.emb_dim,
                                             name="transformer_decoder_MHA1")
        self.attention2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.emb_dim,
                                             name="transformer_decoder_MHA2")

        self.layernorm1 = LayerNormalization(name="transformer_decoder_layer_norm_1")
        self.layernorm2 = LayerNormalization(name="transformer_decoder_layer_norm_2")
        self.layernorm3 = LayerNormalization(name="transformer_decoder_layer_norm_3")

        self.linear_projection = keras.models.Sequential(
            [
                keras.layers.Dense(units=self.dense_dim, activation="relu"),
                keras.layers.Dense(units=self.emb_dim)
            ]
        )

    def call(self, inputs, encoder_outputs, enc_mask, mask=None):
        if mask is not None:
            casual_mask = tf.linalg.band_part(
                tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]], dtype="int32"),
                -1, 0)
            mask = tf.cast(mask[:, newaxis, :], dtype="int32")
            T = tf.shape(mask)[2]
            mask = tf.repeat(mask, T, axis=1)

            enc_mask = tf.cast(enc_mask[:, newaxis, :], dtype="int32")
            enc_mask = tf.repeat(enc_mask, T, axis=1)
            mask = tf.minimum(casual_mask, mask)

        attention1_out = self.attention1(query=inputs, key=inputs, value=inputs, attention_mask=mask)
        norm1 = self.layernorm1(attention1_out)

        attention2_out = self.attention2(query=norm1, key=encoder_outputs, value=encoder_outputs,
                                         attention_mask=enc_mask)
        norm2 = self.layernorm2(attention2_out + norm1)

        linear_proj = self.linear_projection(norm2)

        norm3 = self.layernorm3(linear_proj + norm2)

        return norm3


# emb = Embeddings(1000,256,sequence_length=64)
# a=np.ones((8,8),dtype="float32")
# a = a.flatten()
# print(a.shape)
# a=tf.expand_dims(a,axis=0)
# print(a.shape)
# embeddings_ouput=emb(a)
# mask = emb.compute_mask1(a)
# print("mask :{}".format(tf.cast(mask,dtype="int32")))
#
# num_heads=2
# dense_dim=256
# emd_dim=256
#
# tenc = TransformerEncoder(num_heads,dense_dim,emd_dim)
# encoder_outputs=tenc(embeddings_ouput)
#
# trafdec = TransformerDecoder(num_heads,dense_dim,emd_dim)
# transfoutput = trafdec(embeddings_ouput,encoder_outputs,enc_mask=mask)