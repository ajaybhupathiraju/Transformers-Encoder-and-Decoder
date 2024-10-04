import tensorflow as tf
import keras
from keras.layers import Input, MultiHeadAttention, LayerNormalization
from tensorflow import newaxis


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, dense_dim, emd_dim):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.emd_dim = emd_dim

        self.attention1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.emd_dim)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.linear_projection = keras.models.Sequential(
            [
                keras.layers.Dense(units=self.dense_dim, activation="relu"),
                keras.layers.Dense(units=self.emd_dim)
            ]
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask[:, newaxis, :], dtype="int32")
            T = tf.shape(mask)[2]
            mask = tf.repeat(mask, T, axis=1)

        attention_output1 = self.attention1(query=inputs, key=inputs, value=inputs, attention_mask=mask)

        norm1 = self.layernorm1(attention_output1 + inputs)
        linear_proj = self.linear_projection(norm1)
        return self.layernorm2(linear_proj + norm1)


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
# tenc(embeddings_ouput)