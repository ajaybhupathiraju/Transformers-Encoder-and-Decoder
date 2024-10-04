import numpy as np
import pandas as pd
import tensorflow as tf
import keras


def positional_embedding(emb_dim=256, sequence_length=64):
    """positional embedding for each input token"""
    output = []
    for pos in range(sequence_length):
        PE = np.zeros(emb_dim,dtype="float32");
        for i in range(emb_dim):
            if i % 2 == 0:
                PE[i] = np.sin(pos / 10000 ** (i / emb_dim))
            else:
                PE[i] = np.cos(pos / 10000 ** ((i - 1) / emb_dim))
        output.append(tf.expand_dims(PE, axis=0))
    out = tf.concat(output,axis=0)
    out = tf.expand_dims(out, axis=0)
    return out


#print(positional_embedding(256,64).shape)
