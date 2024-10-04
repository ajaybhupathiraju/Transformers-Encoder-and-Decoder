import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Input, Dropout

from com.iqvia.Embeddings import Embeddings
from com.iqvia.TransformerDecoder import TransformerDecoder
from com.iqvia.TransformerEncoder import TransformerEncoder

vocab_size = 1000
english_sequence_length = 64
french_sequence_length = 64
emb_dim = 256
batch_size=64
num_heads = 2
num_layers = 1
dense_dim = 256
### loading data
text_dataset = tf.data.TextLineDataset("G:\\Ajay\\dataset\\Encoder_Decoder\\fra.txt")

text_dataset=text_dataset.take(1000)

english_vectorization_layer = keras.layers.TextVectorization(max_tokens=vocab_size,
                                                             standardize="lower_and_strip_punctuation",
                                                             output_mode="int",
                                                             output_sequence_length=english_sequence_length)

french_vectorization_layer = keras.layers.TextVectorization(max_tokens=vocab_size,
                                                            standardize="lower_and_strip_punctuation",
                                                            output_mode="int",
                                                            output_sequence_length=french_sequence_length)


def selector(input_text):
    split_text = tf.strings.split(input_text, "\t")
    return {
        "input_1": split_text[0],
        "input_2": "start " + split_text[1]
    }, split_text[1] + " end"


split_dataset = text_dataset.map(selector)


def seperator(input_text):
    split_text = tf.strings.split(input_text, "\t")
    return split_text[0], "start " + split_text[1] + " end"


init_dataset = text_dataset.map(seperator)

english_text = init_dataset.map(lambda x, y: x)
french_text = init_dataset.map(lambda x, y: y)

english_vectorization_layer.adapt(english_text)
french_vectorization_layer.adapt(french_text)


def vectorizer(inputs, outputs):
    return {
        "input_1": english_vectorization_layer(inputs["input_1"]),
        "input_2": french_vectorization_layer(inputs["input_2"]),
    },
    french_vectorization_layer(outputs)

print("split_dataset vectorizer begins..")
dataset = split_dataset.map(vectorizer);
print("split_dataset vectorizer ends..")
dataset = dataset.shuffle(2048).unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
NUM_BATCHES = 10000/batch_size

train_dataset = dataset.take(int(0.9*NUM_BATCHES))
val_dataset = dataset.take(int(0.1*NUM_BATCHES))
print("train_dataset & val_dataset split done")

################### Building Model ##########################
encoder_input = keras.layers.Input(shape=(None,))
emb = Embeddings(vocab_size, emb_dim, sequence_length=english_sequence_length)
x=emb(encoder_input)
enc_mask = emb.compute_mask1(encoder_input)

for i in range(num_layers):
    x=TransformerEncoder(num_heads, dense_dim, emb_dim)(x)
encoder_outputs = x


decoder_input = keras.layers.Input(shape=(None,))
emb = Embeddings(vocab_size, emb_dim, sequence_length=french_sequence_length)
x=emb(decoder_input)

for i in range(num_layers):
    x=TransformerDecoder(num_heads, dense_dim, emb_dim)(x,encoder_outputs,enc_mask)

x = Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(vocab_size,activation="softmax")(x)

model = keras.models.Model(inputs=[encoder_input,decoder_input],outputs=decoder_outputs)
print(model.summary())

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")

model.fit(train_dataset,validation_data=val_dataset,epochs=1)
print(model.evaluate(val_dataset))