#!/usr/bin/env python

import argparse
import tqdm
import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from qtransformer_tf import TextClassifierTF


BUFFER_SIZE = 10000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--q_device', default='default.qubit', type=str)
    parser.add_argument('-B', '--batch_size', default=32, type=int)
    parser.add_argument('-E', '--n_epochs', default=5, type=int)
    parser.add_argument('-C', '--n_classes', default=2, type=int)
    parser.add_argument('-l', '--lr', default=0.001, type=float)
    parser.add_argument('-v', '--vocab_size', default=20000, type=int)
    parser.add_argument('-e', '--embed_dim', default=8, type=int)
    parser.add_argument('-s', '--max_seq_len', default=64, type=int)
    parser.add_argument('-f', '--ffn_dim', default=8, type=int)
    parser.add_argument('-t', '--n_transformer_blocks', default=1, type=int)
    parser.add_argument('-H', '--n_heads', default=2, type=int)
    parser.add_argument('-q', '--n_qubits_transformer', default=0, type=int)
    parser.add_argument('-Q', '--n_qubits_ffn', default=0, type=int)
    parser.add_argument('-L', '--n_qlayers', default=1, type=int)
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float)
    args = parser.parse_args()

    model = TextClassifierTF(
        num_layers=args.n_transformer_blocks,
        d_model=args.embed_dim,
        num_heads=args.n_heads,
        dff=args.ffn_dim,
        vocab_size=args.vocab_size,
        num_classes=args.n_classes,
        maximum_position_encoding=1024,
        dropout_rate=args.dropout_rate,
        n_qubits_transformer=args.n_qubits_transformer,
        n_qubits_ffn=args.n_qubits_ffn,
        n_qlayers=args.n_qlayers,
        q_device=args.q_device)

    assert args.n_classes >= 2

    if args.n_classes == 2:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    else:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparese_categorical_accuracy'])
    #print(model.summary())

    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=args.vocab_size)

    train_data = pad_sequences(train_data, maxlen=args.max_seq_len, padding='pre', truncating='pre')
    test_data = pad_sequences(test_data, maxlen=args.max_seq_len, padding='pre', truncating='pre')
    
    history = model.fit(train_data, train_labels,
                    epochs=args.n_epochs,
                    validation_data=(test_data, test_labels),
                    batch_size=args.batch_size,
                    verbose=1)
    
    model.save('model_tf')