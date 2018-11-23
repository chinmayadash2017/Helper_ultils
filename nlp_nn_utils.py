# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:57:48 2018

@author: chinmaya
"""
# load the pre-trained glove vector


import numpy as np

# load the pre-trained globe vector and return word2vec dict
# load the pre-trained globe vector and return word2vec dict
def load_glove_vec(glove_path, embedding_dim = 100):
    # all available embedding sizes are 50,100,200,300
   word2vec = {}
   with open(glove_path + "/" +'glove.6B.{}d.txt'.format(embedding_dim), encoding="utf8") as file:
       for line in file:
           values = line.split()
           word = values[0]
           vec = np.asarray(values[1:], dtype = 'float32')
           word2vec[word] = vec
   print("Number of words found in word2vec: {}".format(len(word2vec)))
   return word2vec

# prepare embedding matrix
# glove word2vec, word2idx dict and max_vocab_size as argument
# return word embedding matrix of shape (vocab_size, embedding dim)
def prepare_embedding_matrix(word2vec, word2idx, EMBEDDING_DIM, max_vocab_size = 20000):
    num_words = min(max_vocab_size, len(word2idx))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, idx in word2idx.items():
        if idx < max_vocab_size:
            embedding_vec = word2vec.get(word)
            if embedding_vec is not None:
                embedding_matrix[idx] = embedding_vec
    return embedding_matrix
    
    
    
