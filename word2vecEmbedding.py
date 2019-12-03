from gensim.models import KeyedVectors
from hyperparameters import *
import torch
if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format(EMBEDDING_PATH,binary=True)
    vector = model['grant is great']