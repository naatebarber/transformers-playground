import torch
import torch.nn as nn
import torch.utils as utils
from sklearn.utils import shuffle
import pickle

import os

print(os.getcwd())


def load_file(path: str, n_sentences: int):
    dataset = pickle.load(open(path, "rb"))
    # if n_sentences:
    #     dataset = dataset[:n_sentences:]

    for i in range(dataset[:, 0].size):
        dataset[i, 0] = f"<START>{dataset[i, 0]}<EOS>"
        dataset[i, 1] = f"<START>{dataset[i, 1]}<EOS>"

    dataset = shuffle(dataset)

    print(dataset)


path = "./datasets/eng-ge/english-german-test.pkl"

load_file(path=path, n_sentences=30)
