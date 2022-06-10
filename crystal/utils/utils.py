import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import numpy.linalg as la
import torch.nn as nn
import csv
import os

def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def get_middle(array):
    array.sort()
    index = len(array) // 2
    return (array[index] + array[~index]) / 2
