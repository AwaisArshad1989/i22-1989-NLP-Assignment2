import numpy as np
import pandas as pd
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'num_sentences': 500,
    'sentences_per_topic': 100,
    'embedding_dim': 100,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.5,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 32,
    'epochs': 1,  # Quick test
    'patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Load data
with open('cleaned.txt', 'r', encoding='utf-8') as f:
    cleaned_corpus = f.read()

with open('Metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Load embeddings
word_embeddings = np.load('embeddings/embeddings_w2v.npy')
with open('embeddings/word2idx.json', 'r', encoding='utf-8') as f:
    word2idx = json.load(f)

print("Data loaded successfully")

# Simple test
print("Part 2 training setup complete - ready for full implementation!")