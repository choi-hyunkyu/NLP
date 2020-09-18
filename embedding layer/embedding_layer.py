'''
임베딩 레이어 생성
'''
import torch
import torch.nn as nn
embedding_layer = nn.Embedding(
    num_embeddings = len(vocab),
    embedding_dim = 3,
    padding_idx = 1
    )
embedding_layer.weight