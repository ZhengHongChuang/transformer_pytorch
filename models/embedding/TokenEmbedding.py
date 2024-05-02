import torch.nn as nn


"""
    Transformer 的词编码
    vocab_size: 词表大小
    d_model: 词向量维度
    padding_idx: 扩充默认值
"""
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model,padding_idx = 1) :
        super(TokenEmbedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model,padding_idx=1)
    def forward(self, x):
        return self.embedding(x)