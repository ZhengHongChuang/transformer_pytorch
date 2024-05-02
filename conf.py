import os
import torch
from datetime import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_dir = os.path.join("logs",datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


# model parameter setting
batch_size = 128
max_len = 256
d_model = 512
# d_model = 1024
n_layers = 6
n_heads = 8
# n_heads = 16
ffn_hidden = 2048
# ffn_hidden = 4096
drop_prob = 0.1
# drop_prob = 0.3

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
