import torch.nn as nn
from models.blocks import EncoderLayer
from models.embedding.TransformerEmbedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self,vocab_size,max_len,d_model,ffn_hiden,n_head,n_layers, drop_prob, device):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size,d_model,max_len,drop_prob,device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,ffn_hiden,n_head,drop_prob) for _ in range(n_layers)])
    def forward(self,x,mask):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x,mask)
        return x

# if __name__ == "__main__":
#     import torch
#     vocab_size = 10000
#     max_len = 100
#     d_model = 512
#     ffn_hidden = 2048
#     n_head = 8
#     n_layers = 6
#     drop_prob = 0.1
#     device = torch.device( "cpu")

#     encoder = Encoder(vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
#     x = torch.randint(0, vocab_size, (1, max_len), dtype=torch.long, device=device)
#     mask = torch.ones(1, 1, max_len, max_len, device=device)  # 示例掩码
#     output = encoder(x, mask)
#     print( output.shape)