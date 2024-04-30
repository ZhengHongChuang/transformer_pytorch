import torch.nn as nn
from models.blocks.DecoderLayer import DecoderLayer
from models.embedding.TransformerEmbedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self,dec_vocab_size,max_len,d_model,ffn_hiden,n_head,n_layers, drop_prob, device):
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(dec_vocab_size,d_model,max_len,drop_prob,device)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,ffn_hiden,n_head,drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model,dec_vocab_size)

    def forward(self,trg,enc_src,trg_mask,src_mask):
        x = self.embedding(trg)
        for layer in self.decoder_layers:
            x = layer(x,enc_src,trg_mask,src_mask)
        x = self.linear(x)
        return x
# # 测试主函数
# if __name__ == "__main__":
#     import torch
#     dec_vocab_size = 10000
#     max_len = 100
#     d_model = 512
#     ffn_hidden = 2048
#     n_head = 8
#     n_layers = 6
#     drop_prob = 0.1
#     device = torch.device("cpu")

#     decoder = Decoder(dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
#     trg = torch.randint(0, dec_vocab_size, (1, max_len), dtype=torch.long, device=device)
#     enc_src = torch.randn(1, max_len, d_model, device=device)
#     trg_mask = torch.ones(1, 1, max_len, max_len, device=device)  # 示例目标掩码
#     src_mask = torch.ones(1, 1, max_len, max_len, device=device)  # 示例源掩码
#     output = decoder(trg, enc_src, trg_mask, src_mask)
#     print( output.shape)