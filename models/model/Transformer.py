import torch
import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder



class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super(Transformer,self).__init__()
        # init 
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        # model
        self.encoder = Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device)
        self.decoder = Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device)


    def forward(self,src,trg):
        src_mask = self.src_mask(src)
        trg_mask = self.trg_mask(trg)
        enc_src = self.encoder(src,src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
    def src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    def trg_mask(self,trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    


# if __name__ == "__main__":
#     # 模型参数
#     src_pad_idx = 0
#     trg_pad_idx = 0
#     trg_sos_idx = 1
#     enc_voc_size = 10000
#     dec_voc_size = 10000
#     d_model = 512
#     n_head = 8
#     max_len = 100
#     ffn_hidden = 2048
#     n_layers = 6
#     drop_prob = 0.1
#     device = torch.device( "cpu")

#     # 初始化 Transformer 模型
#     transformer = Transformer(src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head,
#                               max_len, ffn_hidden, n_layers, drop_prob, device)

#     # 生成随机输入序列
#     src = torch.randint(0, enc_voc_size, (1, 50))  # 假设编码器输入长度为 50
#     trg = torch.randint(0, dec_voc_size, (1, 60))  # 假设解码器输入长度为 60

#     # 调用模型的前向传播方法
#     output = transformer(src, trg)

#     # 输出结果
#     print( output.shape)

