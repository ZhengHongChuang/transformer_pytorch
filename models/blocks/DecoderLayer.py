import torch.nn as nn
from  models.layers import MultiHeadAttention,PositionwiseFeedForward,LayerNorm
class DecoderLayer(nn.Module):
    def __init__(self,d_model, ffn_hidden, n_head, drop_prob) :
        super(DecoderLayer,self).__init__()
        self.attention1 = MultiHeadAttention(d_model,n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.attention2 = MultiHeadAttention(d_model,n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
        
    def forward(self, dec, enc, mask):
        _x = dec
        x = self.attention1(q=dec,k=dec,v=dec,mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if enc is not None:
            _x = x
            x = self.attention2(q=x,k=enc,v=enc,mask=mask)
            x =self.dropout2(x)
            x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
# if __name__ == "__main__":
#     d_model = 512
#     ffn_hidden = 2048
#     n_head = 8
#     drop_prob = 0.1
#     decoder = DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
#     import torch
#     dec = torch.rand(1, 10, d_model)  # 示例输入
#     enc = torch.rand(1, 10, d_model)  # 示例输入
#     mask = torch.rand(1, 1, 10, 10)  # 示例输入
#     output = decoder(dec, enc, mask)
#     print( output.shape)
    

    
