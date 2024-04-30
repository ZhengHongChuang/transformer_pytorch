import torch.nn as nn
from models.layers.LayerNorm import LayerNorm
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.PositionwiseFeedForward import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(self,d_model, ffn_hidden, n_head, drop_prob) :
        super(Encoder,self).__init__()
        self.attention = MultiHeadAttention(d_model,n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        
    def forward(self,x,mask):
        _x = x
        x = self.attention(q=x,k=x,v=x,mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x =self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
