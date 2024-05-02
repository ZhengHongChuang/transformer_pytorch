import math
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self) :
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,q,k,v,mask = None,e=1e-12):
        """
            k的形状 : [batch_size,head,max_len,d_tensor]
        """
        _,_,_,d_tensor = k.size()
        k_t = k.transpose(2,3)
        score = (q@k_t)/math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask==0,e)
        score = self.softmax(score)
        v = score @ v
        return v ,score