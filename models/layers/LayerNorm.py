import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    """
        模型归一化
    """
    def __init__(self,d_model, eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        out = (x-mean)/torch.sqrt(var+self.eps)
        out = self.gamma *out +self.beta
        return out
# if __name__ == "__main__":
#     d_model = 512
#     layer_norm = LayerNorm(d_model)
#     x = torch.randn(1, 10, d_model) 
#     output = layer_norm(x)
#     print(output.shape)