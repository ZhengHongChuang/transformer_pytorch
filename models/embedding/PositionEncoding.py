import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device) :
        super(PositionalEncoding,self).__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device,requires_grad=False)
        pos = torch.arange(0,max_len,device=device).float().unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0,d_model,2,device=device)*
                             -(math.log(10000.0)/d_model))
        self.encoding[:,0::2] = torch.sin(pos*div_term)
        self.encoding[:,1::2] = torch.cos(pos*div_term)
    def forward(self,x):
        _,seq_len = x.size()
        print(x.size())
        return self.encoding[:seq_len,:]
# if __name__=='__main__':
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     batch_size = 128
#     d_model = 512
#     max_len = 30
#     input = torch.rand([batch_size,max_len])
#     positionalEncoding = PositionalEncoding(d_model,max_len,device)
#     output = positionalEncoding.forward(input)
#     print(output.shape)
