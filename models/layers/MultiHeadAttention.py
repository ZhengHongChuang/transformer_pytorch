import torch.nn as nn
from .ScaledDotProductAttention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head) :
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # init k,q,v
        self.w_k = nn.Linear(d_model,d_model)
        self.w_q = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        self.w_concat = nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        k,q,v = self.split(self.w_k(k)),self.split(self.w_q(q)),self.split(self.w_v(v))
        out,_ = self.attention(q,k,v,mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def concat(self,tensor):
        """
            concat head: [head,d_tensor] -->[d_model]
        """
        batch_size,head,max_len,d_tensor = tensor.size()
        d_model = head*d_tensor
        tensor = tensor.transpose(1,2).contiguous().view(batch_size,max_len,d_model)
        return tensor

    def split(self,tensor):
        """
            split d_model: [d_model] -->[head,d_tensor]
        """
        batch_size,max_len,d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size,max_len ,self.n_head,d_tensor).transpose(1,2)

        return tensor
    
# if __name__ == "__main__":
#     import torch
#     d_model = 512
#     n_head = 8
#     multi_head_attention = MultiHeadAttention(d_model, n_head)
#     q = torch.rand(1, 10, d_model)  # 示例输入
#     k = torch.rand(1, 10, d_model)  # 示例输入
#     v = torch.rand(1, 10, d_model)  # 示例输入
#     output = multi_head_attention.forward(q, k, v)
#     print("Output shape:", output.shape)