import torch.nn as nn
from PositionEncoding import PositionalEncoding
from TokenEmbedding import TokenEmbedding
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device) :
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size,d_model,device)
        self.pos_emb = PositionalEncoding(d_model,max_len,device)
        self.drop = nn.Dropout(drop_prob)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return(tok_emb+pos_emb)
if __name__=='__main__':
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    d_model = 512
    max_len = 30
    drop_prob = 0.1
    vocab_size = 100000
    x = torch.randint(0, vocab_size, (batch_size, max_len), dtype=torch.long, device=device)
    transformerEmbedding = TransformerEmbedding(vocab_size,d_model,max_len,drop_prob,device)
    output = transformerEmbedding.forward(x)
    print(output.shape)