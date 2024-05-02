import torch
from utils.bleu import idx_to_word
from data import *
from models.model import Transformer
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

def translate_sentence(sentence, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        src_indexes = [loader.source.vocab.stoi[token] for token in sentence.split()]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.src_mask(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)
        trg_indexes = [loader.target.vocab.stoi[loader.target.init_token]]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.trg_mask(trg_tensor)
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == loader.target.vocab.stoi[loader.target.eos_token]:
                break
        print(idx_to_word(trg_indexes,loader.target.vocab))


if __name__ == '__main__':
    sentence = ""
    weights_path = ''
    translate_sentence(sentence,weights_path)