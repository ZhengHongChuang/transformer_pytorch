
from conf import *
from utils.data_loader import DataLoader
from utils.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()


loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)


# for i in range(5):
#     example = train.examples[i]
#     src_sentence = example.src
#     trg_sentence = example.trg
#     src_ids = [loader.source.vocab.stoi[token] for token in src_sentence]
#     trg_ids = [loader.target.vocab.stoi[token] for token in trg_sentence]
#     print(f"Source sentence: {' '.join(src_sentence)}")
#     print(f"Source ids: {src_ids}")
#     print(f"Target sentence: {' '.join(trg_sentence)}")
#     print(f"Target ids: {trg_ids}")
#     print()
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

# print(loader.source.vocab[10])
# print(src_pad_idx)
# print(trg_pad_idx)
# print(trg_sos_idx)

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
# print(enc_voc_size)
# print(dec_voc_size)