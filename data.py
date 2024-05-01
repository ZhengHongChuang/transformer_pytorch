
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

# 获取一个批次的数据
# batch = next(iter(train_iter))

# # 将源序列和目标序列转换为句子
# src_sentences = []
# for src in batch.src:
#     tokens = [loader.source.vocab.itos[i] for i in src]
#     sentence = " ".join(tokens)
#     src_sentences.append(sentence)

# trg_sentences = []
# for trg in batch.trg:
#     tokens = [loader.target.vocab.itos[i] for i in trg]
#     sentence = " ".join(tokens)
#     trg_sentences.append(sentence)

# # 打印源序列和目标序列的句子
# for src, trg in zip(src_sentences, trg_sentences):
#     print("Source:", src)
#     print("Target:", trg)
#     print()


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
# print(loader.target.vocab.stoi['Zwei'])
# print(loader.target.vocab.itos[loader.target.vocab.stoi['Zwei']])
# print(loader.target.vocab.itos[2])
# print(loader.target.vocab.itos[5])
# 查看前20个单词
# for i in range(20):
#     print(loader.target.vocab.itos[i])
# print()
# print(loader.target.vocab.stoi["zwei"])
# print(src_pad_idx)
# print(trg_pad_idx)
# print(trg_sos_idx)

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
# print(enc_voc_size)
# print(dec_voc_size)
# sentence = "hello world"
# src_indexes = [loader.source.vocab.stoi[token] for token in sentence.split()]
# print(src_indexes)
# print(loader.target.init_token)
# trg_indexes = [loader.target.vocab.stoi[loader.target.init_token]]
# print(len(trg_indexes))
# print(loader.target.vocab.itos[10])