
from torch import nn, optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from data import *
from models.model.Transformer import Transformer
from utils.bleu import idx_to_word, get_bleu
writer = SummaryWriter(log_dir=log_dir)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    d_model=d_model,
                    n_head=n_heads,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)




def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
            
            

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    

    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    step = 0
    for step in range(total_epoch):
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        if step > warmup:
            scheduler.step(valid_loss)
        writer.add_scalar('Train Loss', train_loss, step+1)
        writer.add_scalar('Validation Loss', valid_loss, step+1)
        writer.add_scalar('BLEU Score', bleu, step+1)

      
        if valid_loss < best_loss:
            best_loss = valid_loss
            folder = 'weights'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            torch.save(model.state_dict(), 'weights/model-{0}.pt'.format(valid_loss))

    writer.close()

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
