
from torch import nn, optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from data import *
from models.model import Transformer
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
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
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




def train(model, iterator, optimizer, criterion, clip, step):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
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
        writer.add_scalar('Train Loss', loss.item(), step)
        step += 1

    return epoch_loss / len(iterator), step

def evaluate(model, iterator, criterion, step):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
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
            
            writer.add_scalar('BLEU Score', total_bleu, step)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    writer.add_scalar('Validation Loss', epoch_loss / len(iterator), step)

    return epoch_loss / len(iterator), batch_bleu, step

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    step = 0
    for step in range(total_epoch):
        train_loss, step = train(model, train_iter, optimizer, criterion, clip, step)
        valid_loss, bleu, step = evaluate(model, valid_iter, criterion, step)
        if step > warmup:
            scheduler.step(valid_loss)
        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'weights/model-{0}.pt'.format(valid_loss))

    writer.close()

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
