import os
import torch
import argparse
from utils import utils
from dataset import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True)
    parser.add_argument('--train_step', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--filter_size', type=int, default=2048)
    parser.add_argument('--warmup', type=int, default=16000)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--summary_grad', action='store_true')
    opt = parser.parse_args()

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    if not os.path.exists(opt.output_dir + '/last/models'):
        os.makedirs(opt.output_dir + '/last/models')
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
    train_data, validation_data, i_vocab_size, t_vocab_size, opt = \
        prepare(opt.problem, opt.data_dir, opt.max_length,
                        opt.batch_size, device, opt)
    if i_vocab_size is not None:
        print("# of vocabs (input):", i_vocab_size)
    print("# of vocabs (target):", t_vocab_size)

    if opt.model == 'transformer':
        from models.model import Transformer
        model_fn = Transformer
    if os.path.exists(opt.output_dir + '/last/models/last_model.pt'):
        print("加载权重")
        last_model_path = opt.output_dir + '/last/models'
        model,global_step = utils.load_checkpoint(last_model_path,device,is_eval=False)
    else:
        model = model_fn(opt.s)    
        






if __name__ == '__main__':
    main()
