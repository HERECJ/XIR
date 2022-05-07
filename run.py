from xml.dom import NotSupportedErr
from framework.trainer import Trainer, Trainer_Resample, Trainer_MixNeg, Trainer_WithLast, Trainer_Re_WithLast
from framework.trainer_cache import Trainer_Cache
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--model', default='MF')
    parser.add_argument('--log_path',default='logs', type=str, help='path for log files')
    parser.add_argument('--data_name', default='ml-100k', type=str, help='name of dataset')
    parser.add_argument('--data_dir', default='datasets/clean_data', type=str, help='data dir')
    parser.add_argument('--split_ratio', default=0.8, type=float)
    parser.add_argument('--num_workers', default=8, type=int) 
    parser.add_argument('--fix_seed', action='store_false', help='whether to fix the seed values')
    parser.add_argument('--seed', default=10, type=int, help='random seeds')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizers')
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=17, type=int)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--metrics', default=['ndcg', 'recall'])
    parser.add_argument('--valid_interval', default=5, type=int)
    parser.add_argument('--topk', default=100, type=int, help='cutoff for evaluators')
    parser.add_argument('--cutoffs', default=[10, 20, 50], nargs='+', type=int)
    parser.add_argument('--steprl', action='store_false', help='whether to use steprl, default true')
    parser.add_argument('--step_size', default=5, type=int, help='step size for stepRL')
    parser.add_argument('--step_gamma', default=0.95, type=float, help='step discount for stepRL')
    parser.add_argument('--debias', default=2, type=int, help='the debias method')
    parser.add_argument('--sample_from_batch', action='store_true', help='indicate whether sampling from batch')
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--lambda', default=0.5, type=float, help='the coefficient to controll the cache')
    parser.add_argument('--alpha', default=1e-4, type=float, help='the lr of the streaming frequency estimation algorithm')
    parser.add_argument('--pop_mode', default=2, type=int, help='the mode of pop normalization')

    config = vars(parser.parse_args())


    if config['debias'] in [1,2,7]:
        trainer = Trainer(config)
    elif config['debias'] in [3]:
        trainer = Trainer_Resample(config)
    elif config['debias'] in [4]:
        trainer = Trainer_MixNeg(config)
    elif config['debias'] in [5]:
        trainer = Trainer_WithLast(config)
    elif config['debias'] in [6]:
        trainer = Trainer_Re_WithLast(config)
    elif config['debias'] in [8]:
        trainer = Trainer_Cache(config)
    else:
        raise NotSupportedErr
    
    train_mat, test_mat = trainer.load_dataset()
    trainer.fit(train_mat, test_mat)

