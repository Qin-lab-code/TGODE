import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Graph ODE for recommendation")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help="available datasets: ['ml-100k', 'ml-1m', 'amazon-2014/Sports_and_Outdoors', 'amazon-2014/Beauty', 'amazon-2014/Toys_and_Games', 'amazon-2014/Video_Games']")
    parser.add_argument('--n_layers', type=int, default=2,
                        help='n_layers')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='n_heads')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden_size')
    parser.add_argument('--inner_size', type=int, default=256,
                        help='inner_size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.5,
                        help='hidden_dropout_prob')
    parser.add_argument('--attn_dropout_prob', type=float, default=0.5,
                        help='attn_dropout_prob')
    parser.add_argument('--hidden_act', type=str, default='gelu',
                        help='hidden_act')
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12,
                        help='layer_norm_eps')
    parser.add_argument('--initializer_range', type=float, default=0.02,
                        help='initializer_range')
    parser.add_argument('--loss_type', type=str, default='CE',
                        help="loss_type ['BPR', 'CE']")

    parser.add_argument('--joint', action='store_true', default=False,
                        help="use joint optimization for ODE backwards")
    parser.add_argument('--hidden', type=int, default=64,
                        help="node embedding size")
    parser.add_argument('--hidden_temp', type=int, default=16,
                        help="temporal embedding of nodes")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="hyper rescaling argument for regularize Adj matrix")
    parser.add_argument('--dropout', action='store_true', default=False,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.8,
                        help="dropout probalitity")
    parser.add_argument('--ode_step', type=float, default=0.2,
                        help="step size of RK-4 ODE solver")


    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--batch', type=int, default=256,
                        help="the batch size for training procedure")
    parser.add_argument('--slice_num', type=int, default=2,
                        help="number of time slices")
    parser.add_argument('--testbatch', type=int, default=256,
                        help="the batch size of users for testing")
    parser.add_argument('--evalnegsample', type=int, default=100,
                        help="num of negative samples when evaluation, -1 means use all negative items")
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help="if use pretrain embeddings")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3,
                        help="weight decay for l2 normalizaton")
    parser.add_argument('--patience', type=int, default=2,
                        help="early stop patience")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help='path to save weights')
    parser.add_argument('--recent', action='store_true', default=False,
                        help="if sample the most recent neighbors")
    parser.add_argument('--log', type=str, default=None,
                        help="log file path")
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--gpu', type=str, default=3,
                        help='training device')
    return parser.parse_args()

