import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Graph ODE for recommendation")
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        help="available datasets: ['ml-100k', 'ml-1m', 'amazon-2014/Electronics', 'amazon-2014/Beauty', 'amazon-2014/Electronics']")
    """What's new"""
    parser.add_argument('--noise_schedule', type=str, default='linear',
                        help="(str) The schedule for noise generating: [linear, linear-var, cosine, binomial]")
    parser.add_argument('--noise_scale', type=float, default=0.001,
                        help="(float) The scale for noise generating")
    parser.add_argument('--noise_min', type=float, default=0.0005,
                        help="(float) Noise lower bound for noise generating")
    parser.add_argument('--noise_max', type=float, default=0.005,
                        help="(float) Noise upper bound for noise generating")
    parser.add_argument('--sampling_noise', type=bool, default=False,
                        help="(bool) Whether to use sampling noise")
    parser.add_argument('--sampling_steps', type=int, default=0,
                        help="(int) Steps of the forward process during inference")
    parser.add_argument('--reweight', type=bool, default=True,
                        help="(bool) Assign different weight to different timestep or not")
    parser.add_argument('--mean_type', type=str, default='x0',
                        help="(str) MeanType for diffusion: [x0, eps]")
    parser.add_argument('--steps', type=int, default=5,
                        help="(int) Diffusion steps")
    parser.add_argument('--history_num_per_term', type=int, default=10,
                        help="(int) The number of history items needed to calculate loss weight")
    parser.add_argument('--beta_fixed', type=bool, default=True,
                        help="(bool) Whether to fix the variance of the first step to prevent overfitting")
    parser.add_argument('--dims_dnn', type=list, default=[300],
                        help="(list of int) The dims for the DNN")
    parser.add_argument('--embedding_size', type=int, default=10,
                        help="(int) Timestep embedding size")
    parser.add_argument('--mlp_act_func', type=str, default='tanh',
                        help="(str) Activation function for MLP")
    parser.add_argument('--time_aware', type=bool, default=False,
                        help="(bool) T-DiffRec or not")
    parser.add_argument('--w_max', type=float, default=1,
                        help="(float) The upper bound of the time-aware interaction weight")
    parser.add_argument('--w_min', type=float, default=0.1,
                        help="(float) The lower bound of the time-aware interaction weight")

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
    parser.add_argument('--patience', type=int, default=5,
                        help="early stop patience")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help='path to save weights')
    parser.add_argument('--recent', action='store_true', default=False,
                        help="if sample the most recent neighbors")
    parser.add_argument('--log', type=str, default=None,
                        help="log file path")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--gpu', type=str, default=0,
                        help='training device')

    # model parse
    """ embedding_size: 64              # (int) The number of features in the hidden state.
    inner_size: 256                 # (int) The inner hidden size in feed-forward layer.
    n_layers: 2                     # (int) The number of transformer layers in transformer encoder.
    n_heads: 2                      # (int) The number of attention heads for multi-head attention layer.
    hidden_dropout_prob: 0.5        # (float) The probability of an element to be zeroed.
    attn_dropout_prob: 0.5          # (float) The probability of an attention score to be zeroed.
    hidden_act: gelu                # (str) The activation function in feed-forward layer.
    layer_norm_eps: 1e-12           # (float) A value added to the denominator for numerical stability.
    initializer_range: 0.02         # (float) The standard deviation for normal initialization.
    loss_type: CE                   # (str) The type of loss function. Range in [CE, BPR].
    dnn_type: trm                   # (str) The type of DNN. Range in [trm, ave].

    sess_dropout: 0.2               # (float) The probability of item embeddings in a session to be zeroed.
    item_dropout: 0.2               # (float) The probability of candidate item embeddings to be zeroed.
    temperature: 0.07               # (float) Temperature for contrastive loss. """

    return parser.parse_args()
