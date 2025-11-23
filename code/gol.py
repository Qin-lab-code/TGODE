import torch, os, logging, random, multiprocessing
import numpy as np
from parse import parse_args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

ARG = parse_args()
os.environ['NUMEXPR_MAX_THREADS'] = '16'
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"
if ARG.log is not None:
    logging.basicConfig(filename=ARG.log, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
else:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

def pLog(s: str):
    logging.info(s)

device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
print(device)
CORES = multiprocessing.cpu_count() // 4

SEED = ARG.seed
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
EPOCH = ARG.epoch
PATH = ARG.path
DATA_PATH = '../data/'
FILE_PATH = './checkpoints/'
SLICE_NUM = ARG.slice_num
dataset = ARG.dataset
recent = ARG.recent
eval_sample = ARG.evalnegsample
patience = ARG.patience
norm = True
n_items = 0
max_seq_len = 0
beta = 12

SAVE = ARG.save
LOAD = ARG.load

seed_torch(SEED)

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

conf = {'lr': ARG.lr, 'decay': ARG.decay,  'n_layers': ARG.n_layers, 'n_heads': ARG.n_heads, 'hidden_size': ARG.hidden_size,
        'inner_size': ARG.inner_size, 'hidden_dropout_prob': ARG.hidden_dropout_prob, 'attn_dropout_prob': ARG.attn_dropout_prob,
        'hidden_act': ARG.hidden_act, 'layer_norm_eps': ARG.layer_norm_eps, 'initializer_range': ARG.initializer_range, 'loss_type': ARG.loss_type,
        'hidden': ARG.hidden, 'dropout': ARG.dropout,
        'keepprob': ARG.keepprob, 'pretrain': ARG.pretrain, 'evalneg': ARG.evalnegsample, 'joint': ARG.joint,
        'alpha': ARG.alpha, 'hidden_temp': ARG.hidden_temp, 'step': ARG.ode_step
        }
