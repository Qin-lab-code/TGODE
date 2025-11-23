import torch, os, logging, random, multiprocessing
import numpy as np
from parse1 import parse_args

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
lr = ARG.lr
decay = ARG.decay
n_items = 0
n_users = 0
max_seq_len = 0

"New"
mean_type = ARG.mean_type
time_aware = ARG.time_aware
w_max = ARG.w_max
w_min = ARG.w_min
noise_schedule = ARG.noise_schedule
noise_scale = ARG.noise_scale
noise_max = ARG.noise_max
noise_min = ARG.noise_min
steps = ARG.steps
beta_fixed = ARG.beta_fixed
embedding_size = ARG.embedding_size
norm = None
reweight = ARG.reweight
sampling_steps = ARG.sampling_steps
sampling_noise = ARG.sampling_noise
mlp_act_func = ARG.mlp_act_func
history_num_per_term = ARG.history_num_per_term
dims_dnn = ARG.dims_dnn
hid_temp = 16
emb_size = 64

SAVE = ARG.save
LOAD = ARG.load

seed_torch(SEED)

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# model parse


