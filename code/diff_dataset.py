import torch
import numpy as np, pandas as pd
from torch.utils.data import Dataset
import gol
from gol import pLog
from os.path import join, exists
import pickle as pkl
import torch.nn.functional as F
import dgl

class Data(Dataset):
    def __init__(self, n_user, n_item, max_n_node, trn_seq, trn_time, val_seq = None, val_time = None, tst_seq = None, tst_time = None, is_eval = False):
        self.n_user, self.n_item = n_user, n_item

        self.is_eval = is_eval

        items, times, num_node, target, target_time, session_len, user_id = [], [], [], [], [], [], []

        if val_seq == None:

            for idx, (session, t) in enumerate(zip(trn_seq, trn_time)):
                for i in range(len(session)-1):
                    target.append(session[i+1])
                    items.append(session[:i+1])
                    times.append(t[:i+1])
                    user_id.append(idx)


        elif tst_seq == None:

            for idx, (session1, t1, session2, t2) in enumerate(zip(trn_seq, trn_time, val_seq, val_time)):
                for i in range(len(session2)):
                    target.append(session2[i])
                    temp = session1 + session2[:i]
                    temp_t = t1 + t2[:i]

                    items.append(temp)
                    times.append(temp_t)
                    user_id.append(idx)


        else:

            for idx, (session1, t1, session2, t2, session3, t3) in enumerate(zip(trn_seq, trn_time, val_seq, val_time, tst_seq, tst_time)):
                session = session1 + session2
                t = t1 + t2
                for i in range(len(session3)):
                    target.append(session3[i])
                    temp = session + session3[:i]
                    temp_t = t + t3[:i]

                    items.append(temp)
                    times.append(temp_t)
                    user_id.append(idx)

        self.target = target
        self.items = items
        self.times = times
        self.user_id = user_id
        # self.target = torch.LongTensor(target).to(gol.device)
        # self.session_len = torch.LongTensor(session_len).to(gol.device)
        # self.items = torch.LongTensor(items).to(gol.device)
        # self.times = torch.LongTensor(times).to(gol.device)

        self.seq_len = len(self.items)

    def __len__(self):
        return self.seq_len

    def __getitem__(self, index):
        return self.items[index], self.times[index], self.target[index], self.user_id[index]

def label_last(g, last_nid):
    is_last = torch.zeros(g.num_nodes(), dtype=torch.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g

def seq_to_temporal_session_graph(seq, times):
    items, indices = np.unique(seq, return_index=True)
    num_nodes = len(items)
    if num_nodes == 0:
        num_nodes = 1
        seq = [0]
        times = [0.0]
        items, indices = np.unique(seq, return_index=True)
    iid2nid = {iid: i for i, iid in enumerate(items)}


    seq_nid = [iid2nid[iid] for iid in seq]
    # counter = Counter(
    #     [(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]
    # )
    edges = [[seq_nid[i], seq_nid[i + 1]] for i in range(len(seq) - 1)]
    # edges = counter.keys()

    if len(edges) > 0:
        src, dst = zip(*edges)
        weight = torch.ones(len(edges)).long()
    else:
        src, dst = [0], [0]
        weight = torch.ones(1).long()

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    # print(len(edges), g.number_of_edges())
    g.edata['w'] = weight
    # print(len(times), g.number_of_nodes())
    # g.ndata['t']  = th.tensor(times)[indices]
    g.ndata['t'] = torch.ones(g.num_nodes()) * max(times)
    # print(g.edata, times, g.number_of_edges(), g.number_of_nodes())
    if g.number_of_edges() == 1 and g.number_of_nodes() == 1:
        g.edata['t'] = torch.tensor(times)[0].unsqueeze(-1)
    else:
        g.edata['t'] = torch.tensor(times)[1:]  # [-g.number_of_edges():]

    # print(g.edata)

    g.ndata['iid'] = torch.from_numpy(items)
    label_last(g, iid2nid[seq[-1]])

    return g


def collate_fn_factory_temporal(*seq_to_graph_fns):
    def collate_fn(samples):
        seqs, times, labels, user_id = zip(*samples)
        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, seqs, times))
            num_nodes = torch.tensor([graph.number_of_nodes() for graph in graphs], dtype=torch.long)
            max_num = max(num_nodes)
            embeds_id = torch.vstack(
                [F.pad(graph.ndata['iid'], (0, max_num - len(graph.ndata['iid'])), value=graph.ndata['iid'][-1]) for
                 graph in graphs])
            times = torch.vstack(
                [F.pad(graph.ndata['t'], (0, max_num - len(graph.ndata['iid'])), value=graph.ndata['iid'][-1]) for graph
                 in graphs])
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = torch.LongTensor(labels)
        user_id = torch.LongTensor(user_id)
        # print(inputs[0].edata)
        return inputs, labels.to(gol.device), embeds_id.to(gol.device), times.to(gol.device), num_nodes.to(gol.device), user_id.to(gol.device)

    return collate_fn



def collate_edge(batch):
    u, p, n, t = tuple(zip(*batch))
    return torch.LongTensor(u).to(gol.device), torch.LongTensor(p).to(gol.device), \
        torch.LongTensor(n).to(gol.device),torch.Tensor(t).to(gol.device)

def collate_eval(batch):
    u, label, exclude_mask = tuple(zip(*batch))
    return torch.LongTensor(u).to(gol.device), torch.cat(label, dim=0), \
        torch.cat(exclude_mask, dim=0)

def getDatasets(path='../data', dataset='ml-100k'):
    data_path = join(path, dataset, 'data.csv')
    df = pd.read_csv(data_path)[['uid', 'sid', 'timestamp']]
    df.timestamp = (df.timestamp - df.timestamp.min()) / (10 * 3600)
    df.timestamp = df.timestamp / df.timestamp.max()
    time_list, user_list, item_list = df['timestamp'].values, df['uid'].values, df['sid'].values
    if min(item_list) == 0:
        judge_item = 1
    else:
        judge_item = 0
    n_user, n_item = user_list.max() + 1, item_list.max() + 1 + judge_item
    trn_user, trn_item, trn_time, val_user, val_item, val_time, tst_user, tst_item, tst_time = \
        [], [], [], [], [], [], [], [], []
    trn_dict = {u: set() for u in range(n_user)}
    processed_path = join(path, dataset, 'diffrec_processed.pkl')
    if exists(processed_path):
        with open(processed_path, 'rb') as f:
            trn_seq, trn_t, trn_user, trn_item, trn_time = pkl.load(f)
            val_seq, val_t = pkl.load(f)
            tst_seq, tst_t = pkl.load(f)
            # trn_dict = pkl.load(f)
        gol.pLog(f'Loading from {processed_path}')
    else:
        gol.pLog(f'Generating data...')
        for uid, line in df.groupby('uid'):
            val_piv, test_piv = np.quantile(line['timestamp'], [0.8, 0.9])
            trn_df = line[line['timestamp'] < val_piv]
            val_df = line[(line['timestamp'] >= val_piv) & (line['timestamp'] < test_piv)]
            tst_df = line[line['timestamp'] >= test_piv]

            trn_user.append(trn_df.uid.values)
            trn_item.append(trn_df.sid.values + judge_item)
            trn_time.append(trn_df.timestamp.values)
            trn_dict[uid] |= set(trn_df.sid.values.tolist())

            val_user.append(val_df.uid.values)
            val_item.append(val_df.sid.values + judge_item)
            val_time.append(val_df.timestamp.values)

            tst_user.append(tst_df.uid.values)
            tst_item.append(tst_df.sid.values + judge_item)
            tst_time.append(tst_df.timestamp.values)

        trn_user, trn_item, trn_time = np.concatenate(trn_user), np.concatenate(trn_item), np.concatenate(trn_time)
        val_user, val_item, val_time = np.concatenate(val_user), np.concatenate(val_item), np.concatenate(val_time)
        tst_user, tst_item, tst_time = np.concatenate(tst_user), np.concatenate(tst_item), np.concatenate(tst_time)

        trn_t = [[] for i in range(n_user)]
        trn_seq = [[] for i in range(n_user)]
        # u_i_graph = zip(trn_user, trn_item)
        for src, dst, t in zip(trn_user, trn_item, trn_time):
            trn_seq[src.item()].append(dst.item())
            trn_t[src.item()].append(t.item())
        # trn_seq = list(filter(lambda x: x != [], trn_seq))

        val_t = [[] for i in range(n_user)]
        val_seq = [[] for i in range(n_user)]
        for src, dst, t in zip(val_user, val_item, val_time):
            val_seq[src.item()].append(dst.item())
            val_t[src.item()].append(t.item())
        # val_seq = list(filter(lambda x: x != [], val_seq))

        tst_t = [[] for i in range(n_user)]
        tst_seq = [[] for i in range(n_user)]
        for src, dst, t in zip(tst_user, tst_item, tst_time):
            tst_seq[src.item()].append(dst.item())
            tst_t[src.item()].append(t.item())
        # tst_seq = list(filter(lambda x: x != [], tst_seq))

        with open(processed_path, 'wb') as f:
            pkl.dump((trn_seq, trn_t, trn_user, trn_item, trn_time), f, pkl.HIGHEST_PROTOCOL)
            pkl.dump((val_seq, val_t), f, pkl.HIGHEST_PROTOCOL)
            pkl.dump((tst_seq, tst_t), f, pkl.HIGHEST_PROTOCOL)
            # pkl.dump(trn_dict, f, pkl.HIGHEST_PROTOCOL)

        gol.pLog(f'Data generated to {processed_path}')

    num_node = []
    for s1, s2, s3 in zip(trn_seq, val_seq, tst_seq):
        num_node.append(len(s1)+len(s2)+len(s3))
    max_n_node = np.max(num_node)


    tr_set = Data(n_user, n_item, max_n_node, trn_seq, trn_t, None, None, is_eval=False)
    va_set = Data(n_user, n_item, max_n_node, trn_seq, trn_t, val_seq, val_t, None, is_eval=True)
    te_set = Data(n_user, n_item, max_n_node, trn_seq, trn_t, val_seq, val_t, tst_seq, tst_t, is_eval=True)

    max_seq_len = max_n_node

    pLog(f'{dataset.upper()} loaded. #User: {n_user}, #Item: {n_item}')
    whole_graph = torch.from_numpy(np.stack((trn_user, trn_item), axis=0)).long().contiguous()
    whole_graph = torch.cat((
        whole_graph, whole_graph[[1, 0], :]
    ), dim=-1)
    graph_time = np.concatenate((trn_time, trn_time))

    return n_item, n_user, max_seq_len, tr_set, va_set, te_set, trn_user, trn_item, trn_time
