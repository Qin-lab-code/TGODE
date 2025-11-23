import torch
import numpy as np, pandas as pd
from torch.utils.data import Dataset
import gol1 as gol
from gol1 import pLog
from os.path import join, exists
import pickle as pkl


class TimeGraph(Dataset):
    def __init__(self, n_user, n_item, userList, itemList, timeList, is_eval=False, tr_dict=None):
        self.n_user, self.n_item = n_user, n_item
        self.uniqueUsers = list(set(userList))
        self.is_eval = is_eval
        self.tr_dict = tr_dict
        self.posDict = self.getPosDict(zip(userList, itemList))

        # time_idx = np.argsort(timeList)
        # self.userList, self.itemList, self.timeList = userList[time_idx], itemList[time_idx], timeList[time_idx]
        self.userList, self.itemList, self.timeList = userList, itemList, timeList
        self.userSet = list(set(self.userList))
        self.len = len(self.userList) if not self.is_eval else len(self.userSet)

        self.sp_graph = []
        for u, i, t in zip(self.userList, self.itemList, self.timeList):
            self.sp_graph.append([u, i + self.n_user])
            # self.sp_graph.append([i + self.n_user, u])
        self.sp_graph = torch.LongTensor(self.sp_graph).T

    def getPosDict(self, u_i_graph):
        pos_dict = {x: set() for x in range(self.n_user + self.n_item)}
        for edge in u_i_graph:
            pos_dict[edge[0].item()].add(edge[1].item())
        return pos_dict

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.is_eval:
            user = self.userList[index]
            pos_item = self.itemList[index]
            clk_time = self.timeList[index]

            pos_set = self.posDict[user]
            neg_item = np.random.randint(0, self.n_item)
            while neg_item in pos_set:
                neg_item = np.random.randint(0, self.n_item)
            return user, pos_item, neg_item, clk_time
        else:
            user = self.userSet[index]
            pos_set = torch.LongTensor(list(self.posDict[user]))
            exclude_set = torch.LongTensor(list(self.tr_dict[user]))

            labels = torch.zeros((self.n_item, )).long()
            labels[pos_set] = 1

            exclude_mask = torch.zeros((self.n_item,)).long()
            exclude_mask[exclude_set] = 1

            return user, labels.unsqueeze(0), exclude_mask.unsqueeze(0)


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
    time_list, user_list, item_list = df['timestamp'].values, df['uid'].values, df['sid'].values
    n_user, n_item = user_list.max() + 1, item_list.max() + 1
    trn_user, trn_item, trn_time, val_user, val_item, val_time, tst_user, tst_item, tst_time = \
        [], [], [], [], [], [], [], [], []
    trn_dict = {u: set() for u in range(n_user)}
    processed_path = join(path, dataset, 'processed.pkl')
    if exists(processed_path):
        with open(processed_path, 'rb') as f:
            trn_user, trn_item, trn_time = pkl.load(f)
            val_user, val_item, val_time = pkl.load(f)
            tst_user, tst_item, tst_time = pkl.load(f)
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
            trn_item.append(trn_df.sid.values)
            trn_time.append(trn_df.timestamp.values)
            trn_dict[uid] |= set(trn_df.sid.values.tolist())

            val_user.append(val_df.uid.values)
            val_item.append(val_df.sid.values)
            val_time.append(val_df.timestamp.values)

            tst_user.append(tst_df.uid.values)
            tst_item.append(tst_df.sid.values)
            tst_time.append(tst_df.timestamp.values)

        trn_user, trn_item, trn_time = np.concatenate(trn_user), np.concatenate(trn_item), np.concatenate(trn_time)
        val_user, val_item, val_time = np.concatenate(val_user), np.concatenate(val_item), np.concatenate(val_time)
        tst_user, tst_item, tst_time = np.concatenate(tst_user), np.concatenate(tst_item), np.concatenate(tst_time)

        with open(processed_path, 'wb') as f:
            pkl.dump((trn_user, trn_item, trn_time), f, pkl.HIGHEST_PROTOCOL)
            pkl.dump((val_user, val_item, val_time), f, pkl.HIGHEST_PROTOCOL)
            pkl.dump((tst_user, tst_item, tst_time), f, pkl.HIGHEST_PROTOCOL)
            # pkl.dump(trn_dict, f, pkl.HIGHEST_PROTOCOL)
        gol.pLog(f'Data generated to {processed_path}')

    time_piv = np.quantile(trn_time, np.linspace(0, 1, gol.SLICE_NUM + 1))
    # time_piv = np.linspace(0, time_list.max() + 1e-10, num=gol.SLICE_NUM + 1)

    tr_set = TimeGraph(n_user, n_item, trn_user, trn_item, trn_time)
    va_set = TimeGraph(n_user, n_item, val_user, val_item, val_time, is_eval=True, tr_dict=trn_dict)
    te_set = TimeGraph(n_user, n_item, tst_user, tst_item, tst_time, is_eval=True, tr_dict=trn_dict)

    pLog(f'{dataset.upper()} loaded. #User: {n_user}, #Item: {n_item}')
    pLog(f'#Train: {trn_user.shape[0]}, #Valid: {val_user.shape[0]}, #Test: {tst_user.shape[0]}')
    whole_graph = torch.from_numpy(np.stack((trn_user, trn_item + n_user), axis=0)).long().contiguous()
    whole_graph = torch.cat((
        whole_graph, whole_graph[[1, 0], :]
    ), dim=-1)
    graph_time = np.concatenate((trn_time, trn_time))
    return n_user, n_item, tr_set, va_set, te_set, time_piv, whole_graph, graph_time