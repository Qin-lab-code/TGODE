import gol
import torch, torch.nn as nn
import numpy as np
import torch.nn.functional as F
from seq_dataset import seq_to_temporal_session_graph
import dgl
import time

def select_slice(times, seq_len, target_time):
    all_time_slices = torch.arange(0, 1.1, 0.1)

    slice_list = []
    max_length = 0
    slices_num = []

    for i, t in enumerate(times):
        covered_slices = set()
        length = seq_len[i]
        time_l = t[:length]

        for value in time_l:
            slice_index = int(value * 10)
            covered_slices.add(slice_index)


        uncovered_slices = sorted(set(range(11)) - covered_slices)
        if len(uncovered_slices) == 0:
            uncovered_slices = [10]
            covered_slices_num = 1
        else:
            covered_slices_num = max(1, int(length / len(uncovered_slices)))
        max_length = max(max_length, len(uncovered_slices))
        slice_list.append(uncovered_slices)
        slices_num.append(covered_slices_num)
    max_length += 1

    padding_list = []
    for idx, sublist in enumerate(slice_list):
        sublist = [sub / 10.0 for sub in sublist]
        padding_needed = max_length - len(sublist)

        sublist.extend([target_time[idx]] * padding_needed)
        padding_list.append(sublist)

    # slice_tensor = torch.FloatTensor(padding_list).to(gol.device)

    return padding_list, slices_num


def dgl_graph_construction(seq, times):
    graphs = list(map(seq_to_temporal_session_graph, seq, times))
    bg = dgl.batch(graphs)
    bg = bg.to(gol.device)
    return bg


def constract_local_graph(time_slice, Diffusion, seqs, times, item_seq, tensor_times, seq_len, slice_num, seqEmb):
    # K = 1
    item_list = seqs
    time_list = times

    for idx in range(len(time_slice[0])):
        add_time = [t[idx] for t in time_slice]
        add_time_tensor = torch.FloatTensor(add_time).to(gol.device)
        x = Diffusion.full_sort_predict(item_seq, tensor_times, seq_len, add_time_tensor, seqEmb)

        K = max(slice_num)
        topk_indices = x.topk(k=K, dim=1).indices

        for i in range(topk_indices.size(0)):
            line_k = slice_num[i]
            for j in range(line_k):
                col_index = topk_indices[i][j].item()
                item_list[i].append(col_index)
                time_list[i].append(add_time[i])
    out_graph = dgl_graph_construction(item_list, time_list)

    return out_graph
















