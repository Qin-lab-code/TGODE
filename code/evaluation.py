import numpy as np
import torch
import math
from torch.utils.data import DataLoader
import gol
from model import TGODE
from diffusion import TimeDiff
from seq_dataset import getDatasets, seq_to_temporal_session_graph, collate_fn_factory_temporal, collate_fn_factory_temporal2
from utils import *
import copy
from tqdm import tqdm
from numba import jit

Ks = [1, 5, 10, 20, 50]


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores

def eval_model(model: TGODE, Diffusion: TimeDiff, eval_set):
    top_K = [1, 5, 10, 20, 50]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []

    collate_fn = collate_fn_factory_temporal(seq_to_temporal_session_graph)
    collate_fn2 = collate_fn_factory_temporal2(seq_to_temporal_session_graph)
    eval_loader = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_fn)
    eval_loader2 = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_fn2)

    with torch.no_grad():
        Diffusion.eval()
        model.eval()
        out_graph_list = []
        for idx, batch in enumerate(tqdm(eval_loader2)):
            out_graph, labels, seqs, times, item_seq, tensor_times, session_len, target_time = batch
            seqs = copy.deepcopy(seqs)
            times = copy.deepcopy(times)
            target = labels
            seq_len = torch.squeeze(session_len)
            seqEmb = model.calculate_seq_output(*out_graph, target, seq_len, item_seq, target_time)
            time_slice, slice_num = select_slice(times, seq_len, target_time)
            out_graph = constract_local_graph(time_slice, Diffusion, seqs, times, item_seq, tensor_times, seq_len,
                                              slice_num, seqEmb)
            out_graph_list.append(out_graph)



        test_outputs, tot_cnt = [], 0
        for idx, batch in enumerate(tqdm(eval_loader)):
            out_graph, labels, item_seq, times, session_len, target_time = batch
            tar = target = labels
            seq_len = torch.squeeze(session_len)
            out_graph = out_graph_list[idx]
            item_score = model.full_sort_predict(out_graph, seq_len, item_seq, target_time)
            # item_score[exclude_mask] = -1e10

            item_score = item_score.cpu().detach().numpy()
            index = []
            loop_len = len(item_score)
            for idd in range(loop_len):
                index.append(find_k_largest(50, item_score[idd]))
            index = np.array(index)
            tar = tar.cpu().detach().numpy()
            for K in top_K:
                for prediction, target in zip(index[:, :K], tar):
                    prediction_list = prediction.tolist()
                    epsilon = 0.1 ** 10
                    DCG = 0
                    IDCG = 0
                    for j in range(K):
                        if prediction_list[j] == target:
                            DCG += 1 / math.log2(j + 2)
                    for j in range(min(1, K)):
                        IDCG += 1 / math.log2(j + 2)
                    metrics['ndcg%d' % K].append(DCG / max(IDCG, epsilon))
                    metrics['hit%d' % K].append(np.isin(target, prediction))
                    if len(np.where(prediction == target)[0]) == 0:
                        metrics['mrr%d' % K].append(0)
                    else:
                        metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))


    return metrics