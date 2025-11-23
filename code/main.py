import torch
import gol, gol1
import numpy as np
from seq_dataset import getDatasets, seq_to_temporal_session_graph, collate_fn_factory_temporal, collate_fn_factory_temporal2
from torch.utils.data import DataLoader
from model import TGODE
from diffusion import TimeDiff
from evaluation import eval_model
from pprint import pformat
from tqdm import tqdm
import copy
from utils import *
import time


def train_eval(model: TGODE, Diffusion: TimeDiff, tr_set, va_set, te_set):
    collate_fn = collate_fn_factory_temporal(seq_to_temporal_session_graph)
    collate_fn2 = collate_fn_factory_temporal2(seq_to_temporal_session_graph)
    tr_loader = DataLoader(tr_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_fn)
    tr_loader2 = DataLoader(tr_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_fn2)
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])#, amsgrad=True)
    opt2 = torch.optim.AdamW(Diffusion.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])  # , amsgrad=True)
    batch_num = len(tr_set) // gol.BATCH_SZ
    ave_loss, best_val_epoch, best_val_hit, best_val_mrr, best_val_ndcg = 0., 0, 0., 0., 0.
    top_K = [1, 5, 10, 20, 50]
    te_result = None
    best_result = None

    for epoch in range(gol.EPOCH):

        # Diffusion process

        Diffusion.eval()
        model.eval()
        out_graph_list = []
        for idx, batch in enumerate(tqdm(tr_loader2)):
            out_graph, labels, seqs, times, item_seq, tensor_times, session_len, target_time = batch
            seqs = copy.deepcopy(seqs)
            times = copy.deepcopy(times)
            target = labels
            seq_len = torch.squeeze(session_len)
            seqEmb = model.calculate_seq_output(*out_graph, target, seq_len, item_seq, target_time)
            time_slice, slice_num = select_slice(times, seq_len, target_time)
            out_graph = constract_local_graph(time_slice, Diffusion, seqs, times, item_seq, tensor_times, seq_len, slice_num, seqEmb)
            out_graph_list.append(out_graph)

        model.train()

        for idx, batch in enumerate(tqdm(tr_loader)):
            out_graph, labels, item_seq, times, session_len, target_time = batch
            target = labels
            seq_len = torch.squeeze(session_len)
            out_graph = out_graph_list[idx]
            bpr_loss = model.calculate_loss(out_graph, target, seq_len, item_seq, target_time)
            tot_loss = bpr_loss

            opt.zero_grad()
            tot_loss.backward()
            opt.step()
            ave_loss += tot_loss.item()
        ave_loss /= batch_num

        Diffusion.train()

        for idx, batch in enumerate(tqdm(tr_loader)):
            out_graph, labels, item_seq, times, session_len, target_time = batch
            target = labels
            seq_len = torch.squeeze(session_len)
            out_graph = out_graph_list[idx]
            seqEmb = model.calculate_seq_output(out_graph, target, seq_len, item_seq, target_time)
            bpr_loss = Diffusion.calculate_loss(item_seq, times, seq_len, target, seqEmb)
            tot_loss = bpr_loss

            opt2.zero_grad()
            tot_loss.backward()
            opt2.step()

        metrics = eval_model(model, Diffusion, va_set)

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K])
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K])
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K])

        gol.pLog(f'Epoch {epoch} / {gol.EPOCH}, Loss: {ave_loss:.5f}')

        for K in top_K:
            print('Recall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f' %
                  (K, metrics['hit%d' % K], K, metrics['mrr%d' % K],
                   K, metrics['ndcg%d' % K]))


        if epoch - best_val_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without valid improvement.')
            break

        if metrics["mrr10"] > best_val_mrr or epoch == 0:
            best_val_epoch, best_val_hit, best_val_mrr, best_val_ndcg= epoch, metrics["hit10"], metrics["mrr10"], metrics["ndcg10"]
            te_result = eval_model(model, Diffusion, te_set)
            if epoch == 0:
                best_result = te_result
            for K in top_K:
                te_result['hit%d' % K] = np.mean(te_result['hit%d' % K])
                te_result['mrr%d' % K] = np.mean(te_result['mrr%d' % K])
                te_result['ndcg%d' % K] = np.mean(te_result['ndcg%d' % K])
            for K in top_K:
                best_result['hit%d' % K] = max(best_result['hit%d' % K], te_result['hit%d' % K])
                best_result['mrr%d' % K] = max(best_result['mrr%d' % K], te_result['mrr%d' % K])
                best_result['ndcg%d' % K] = max(best_result['ndcg%d' % K], te_result['ndcg%d' % K])

                print('Test Recall@%d: %.4f\tTest MRR%d: %.4f\tTest NDCG%d: %.4f' %
                      (K, te_result['hit%d' % K], K, te_result['mrr%d' % K],
                       K, te_result['ndcg%d' % K]))
            if gol.SAVE:
                torch.save(model.cpu(), gol.FILE_PATH + 'model.pt')
                model = model.to(gol.device)
                torch.save(Diffusion.cpu(), gol.FILE_PATH + 'Diffusion.pt')
                Diffusion = Diffusion.to(gol.device)

        gol.pLog(f'Best valid MRR at epoch {best_val_epoch}')
    for K in top_K:
        print('Test Recall@%d: %.4f\tTest MRR%d: %.4f\tTest NDCG%d: %.4f' %
              (K, best_result['hit%d' % K], K, best_result['mrr%d' % K],
               K, best_result['ndcg%d' % K]))
    return best_result, best_val_epoch




if __name__ == '__main__':
    n_item, max_seq_len, train_set, val_set, test_set, time_piv, whole_graph, whole_time = getDatasets(gol.DATA_PATH, gol.dataset)
    gol.n_items = n_item
    gol1.n_items = n_item
    # gol.n_users = n_user
    gol.max_seq_len = max_seq_len
    if gol.LOAD:
        recModel = torch.load(gol.FILE_PATH + 'model.pt').to(gol.device)
        Diffusion = torch.load(gol.FILE_PATH + 'diffusion.pt').to(gol.device)
        gol.pLog('Load done')
    else:
        recModel = TGODE(whole_graph, whole_time, train_set, time_piv, max_seq_len).to(gol.device)
        Diffusion = TimeDiff(gol.device, gol.n_items, gol1.n_users, gol1.mean_type,
                           gol1.time_aware, gol1.w_max, gol1.w_min, gol1.noise_schedule, gol1.noise_scale, gol1.noise_max,
                           gol1.noise_min, gol1.steps, gol1.beta_fixed, gol1.embedding_size, gol1.emb_size, gol1.hid_temp, gol1.norm, gol1.reweight,
                           gol1.sampling_steps, gol1.sampling_noise, gol1.mlp_act_func, gol1.history_num_per_term,
                           gol1.dims_dnn).to(gol.device)

    gol.pLog('Start Training\n')
    test_result, best_epoch = train_eval(recModel, Diffusion, train_set, val_set, test_set)
    gol.pLog(f'Training on {gol.dataset.upper()} finished, best valid Recall@20 at epoch {best_epoch}')

