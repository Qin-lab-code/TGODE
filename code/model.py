import gol
import torch, torch.nn as nn
import numpy as np
from dataset import TimeGraph
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from Transformer import TransformerEncoder
from BPRLoss import BPRLoss
from torchdiffeq import odeint
import dgl
from dgl.nn.pytorch import GraphConv, GATConv

class TGODE(nn.Module):
    def __init__(self, whole_graph, whole_time, dataset: TimeGraph, time_piv, max_seq_len):
        super(TGODE, self).__init__()
        self.config = gol.conf
        self.dataset = dataset
        self.time_piv = time_piv
        self.time_list = torch.FloatTensor(whole_time).to(gol.device)
        self.ode_time_max = max(self.time_list)
        self.ode_solver = 'rk4' # 'dopri5'
        self.num_splits = 3
        self.step_size = self.config['step']
        self.norm = gol.norm

        self.sp_graph = whole_graph.to(gol.device)

        self.__init_weight()

        self.n_layers = self.config['n_layers']
        self.n_heads = self.config['n_heads']
        self.hidden_size = self.hid_dim  # same as embedding_size
        self.inner_size = self.config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = self.config['hidden_dropout_prob']
        self.attn_dropout_prob = self.config['attn_dropout_prob']
        self.hidden_act = self.config['hidden_act']
        self.layer_norm_eps = self.config['layer_norm_eps']
        self.initializer_range = self.config['initializer_range']
        self.loss_type = self.config['loss_type']
        self.max_seq_length = max_seq_len
        self.position_embedding = nn.Embedding(self.max_seq_length * 2, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

            # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init_weight(self):
        self.n_users = self.dataset.n_user
        self.n_items = self.dataset.n_item
        self.n_nodes = self.n_items
        self.hid_dim = self.config['hidden']
        self.hid_temp = self.config['hidden_temp']
        self.keep_prob = self.config['keepprob']
        self.embed = nn.Embedding(self.n_items, self.hid_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hid_dim)
        feat_drop = 0.0
        self.ODEFunc = GraphGRUODE(self.hid_dim, self.hid_dim, gnn='MLP', device=gol.device)
        self.readout = AttnReadout(
            self.hid_dim,
            self.hid_dim,
            self.hid_dim,
            batch_norm=None,
            feat_drop=feat_drop,
            activation=None,
        )
        self.fc_sr = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.layers = nn.ModuleList()
        self.graph_aggr = TempGAT(self.hid_dim, self.hid_temp)

        self.layers = TempGAT(self.hid_dim, self.hid_temp)

        self.temp_enc = TimeEncode(self.hid_temp)

        self.operator = ODEBlock(self.n_nodes, self.hid_dim, self.sp_graph)
        self.emb_drop = nn.Dropout(p=1 - gol.conf['keepprob'])

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, local_graph, item_seq, target_time):
        init_emb = self.emb_drop(self.embed) if gol.conf['dropout'] else self.embed.weight
        gnn = self.graph_aggr
        edge_enc = self.temp_enc(self.time_list)
        init_emb = gnn(init_emb, self.sp_graph, edge_enc)

        local_emb = self.personal_forward(local_graph, init_emb[item_seq], target_time)

        int_time = torch.Tensor([0, torch.max(target_time)]).to(gol.device)
        ode_output = odeint(self.operator(init_emb), init_emb, int_time, method=self.ode_solver,
                            options=dict(step_size=self.step_size))
        init_emb = ode_output[-1]

        return init_emb, local_emb

    def personal_forward(self, local_graph, feat, target_time):

        mg = local_graph
        if self.norm:
            feat = nn.functional.normalize(feat)
        self.ODEFunc.set_graph(mg)
        self.ODEFunc.set_x(feat)
        t_end = torch.max(target_time)
        t_start = mg.edata['t'].min()
        # step_t = th.unique(mg.edata['t'].sort()[0]
        t = torch.tensor([t_start, t_end], device=mg.device)
        if self.ode_solver != "dopri5":
            feat = odeint(self.ODEFunc, feat, t=t, method=self.ode_solver,
                          options={"perturb": "True", "step_size": t_end / self.num_splits})[-1]  # .mean(0)
        else:
            # feat = odeint(self.ODEFunc, feat, t=t, rtol=1e-1, atol=1e-2, options={"first_step": 0.2})[-1]
            feat = odeint(self.ODEFunc, feat, t=t, rtol=1e-4, atol=1e-5)[-1]  # , options={"first_step": 0.0})[-1]
        return feat

    def combanation(self, item_seq, item_seq_len, item_emb, local_item_emb):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = item_emb[item_seq]

        input_emb = item_emb + position_embedding
        local_item_emb = local_item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        local_item_emb = self.LayerNorm(local_item_emb)
        input_emb = self.dropout(input_emb)
        local_item_emb = self.dropout(local_item_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        local_trm_output = self.trm_encoder(local_item_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        local_output = local_trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        local_output = self.gather_indexes(local_output, item_seq_len - 1)

        return output, local_output  # [B H]

    def calculate_seq_output(self, local_graph, pos_items, item_seq_len, item_seq, target_time):
        item_emb, local_item_emb = self.forward(local_graph, item_seq, target_time)
        item_emb = item_emb.div(torch.norm(item_emb, p=2, dim=-1, keepdim=True))
        local_item_emb = local_item_emb.div(torch.norm(local_item_emb, p=2, dim=-1, keepdim=True))

        output, local_output = self.combanation(item_seq, item_seq_len, item_emb, local_item_emb)
        seq_output = local_output + output

        seq_output = seq_output.div(torch.norm(seq_output, p=2, dim=-1, keepdim=True) + 1e-12)

        return seq_output


    def calculate_loss(self, local_graph, pos_items, item_seq_len, item_seq, target_time):
        item_emb, local_item_emb = self.forward(local_graph, item_seq, target_time)
        item_emb = item_emb.div(torch.norm(item_emb, p=2, dim=-1, keepdim=True))
        local_item_emb = local_item_emb.div(torch.norm(local_item_emb, p=2, dim=-1, keepdim=True))

        output, local_output = self.combanation(item_seq, item_seq_len, item_emb, local_item_emb)
        seq_output = local_output + output

        seq_output = seq_output.div(torch.norm(seq_output, p=2, dim=-1, keepdim=True) + 1e-12)
        if self.loss_type == 'BPR':
            neg_items = pos_items
            pos_items_emb = self.embed(pos_items)
            neg_items_emb = self.embed(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_items_emb = self.embed.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
            # loss = self.loss_fct(logits, pos_items)

            scores = torch.log_softmax(gol.beta * scores, dim=-1)
            loss = torch.nn.functional.nll_loss(scores, pos_items)

            return loss

    def full_sort_predict(self, local_graph, item_seq_len, item_seq, target_time):
        item_emb, local_item_emb = self.forward(local_graph, item_seq, target_time)

        item_emb = item_emb.div(torch.norm(item_emb, p=2, dim=-1, keepdim=True))
        local_item_emb = local_item_emb.div(torch.norm(local_item_emb, p=2, dim=-1, keepdim=True))
        output, local_output = self.combanation(item_seq, item_seq_len, item_emb, local_item_emb)
        seq_output = local_output + output
        seq_output = seq_output.div(torch.norm(seq_output, p=2, dim=-1, keepdim=True) + 1e-12)
        test_items_emb = self.embed.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        scores = torch.log_softmax(gol.beta * scores, dim=-1)
        return scores


class ODEBlock(nn.Module):
    def __init__(self, n_nodes, hid_dim, sp_graph):
        super(ODEBlock, self).__init__()
        self.n_nodes = n_nodes
        self.hid_dim = hid_dim
        self.gnn_class = CGNN
        self.sp_graph = sp_graph

    def forward(self, ori_emb):
        sp_graph = self.sp_graph
        sp_graph, edge_weight = gcn_norm(sp_graph, add_self_loops=True)
        return self.gnn_class(ori_emb, sp_graph, edge_weight)


class CGNN(MessagePassing):
    def __init__(self, ori_emb, sp_graph, edge_weight):
        super(CGNN, self).__init__()
        self.aggr = 'add'
        self.ori_emb = ori_emb
        self.sp_graph = sp_graph
        self.edge_weight = edge_weight
        self.sp_loop, self.loop_weight = add_self_loops(self.sp_graph, self.edge_weight, fill_value=-1.)

    def forward(self, t, x):
        message_emb = self.propagate(self.sp_graph, x=x, edge_weight=self.edge_weight)
        return message_emb + self.ori_emb

    def message(self, x_j, x_i, edge_weight):
        key = 1
        if key:
            return edge_weight.view(-1, 1) * x_j * x_i
        else:
            return edge_weight.view(-1, 1) * x_j


class TimeEncode(nn.Module):
    def __init__(self, time_dim, factor=5):
        super().__init__()
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())


    def forward(self, ts):
        map_ts = ts.unsqueeze(-1) * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic.squeeze(0)


class TempGAT(MessagePassing):
    def __init__(self, hiddim, hid_temp, attn_dropout=0.2):
        super(TempGAT, self).__init__(aggr='add')
        self.hid_sqrt = (2 * hiddim + hid_temp) ** -0.5
        self.K = nn.Linear(hiddim, hiddim)
        self.Q = nn.Linear(hiddim, hiddim)
        # self.V = nn.Linear(hiddim, hiddim)
        self.alpha = nn.Linear(2 * hiddim + hid_temp, 1, bias=False)

        self.FC = nn.Linear(2 * hiddim, hiddim)
        self.dropout = nn.Dropout(attn_dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, time_enc):
        edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=False, dtype=x.dtype)
        conv_out = self.propagate(edge_index, x=x, edge_weight=edge_weight, time_enc=time_enc)
        # fc_in = torch.cat((x, conv_out), dim=-1)
        # fc_in = self.dropout(fc_in)
        return (conv_out + x) / 2

    def message(self, x_j, x_i, edge_weight, time_enc):
        key, query = self.K(x_j), self.Q(x_i)
        logits = torch.sigmoid(self.alpha(torch.cat((key, query, time_enc), dim=-1)))

        v = x_j * edge_weight.unsqueeze(-1)
        # v = self.V(x_j) * edge_weight.unsqueeze(-1)
        return v * logits

class GGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, feat_drop=0.0, activation=None):
        super().__init__()
        self.dropout = nn.Dropout(feat_drop)
        self.gru = nn.GRUCell(2 * output_dim, input_dim)
        self.W1 = nn.Linear(input_dim, output_dim, bias=False)
        self.W2 = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def messager(self, edges):

        return {'m': edges.src['ft'] * edges.data['w'].unsqueeze(-1), 'w': edges.data['w']}

    def reducer(self, nodes):

        m = nodes.mailbox['m']
        w = nodes.mailbox['w']
        hn = m.sum(dim=1) / w.sum(dim=1).unsqueeze(-1)

        return {'neigh': hn}

    def forward(self, mg, feat):
        with mg.local_scope():
            mg.ndata['ft'] = self.dropout(feat)
            if mg.number_of_edges() > 0:
                mg.update_all(self.messager, self.reducer)
                neigh1 = mg.ndata['neigh']
                mg1 = mg.reverse(copy_edata=True)
                mg1.update_all(self.messager, self.reducer)
                neigh2 = mg1.ndata['neigh']
                neigh1 = self.W1(neigh1)
                neigh2 = self.W2(neigh2)
                hn = torch.cat((neigh1, neigh2), dim=1)
                rst = self.gru(hn, feat)
            else:
                rst = feat
        if self.activation is not None:
            rst = self.activation(rst)
        return rst

class GraphGRUODE(nn.Module):

    def __init__(self, in_dim, hid_dim, device, gnn='GATConv', bias=True, **kwargs):

        super(GraphGRUODE, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.device = device
        self.gnn = gnn
        self.bias = bias
        self.dropout = nn.Dropout(0.1)

        if self.gnn == 'GCNConv':
            self.lin_xz = GraphConv(self.in_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_xr = GraphConv(self.in_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_xh = GraphConv(self.in_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hz = GraphConv(self.hid_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hr = GraphConv(self.hid_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
            self.lin_hh = GraphConv(self.hid_dim, self.hid_dim, bias=self.bias, allow_zero_in_degree=True)
        elif self.gnn == 'GATConv':
            self.lin_xz = GATConv(self.in_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_xr = GATConv(self.in_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_xh = GATConv(self.in_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hz = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hr = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
            self.lin_hh = GATConv(self.hid_dim, self.hid_dim, bias=self.bias, num_heads=1, allow_zero_in_degree=True)
        elif self.gnn == 'MLP':
            self.enc = nn.Linear(self.hid_dim, self.hid_dim)
        else:
            raise NotImplementedError

        self.edge_index = None
        self.x = None



    def set_graph(self, graph: dgl.DGLGraph):

        self.graph = graph

    def set_x(self, x):
        self.x = x.to(self.device)

    def forward(self, t, h):

        node_idx = self.graph.filter_nodes(lambda nodes: nodes.data['t'] >= t)  # filter out sessions already computed
        edge_idx = self.graph.filter_edges(lambda edges: edges.data['t'] <= t)
        edge_index = (self.graph.edges()[0][edge_idx], self.graph.edges()[1][edge_idx])
        graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=self.graph.number_of_nodes(), device=self.device)
        graph = dgl.node_subgraph(graph, node_idx)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_reverse_edges(graph)

        ht = h
        x = self.x


        if self.gnn == 'GATConv':
            xr, xz, xh = self.lin_xr(graph, x).max(1)[0], self.lin_xz(graph, x).max(1)[0], self.lin_xh(graph, x).max(1)[
                0]
            r = torch.sigmoid(xr + self.lin_hr(graph, ht).max(1)[0])
            z = torch.sigmoid(xz + self.lin_hz(graph, ht).max(1)[0])
            u = torch.tanh(xh + self.lin_hh(graph, r * ht).max(1)[0])
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'GCNConv':
            xr, xz, xh = self.lin_xr(graph, x), self.lin_xz(graph, x), self.lin_xh(graph, x)
            r = torch.sigmoid(xr + self.lin_hr(graph, ht))
            z = torch.sigmoid(xz + self.lin_hz(graph, ht))
            u = torch.tanh(xh + self.lin_hh(graph, r * ht))
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'Linear':
            xr, xz, xh = self.lin_xr(x), self.lin_xz(x), self.lin_xh(x)
            r = torch.sigmoid(xr + self.lin_hr(ht))
            z = torch.sigmoid(xz + self.lin_hz(ht))
            u = torch.tanh(xh + self.lin_hh(r * ht))
            dh = (1 - z) * (u - ht)
        elif self.gnn == 'MLP':
            dh = self.enc(ht)
        Dh = dh
        return Dh


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim
            else None
        )
        self.activation = activation

    def forward(self, g, feat, last_nodes):
        feat = feat[g.ndata['iid']]
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat)
        feat_u = self.fc_u(feat)
        feat_v = self.fc_v(feat[last_nodes])
        feat_v = dgl.broadcast_nodes(g, feat_v)
        e = self.fc_e(torch.sigmoid(feat_u + feat_v))
        alpha = dgl.ops.segment.segment_softmax(g.batch_num_nodes(), e)
        feat_norm = feat * alpha
        rst = dgl.ops.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum')
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst