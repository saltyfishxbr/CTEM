import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv
from cand_model import *
from utils import *

def _as_scalar(x):
    import torch
    if isinstance(x, (list, tuple)):
        return _as_scalar(x[0])
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.view(-1)[0].item()
    return x

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x, mask):
        if x is None or x.numel() == 0 or x.size(0) == 0:
            return x.new_zeros((0, self.rnn.hidden_size))

        lengths = mask.sum(dim=1).long() if mask is not None else torch.full(
            (x.size(0),), x.size(1), device=x.device, dtype=torch.long
        )
        lengths = torch.clamp(lengths, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        idx = (lengths - 1).to(out.device)  # [N]
        last = out[torch.arange(out.size(0), device=out.device), idx]  # [N,H]
        return last



class EdgeEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class BeliefAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.score_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, ego_h, cand_h, cand_mask=None):
        q = self.query(ego_h).unsqueeze(1)  # [B, 1, D]
        k = self.key(cand_h)                # [B, N, D]
        score = (q * k).sum(dim=-1)         # [B, N]
        if cand_mask is not None:
            score = score.masked_fill(~cand_mask, -1e9)
        attn = torch.softmax(score, dim=1)
        context = torch.sum(attn.unsqueeze(-1) * cand_h, dim=1)
        cte_score = self.score_head(cand_h).squeeze(-1)  # optional distillation target
        return context, attn, cte_score


class EgoTrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, horizon=30):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2)
        self.horizon = horizon

    def forward(self, context, ego_emb):
        z = torch.cat([context, ego_emb], dim=-1)
        z = z.unsqueeze(1).repeat(1, self.horizon, 1)
        out, _ = self.gru(z)
        return self.out(out)


class CTEGraphPredictor(nn.Module):
    def __init__(self,
                 node_feat_dim=4,
                 edge_feat_dim=6,
                 hidden_dim=64,
                 horizon=50,
                 rollout_model=None,
                 use_hetero=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.node_types = ['small', 'large', 'motor']
        # self.edge_types = [
        #     ('small', 'interacts', 'small'),
        #     ('small', 'interacts', 'large'),
        #     ('small', 'interacts', 'motor'),
        #     ('large', 'interacts', 'small'),
        #     ('large', 'interacts', 'motor'),
        #     ('motor', 'interacts', 'small'),
        # ]
        self.node_types = ['acc', 'small', 'large']
        self.edge_types = [
            (src, 'interacts', dst)
            for src in self.node_types
            for dst in self.node_types
        ]
        self.use_hetero = use_hetero

        self.encoders = nn.ModuleDict({
            node_type: TrajectoryEncoder(node_feat_dim, hidden_dim)
            for node_type in self.node_types
        })

        self.edge_embed = EdgeEmbedding(edge_feat_dim, hidden_dim)

        self.gnn = HeteroConv({
            etype: GATConv((-1, -1), hidden_dim, edge_dim=hidden_dim,add_self_loops=False)
            for etype in self.edge_types
        }, aggr='sum')

        self.attn = BeliefAttention(hidden_dim)
        self.decoder = EgoTrajectoryDecoder(hidden_dim * 2, hidden_dim, horizon)
        self.rollout_model = rollout_model or ConstantVelocityRollout(pred_horizon=horizon)

    def forward(self, data):
        device=next(self.parameters()).device
        h_dict = {}
        raw_ego_type = data['ego_type']
        raw_ego_index = data['ego_index']
        ego_type = _as_scalar(raw_ego_type)  # e.g. 'small' / 'large' / 'motor'
        ego_index = int(_as_scalar(raw_ego_index))
        # for node_type in self.node_types:
        #     x = data[node_type].x
        #     mask = data[node_type].mask
        #     h_dict[node_type] = self.encoders[node_type](x, mask)
        #     data[node_type].h = h_dict[node_type]

        for node_type in self.node_types:
            if node_type in data.node_types:
                x = data[node_type].x
                mask = data[node_type].mask
                if x.size(0) == 0:
                    h_dict[node_type] = x.new_zeros((0, self.hidden_dim),device=device)
                else:
                    h_dict[node_type] = self.encoders[node_type](x, mask)
                data[node_type].h = h_dict[node_type]
            else:
                h_dict[node_type] = torch.zeros((0, self.hidden_dim),device=device)
                data[node_type].h = h_dict[node_type]

        for etype in self.edge_types:
            if etype in data.edge_types:
                data[etype].edge_attr = self.edge_embed(data[etype].edge_attr)

        edge_index_dict = {
            etype: data[etype].edge_index for etype in self.edge_types if etype in data.edge_types
        }
        edge_attr_dict = {
            etype: data[etype].edge_attr for etype in self.edge_types if etype in data.edge_types
        }

        h_out = self.gnn(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        for nt in self.node_types:
            if nt not in h_out:
                h_out[nt] = h_dict[nt]

        # ego_type = data['ego_type']
        # ego_index = int(data['ego_index'])

        ego_h = h_out[ego_type][ego_index:ego_index+1]
        cand_h_list = []
        for node_type in self.node_types:
            h = h_out.get(node_type, None)
            if h is None or h.size(0) == 0:
                continue
            if node_type == ego_type:
                mask = torch.ones(len(h), dtype=torch.bool, device=h.device)
                mask[ego_index] = False
                h = h[mask]
            if h.size(0) > 0:
                cand_h_list.append(h)

        if len(cand_h_list) == 0:
            cand_h = torch.zeros(1, 0, ego_h.size(-1), device=ego_h.device)
        else:
            cand_h = torch.cat(cand_h_list, dim=0).unsqueeze(0)

        context, attn, cte_pred = self.attn(ego_h, cand_h)
        ego_pred = self.decoder(context, ego_h)

        Nc = cand_h.size(1)
        if Nc > 0:
            cand_last_pos, cand_last_vel = [], []
            for nt in self.node_types:
                if nt not in data.node_types:
                    continue
                X = data[nt].x  # [Ni, T, 4]
                M = data[nt].mask  # [Ni, T]
                if X.size(0) == 0:
                    continue
                for local_i in range(X.size(0)):
                    if nt == ego_type and local_i == ego_index:
                        continue
                    valid = (M[local_i] > 0.5).nonzero(as_tuple=False).flatten()
                    t_last = int(valid[-1]) if valid.numel() > 0 else X.size(1) - 1
                    cand_last_pos.append(X[local_i, t_last, :2])
                    cand_last_vel.append(X[local_i, t_last, 2:4])
            if len(cand_last_pos) > 0:
                cand_last_pos = torch.stack(cand_last_pos, dim=0)
                cand_last_vel = torch.stack(cand_last_vel, dim=0)
                cand_rollout_result = self.rollout_model(cand_last_pos, cand_last_vel)
            else:
                cand_rollout_result = torch.zeros(0, self.decoder.horizon, 2, device=ego_h.device)
        # if Nc > 0:
        #     cand_hist_trajs = []
        #     for nt in self.node_types:
        #         if nt not in data.node_types:
        #             continue
        #         X = data[nt].x  # [Ni, T, 4]
        #         if X.size(0) == 0:
        #             continue
        #         for local_i in range(X.size(0)):
        #             if nt == ego_type and local_i == ego_index:
        #                 continue
        #             cand_hist_trajs.append(X[local_i, :, :2])  # 只取位置，不要速度
        #     if len(cand_hist_trajs) > 0:
        #         cand_rollout_result = torch.stack(cand_hist_trajs, dim=0)  # [N_cand, T_hist, 2]
        #     else:
        #         cand_rollout_result = torch.zeros(0, X.size(1), 2, device=ego_h.device)
        # else:
        #     cand_rollout_result = torch.zeros(0, self.decoder.horizon, 2, device=ego_h.device)

        cte_true=compute_cte(ego_pred.detach(),cand_rollout_result)
        cte_pred = cte_pred.squeeze(0)
        return ego_pred.squeeze(0), attn, cte_pred,cte_true