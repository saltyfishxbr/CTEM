import os, json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from train_eval import NGSIMDataset
from network import CTEGraphPredictor
from cand_model import ConstantVelocityRollout
from utils import compute_cte

ORDER_TYPES = ['acc','small', 'large']

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _as_scalar(x: Any):
    if isinstance(x, (list, tuple)):
        return _as_scalar(x[0])
    if torch.is_tensor(x):
        return x.view(-1)[0].item()
    return x


def extract_last_pos_vel(X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    N, T, _ = X.shape
    device = X.device
    pos = torch.zeros((N, 2), device=device)
    vel = torch.zeros((N, 2), device=device)
    for i in range(N):
        valid = (M[i] > 0.5).nonzero(as_tuple=False).flatten()
        t_last = int(valid[-1].item()) if valid.numel() > 0 else T - 1
        pos[i] = X[i, t_last, :2]
        vel[i] = X[i, t_last, 2:4]
    return pos, vel


def load_schema(npz_path: str):
    schema_path = Path(npz_path).with_suffix("").with_name("edge_attr_schema.json")
    if schema_path.exists():
        try:
            with open(schema_path, 'r') as f:
                js = json.load(f)
            cols = js.get('columns', [])
            if isinstance(cols, list) and len(cols) >= 6:
                return cols
        except Exception:
            pass
    return ['te_fwd', 'te_rev', 'dist', 'rel_v', 'ttc', 'same_lane']

def try_get_candidate_meta_from_model_output(model_outputs):
    if isinstance(model_outputs, (list, tuple)) and len(model_outputs) >= 5:
        meta = model_outputs[4]
        pred_cte = model_outputs[2]
        if isinstance(meta, (list, tuple)) and pred_cte is not None and len(meta) == pred_cte.numel():
            return meta
    return None


def reconstruct_candidate_meta(data, pred_cte_len: int):
    ego_type = _as_scalar(data['ego_type'])
    ego_idx = int(_as_scalar(data['ego_index']))
    meta = []
    for nt in ORDER_TYPES:
        if nt not in data.node_types:
            continue
        X, M = data[nt].x, data[nt].mask
        if X.size(0) == 0:
            continue
        for nid in range(X.size(0)):
            if nt == ego_type and nid == ego_idx:
                continue
            pos, _ = extract_last_pos_vel(X[nid:nid+1], M[nid:nid+1])
            meta.append({'type': nt, 'node_id': nid, 'edge_id': None,
                         'ego_type': ego_type,
                         'pos_tlast': (float(pos[0,0].item()), float(pos[0,1].item()))})
    if len(meta) >= pred_cte_len:
        return meta[:pred_cte_len]
    while len(meta) < pred_cte_len:
        meta.append({'type': 'small', 'node_id': -1, 'edge_id': None, 'ego_type': ego_type, 'pos_tlast': (0.0, 0.0)})
    return meta

def get_edge_feats_in_meta_order(meta, raw_edge_map: Dict[tuple, np.ndarray], schema_cols: List[str]) -> np.ndarray:
    D = len(schema_cols)
    out = np.zeros((len(meta), D), dtype=np.float32)
    type_counters = {k: 0 for k in ORDER_TYPES}

    for i, m in enumerate(meta):
        nt = m['type']
        ego_t = m.get('ego_type')
        et = (nt, 'interacts', ego_t)
        arr = raw_edge_map.get(et, None)
        idx = type_counters[nt]
        if arr is not None and idx < arr.shape[0]:
            row = arr[idx]
            if row.shape[-1] >= D:
                out[i, :] = row[:D]
            else:
                out[i, :row.shape[-1]] = row
        type_counters[nt] += 1
    return out

def plot_causal_graph_directed(path: str, meta, pred_cte: torch.Tensor, coord_lim: float = 20.0, viz_nonneg=True):
    cte = pred_cte.detach().cpu().numpy().reshape(-1)
    if viz_nonneg:
        cte = np.log1p(np.exp(cte))
    lw = 1.0 + 4.0 * (cte - cte.min()) / (cte.ptp() + 1e-6)

    xs = [m.get('pos_tlast', (0.0,0.0))[0] for m in meta]
    ys = [m.get('pos_tlast', (0.0,0.0))[1] for m in meta]

    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('viridis')
    sc = plt.scatter(xs, ys, s=60, c=cte, cmap=cmap, alpha=0.9)
    plt.scatter(0, 0, s=120, c='k', marker='*')

    for i, (x,y) in enumerate(zip(xs, ys)):
        plt.annotate('', xy=(0, 0), xytext=(x, y),
                     arrowprops=dict(arrowstyle='->', lw=lw[i],
                                     color=cmap((cte[i] - cte.min())/(cte.ptp()+1e-6))))
        plt.text(x, y, f"{cte[i]:.3f}", fontsize=8, ha='center', va='bottom')

    plt.colorbar(sc, label='pred CTE (viz)')
    plt.xlabel('dx (m)'); plt.ylabel('dy (m)'); plt.title('Causal Graph (directed)')
    plt.xlim(-coord_lim, coord_lim); plt.ylim(-coord_lim, coord_lim)
    plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()


def plot_ade_vs_cte(path: str, ades: List[float], mean_cte: List[float]):
    plt.figure(figsize=(6, 4))
    plt.scatter(mean_cte, ades, s=26, alpha=0.75)
    plt.xlabel('mean pred CTE'); plt.ylabel('ADE (m)'); plt.title('ADE vs CTE')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_delta_te_hist(path: str, feats: np.ndarray, schema_cols: List[str]):
    if feats.size == 0: return
    i_fwd = schema_cols.index('te_fwd') if 'te_fwd' in schema_cols else 0
    i_rev = schema_cols.index('te_rev') if 'te_rev' in schema_cols else 1
    delta = feats[:, i_fwd] - feats[:, i_rev]
    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=40, alpha=0.85)
    plt.xlabel('ΔTE = TE_fwd - TE_rev'); plt.ylabel('count'); plt.title('Directionality Distribution')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_attn_vs_cte(path: str, attn_all: List[float], cte_all: List[float]):
    a = np.asarray(attn_all).reshape(-1)
    c = np.asarray(cte_all).reshape(-1)
    plt.figure(figsize=(5, 4))
    plt.scatter(a, c, s=14, alpha=0.7)
    plt.xlabel('attention weight'); plt.ylabel('pred CTE'); plt.title('Attention–CTE consistency')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_type_contribution(path: str, type2vals: Dict[str, List[float]]):
    keys = ORDER_TYPES
    vals = [np.mean(type2vals[k]) if (k in type2vals and len(type2vals[k])>0) else 0.0 for k in keys]
    plt.figure(figsize=(5, 4))
    plt.bar(keys, vals)
    plt.ylabel('mean pred CTE'); plt.title('Type-wise CTE contribution')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_dynamic_cte(path: str, ego_future_pred: torch.Tensor, cand_pos: torch.Tensor, cand_vel: torch.Tensor,
                     dt: float, use_velocity=False, base_k=5, noise_std=0.3):
    steps = ego_future_pred.size(0)
    roller = ConstantVelocityRollout(dt=dt, pred_horizon=steps)
    vel_jitter = cand_vel + noise_std * torch.randn_like(cand_vel)
    cand_roll = roller(cand_pos, vel_jitter)  # [Nc,T,2]

    Ts = []
    curves = []
    for i in range(cand_roll.size(0)):
        vals = []
        for t in range(6, steps+1):
            T_eff = t - 1 if use_velocity else t
            k = int(min(base_k, max(2, T_eff//4)))
            if T_eff < 3:
                continue
            s = compute_cte(
                ego_future_pred[:t].unsqueeze(0),
                cand_roll[i,:t,:].unsqueeze(0),
                lag=1, k=k, use_velocity=use_velocity
            )
            vals.append(float(s.view(-1).mean().item()))
            if i == 0: Ts.append(t)
        curves.append(vals)
    if len(Ts) == 0: return
    plt.figure(figsize=(7, 4))
    for vals in curves:
        if len(vals) == 0: continue
        plt.plot(Ts[:len(vals)], vals, lw=1.2, alpha=0.8)
    plt.xlabel('prefix length (frames)'); plt.ylabel('approx CTE')
    plt.title(f'Dynamic CTE (CV+noise, use_velocity={use_velocity})')
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

if __name__ == '__main__':
    npz_path    = './ProcessedData/TGSIM_L1/_topk_dataset.npz'
    ckpt_path   = './ckpt/TGSIM_L1/cv/50/best_model.pth'
    out_dir     = './viz_out_TGSIM_L1'
    split       = 'val'
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_batches = 60
    coord_limit_m = 20.0
    viz_nonneg  = True

    ensure_dir(out_dir)
    schema_cols = load_schema(npz_path)

    ds = NGSIMDataset(npz_path, split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Model
    model = CTEGraphPredictor(node_feat_dim=4, edge_feat_dim=6, hidden_dim=64, horizon=ds.ego_future.shape[1])
    model.to(device)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get('state_dict', state)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print('[WARN] checkpoint not found')
    model.eval()

    ades, mean_cte_list = [], []
    all_edge_feats = []
    all_attn, all_pred_cte = [], []
    type2vals: Dict[str, List[float]] = {k: [] for k in ORDER_TYPES}

    with torch.no_grad():
        for bi, data in enumerate(loader):
            if bi >= num_batches: break
            data = data.to(device)

            raw_edge_attrs = {}
            ego_type = _as_scalar(data['ego_type'])
            for nt in ORDER_TYPES:
                et = (nt, 'interacts', ego_type)
                if et in data.edge_types and hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None:
                    raw_edge_attrs[et] = data[et].edge_attr.detach().cpu().numpy()  # [E, 6]
                else:
                    raw_edge_attrs[et] = None

            outputs = model(data)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 4:
                pred, attn, pred_cte, true_cte = outputs[:4]
            else:
                raise RuntimeError('Model forward must return at least (pred, attn, pred_cte, true_cte)')

            meta = try_get_candidate_meta_from_model_output(outputs)
            if meta is None:
                meta = reconstruct_candidate_meta(data, pred_cte_len=pred_cte.numel())

            ego_index = int(_as_scalar(data['ego_index']))
            gt = data[ego_type].y[ego_index].to(device)
            mask = data[ego_type].y_mask[ego_index].to(device)
            valid = (mask > 0.5).nonzero(as_tuple=False).flatten()
            t_last = int(valid[-1].item()) if valid.numel() > 0 else gt.size(0)-1
            ade = torch.norm(pred[:t_last+1] - gt[:t_last+1], dim=-1).mean().item()
            ades.append(ade)
            mean_cte_list.append(pred_cte.mean().item() if pred_cte.numel() > 0 else 0.0)

            if bi < 12:
                try:
                    plot_causal_graph_directed(
                        os.path.join(out_dir, f'causal_graph_edges_{bi:03d}.png'),
                        meta, pred_cte, coord_lim=coord_limit_m, viz_nonneg=viz_nonneg)
                except Exception as e:
                    print(f'[skip causal graph {bi}] {e}')
                try:
                    xs = [m.get('pos_tlast', (0.0,0.0))[0] for m in meta]
                    ys = [m.get('pos_tlast', (0.0,0.0))[1] for m in meta]
                    if len(xs) > 0:
                        cand_pos = torch.tensor(np.c_[xs, ys], dtype=torch.float32, device=device)
                        cand_vel = torch.zeros_like(cand_pos)
                        plot_dynamic_cte(
                            os.path.join(out_dir, f'cte_time_series_{bi:03d}.png'),
                            pred, cand_pos, cand_vel, dt=float(ds.dt),
                            use_velocity=False, base_k=5, noise_std=0.3)
                except Exception as e:
                    print(f'[skip dynamic cte {bi}] {e}')

            feats = get_edge_feats_in_meta_order(meta, raw_edge_attrs, schema_cols)
            if feats.size > 0:
                all_edge_feats.append(feats)
                for i, m in enumerate(meta):
                    if i < pred_cte.numel():
                        type2vals[m['type']].append(float(pred_cte.view(-1)[i].item()))

            # attention–CTE
            if attn is not None and pred_cte is not None and pred_cte.numel() == attn.numel():
                all_attn += attn.view(-1).detach().cpu().tolist()
                all_pred_cte += pred_cte.view(-1).detach().cpu().tolist()

    plot_ade_vs_cte(os.path.join(out_dir, 'ade_vs_cte.png'), ades, mean_cte_list)

    if len(all_edge_feats) > 0:
        all_edge_feats = np.concatenate(all_edge_feats, axis=0)
        plot_delta_te_hist(os.path.join(out_dir, 'delta_te_hist.png'), all_edge_feats, schema_cols)
        def col(name, default_idx):
            return all_edge_feats[:, schema_cols.index(name) if name in schema_cols else default_idx]
        cte_fwd, cte_rev, dist_m, rel_v, ttc_m = col('cte_fwd',0), col('cte_rev',1), col('dist',2), col('rel_v',3), col('ttc',4)
        lines = [
            f"Spearman(CTE_fwd, dist) = {spearmanr(cte_fwd, dist_m, nan_policy='omit').correlation:.3f}",
            f"Spearman(CTE_fwd, rel_v) = {spearmanr(cte_fwd, rel_v,  nan_policy='omit').correlation:.3f}",
            f"Spearman(CTE_rev, dist) = {spearmanr(cte_rev, dist_m, nan_policy='omit').correlation:.3f}",
            f"Spearman(CTE_rev, TTC)  = {spearmanr(cte_rev, ttc_m,  nan_policy='omit').correlation:.3f}",
            f"Spearman(CTE_rev, rel_v) = {spearmanr(cte_rev, rel_v,  nan_policy='omit').correlation:.3f}",
        ]
        with open(os.path.join(out_dir, 'te_phys_correlations.txt'), 'w') as f:
            f.write('\n'.join(lines))

    if len(all_attn) > 0 and len(all_pred_cte) > 0:
        plot_attn_vs_cte(os.path.join(out_dir, 'attn_vs_predcte_scatter.png'), all_attn, all_pred_cte)
        rho = spearmanr(all_attn, all_pred_cte, nan_policy='omit').correlation
        msg = [f'Spearman(attn, pred_cte) = {rho:.4f}']
        if len(all_edge_feats) > 0 and 'dist' in schema_cols:
            dist_m = all_edge_feats[:, schema_cols.index('dist')]
            bins = np.quantile(dist_m, [0.0, 0.33, 0.66, 1.0])
            for b in range(3):
                mask = (dist_m >= bins[b]) & (dist_m <= bins[b+1])
                aa = np.asarray(all_attn)[mask]
                cc = np.asarray(all_pred_cte)[mask]
                if aa.size > 5 and cc.size > 5:
                    msg.append(f'  bin{b+1} [{bins[b]:.2f},{bins[b+1]:.2f}] → ρ={spearmanr(aa, cc, nan_policy="omit").correlation:.3f}')
        with open(os.path.join(out_dir, 'attn_cte_spearman.txt'), 'w') as f:
            f.write('\n'.join(msg))

    plot_type_contribution(os.path.join(out_dir, 'cte_type_contribution.png'), type2vals)

    print('All figures saved to:', out_dir)
