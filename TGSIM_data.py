import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from copent import transent

@dataclass
class WindowConfig:
    dt: float = 0.1
    t_hist_s: float = 3.0
    t_pred_s: float = 5.0
    hist_frames: int = 30    # 3s / 0.1
    pred_frames: int = 50    # 5s / 0.1
    min_len_for_te: int = 10
    lag_frames: int = 1
    step_center: int = 1


@dataclass
class RangeConfig:
    y_front_m: float = 100.0
    y_back_m: float = 60.0


@dataclass
class TEConfig:
    top_k: int = 6


@dataclass
class PhysConfig:
    lane_width: float = 3.6
    ttc_cap: float = 10.0


@dataclass
class DatasetConfig:
    train_ratio: float = 0.8


def load_preprocessed(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    needed_cols = [
        "run_index",
        "Vehicle_ID",
        "Frame_ID",
        "Local_X_m",
        "Local_Y_m",
        "v_Vel_mps",
        "v_x",
        "v_y",
        "lane_kf",
        "v_Length_m",
        "v_Width_m",
        "veh_type",
    ]
    for c in needed_cols:
        if c not in df.columns:
            raise KeyError(f"预处理 CSV 缺少列: {c}")

    df["run_index"] = df["run_index"].astype(int)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["veh_type"] = df["veh_type"].astype(np.int32)

    df = df.sort_values(["run_index", "Frame_ID", "Vehicle_ID"]).reset_index(drop=True)
    return df


def build_frame_index(df: pd.DataFrame) -> Dict[Tuple[int, int], pd.DataFrame]:
    frame_index: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (run_idx, f_id), g in df.groupby(["run_index", "Frame_ID"], sort=False):
        use_cols = [
            "Vehicle_ID",
            "Local_Y_m",
            "Local_X_m",
            "v_Vel_mps",
            "v_x",
            "v_y",
            "veh_type",
            "v_Length_m",
            "v_Width_m",
        ]
        use_cols = [c for c in use_cols if c in g.columns]
        gg = g[use_cols].sort_values("Local_Y_m").reset_index(drop=True)
        frame_index[(int(run_idx), int(f_id))] = gg
    return frame_index


def build_vid_index(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    idx: Dict[int, pd.DataFrame] = {}
    for vid, g in df.groupby("Vehicle_ID", sort=False):
        idx[int(vid)] = g.sort_values("Frame_ID").reset_index(drop=True)
    return idx


def query_candidates_by_longitudinal_range(
    frame_df: pd.DataFrame,
    y_center: float,
    y_back_m: float,
    y_front_m: float,
    ego_vid: int,
) -> List[int]:
    y = frame_df["Local_Y_m"].to_numpy(dtype=float)
    lo = y_center - y_back_m
    hi = y_center + y_front_m
    mask = (y >= lo) & (y <= hi)
    vids = frame_df.loc[mask, "Vehicle_ID"].tolist()
    return [int(v) for v in vids if int(v) != int(ego_vid)]


def slice_recent_history(
    df_by_vid: Dict[int, pd.DataFrame],
    vid: int,
    center_frame: int,
    max_len: int,
) -> Optional[pd.DataFrame]:
    g = df_by_vid.get(int(vid))
    if g is None:
        return None
    lo = center_frame - max_len + 1
    hi = center_frame
    sel = g[(g["Frame_ID"] >= lo) & (g["Frame_ID"] <= hi)]
    if sel.empty:
        return None
    return sel.copy()


def align_on_common_frames(ego_hist: pd.DataFrame, cand_hist: pd.DataFrame):
    cols_needed = ["Frame_ID", "v_Vel_mps"]
    for df_ in (ego_hist, cand_hist):
        for c in cols_needed:
            if c not in df_.columns:
                raise KeyError(f"历史序列缺少列 {c}")
    merged = ego_hist[cols_needed].merge(
        cand_hist[cols_needed],
        on="Frame_ID",
        suffixes=("_ego", "_cand"),
    )
    if merged.empty:
        return np.array([]), np.array([])
    merged = merged.sort_values("Frame_ID")
    x_ego = merged["v_Vel_mps_ego"].to_numpy(dtype=float)
    x_cand = merged["v_Vel_mps_cand"].to_numpy(dtype=float)
    return x_ego, x_cand


def valid_center_frames_for_ego(ego_df: pd.DataFrame, win: WindowConfig) -> List[int]:
    frames = ego_df["Frame_ID"].to_numpy(dtype=int)
    if len(frames) < (win.hist_frames + win.pred_frames):
        return []
    start = int(frames.min())
    end = int(frames.max())
    lo = start + win.hist_frames - 1
    hi = end - win.pred_frames
    if hi < lo:
        return []
    centers = np.arange(lo, hi + 1, win.step_center, dtype=int).tolist()
    return centers


def compute_phys_series(
    dx: np.ndarray,
    dy: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    lane_width: float = 3.6,
    ttc_cap: float = 10.0,
) -> np.ndarray:
    T = len(dx)
    out = np.zeros((T, 4), dtype=np.float32)
    eps = 1e-6
    for t in range(T):
        dist = float(np.hypot(dx[t], dy[t]))
        rel_speed = float(np.hypot(dvx[t], dvy[t]))
        rdotv = dx[t] * dvx[t] + dy[t] * dvy[t]
        rnorm = max(dist, eps)
        closing = max(0.0, -rdotv / rnorm)
        ttc = dist / max(closing, eps) if closing > eps else ttc_cap
        same_lane = 1.0  # 有需要可以之后用 lane_kf 判断
        out[t, 0] = dist
        out[t, 1] = rel_speed
        out[t, 2] = min(ttc, ttc_cap)
        out[t, 3] = same_lane
    return out


def masked_mean_std(arr: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6):
    if mask is None:
        m = arr.mean(axis=(0, 1), keepdims=False)
        s = arr.std(axis=(0, 1), keepdims=False)
        return m, np.maximum(s, eps)
    else:
        m_list, s_list = [], []
        D = arr.shape[-1]
        flat_mask = mask > 0.5
        for d in range(D):
            valid = arr[..., d][flat_mask]
            if valid.size == 0:
                m_list.append(0.0)
                s_list.append(1.0)
            else:
                m_list.append(float(valid.mean()))
                s_list.append(float(max(valid.std(), eps)))
        return np.array(m_list), np.array(s_list)


def norm_inplace(arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    reshape_shape = (1,) * (arr.ndim - 1) + (-1,)
    arr -= mean.reshape(reshape_shape)
    arr /= std.reshape(reshape_shape)

def main(preprocessed_csv: str, out_path: str):
    os.makedirs(out_path, exist_ok=True)

    win_cfg = WindowConfig()
    range_cfg = RangeConfig()
    te_cfg = TEConfig()
    phys_cfg = PhysConfig()
    dataset_cfg = DatasetConfig()

    df = load_preprocessed(preprocessed_csv)

    vid_counts = df.groupby("Vehicle_ID", sort=False)["Frame_ID"].count()
    ego_vids = vid_counts[
        vid_counts >= (win_cfg.hist_frames + win_cfg.pred_frames)
    ].index.tolist()

    frame_index = build_frame_index(df)# key: (run_index, Frame_ID)
    df_by_vid = build_vid_index(df)

    ego_hist_list: List[np.ndarray] = []
    ego_future_list: List[np.ndarray] = []
    ego_type_list: List[int] = []
    cand_hist_list: List[np.ndarray] = []
    cand_mask_list: List[np.ndarray] = []
    cand_type_list: List[np.ndarray] = []
    topk_te_list: List[np.ndarray] = []
    reverse_te_list: List[np.ndarray] = []
    cand_phys_list: List[np.ndarray] = []
    meta_rows: List[dict] = []

    for ego_vid in tqdm(ego_vids, total=len(ego_vids), desc="build samples"):
        ego_df = df_by_vid[int(ego_vid)]
        run_idx_ego = int(ego_df["run_index"].iloc[0])

        centers = valid_center_frames_for_ego(ego_df, win_cfg)
        if not centers:
            continue

        ego_frames = ego_df["Frame_ID"].to_numpy(dtype=int)
        ego_y_vals = ego_df["Local_Y_m"].to_numpy(dtype=float)
        ego_x_vals = ego_df["Local_X_m"].to_numpy(dtype=float)
        ego_vx = ego_df["v_x"].to_numpy(dtype=float)
        ego_vy = ego_df["v_y"].to_numpy(dtype=float)
        ego_type_seq = ego_df["veh_type"].to_numpy(np.int32)

        frame_to_index = {int(f): i for i, f in enumerate(ego_frames)}

        for t_c in centers:
            key = (run_idx_ego, int(t_c))
            if key not in frame_index:
                continue
            if t_c not in frame_to_index:
                continue

            idx_c = frame_to_index[t_c]
            if idx_c - (win_cfg.hist_frames - 1) < 0 or idx_c + win_cfg.pred_frames >= len(ego_frames):
                continue

            hist_idx = np.arange(idx_c - win_cfg.hist_frames + 1, idx_c + 1)
            fut_idx = np.arange(idx_c + 1, idx_c + 1 + win_cfg.pred_frames)

            ego_hist_abs = np.stack(
                [ego_x_vals[hist_idx], ego_y_vals[hist_idx], ego_vx[hist_idx], ego_vy[hist_idx]],
                axis=-1,
            ).astype(np.float32)  # [T_hist, 4]

            ego_future_abs = np.stack(
                [ego_x_vals[fut_idx], ego_y_vals[fut_idx]],
                axis=-1,
            ).astype(np.float32)  # [T_pred, 2]

            x0 = float(ego_x_vals[idx_c])
            y0 = float(ego_y_vals[idx_c])
            ego_hist_rel = ego_hist_abs.copy()
            ego_hist_rel[:, 0] -= x0
            ego_hist_rel[:, 1] -= y0
            ego_future_rel = ego_future_abs.copy()
            ego_future_rel[:, 0] -= x0
            ego_future_rel[:, 1] -= y0

            frdf = frame_index[key]
            ego_y = float(ego_y_vals[idx_c])

            cand_ids = query_candidates_by_longitudinal_range(
                frdf, ego_y, range_cfg.y_back_m, range_cfg.y_front_m, ego_vid
            )
            if len(cand_ids) == 0:
                continue

            ego_hist_df = slice_recent_history(df_by_vid, ego_vid, int(ego_frames[idx_c]), win_cfg.hist_frames)
            if ego_hist_df is None or ego_hist_df.empty:
                continue

            te_scores: List[Tuple[int, float, float]] = []
            cand_window_cache: Dict[int, pd.DataFrame] = {}

            for cvid in cand_ids:
                cand_hist_df = slice_recent_history(df_by_vid, cvid, int(ego_frames[idx_c]), win_cfg.hist_frames)
                if cand_hist_df is None or cand_hist_df.empty:
                    continue

                ego_spd, cand_spd = align_on_common_frames(ego_hist_df, cand_hist_df)
                if len(ego_spd) < win_cfg.min_len_for_te or len(cand_spd) < win_cfg.min_len_for_te:
                    continue

                try:
                    te_forward = float(transent(cand_spd, ego_spd, win_cfg.lag_frames))   # cand -> ego
                    te_reverse = float(transent(ego_spd, cand_spd, win_cfg.lag_frames))   # ego  -> cand
                except Exception:
                    continue
                if not (np.isfinite(te_forward) and np.isfinite(te_reverse)):
                    continue

                te_scores.append((int(cvid), te_forward, te_reverse))
                cand_window_cache[int(cvid)] = cand_hist_df

            if not te_scores:
                continue

            te_scores.sort(key=lambda x: x[1], reverse=True)
            topk = te_scores[: te_cfg.top_k]

            K = te_cfg.top_k
            T = win_cfg.hist_frames
            D_nei = 4
            cand_hist = np.zeros((K, T, D_nei), dtype=np.float32)
            cand_mask = np.zeros((K, T), dtype=np.float32)
            cand_type = np.zeros((K,), dtype=np.int64)
            cand_phys = np.zeros((K, T, 4), dtype=np.float32)
            topk_te = np.zeros((K,), dtype=np.float32)
            reverse_te = np.zeros((K,), dtype=np.float32)

            for k_idx, (cvid, te_fwd, te_rev) in enumerate(topk):
                ch = cand_window_cache.get(int(cvid))
                if ch is None or ch.empty:
                    continue
                ch = ch[["Frame_ID", "Local_X_m", "Local_Y_m", "v_x", "v_y", "veh_type"]].copy()
                ch = ch.sort_values("Frame_ID")
                ch_map = {
                    int(r.Frame_ID): (
                        float(r.Local_X_m),
                        float(r.Local_Y_m),
                        float(r.v_x),
                        float(r.v_y),
                        int(r.veh_type),
                    )
                    for _, r in ch.iterrows()
                }

                for t_i, f_id in enumerate(ego_frames[hist_idx]):
                    tup = ch_map.get(int(f_id))
                    if tup is None:
                        continue
                    x, y, vx, vy, vtype = tup

                    cand_hist[k_idx, t_i, 0] = x - ego_x_vals[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 1] = y - ego_y_vals[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 2] = vx - ego_vx[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 3] = vy - ego_vy[hist_idx[t_i]]
                    cand_mask[k_idx, t_i] = 1.0
                    cand_type[k_idx] = int(vtype)

                dx = cand_hist[k_idx, :, 0]
                dy = cand_hist[k_idx, :, 1]
                dvx = cand_hist[k_idx, :, 2]
                dvy = cand_hist[k_idx, :, 3]
                cand_phys[k_idx] = compute_phys_series(
                    dx, dy, dvx, dvy,
                    lane_width=phys_cfg.lane_width,
                    ttc_cap=phys_cfg.ttc_cap,
                )

                topk_te[k_idx] = te_fwd     # cand -> ego
                reverse_te[k_idx] = te_rev  # ego  -> cand

            ego_type_id = int(ego_type_seq[frame_to_index[t_c]])  # 1=acc,2=small,3=large

            ego_hist_list.append(ego_hist_rel)
            ego_future_list.append(ego_future_rel)
            ego_type_list.append(ego_type_id)
            cand_hist_list.append(cand_hist)
            cand_mask_list.append(cand_mask)
            cand_type_list.append(cand_type)
            topk_te_list.append(topk_te)
            reverse_te_list.append(reverse_te)
            cand_phys_list.append(cand_phys)

            meta_rows.append(
                {
                    "ego_id": int(ego_vid),
                    "ego_type": int(ego_type_id),
                    "run_index": int(run_idx_ego),
                    "center_frame": int(t_c),
                    "num_cands_seen": len(te_scores),
                }
            )

    if not meta_rows:
        raise RuntimeError("No sample, check your data")

    ego_hist_arr = np.stack(ego_hist_list, axis=0)
    ego_future_arr = np.stack(ego_future_list, axis=0)
    ego_type_arr = np.array(ego_type_list, dtype=np.int64)
    cand_hist_arr = np.stack(cand_hist_list, axis=0)
    cand_mask_arr = np.stack(cand_mask_list, axis=0)
    cand_type_arr = np.stack(cand_type_list, axis=0)
    topk_te_arr = np.stack(topk_te_list, axis=0)
    reverse_te_arr = np.stack(reverse_te_list, axis=0)
    cand_phys_arr = np.stack(cand_phys_list, axis=0)
    meta_df = pd.DataFrame(meta_rows)

    N, T, _ = ego_hist_arr.shape
    Tp = ego_future_arr.shape[1]
    K = cand_hist_arr.shape[1]
    print(f"Final sample N={N}, T_hist={T}, T_pred={Tp}, K={K}")

    # train/val
    idx_all = np.arange(N)
    np.random.shuffle(idx_all)
    n_train = int(N * dataset_cfg.train_ratio)
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]

    ego_hist_mean, ego_hist_std = masked_mean_std(ego_hist_arr[train_idx])
    ego_future_mean, ego_future_std = masked_mean_std(ego_future_arr[train_idx])

    cand_hist_train = cand_hist_arr[train_idx]
    cand_phys_train = cand_phys_arr[train_idx]
    cand_mask_train = cand_mask_arr[train_idx]

    cm = cand_mask_train.reshape(-1, T)
    ch = cand_hist_train.reshape(-1, T, cand_hist_train.shape[-1])
    cp = cand_phys_train.reshape(-1, T, cand_phys_train.shape[-1])
    cand_mean, cand_std = masked_mean_std(ch, mask=cm)
    cand_phys_mean, cand_phys_std = masked_mean_std(cp, mask=cm)

    norm_stats = {
        "ego_hist_mean": ego_hist_mean.tolist(),
        "ego_hist_std": ego_hist_std.tolist(),
        "ego_future_mean": ego_future_mean.tolist(),
        "ego_future_std": ego_future_std.tolist(),
        "cand_hist_mean": cand_mean.tolist(),
        "cand_hist_std": cand_std.tolist(),
        "cand_phys_mean": cand_phys_mean.tolist(),
        "cand_phys_std": cand_phys_std.tolist(),
    }

    ego_hist_n = ego_hist_arr.copy()
    ego_future_n = ego_future_arr.copy()
    ego_type_n = ego_type_arr.copy()
    cand_hist_n = cand_hist_arr.copy()
    cand_phys_n = cand_phys_arr.copy()

    norm_inplace(ego_hist_n, ego_hist_mean, ego_hist_std)
    norm_inplace(ego_future_n, ego_future_mean, ego_future_std)
    norm_inplace(cand_hist_n, cand_mean, cand_std)
    norm_inplace(cand_phys_n, cand_phys_mean, cand_phys_std)

    npz_path = os.path.join(out_path, "_topk_dataset.npz")
    np.savez_compressed(
        npz_path,
        ego_hist=ego_hist_n,
        ego_future=ego_future_n,
        ego_type=ego_type_n,
        cand_hist=cand_hist_n,
        cand_mask=cand_mask_arr,
        cand_type=cand_type_arr,
        topk_te=topk_te_arr,
        reverse_te=reverse_te_arr,
        cand_phys=cand_phys_n,
        train_idx=train_idx,
        val_idx=val_idx,
        meta=json.dumps(meta_df.to_dict(orient="list")).encode("utf-8"),
        dt=np.array([win_cfg.dt], dtype=np.float32),
    )

    with open(os.path.join(out_path, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    with open(os.path.join(out_path, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f, indent=2)

    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_csv",
        type=str,
        default="./ProcessedData/TGSIM_L2/TGSIM_L2.csv",
        help="preprocess_tgsim.py CSV path",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./ProcessedData/TGSIM_topk",
        help="output path",
    )
    args = parser.parse_args()
    main(args.preprocessed_csv, args.out_path)
