import argparse
import glob
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict,List,Tuple,Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from copent import transent

@dataclass
class WindowConfig:
    dt=0.1
    t_hist_s=3.0
    t_pred_s=5.0
    hist_frames=30
    pred_frames=50
    min_len_for_te=10
    lag_frames=1#TE lag=1*dt
    step_center=1#slide by 1 frame among valid centers

@dataclass
class RangeConfig:
    y_front_m=100.0
    y_back_m=60.0

@dataclass
class TEConfig:
    top_k=6

@dataclass
class PhysConfig:
    lane_width=3.6
    ttc_cap=10.0

@dataclass
class DatasetConfig:
    train_ratio=0.8

FT_TO_M=0.3048

NGSIM_COLS=[
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc",
    "Lane_ID", "Preceeding", "Following", "Space_Hdwy", "Time_Hdwy"
]

def load_df(paths):
    dfs=[]
    for p in paths:
        df=pd.read_csv(p,usecols=[c for c in NGSIM_COLS if c in pd.read_csv(p,nrows=1).columns])
        dfs.append(df)
    df=pd.concat(dfs,ignore_index=True)
    df=df.sort_values(["Frame_ID","Vehicle_ID"]).reset_index(drop=True)

    for col in ["Local_X","Local_Y","Global_X","Global_Y","v_Length","v_Width","Space_Hdwy"]:
        if col in df.columns:
            df[col+"_m"]=df[col].apply(lambda x:x*FT_TO_M)

    if "v_Vel" in df.columns:
        df["v_Vel_mps"]=df["v_Vel"].apply(lambda x:x*FT_TO_M)
    if "v_Acc" in df.columns:
        df["v_Acc_mps2"]=df["v_Acc"].apply(lambda x:x*FT_TO_M)

    if "v_Class" in df.columns:
        df["veh_type"]=df["v_Class"]

    return df

def add_diff_velocity(df,smooth_window=7,poly=2):
    dfs=[]
    for vid,g in df.groupby("Vehicle_ID",sort=False):
        g=g.sort_values(["Frame_ID"]).copy()
        for comp,col in [("x","Local_X_m"),("y","Local_Y_m")]:
            if col not in g.columns:
                continue
            val=g[col].to_numpy()
            dv=np.zeros_like(val,dtype=float)
            dv[1:-1]=(val[2:]-val[:-2])/(2*0.1)
            if len(val)>=2:
                dv[0]=(val[1]-val[0])/0.1
                dv[-1]=(val[-1]-val[-2])/0.1
            g[f"v_{comp}"]=dv

        for comp in ["x","y"]:
            v=g[f"v_{comp}"].to_numpy()
            av=np.zeros_like(v,dtype=float)
            if len(v)>=3:
                av[1:-1]=(v[2:]-v[:-2])/(2*0.1)
                av[0]=(v[-1]-v[0])/0.1 if len(v)>=2 else 0
                av[-1]=(v[-1]-v[-2])/0.1 if len(v)>=2 else 0
            g[f"a_{comp}"]=av

        dfs.append(g)

    out=pd.concat(dfs,ignore_index=True)
    out=out.sort_values(["Frame_ID","Vehicle_ID"]).reset_index(drop=True)
    return out

def build_frame_index(df):
    frame_index={}
    for f_id,g in df.groupby("Frame_ID",sort=False):
        use_cols=["Vehicle_ID","Local_Y_m","Local_X_m","v_Vel_mps","v_y","v_x","veh_type","v_Length_m","v_Width_m"]
        use_cols=[c for c in use_cols if c in g.columns]
        gg=g[use_cols].sort_values("Local_Y_m").reset_index(drop=True)
        frame_index[f_id]=gg
    return frame_index

def query_candidates_by_longitudinal_range(frame_df,y_center,y_back_m,y_front_m,ego_vid):
    y=frame_df["Local_Y_m"].to_numpy()
    lo=y_center-y_back_m
    hi=y_center+y_front_m
    mask=(y>=lo) & (y<=hi)
    vids=frame_df.loc[mask,"Vehicle_ID"].tolist()
    return [vid for vid in vids if vid!=ego_vid]

def slice_recent_history(df_by_vid,vid,center_frame,max_len):
    g=df_by_vid.get(vid)
    if g is None:
        return None
    lo=center_frame-max_len+1
    hi=center_frame
    sel=g[(g["Frame_ID"]>=lo) & (g["Frame_ID"]<=hi)]
    if sel.empty:
        return None
    return sel.copy()

def align_on_common_frames(ego_hist,cand_hist):
    cols_needed=["Frame_ID",'v_Vel_mps']
    for df_ in (ego_hist,cand_hist):
        for c in cols_needed:
            if c not in df_.columns:
                raise KeyError(f"Missing column {c} in history DataFrame")
    merged=ego_hist[cols_needed].merge(cand_hist[cols_needed],on="Frame_ID",suffixes=("_ego","_cand"))
    if merged.empty:
        return np.array([]),np.array([])
    merged=merged.sort_values("Frame_ID")
    x_ego=merged["v_Vel_mps_ego"].to_numpy(dtype=float)
    x_cand=merged["v_Vel_mps_cand"].to_numpy(dtype=float)
    return x_ego,x_cand

def build_vid_index(df):
    idx={}
    for vid,g in df.groupby("Vehicle_ID",sort=False):
        idx[vid]=g.sort_values("Frame_ID").reset_index(drop=True)
    return idx

def valid_center_frames_for_ego(ego_df,win):
    frames=ego_df["Frame_ID"].to_numpy()
    if len(frames)<(win.hist_frames+win.pred_frames):
        return []
    start=frames.min()
    end=frames.max()
    lo=start+win.hist_frames-1
    hi=end-win.pred_frames
    if hi<lo:
        return []
    centers=np.arange(lo,hi+1,win.step_center,dtype=int).tolist()
    return centers

def compute_phys_series(dx,dy,dvx,dvy,lane_width=3.6,ttc_cap=10.0):
    T=len(dx)
    out=np.zeros((T,4),dtype=np.float32)
    eps=1e-6
    for t in range(T):
        dist=float(np.hypot(dx[t],dy[t]))
        rel_speed=float(np.hypot(dvx[t],dvy[t]))
        rdotv=dx[t]*dvx[t]+dy[t]*dvy[t]
        rnorm=max(dist,eps)
        closing=max(0.0,-rdotv/rnorm)
        ttc=dist/max(closing,eps) if closing>eps else ttc_cap
        same_lane=1.0 if abs(dx[t])<(lane_width*0.5) else 0.0
        out[t,0]=dist
        out[t,1]=rel_speed
        out[t,2]=min(ttc,ttc_cap)
        out[t,3]=same_lane
    return out

def masked_mean_std(arr,mask=None,eps=1e-6):
    if mask is None:
        m=arr.mean(axis=(0,1),keepdims=False)
        s=arr.std(axis=(0,1),keepdims=False)
        return m,np.maximum(s,eps)
    else:
        m_list,s_list=[],[]
        D=arr.shape[-1]
        for d in range(D):
            valid=arr[...,d][mask>0.5]
            if valid.size==0:
                m_list.append(0.0)
                s_list.append(1.0)
            else:
                m_list.append(valid.mean())
                s_list.append(max(valid.std(),eps))
        return np.array(m_list),np.array(s_list)

def main(file_path,out_path):
    win_cfg=WindowConfig()
    range_cfg=RangeConfig()
    te_cfg=TEConfig()
    phys_cfg=PhysConfig()
    dataset_cfg=DatasetConfig()
    files=sorted(glob.glob(file_path))
    if not files:
        raise FileNotFoundError(f"File {file_path} not found")

    df=load_df(files)
    df=add_diff_velocity(df)

    print("Selecting ego-eligible vehicles...")
    vid_counts=df.groupby("Vehicle_ID",sort=False)["Frame_ID"].count()
    ego_vids=vid_counts[vid_counts>=(win_cfg.hist_frames+win_cfg.pred_frames)].index.tolist()
    print(f'eligible ego vehicles:{len(ego_vids)}')

    print("building indices...")
    frame_index=build_frame_index(df)
    df_by_vid=build_vid_index(df)

    ego_hist_list=[]
    ego_future_list=[]
    ego_type_list=[]
    cand_hist_list=[]
    cand_mask_list=[]
    cand_type_list=[]
    topk_te_list=[]
    reverse_te_list=[]
    cand_phys_list=[]
    meta_rows=[]

    for ego_vid in tqdm(ego_vids, total=len(ego_vids)):
        ego_df = df_by_vid[ego_vid]
        centers = valid_center_frames_for_ego(ego_df, win_cfg)
        if not centers:
            continue

        ego_frames = ego_df["Frame_ID"].to_numpy()
        ego_y_vals = ego_df["Local_Y_m"].to_numpy()
        ego_x_vals = ego_df["Local_X_m"].to_numpy()
        ego_vx = ego_df["v_x"].to_numpy()
        ego_vy = ego_df["v_y"].to_numpy()
        ego_type_seq = ego_df["veh_type"].to_numpy(np.int32)  # per-frame

        frame_to_index = {f: i for i, f in enumerate(ego_frames)}

        for t_c in centers:
            if t_c not in frame_index:
                continue
            if t_c not in frame_to_index:
                continue

            idx_c = frame_to_index[t_c]
            if idx_c - (win_cfg.hist_frames - 1) < 0 or idx_c + win_cfg.pred_frames >= len(ego_frames):
                continue

            hist_idx = np.arange(idx_c - win_cfg.hist_frames + 1, idx_c + 1)
            fut_idx = np.arange(idx_c + 1, idx_c + 1 + win_cfg.pred_frames)

            ego_hist_abs = np.stack([
                ego_x_vals[hist_idx],
                ego_y_vals[hist_idx],
                ego_vx[hist_idx],
                ego_vy[hist_idx],
            ], axis=-1).astype(np.float32)  # [T_hist, 4]

            ego_future_abs = np.stack([
                ego_x_vals[fut_idx],
                ego_y_vals[fut_idx],
            ], axis=-1).astype(np.float32)  # [T_pred, 2]

            x0 = float(ego_x_vals[idx_c])
            y0 = float(ego_y_vals[idx_c])
            ego_hist_rel = ego_hist_abs.copy()
            ego_hist_rel[:, 0] -= x0
            ego_hist_rel[:, 1] -= y0
            ego_future_rel = ego_future_abs.copy()
            ego_future_rel[:, 0] -= x0
            ego_future_rel[:, 1] -= y0

            frdf = frame_index[t_c]
            ego_y = float(ego_y_vals[idx_c])
            cand_ids = query_candidates_by_longitudinal_range(
                frdf, ego_y, range_cfg.y_back_m, range_cfg.y_front_m, ego_vid
            )
            if len(cand_ids) == 0:
                continue

            ego_hist_df = slice_recent_history(df_by_vid, ego_vid, int(ego_frames[idx_c]), win_cfg.hist_frames)
            if ego_hist_df is None or ego_hist_df.empty:
                continue

            te_scores = []  # list of (cand_vid, te_forward(cand->ego), te_reverse(ego->cand))
            cand_window_cache = {}

            for cvid in cand_ids:
                cand_hist_df = slice_recent_history(df_by_vid, cvid, int(ego_frames[idx_c]), win_cfg.hist_frames)
                if cand_hist_df is None or cand_hist_df.empty:
                    continue

                ego_spd, cand_spd = align_on_common_frames(ego_hist_df, cand_hist_df)
                if len(ego_spd) < win_cfg.min_len_for_te or len(cand_spd) < win_cfg.min_len_for_te:
                    continue

                try:
                    te_forward = float(transent(ego_spd, cand_spd, win_cfg.lag_frames))
                    # 反向：ego -> cand
                    te_reverse = float(transent(cand_spd, ego_spd, win_cfg.lag_frames))
                except Exception:
                    continue
                if not (np.isfinite(te_forward) and np.isfinite(te_reverse)):
                    continue

                te_scores.append((int(cvid), te_forward, te_reverse))
                cand_window_cache[int(cvid)] = cand_hist_df

            if not te_scores:
                continue

            te_scores.sort(key=lambda x: x[1], reverse=True)
            topk = te_scores[:TEConfig.top_k]

            K = TEConfig.top_k
            T = win_cfg.hist_frames
            D_nei = 4  # [dX, dY, dVx, dVy]
            cand_hist = np.zeros((K, T, D_nei), dtype=np.float32)
            cand_mask = np.zeros((K, T), dtype=np.float32)
            cand_type = np.zeros((K,), dtype=np.int64)
            cand_phys = np.zeros((K, T, 4), dtype=np.float32)  # [dist, rel_speed, TTC, same_lane]
            topk_te = np.zeros((K,), dtype=np.float32)  # cand->ego
            reverse_te = np.zeros((K,), dtype=np.float32)  # ego->cand

            for k_idx, (cvid, te_fwd, te_rev) in enumerate(topk):
                ch = cand_window_cache.get(int(cvid))
                if ch is None or ch.empty:
                    continue
                ch = ch[["Frame_ID", "Local_X_m", "Local_Y_m", "v_x", "v_y", "veh_type"]].copy()
                ch = ch.sort_values("Frame_ID")
                ch_map = {
                    int(r.Frame_ID): (float(r.Local_X_m), float(r.Local_Y_m),
                                      float(r.v_x), float(r.v_y),
                                      int(r.veh_type) if "veh_type" in ch.columns else 0)
                    for _, r in ch.iterrows()
                }

                for t_i, f_id in enumerate(ego_frames[hist_idx]):
                    tup = ch_map.get(int(f_id), None)
                    if tup is None:
                        continue
                    x, y, vx, vy, vtype = tup

                    cand_hist[k_idx, t_i, 0] = x - ego_x_vals[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 1] = y - ego_y_vals[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 2] = vx - ego_vx[hist_idx[t_i]]
                    cand_hist[k_idx, t_i, 3] = vy - ego_vy[hist_idx[t_i]]
                    cand_mask[k_idx, t_i] = 1.0
                    cand_type[k_idx] = vtype

                dx = cand_hist[k_idx, :, 0]
                dy = cand_hist[k_idx, :, 1]
                dvx = cand_hist[k_idx, :, 2]
                dvy = cand_hist[k_idx, :, 3]
                cand_phys[k_idx] = compute_phys_series(
                    dx, dy, dvx, dvy, lane_width=PhysConfig.lane_width, ttc_cap=PhysConfig.ttc_cap
                )

                topk_te[k_idx] = te_fwd  # cand->ego
                reverse_te[k_idx] = te_rev  # ego->cand

            ego_type_id = int(ego_type_seq[idx_c])  # 1=motor, 2=small, 3=large

            ego_hist_list.append(ego_hist_rel)  # [T,4]
            ego_future_list.append(ego_future_rel)  # [T_pred,2]
            ego_type_list.append(ego_type_id)
            cand_hist_list.append(cand_hist)  # [K,T,4]
            cand_mask_list.append(cand_mask)  # [K,T]
            cand_type_list.append(cand_type)  # [K]
            topk_te_list.append(topk_te)  # [K]
            reverse_te_list.append(reverse_te)  # [K]
            cand_phys_list.append(cand_phys)  # [K,T,4]

            meta_rows.append({
                "ego_id": int(ego_vid),
                "ego_type": int(ego_type_id),
                "center_frame": int(ego_frames[idx_c]),
                "num_cands_seen": len(te_scores),
            })

    if len(meta_rows)==0:
        raise RuntimeError("No samples built")

    ego_hist_arr=np.stack(ego_hist_list,axis=0)#[N,T,4]
    ego_future_arr=np.stack(ego_future_list,axis=0)#[N,T_pred,2]
    ego_type_arr = np.array(ego_type_list, dtype=np.int64)
    cand_hist_arr=np.stack(cand_hist_list,axis=0)#[N,K,T,4]
    cand_mask_arr=np.stack(cand_mask_list,axis=0)#[N,K,T]
    cand_type_arr=np.stack(cand_type_list,axis=0)#[N,K]
    topk_te_arr=np.stack(topk_te_list,axis=0)#[N,K]
    reverse_te_arr=np.stack(reverse_te_list,axis=0)#[N,K]
    cand_phys_arr=np.stack(cand_phys_list,axis=0)#[N,K,T,4]
    meta_df=pd.DataFrame(meta_rows)

    N,T,_=ego_hist_arr.shape
    Tp=ego_future_arr.shape[1]
    K=cand_hist_arr.shape[1]

    print(f"build arrays:N={N},T={T},Tp={Tp},K={K}")

    idx=np.arange(N)
    np.random.shuffle(idx)
    n_train=int(N*dataset_cfg.train_ratio)
    train_idx=idx[:n_train]
    val_idx=idx[n_train:]

    ego_hist_mean,ego_hist_std=masked_mean_std(ego_hist_arr[train_idx])
    ego_future_mean,ego_future_std=masked_mean_std(ego_future_arr[train_idx])

    cand_hist_train=cand_hist_arr[train_idx]
    cand_phys_train = cand_phys_arr[train_idx]
    cand_mask_train=cand_mask_arr[train_idx]

    cm=cand_mask_train.reshape(-1,T)
    ch=cand_hist_train.reshape(-1,T,cand_hist_train.shape[-1])#[n*K,T,4]
    cp=cand_phys_train.reshape(-1,T,cand_phys_train.shape[-1])#[n*K,T,4]
    cand_mean,cand_std=masked_mean_std(ch,mask=cm)
    cand_phys_mean,cand_phys_std=masked_mean_std(cp,mask=cm)



    norm_stats={
        "ego_hist_mean":ego_hist_mean.tolist(),
        "ego_hist_std":ego_hist_std.tolist(),
        "ego_future_mean":ego_future_mean.tolist(),
        "ego_future_std":ego_future_std.tolist(),
        "cand_hist_mean":cand_mean.tolist(),
        "cand_hist_std":cand_std.tolist(),
        "cand_phys_mean":cand_phys_mean.tolist(),
        "cand_phys_std":cand_phys_std.tolist(),
    }

    def norm_inplace(arr,mean,std):
        arr -= mean.reshape(*(1,) * (arr.ndim - 1), -1)
        arr /= std.reshape(*(1,) * (arr.ndim - 1), -1)

    ego_hist_n=ego_hist_arr.copy()
    ego_future_n=ego_future_arr.copy()
    ego_type_n=ego_type_arr.copy()
    cand_hist_n=cand_hist_arr.copy()
    cand_phys_n=cand_phys_arr.copy()

    norm_inplace(ego_hist_n,ego_hist_mean,ego_hist_std)
    norm_inplace(ego_future_n,ego_future_mean,ego_future_std)
    norm_inplace(cand_hist_n,cand_mean,cand_std)
    norm_inplace(cand_phys_n,cand_phys_mean,cand_phys_std)

    print("____saving____")
    npz_path=os.path.join(out_path,"_topk_dataset.npz")
    np.savez_compressed(
        npz_path,
        ego_hist=ego_hist_n,#[N,T,4]
        ego_future=ego_future_n,#[N,T_pred,2]
        ego_type=ego_type_n,#[N]
        cand_hist=cand_hist_n,#[N,K,T,4]
        cand_mask=cand_mask_arr,#【N,K,T】
        cand_type=cand_type_arr,#[N,K]
        topk_te=topk_te_arr,#[N,K]
        reverse_te=reverse_te_arr,#[N,K]
        cand_phys=cand_phys_n,#[N,K,T,4]
        train_idx=train_idx,val_idx=val_idx,
        meta=json.dumps(meta_df.to_dict(orient='list')).encode("utf-8"),
        dt=np.array([0.1],dtype=np.float32),
    )
    with open(os.path.join(out_path, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    with open(os.path.join(out_path, "splits.json"), "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f, indent=2)

    print("done")


if __name__=="__main__":
    file_path="./RawDataset/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.csv"
    out_path="./ProcessedData/NGSIM"
    main(file_path,out_path)