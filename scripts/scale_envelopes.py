import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.manifold import TSNE
import pandas as pd
import random
from phd_qwen_CUDA_clean import get_phd, load_qwen_model, load_roberta_model, get_embeds
from copy import deepcopy
from tqdm import tqdm
from GPTID.IntrinsicDimCUDA_clean import pairwise_distances
import torch


def get_mst_edge_lengths(points):
    """
    points: numpy array of shape [n, d]
    returns: numpy vector of length n-1 containing MST edge weights
    """
    # 1. Compute all-pairs Euclidean distances
    # pdist returns a condensed distance vector
    adj_matrix = pairwise_distances(torch.Tensor(points).to('cuda:1')).float().cpu().numpy().astype(float)
#    print(dist_vector.shape)
    np.fill_diagonal(adj_matrix, 0)

    # 2. Convert to a square adjacency matrix (full graph)
#    adj_matrix = squareform(dist_vector)
#    print(adj_matrix.shape)
    
    # 3. Calculate the Minimum Spanning Tree
    # returns a compressed sparse row (CSR) matrix
    mst_matrix = minimum_spanning_tree(adj_matrix)
    
    # 4. Extract edge weights into a numpy vector
    # .data contains only the non-zero entries (the n-1 MST edges)
    return mst_matrix.data


def lower_quantile_trimmed_mst_sum(mst_lens, p, alpha=1.0):
    """
    Remove the shortest p-fraction of MST edges and sum the rest:
      p=0.0 -> full sum
      p=0.9 -> keep only the top 10% longest edges
    """
    if mst_lens.size == 0:
        return 0.0
    m = mst_lens.size  # = n-1
    k = int(np.floor(p * m))  # number of shortest edges to drop
    lens_sorted = np.sort(mst_lens)  # ascending
    return float((lens_sorted[k:] ** alpha).sum())


def upper_quantile_trimmed_mst_sum(mst_lens, p, alpha=1.0):
    """
    Remove the shortest p-fraction of MST edges and sum the rest:
      p=0.0 -> full sum
      p=0.9 -> keep only the top 10% longest edges
    """
    if mst_lens.size == 0:
        return 0.0
    m = mst_lens.size  # = n-1
    k = int(np.floor(p * m))  # number of longest edges to drop
    lens_sorted = np.sort(mst_lens)[::-1]  # descending
    #print(k, m, lens_sorted[0], lens_sorted[k], lens_sorted[-1])
    #return float((lens_sorted[k:k_max+1] ** alpha).sum())
    return float((lens_sorted[k:] ** alpha).sum())


def double_quantile_trimmed_mst_sum(mst_lens, p_lower, p_upper, alpha=1.0):
    """
    Remove the shortest p-fraction of MST edges and sum the rest:
      p=0.0 -> full sum
      p=0.9 -> keep only the top 10% longest edges
    """
    if mst_lens.size == 0:
        return 0.0
    if p_upper <= p_lower:
        p_upper = 1
    m = mst_lens.size  # = n-1
    k_min = int(np.floor(p_lower * m))  # number of shortest edges to drop
    k_max = int(np.ceil(p_upper * m))  # number of longest edges to drop
    lens_sorted = np.sort(mst_lens)  # ascending
    return float((lens_sorted[k_min:min(m,k_max)] ** alpha).sum())


def plot_tsne(X, perplexity=30, n_iter=1500, seed=0, title="t-SNE"):
    """
    X: (N, D) numpy array
    """
    X = np.asarray(X, dtype=np.float32)

    # t-SNE can be slow; if N is huge, subsample:
    # X = X[np.random.default_rng(seed).choice(len(X), size=5000, replace=False)]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        verbose=1,
    )
    Y = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(Y[:, 0], Y[:, 1], s=5)
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.grid(True, alpha=0.3)

    return plt




def _fit_loglog_slope(x_vals, y_vals):
    """
    Fit y = a + b*x in least squares, return (slope=b, intercept=a, r2).
    Inputs should be 1D arrays of already-log-transformed values.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, np.nan

    # slope/intercept
    b, a = np.polyfit(x, y, deg=1)  # y ≈ a + b x  (polyfit returns [b,a])
    y_hat = a + b * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    return b, a, r2


def _safe_d_from_barcode_slope(slope):
    """
    If log b_n(u) ≈ const + slope*log n, then slope ≈ -1/d.
    """
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return np.nan
    return -1.0 / slope


def _safe_d_from_energy_slope(alpha, slope):
    """
    If log S_{alpha,p}(n) ≈ const + slope*log n and slope ≈ 1 - alpha/d,
    then d = alpha / (1 - slope).
    """
    if not np.isfinite(slope):
        return np.nan
    denom = 1.0 - slope
    if abs(denom) < 1e-12:
        return np.nan
    return float(alpha) / denom


def plot_median_by_param_value(
    df_en,
    d_energy_stats_df_list,
    limit=3000,
    min_count_plot=100,
    obj_name='d_energy',
    xlim=0.5,
    filename_save='figures/default'
):
    model2count = df_en.iloc[:limit, :].groupby('model').count()[['source']]
    models = model2count.query(f"source > {min_count_plot}").index.tolist()
    df_filter = df_en.iloc[:limit, :]
    for idx, d_energy_df in enumerate(d_energy_stats_df_list):
        d_energy_df['text'] = df_filter.iloc[idx, :]['text']
    d_energies = pd.concat(d_energy_stats_df_list)
    df_joined = d_energies.set_index('text').join(df_filter.set_index('text')).reset_index()
    df_joined_filtered = df_joined.query("model in @models")
    for model_idx, model_name in enumerate(models):
        df_model = df_joined_filtered.query("model == @model_name")
        df_model = df_model.query(f"param_value < {xlim}").query("d_hat > 0")
        df_model.groupby("param_value")['d_hat'].median().plot(label=model_name, figsize=(10, 10), c='C' + str(model_idx))
     #   plt.axhline(df_joined_filtered.groupby("model")['phd_gemma'].median()[model_name], c='C' + str(model_idx))
    #    plt.fill_between(
    #        sorted(df_model['param_value'].drop_duplicates().values.tolist()),
    #        df_model.groupby("param_value")['d_hat'].quantile(0.95),
    #        df_model.groupby("param_value")['d_hat'].quantile(0.05),
    #        alpha=0.2
    #    )
        plt.ylabel(obj_name)
        #, count texts = {df_model.shape[0]}")
        plt.legend()
    plt.title(f"median d_energy as function of param_value")
    plt.savefig(filename_save + '.png')
    plt.figure()
#    plt.show()
     #plt.figure((10, 10))
    for model_idx, model_name in enumerate(models):
        df_model = df_joined_filtered.query("model == @model_name")
        df_model = df_model.query(f"param_value < {xlim}").query("d_hat > 0")
        df_model.groupby("param_value")['d_hat'].median().plot(label=model_name, figsize=(10, 10), c='C' + str(model_idx))
        plt.axhline(
            df_joined_filtered.groupby("model")['phd_gemma'].median()[model_name],
            c='C' + str(model_idx + 1),
            label='median phd value'
        )
        plt.fill_between(
            sorted(df_model['param_value'].drop_duplicates().values.tolist()),
            df_model.groupby("param_value")['d_hat'].quantile(0.8),
            df_model.groupby("param_value")['d_hat'].quantile(0.05),
            alpha=0.2,
            color='C' + str(model_idx + 1)
        )
        plt.ylabel(obj_name)
        plt.title(f"{model_name} param_value vs {obj_name} and 0.8, 0.05 quantiles of {obj_name}")#, count texts = {model2count[model_name].shape[0]}")
        plt.legend()
#        plt.show()
        plt.savefig(filename_save + '_' + model_name + '.png')
        plt.figure()


def log_scale(data, grid):
    if data.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    m = data.size                  
    idx = np.clip((np.ceil(grid * m).astype(int) - 1), 0, m - 1)
    return data[idx]




class PHDimScale:
    def __init__(
        self,
        p_list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        n_fraction_list=[0.2, 0.4, 0.6, 0.8, 1.0],
        alpha=1.0,
        p_range=0.5
    ):
        self.p_list = p_list
        self.n_fraction_list = n_fraction_list
        self.alpha = alpha
        self.p_range = p_range
        u_grid = np.logspace(-2, 0, 80)   # 1e-3 ... 1
        u_grid = np.clip(u_grid, 1e-2, 1.0)
        self.u_grid = u_grid
        
        self.envelopes = dict()
        self.log_envelopes = dict()
        self.env_mean_by_n = {}
        self.log_env_mean_by_n = {}
        self.replicates = 10# count retries for decreasing std

    def calculate(self, embeds, object_name):
        rows_p = []
        self.len_text = embeds.shape[0]
        for n_fraction in self.n_fraction_list:
            for r in range(self.replicates):
                n = int(n_fraction * embeds.shape[0])
                indices = np.random.choice(embeds.shape[0], size=n)
                pts = embeds[indices, :]
                mst_lens = np.sort(get_mst_edge_lengths(pts))
                self.envelopes[(n, r)] = np.sort(mst_lens)
                self.log_envelopes[(n, r)] = log_scale(self.envelopes[(n, r)], grid=self.u_grid)
                for p in self.p_list:
                    s_lower = lower_quantile_trimmed_mst_sum(mst_lens, p=p, alpha=self.alpha)
                    s_upper = upper_quantile_trimmed_mst_sum(mst_lens, p=p, alpha=self.alpha)
                    s_range = double_quantile_trimmed_mst_sum(mst_lens, p_lower=p, p_upper=min(1, p+self.p_range), alpha=self.alpha)
                    rows_p.append({"alpha":self.alpha, "n": n, "rep": r, "p": p, "S_lower": s_lower, "S_upper": s_upper, "S_range": s_range})
            df_p = pd.DataFrame(rows_p)
            self.agg_p_lower = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_lower"].mean()
            self.agg_p_upper = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_upper"].mean()
            self.agg_p_range = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_range"].mean()
        
            self.mats = np.stack([self.envelopes[(n, r)] for r in range(self.replicates)], axis=0)
            self.env_mean_by_n[n] = np.nanmean(self.mats, axis=0)
            self.log_mats = np.stack([self.log_envelopes[(n, r)] for r in range(self.replicates)], axis=0)
            self.log_env_mean_by_n[n] = np.nanmean(self.log_mats, axis=0)
        d_hat_stats_df = self.get_d_hat_stats(object_name)
        d_energy_range_stats_df = self.get_d_energy_stats(object_name, self.agg_p_range, "trimmered_energy_range", "S_range")
        d_energy_upper_stats_df = self.get_d_energy_stats(object_name, self.agg_p_upper, "trimmered_energy_upper", "S_upper")
        d_energy_lower_stats_df = self.get_d_energy_stats(object_name, self.agg_p_lower, "trimmered_energy_lower", "S_lower")

        return d_hat_stats_df, d_energy_range_stats_df, d_energy_upper_stats_df, d_energy_lower_stats_df

    def get_d_hat_stats(self, obj_name):
        all_rows = []
#        obj_name = 'no name'
        for u0 in self.p_list:
            j = int(np.argmin(np.abs(self.u_grid - u0)))
            xs, ys = [], []
            for n in self.n_fraction_list:
                n = int(n * embeds.shape[0])
                b = self.log_env_mean_by_n[n][j]
                if b > 0:
                    xs.append(np.log(n))
                    ys.append(np.log(b))
        
            slope, intercept, r2 = _fit_loglog_slope(xs, ys)
            d_hat = _safe_d_from_barcode_slope(slope)
        
            all_rows.append({
                "object": obj_name,
                "estimator": "barcode_quantile",
                "param_type": "u",
                "param_value": float(u0),
                "alpha": np.nan,
                "fit_range": "full",
                "n_points": len(xs),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "d_hat": d_hat,
            })
        
        return pd.DataFrame(all_rows)
        
    def get_d_energy_stats(self, obj_name, agg_p, estimator_name, key):
        all_rows = []
        # obj_name = 'no name'
        for p in self.p_list:
            sub = agg_p[agg_p["p"] == p].sort_values("n")
            xs = np.log([int(self.len_text * n) for n in self.n_fraction_list])
#            print("before np.log")
            ys = np.log(sub[key].values)
#            print("after np.log")
        
            slope, intercept, r2 = _fit_loglog_slope(xs, ys)
            d_hat = _safe_d_from_energy_slope(self.alpha, slope)
            
            all_rows.append({
                "object": obj_name,
                "estimator": estimator_name,
                "param_type": "p",
                "param_value": float(p),
                "alpha": float(self.alpha),
                "fit_range": "full",
                "n_points": int(sub.shape[0]),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "d_hat": d_hat,
            })
            
        return pd.DataFrame(all_rows)


def plot_barcodes( env_mean_by_n):
    plt.figure(figsize=(8, 5.5))
    step = 0.01
    for n in n_list:
        b = env_mean_by_n[n]
        a = np.linspace(start=0, stop=1, num=len(b))
        plt.plot(a, b, label=f"n={n}")
    plt.xlabel(r"$u$  (rank fraction)")
    plt.ylabel(r"$b_n(u)$  (descending MST edge length)")
    plt.title(f"{obj_name}: Barcode envelope curves for different $n$")
    plt.legend(framealpha=0.3)
    plt.grid(True, alpha=0.3)