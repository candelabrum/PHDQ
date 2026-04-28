import matplotlib.pyplot as plt
import math
import pandas as pd
import random
import torch
import random
import numpy as np
import torch
import os
import umap
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import requests
import json
import time
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython.display import HTML, display_html
from collections import Counter, defaultdict
from statsmodels.stats.proportion import test_proportions_2indep
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA
from phd_qwen_CUDA_clean import get_phd, load_qwen_model, load_roberta_model, get_embeds
from copy import deepcopy
from tqdm import tqdm
from GPTID.IntrinsicDimCUDA_clean import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from phd_qwen_CUDA_clean import get_phd, load_roberta_model, load_qwen_model
from tqdm import tqdm
from scipy.special import softmax



warnings.filterwarnings('ignore')
sns.set_style("darkgrid")



def get_mst_edge_lengths(points, return_matrix=False, device='cuda:1'):
    """
    points: numpy array of shape [n, d]
    returns: numpy vector of length n-1 containing MST edge weights
    """
    # 1. Compute all-pairs Euclidean distances
    # pdist returns a condensed distance vector
    adj_matrix = pairwise_distances(torch.Tensor(points).to(device)).float().cpu().numpy().astype(float)
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

    if return_matrix == True:
        return mst_matrix.toarray()
    return mst_matrix.data


def calculate_second_min(embeds, device='cuda:1'):
    dist_matrix = pairwise_distances(torch.Tensor(embeds).to(device)).float().cpu().numpy().astype(float)
    dist_matrix_partitioned = np.partition(dist_matrix, 1, axis=1)
    second_mins_by_row = dist_matrix_partitioned[:, 1]

    return second_mins_by_row


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


def set_all_seeds(seed_value):
    # Pure Python
    random.seed(seed_value)
    # NumPy
    np.random.seed(seed_value)
    # Python environment hash (helps with reproducibility in some cases)
#    tf.random.set_seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        # For deterministic CUDA behavior
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

def plot_median_by_param_value(
    df_en,
    d_energy_stats_df_list,
    limit=3000,
    min_count_plot=100,
    obj_name='d_energy',
    xlim=0.5,
    filename_save='figures/default',
    save_roc_auc=True,
    roc_auc_path=None
):
    model2count = df_en.iloc[:limit, :].groupby('model').count()[['text']]
    models = model2count.query(f"text > {min_count_plot}").index.tolist()
    df_filter = df_en.iloc[:limit, :]
    for idx, d_energy_df in enumerate(d_energy_stats_df_list):
        d_energy_df['text'] = df_filter.iloc[idx, :]['text']
    d_energies = pd.concat(d_energy_stats_df_list)
    df_joined = d_energies.set_index('text').join(df_filter.set_index('text')).reset_index()
    df_joined_filtered = df_joined.query("model in @models")

    if save_roc_auc:
        roc_rows = []
        for param_value, grp in df_joined_filtered.groupby('param_value'):
            y_true = pd.to_numeric(grp['model'], errors='coerce')
            y_score = pd.to_numeric(grp['d_hat'], errors='coerce')
            valid_mask = y_true.notna() & y_score.notna()
            y_true = y_true[valid_mask]
            y_score = y_score[valid_mask]

            n_samples = int(valid_mask.sum())
            n_pos = int((y_true == 1).sum())
            n_neg = int((y_true == 0).sum())

            if n_samples < 2:
                roc_auc = np.nan
                status = 'insufficient_samples'
            elif n_pos == 0 or n_neg == 0:
                roc_auc = np.nan
                status = 'single_class'
            else:
                roc_auc = float(roc_auc_score(y_true, y_score))
                status = 'ok'

            roc_rows.append({
                'param_value': float(param_value),
                'roc_auc': roc_auc,
                'n_samples': n_samples,
                'n_pos': n_pos,
                'n_neg': n_neg,
                'status': status,
            })

        roc_auc_df = pd.DataFrame(roc_rows).sort_values('param_value').reset_index(drop=True)
        if roc_auc_path is None:
            roc_auc_path = filename_save + '_roc_auc.csv'
        roc_auc_df.to_csv(roc_auc_path, index=False)

    for model_idx, model_name in enumerate(models):
        df_model = df_joined_filtered.query("model == @model_name")
        df_model = df_model.query(f"param_value < {xlim}").query("d_hat > 0")
        df_model.groupby("param_value")['d_hat'].median().plot(label=model_name, figsize=(10, 10), c='C' + str(model_idx))
    #    plt.axhline(df_joined_filtered.groupby("model")['phd_gemma'].median()[model_name], c='C' + str(model_idx))
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
        if'phd_gemma' in df_joined_filtered.columns.tolist():
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
        plt.savefig(filename_save + '_' + model_name[:8] + '.png')
        plt.figure()


class PHDimScale:
    def __init__(
        self,
        p_list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        n_fraction_list=[0.2, 0.4, 0.6, 0.8, 1.0],
        alpha=1.0,
        p_range=0.5,
        replace=True
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
        self.replicates = 10  # count retries for decreasing std
        self.dfs = []
        self.replace=replace

    def calculate(self, embeds, object_name, tokens):
        rows_p = []
        self.len_text = embeds.shape[0]
        tokens_arr = np.asarray(tokens)
        self.n_values = []
        for n_fraction in self.n_fraction_list:
            n = int(n_fraction * self.len_text)
            if n > 1 and n not in self.n_values:
                self.n_values.append(n)

        for n in self.n_values:
            for r in range(self.replicates):
                indices = np.random.choice(embeds.shape[0], size=n, replace=self.replace)
                pts = embeds[indices, :]

                ###
                # print("before mst lengths:") 
                mst_matrix = get_mst_edge_lengths(pts, return_matrix=True)
                # print("after mst lengths:") 
                # print("tokens len: ", len([token for index, token in enumerate(tokens) if index in indices]))
                # print("mst_lengts: ", mst_lengths.shape)
                sampled_tokens = tokens_arr[indices].tolist()
                df_edges = calculate_df_edges(sampled_tokens, mst_matrix)
                # print("after calculate df edges:") 
                # print(df_edges)
                df_edges['quantile'] = (pd.qcut(df_edges['weight'], q=99, duplicates='drop').rank(pct=True) * 100).fillna(50).apply(int)
                self.dfs.append(df_edges.assign(index_text=object_name))
                ###
                mst_lens = np.sort(mst_matrix[mst_matrix > 0])
                self.envelopes[(n, r)] = np.sort(mst_lens)
                self.log_envelopes[(n, r)] = log_scale(self.envelopes[(n, r)], grid=self.u_grid)
                for p in self.p_list:
                    s_lower = lower_quantile_trimmed_mst_sum(mst_lens, p=p, alpha=self.alpha)
                    s_upper = upper_quantile_trimmed_mst_sum(mst_lens, p=p, alpha=self.alpha)
                    s_range = double_quantile_trimmed_mst_sum(mst_lens, p_lower=p, p_upper=min(1, p+self.p_range), alpha=self.alpha)
                    rows_p.append({"alpha":self.alpha, "n": n, "rep": r, "p": p, "S_lower": s_lower, "S_upper": s_upper, "S_range": s_range})

            self.mats = np.stack([self.envelopes[(n, r)] for r in range(self.replicates)], axis=0)
            self.env_mean_by_n[n] = np.nanmean(self.mats, axis=0)
            self.log_mats = np.stack([self.log_envelopes[(n, r)] for r in range(self.replicates)], axis=0)
            self.log_env_mean_by_n[n] = np.nanmean(self.log_mats, axis=0)

        df_p = pd.DataFrame(rows_p)
        self.agg_p_lower = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_lower"].mean()
        self.agg_p_upper = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_upper"].mean()
        self.agg_p_range = df_p.groupby(["n", "p", "alpha"], as_index=False)["S_range"].mean()
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
            for n in self.n_values:
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
            xs = np.log(sub["n"].to_numpy(dtype=float))
            ys = np.log(sub[key].to_numpy(dtype=float))

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


def visualize(sample_text, inverse_weights=False, show=True, reducer_type='tsne', perplexity=30, seed=0, df_edges=pd.DataFrame()):
    
    embeds, inputs = get_embeds(sample_text, tokenizer, model, returns_tokenized=True)
    input2embed = pd.DataFrame({'inputs': inputs, 'embeds': embeds.tolist()})

    if reducer_type == 'umap':
        reducer = umap.UMAP(
            random_state=139892,
            n_neighbors=20
        )
    elif reducer_type == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            verbose=1,
        )
    elif reducer_type == 'pca':
        reducer = PCA(
            n_components=2
        )
    else:
        assert False, 'reducer type is invalid'
        sample_W
    if show:
        embeddings = reducer.fit_transform(embeds)
    
        fig = px.scatter(
            embeddings,
            size_max=5,
            size=[0.3] * embeddings.shape[0],
            x=0,
            y=1,
            color=input2embed['inputs'],  # Shorten the legend text
            hover_name=input2embed['inputs'].apply(
                lambda x: x[:70]),  # Show full text on hover
            #   labels={'type_of_cluster': 'Shortened Message'} 
            width=1000,
            height=800       
        )
        
        for row_idx, row in tqdm(df_edges.iterrows()):
            i = row['row']
            j = row['col']
            weight = row['weight']
            fig.add_shape(
                type='line',
                x0=embeddings[i,0], y0=embeddings[i,1],
                x1=embeddings[j,0], y1=embeddings[j,1],
                line=dict(color='gray', width=weight / 1000),
                layer='below'
            )


        fig.show()

    return None


def log_scale(data, grid):
    if data.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    m = data.size                  
    idx = np.clip((np.ceil(grid * m).astype(int) - 1), 0, m - 1)
    return data[idx]




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


class IdentityTransformer:
    def fit_transform(self, X):
        return X


def get_embeds_tsne(text, tokenizer, model, returns_tokenized=False, reducer_type='pca', raw_input=False):
    if  returns_tokenized:
        embeds, tokens = get_embeds(text, tokenizer, model, returns_tokenized=returns_tokenized, raw_input=raw_input)
    else:
        embeds = get_embeds(text, tokenizer, model, returns_tokenized=returns_tokenized, raw_input=raw_input)
        
    if reducer_type == 'umap':
        reducer = umap.UMAP(
            random_state=139892,
            n_neighbors=20
        )
    elif reducer_type == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=0,
            verbose=1,
        )
    elif reducer_type == 'pca':
        reducer = PCA(
            n_components=2
        )
    elif reducer_type == 'none':
        reducer = IdentityTransformer()
    else:
        assert False, 'reducer type is invalid'

    Y = reducer.fit_transform(embeds)
    

    return Y, tokens


def calculate_df_edges(tokens, mst_lengths):
    rows, cols = mst_lengths.nonzero()
    rows_list = []
    for idx_edge in range(len(rows)):
        row_idx = rows[idx_edge]
        col_idx = cols[idx_edge]
        rows_list.append({
            'row': row_idx,
            'col': col_idx,
            'weight': mst_lengths[row_idx, col_idx],
            'token_first': tokens[row_idx],
            'token_second': tokens[col_idx]
        })

    df_edges = pd.DataFrame(rows_list)
    
    return df_edges 


def draw_html(tokens_and_weights, cmap=plt.get_cmap("bwr"), display=True,
              token_template="""<span style="background-color: {color_hex}">{token}</span>""",
              font_style="font-size:14px;"
             ):
    
    def get_color_hex(weight):
        rgba = cmap(1. / (1 + np.exp(float(weight))), bytes=True)
#        rgba = cmap(float(weight), bytes=True)
        return '#%02X%02X%02X' % rgba[:3]
    
    tokens_html = [
        token_template.format(token=token, color_hex=get_color_hex(weight))
        for token, weight in tokens_and_weights
    ]
    
    
    raw_html = """<p style="{}">{}</p>""".format(font_style, ' '.join(tokens_html))
    if display:
        display_html(HTML(raw_html))
        
    return raw_html


def visualize_text(text, tokenizer, model, display=True):
    df_edges, tokens = get_prompt(text, tokenizer, model)
    df_row = df_edges.groupby('row')[['weight', 'quantile']].mean()
    df_col = df_edges.groupby('col')[['weight', 'quantile']].mean()
    df_concat = pd.concat([df_row, df_col]).reset_index()
    df_mean = df_concat.groupby('index').mean()
    tokens_and_weights = list(zip(tokens, (df_mean['quantile'].values.tolist() - df_mean['quantile'].median()) / 100))
    draw_html(tokens_and_weights, display=display)
    df_mean['token'] = tokens

    return df_mean


def get_target_token(token1, token2, second_minimum):
#    print(token2)
    if second_minimum[token1] > second_minimum[token2]:
        return token1
    return token2


def get_indices(df_edges, embeds, method='mst', q_lower=0.9, q_upper=1.0):  
    q_upper = df_edges['weight'].quantile(q_upper)
    q_lower = df_edges['weight'].quantile(q_lower)
    
    row_mean = df_edges.groupby("row")[['weight']].mean()
    col_mean = df_edges.groupby("col")[['weight']].mean()
    
    df_edges = df_edges.set_index("col").join(
        col_mean.rename(columns={'weight': 'col_weight_mean'})
    ).reset_index().set_index('row').join(
        row_mean.rename(columns={'weight': 'row_weight_mean'})
    ).reset_index()

    df_medium = df_edges.query("@q_upper > weight > @q_lower")   

    if method == 'second_min':
        second_minimum = calculate_second_min(embeds)
        indices = df_medium.apply(lambda x: get_target_token(x['row'], x['col'], second_minimum), axis=1).values.tolist()
    elif method == 'all':
        indices = df_medium['row'].values.tolist() + df_medium['col'].values.tolist()
        indices = sorted(list(set(indices)))
    elif method == 'mst':
        indices = df_medium.apply(lambda x: x['row'] if x['row_weight_mean'] > x['col_weight_mean'] else x['col'], axis=1).values.tolist()
    else:
        assert False, "invalid method value, method value must be in ['all', 'mst', 'second_min'] "

    return indices


def prompt_by_size(n2texts):
    nlist = list(n2texts.keys())
    top_texts = len(n2texts[nlist[0]])
    prompt = f"""
    You will see {len(nlist)} groups text. Each group of text consist of {top_texts} texts separated by \n \n and contains approximatly equal texts lenght.
    Each group separated by "=" * 100 
    In this texts you will see some tokens which highlighted by upper case and borders with |.
    You have to find the distinguishing features between texts lenght in highlighted tokens.

    
    """
    
    for n, texts in n2texts.items():
        prompt += "\n\n".join(texts) + "\n\n" + "=" * 100 + "\n\n"

    return prompt
    

def get_prompt_new(
    texts,
    tokenizer,
    model,
    nlist=[2 ** i for i in range(5, 10)],
    method='all',
    randomized_indices=True,
    q_lower=0.9,
    q_upper=1.0
):
    """
    texts: List[str]
    tokenizer: tokenizer
    model: llm 
    nlist: sizes of texts
    method: method to choose tokens. must be in ['mst', 'all', 'second_min']
    ===============================
    return 
    str - prompt
    """
    n2texts = defaultdict(list)
    top_texts = len(texts)
    stats_by_tokens = dict()
    assert method in ['mst', 'all', 'second_min']
    final_prompt = f"""
    You will see {top_texts} groups text. Each group of text consist of 5 texts separated by \n \n. Each group separated by "=" * 100 
    In this texts you will see some tokens which highlighted by upper case and borders with |.
    You have to find the distinguishing features in the highlighted words from the words that are not highlighted.
    """

    for idx in range(top_texts):
        text = texts[idx]
        embeds, tokens = get_embeds_tsne(text, tokenizer, model, returns_tokenized=True, reducer_type='none')
        for n in nlist:
            start_sentence_count = 0
            count_dot = 0
            assert n <= embeds.shape[0], f"Number of tokens in text is less {n}, embeds.shape is {embeds.shape} nlist={nlist}" 
            if randomized_indices:
                random_indices = np.random.choice(embeds.shape[0], size=n)
            else:
                random_indices = [i for i in range(n)]
            # print("before get_mst_edge_lengths")
            mst_lengths = get_mst_edge_lengths(embeds[random_indices, :], return_matrix=True)
            # print("after get_mst_edge_lengths")
            selected_tokens = [tokens[token_idx] for token_idx in random_indices]
            
            df_edges = calculate_df_edges(selected_tokens, mst_lengths)

            indices = get_indices(
                df_edges,
                method=method,
                q_lower=q_lower,
                q_upper=q_upper,
                embeds=embeds[random_indices, :]
            )
            result_text = ''
            all_medium_tokens = []
            all_tokens = []
            
            for token_idx, token in enumerate(tokens[:n]):
                if token_idx not in indices:
                    result_text += token
                else:
                    result_text += '|' + token.upper() + '|'
                    all_medium_tokens.append(token)
                    start_sentence_count += 1 if token_idx > 0 and tokens[token_idx - 1] == '.' else 0

                count_dot += 1 if token == '.' else 0
                
                all_tokens.append(token)
                
                stats_by_tokens[(idx, n)] = [all_tokens, all_medium_tokens, start_sentence_count, count_dot]

            n2texts[n].append(result_text)
            final_prompt += "\n" + "\n" + f"{result_text}" + "\n" + "\n " 
        final_prompt += "\n" + "\n" + "=" * 100 + "\n" + "\n"

    return final_prompt, all_medium_tokens, prompt_by_size(n2texts), n2texts, all_tokens, n2texts, stats_by_tokens


def get_answer(prompt):
    try:
        response = requests.post(
          url="https://openrouter.ai/api/v1/chat/completions",
          headers={
            "Authorization": "Bearer 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",
            "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
          data=json.dumps({
            "model": "google/gemini-2.5-pro", # Optional
            "messages": [
              {
                "role": "user",
                "content": prompt
              }
            ]
          })
        )
        content = response.json()['choices'][0]['message']['content']
    except:
        time.sleep(3)
        response = requests.post(
          url="https://openrouter.ai/api/v1/chat/completions",
          headers={
            "Authorization": "Bearer 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",
            "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
          data=json.dumps({
            "model": "google/gemini-2.5-pro", # Optional
            "messages": [
              {
                "role": "user",
                "content": prompt
              }
            ]
          })
        )
        content = response.json()['choices'][0]['message']['content']
        
    return content


def calculate_mean_len_token(tokens):
    tokens_len = [len(token) for token in tokens]
    return sum(tokens_len) / len(tokens), np.std(tokens_len)


def get_stats(filename, stats_by_tokens, number_texts):
    all_tokens_from_all_texts = []
    all_medium_tokens_from_all_texts = []
    n2all_dict = defaultdict(list)
    n2med_dict = defaultdict(list)
    start_stats_by_n = defaultdict(list)
    all_stats_by_n = defaultdict(list)

    for text_idx in range(number_texts):
        all_tokens, all_medium_tokens, start_sentence_count, dot_count = stats_by_tokens[(text_idx, 512)]
        all_tokens_from_all_texts.extend(all_tokens)
        all_medium_tokens_from_all_texts.extend(all_medium_tokens)


    for text_idx in range(number_texts):
        for n in range(5, 10):
            all_tokens, all_medium_tokens, start_sentence_count, dot_count = stats_by_tokens[(text_idx, 2 ** n)]
            n2all_dict[n].extend(all_tokens)
            n2med_dict[n].extend(all_medium_tokens)
            start_stats_by_n[n].append(
                start_sentence_count / len(all_medium_tokens)
            )
            all_stats_by_n[n].append(
                dot_count / len(all_tokens)
            )

    med_stat = []
    all_stat = []
    for n in range(5, 10):
        med_stat.append(np.mean(start_stats_by_n[n]))
        all_stat.append(np.mean(all_stats_by_n[n]))

    get_mean_token_plots(
        filename,
        all_tokens_from_all_texts,
        all_medium_tokens_from_all_texts,
        n2all_dict,
        n2med_dict
    )   

    ns = [2 ** i for i in range(5, 10)]
    plt.plot(ns, all_stat, label="all text")
    plt.plot(ns, med_stat, label="highlighted text")
    plt.xlabel("Number of tokens")
    plt.ylabel("Mean start stat")
    plt.title("Mean start stat by group")
    plt.legend()
    plt.savefig(f"figures/stat_start_count/{filename.split('/')[1]}.png")
    plt.show()
    print("get_stats is done")


def calculate_simple_stats(
    filename,
    dict_all,
    n_all_tokens,
    dict_medium,
    n_med_tokens
):
    pvalues_list = []
    tokens_list = []
    count_all_list = []
    count_medium_list = []
    for idx, (token_medium, count_medium) in enumerate(dict_medium.items()):
        count_all = dict_all[token_medium]
        count_all_list.append(count_all / n_all_tokens)
        count_medium_list.append(count_medium / n_med_tokens)
        stat, p_value  = test_proportions_2indep(
            count1=count_all,
            nobs1=n_all_tokens,
            count2=count_medium,
            nobs2=n_med_tokens,
            alternative='two-sided'
        )
#        if p_value < 0.05:
#            print(f"token = {token_medium}, pvalue = {p_value}")
        pvalues_list.append(p_value)
        tokens_list.append(token_medium)
        if idx > 30:
            break

    with open(f"data/strange_tokens/{filename.split('/')[1]}.pickle", "wb") as fd:
        pickle.dump([pvalues_list, tokens_list, count_all_list, count_medium_list], fd)

    print("calculate simple stats plots done!")

    
def get_mean_token_plots(
    filename,
    all_tokens_from_all_texts,
    all_medium_tokens_from_all_texts, 
    n2all_dict,
    n2med_dict
):
    n_all_tokens = len(all_tokens_from_all_texts)
    top_all_tokens = sorted(Counter(all_tokens_from_all_texts).items(), reverse=True, key=lambda x: x[1])
    n_med_tokens = len(all_medium_tokens_from_all_texts)
    top_med_tokens = sorted(Counter(all_medium_tokens_from_all_texts).items(), reverse=True, key=lambda x: x[1])
    
    dict_all = defaultdict(int, **dict(top_all_tokens))
    dict_medium = defaultdict(int, **dict(top_med_tokens))
    
    all_means = []
    all_stds = []
    med_means = []
    med_stds = []
    ns = []
    for n in range(5, 10):
        all_mean, all_std = calculate_mean_len_token(n2all_dict[n])
        med_mean, med_std = calculate_mean_len_token(n2med_dict[n])
        all_means.append(all_mean)
        all_stds.append(all_std)
        med_means.append(med_mean)
        med_stds.append(med_std)
        ns.append(2 ** n)
    #    print(f"all tokens: n = {2 ** n}", calculate_mean_len_token(n2all_dict[n]))
    #    print(f"medium tokens: n = {2 ** n}", calculate_mean_len_token(n2med_dict[n]))
    
    calculate_simple_stats(
        filename,
        dict_all,
        n_all_tokens,
        dict_medium,
        n_med_tokens
    )

    print("get mean token plots done!")
    plt.plot(ns, all_means, label="all text")
    plt.plot(ns, med_means, label="highlighted text")
    plt.xlabel("Number of tokens")
    plt.ylabel("Mean len of token")
    plt.title("Mean len of token by group")
    plt.savefig(f"figures/mean_token_len/mean_{filename.split('/')[1]}.png")
    plt.show()
    plt.plot(ns, all_stds, label="all text")
    plt.plot(ns, med_stds, label="highlighted text")
    plt.xlabel("Number of tokens")
    plt.ylabel("Std len of token")
    plt.title("Std len of token by group")
    plt.savefig(f"figures/mean_token_len/std_{filename.split('/')[1]}.png")
    plt.show()

from collections import defaultdict, Counter


def get_prompt(
    text,
    tokenizer,
    model,
    limit=None
):
    if limit:
        text = ''.join(tokenizer.tokenize(text)[:limit]).replace('▁', ' ')
    embeds, tokens = get_embeds_tsne(text, tokenizer, model, returns_tokenized=True, reducer_type='none')
    mst_lengths = get_mst_edge_lengths(embeds, return_matrix=True)
    df_edges = calculate_df_edges(tokens, mst_lengths)
    df_edges['quantile'] = (pd.qcut(df_edges['weight'], q=99).rank(pct=True) * 100).apply(int)

    return df_edges, tokens


def get_dot_index(dot_indices, x):
    return  x - np.array(dot_indices)[np.argmin((x - np.array(dot_indices))[x - np.array(dot_indices) >= 0])]
    

def show_prob(text, tokenizer, model):
    embeds, tokens = get_embeds(text, tokenizer, model, returns_tokenized=True, last_hidden_state=None)
    
    tokens_ids = [tokenizer.encode(token)[1] if len(tokenizer.encode(token)) == 2 else tokenizer.encode(token)[0] for token in tokens]
    
    assert len(tokens_ids) == embeds.shape[0], f"len(tokens_ids) = {len(tokens_ids)}, embeds.shape[0] = {embeds.shape[0]}"
    probs = []
    # print(tokens_ids)
    for i in range(len(tokens_ids) - 1):
        prob = softmax(embeds[i, :])[tokens_ids[i + 1]]
        # print(prob)
#         print( softmax(embeds[i, :]))
        probs.append(prob)

    return probs
