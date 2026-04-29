import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


sns.set_style("darkgrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PADBEN pickle with PADBEN notebook logic")
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=Path("data/padben_sentence-pair-task1.pickle"),
        help="Path to PADBEN pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures"),
        help="Directory where output figures are saved",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=0.9,
        help="Upper x-limit for param_value filter",
    )
    parser.add_argument(
        "--min-count-ratio",
        type=float,
        default=0.032,
        help="Minimum class frequency ratio for plotting",
    )
    return parser.parse_args()


def validate_payload(payload: object) -> tuple:
    payload = (
        payload['df_en'],
        payload['d_hat_stats_df_list'], 
        payload['d_energy_range_stats_df_list'],
        payload['d_energy_upper_stats_df_list'],
        payload['d_energy_lower_stats_df_list'],
        payload['dfs_list']
    )
    if not isinstance(payload, (list, tuple)) or len(payload) != 6:
        raise ValueError(
            "Unexpected pickle layout. Expected list/tuple with 6 elements: "
            "[df_en, d_hat_stats_df_list, d_energy_range_stats_df_list, "
            "d_energy_upper_stats_df_list, d_energy_lower_stats_df_list, dfs_list]."
        )
    return tuple(payload)


def build_joined_metric_df(
    df_plot: pd.DataFrame,
    stats_list: list,
    metric_col: str,
    limit: int,
    min_count_plot: float,
    xlim: float,
) -> pd.DataFrame:
    model2count = df_plot.iloc[:limit, :].groupby("model").count()[["text"]]
    models = model2count.query(f"text > {min_count_plot}").index.tolist()
    df_filter = df_plot.iloc[:limit, :]

    metric_frames = []
    for idx, metric_df in enumerate(stats_list):
        metric_df_local = metric_df.copy()
        metric_df_local["text"] = df_filter.iloc[idx, :]["text"]
        metric_frames.append(metric_df_local)

    metrics_concat = pd.concat(metric_frames)
    df_joined = metrics_concat.set_index("text").join(df_filter.set_index("text")).reset_index()
    df_joined = df_joined.query("model in @models")
    df_joined = df_joined.query(f"param_value < {xlim}")
    df_joined = df_joined.query(f"{metric_col} > 0")
    return df_joined


def compute_roc_auc_by_param(df_joined: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    roc_rows = []
    for param_value, grp in df_joined.groupby("param_value"):
        y_true = pd.to_numeric(grp["model"], errors="coerce")
        y_score = pd.to_numeric(grp[metric_col], errors="coerce")
        valid_mask = y_true.notna() & y_score.notna()
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]

        n_samples = int(valid_mask.sum())
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        if n_samples < 2 or n_pos == 0 or n_neg == 0:
            roc_auc = np.nan
        else:
            roc_auc = float(roc_auc_score(y_true, y_score))

        roc_rows.append({"param_value": float(param_value), "roc_auc": roc_auc})

    return pd.DataFrame(roc_rows).sort_values("param_value").reset_index(drop=True)


def build_text_dhat_with_phd_df(
    df_en: pd.DataFrame,
    d_energy_range_stats_df_list: list,
) -> pd.DataFrame:
    if "text" not in df_en.columns:
        raise ValueError("Column 'text' is required in df_en.")
    if "phd" not in df_en.columns:
        raise ValueError("Column 'phd' is required in df_en.")

    text_series = df_en["text"].reset_index(drop=True)
    if len(d_energy_range_stats_df_list) != len(text_series):
        raise ValueError(
            "Length mismatch: d_energy_range_stats_df_list and df_en rows must match. "
            f"Got {len(d_energy_range_stats_df_list)} vs {len(text_series)}."
        )

    rows = []
    for idx, stats_df in enumerate(d_energy_range_stats_df_list):
        stats_local = stats_df[["param_value", "d_hat"]].copy()
        stats_local["text"] = text_series.iloc[idx]
        rows.append(stats_local)

    long_df = pd.concat(rows, ignore_index=True)
    wide_df = long_df.pivot_table(
        index="text",
        columns="param_value",
        values="d_hat",
        aggfunc="first",
    ).reset_index()

    def _format_param_col(param_value: float) -> str:
        formatted = format(float(param_value), ".12g")
        formatted = formatted.replace("-", "m").replace(".", "_")
        return f"d_hat_p_{formatted}"

    wide_df.columns = ["text"] + [_format_param_col(col) for col in wide_df.columns[1:]]

    text_phd_df = df_en[["text", "phd"]].drop_duplicates(subset=["text"])
    result_df = wide_df.merge(text_phd_df, on="text", how="left")
    return result_df


def main() -> None:
    args = parse_args()

    if not args.pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {args.pickle_path}")

    with args.pickle_path.open("rb") as fd:
        payload = pickle.load(fd)

    (
        df_en,
        _d_hat_stats_df_list,
        d_energy_range_stats_df_list,
        d_energy_upper_stats_df_list,
        d_energy_lower_stats_df_list,
        _dfs_list,
    ) = validate_payload(payload)

    if "label" not in df_en.columns:
        raise ValueError(
            "Column 'label' is required in df_en. "
            f"Available columns: {list(df_en.columns)}"
        )

    df_plot = df_en.rename(columns={"label": "model"}).copy()
    df_plot["model"] = df_plot["model"].astype(str)
    number_of_texts = df_plot.shape[0]
    min_count_plot = max(1, number_of_texts * args.min_count_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.pickle_path.stem
    if stem.startswith("padben_"):
        suffix = stem[len("padben_") :]
    else:
        suffix = stem

    plot_specs = [
        ("d_energy_range", d_energy_range_stats_df_list),
        ("d_energy_upper", d_energy_upper_stats_df_list),
        ("d_energy_lower", d_energy_lower_stats_df_list),
    ]

    text_dhat_phd_df = build_text_dhat_with_phd_df(df_en, d_energy_range_stats_df_list)
    csv_path = args.output_dir / f"padben_{suffix}_d_energy_range_text_dhat_phd.csv"
    text_dhat_phd_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    roc_curves = {}

    for idx, (metric_name, stats_list) in enumerate(plot_specs):
        ax = axes[idx]
        df_joined = build_joined_metric_df(
            df_plot=df_plot,
            stats_list=stats_list,
            metric_col="d_hat",
            limit=number_of_texts,
            min_count_plot=min_count_plot,
            xlim=args.xlim,
        )

        model_order = sorted(df_joined["model"].dropna().astype(str).unique().tolist())
        for model_name in model_order:
            df_model = df_joined.query("model == @model_name")
            median_by_param = df_model.groupby("param_value")["d_hat"].median().sort_index()
            if not median_by_param.empty:
                ax.plot(median_by_param.index.values, median_by_param.values, label=model_name)

        ax.set_title(f"{metric_name}: median curve by model")
        ax.set_xlabel("param_value")
        ax.set_ylabel(metric_name)
        ax.legend(loc="best", fontsize=8)

        roc_curves[metric_name] = compute_roc_auc_by_param(df_joined, "d_hat")

    roc_ax = axes[3]
    for metric_name, roc_df in roc_curves.items():
        if not roc_df.empty:
            roc_ax.plot(roc_df["param_value"], roc_df["roc_auc"], label=metric_name)

    roc_ax.set_title("ROC-AUC by param_value")
    roc_ax.set_xlabel("param_value")
    roc_ax.set_ylabel("ROC-AUC")
    roc_ax.set_ylim(0.0, 1.0)
    roc_ax.axhline(0.5, linestyle="--", linewidth=1.2, color="gray", label="random (0.5)")
    if 'phd' in df_en:
        roc_ax.axhline(roc_auc_score(df_en['label'], df_en['phd']), linestyle="--", linewidth=1.2, color="red", label="PHD old estimation")
    roc_ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"PADBEN curves and ROC-AUC: {suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = args.output_dir / f"padben_{suffix}_combined.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined figure to: {output_path}")
    print(f"Saved text-dhat-phd CSV to: {csv_path} (shape={text_dhat_phd_df.shape})")


if __name__ == "__main__":
    main()
