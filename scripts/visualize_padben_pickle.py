import argparse
import pickle
from pathlib import Path

from phd_scale import plot_median_by_param_value


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
    if not isinstance(payload, (list, tuple)) or len(payload) != 6:
        raise ValueError(
            "Unexpected pickle layout. Expected list/tuple with 6 elements: "
            "[df_en, d_hat_stats_df_list, d_energy_range_stats_df_list, "
            "d_energy_upper_stats_df_list, d_energy_lower_stats_df_list, dfs_list]."
        )
    return tuple(payload)


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

    for obj_name, stats_list in plot_specs:
        output_base = args.output_dir / f"{obj_name}_{suffix}"
        plot_median_by_param_value(
            df_plot,
            stats_list,
            limit=number_of_texts,
            min_count_plot=min_count_plot,
            obj_name=obj_name,
            xlim=args.xlim,
            filename_save=str(output_base),
        )

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
