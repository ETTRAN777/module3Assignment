# graphMaker.py
# Generate charts for a chosen subset size (N) of games_dataset.csv.
# All output filenames include the chosen N so you can compare 500 vs 10,000, etc.

import os
import ast
import glob
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ---------------- paths & logging ----------------
def script_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    dataset_csv = os.path.join(output_dir, "games_dataset.csv")
    features_csv = os.path.join(output_dir, "features_matrix.csv")
    return script_dir, output_dir, dataset_csv, features_csv

def log_env(script_dir, output_dir, dataset_csv, features_csv, chosen_size, use_sample):
    print("### graphMaker starting")
    print(f"### Script directory : {script_dir}")
    print(f"### Outputs directory: {output_dir}")
    print(f"### Dataset file     : {dataset_csv}")
    print(f"### Features file    : {features_csv}")
    print(f"### Chosen size (N)  : {chosen_size if chosen_size else 'auto'}")
    print(f"### Sampling         : {'random sample' if use_sample else 'head N'}")

# ---------------- helpers ----------------
def coerce_genres_list_col(df: pd.DataFrame) -> pd.DataFrame:
    if "genres_list" not in df.columns:
        df["genres_list"] = [[] for _ in range(len(df))]
        return df

    def fix_cell(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    val = ast.literal_eval(s)
                    return [str(v) for v in val] if isinstance(val, (list, tuple)) else []
                except Exception:
                    pass
            if "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            if s == "" or s.lower() == "nan":
                return []
            return [s]
        return []
    df["genres_list"] = df["genres_list"].apply(fix_cell)
    return df

def infer_current_run_size(features_csv: str, dataset_csv: str) -> int:
    """Prefer features_matrix.csv row count (current run), else fall back to games_dataset.csv row count."""
    if os.path.exists(features_csv):
        try:
            with open(features_csv, "r", encoding="utf-8") as f:
                # count lines excluding header
                n = sum(1 for _ in f) - 1
                if n > 0:
                    return n
        except Exception:
            pass
    if os.path.exists(dataset_csv):
        try:
            with open(dataset_csv, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
                return max(n, 0)
        except Exception:
            return 0
    return 0

def load_dataset_subset(dataset_csv: str, size: int, use_sample: bool) -> pd.DataFrame:
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    df = coerce_genres_list_col(df)

    total = len(df)
    size = min(size, total)
    if use_sample and size < total:
        df = df.sample(size, random_state=42).reset_index(drop=True)
    else:
        df = df.head(size).reset_index(drop=True)
    return df, size, total

# ---------------- charts ----------------
def plot_top10_similarity(csv_path: str, output_dir: str, N: int):
    df = pd.read_csv(csv_path)
    if "name" not in df.columns or "similarity" not in df.columns:
        print(f"X Skipping {csv_path}: required columns not found.")
        return

    stem = os.path.splitext(os.path.basename(csv_path))[0].replace("top10_", "")
    title = f"Top 10 Similar Games to {stem.replace('_', ' ')} ({N} games)"

    plt.figure(figsize=(9, 5))
    names = df["name"].tolist()[::-1]
    sims = df["similarity"].tolist()[::-1]
    plt.barh(names, sims)
    plt.xlabel("Cosine Similarity")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{stem}_top10_similarity_{N}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"### Saved chart: {out_path}")

def plot_genre_distribution(df_subset: pd.DataFrame, output_dir: str, N: int, top_k: int = 15):
    all_genres = [g for lst in df_subset["genres_list"] for g in lst]
    if not all_genres:
        print("X No genres found; skipping genre distribution.")
        return
    counts = Counter(all_genres).most_common(top_k)
    labels, values = zip(*counts)

    plt.figure(figsize=(9, 5))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Count")
    plt.title(f"Top {top_k} Genres in Dataset ({N} games)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"genre_distribution_{N}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"### Saved chart: {out_path}")

def plot_ratings_vs_metacritic(df_subset: pd.DataFrame, output_dir: str, N: int):
    if "rating" not in df_subset.columns or "metacritic" not in df_subset.columns:
        print("X Required columns not found in subset; skipping scatter.")
        return
    sub = df_subset.dropna(subset=["rating", "metacritic"])

    plt.figure(figsize=(6, 6))
    plt.scatter(sub["metacritic"], sub["rating"], alpha=0.5)
    plt.xlabel("Metacritic Score")
    plt.ylabel("User Rating (RAWG, 0–5 scale)")
    plt.title(f"Metacritic vs. User Rating ({N} games)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"ratings_vs_metacritic_{N}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"### Saved chart: {out_path}")

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Generate charts for a chosen dataset size N.")
    parser.add_argument("--size", type=int, default=None, help="Number of games to use from games_dataset.csv")
    parser.add_argument("--sample", action="store_true", help="Use a random sample of N rows instead of head N")
    args = parser.parse_args()

    script_dir, output_dir, dataset_csv, features_csv = script_paths()
    if not os.path.isdir(output_dir):
        print(f"X Outputs directory not found: {output_dir}")
        print("  Run your main similarity script first to generate outputs/")
        return

    # Decide N: user-provided > inferred from features > full dataset
    inferred = infer_current_run_size(features_csv, dataset_csv)
    chosen_N = args.size if args.size is not None else inferred if inferred > 0 else None
    if chosen_N is None or chosen_N <= 0:
        # fallback to entire dataset if nothing else is inferable
        with open(dataset_csv, "r", encoding="utf-8") as f:
            chosen_N = max(sum(1 for _ in f) - 1, 0)

    log_env(script_dir, output_dir, dataset_csv, features_csv, chosen_N, args.sample)

    # Load subset and generate dataset-wide charts
    try:
        df_subset, used_N, total_rows = load_dataset_subset(dataset_csv, chosen_N, args.sample)
    except FileNotFoundError as e:
        print(f"X {e}")
        return

    print(f"### Loaded subset: N={used_N} from total={total_rows}")
    plot_genre_distribution(df_subset, output_dir, used_N, top_k=15)
    plot_ratings_vs_metacritic(df_subset, output_dir, used_N)

    # Per-query Top-10 similarity charts (from CSVs already produced by main script)
    pattern = os.path.join(output_dir, "top10_*.csv")
    top10_files = glob.glob(pattern)
    print(f"### Found {len(top10_files)} Top-10 CSV file(s).")
    for csv_path in top10_files:
        plot_top10_similarity(csv_path, output_dir, used_N)

    print("### graphMaker finished")

if __name__ == "__main__":
    main()
