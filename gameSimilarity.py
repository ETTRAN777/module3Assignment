"""
RAWG Game Similarity (Top-10 per Query)
---------------------------------------
- Uses cached dataset if available and large enough
- Otherwise collects a dataset of games from RAWG (configurable section/subset)
- Encodes numeric + genre features
- Computes cosine similarity
- Prints Top 10 most similar games for each query (query not included)
- Saves CSVs for your Medium figures/tables

Setup:
  pip install requests scikit-learn pandas python-dotenv

Secrets:
  Create a file named 'secrets.env' in the same folder with:
    RAWG_API_KEY=your_rawg_api_key_here

Run:
  python gameSimilarity.py

Customize:
  - Change QUERY_TITLES below
  - Choose DATASET_SOURCE = "all_time_top" | "popular" | "custom"
  - If using "custom", edit CUSTOM_PARAMS
"""

import os
import re
import ast
import time
import math
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# Load RAWG_API_KEY from secrets.env if python-dotenv is available
# --------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    if os.path.exists("secrets.env"):
        load_dotenv("secrets.env")
    else:
        load_dotenv()
except Exception:
    pass

# --------------------------------
# CONFIGURATION
# --------------------------------
BASE_URL = "https://api.rawg.io/api/games"

DATASET_SOURCE = "all_time_top"  # "all_time_top" | "popular" | "custom"

CUSTOM_PARAMS = {
    # Example custom parameters if desired
    # "ordering": "-added",
    # "genres": "indie",
    # "dates": "2023-01-01,2023-12-31",
    # "platforms": "4",
}

DATASET_SIZE = 500
REQUEST_DELAY_SEC = 0.1

QUERY_TITLES = ["The Witcher 3: Wild Hunt","Red Dead Redemption 2","Celeste","Hollow Knight"]

ROW_VIEW_COLS = ["name", "similarity", "rating", "metacritic", "playtime", "ratings_count", "genres_list"]

API_KEY = (os.getenv("RAWG_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("RAWG_API_KEY not found. Add it to secrets.env or set it in your environment.")

CACHE_DIR = "outputs"
CACHE_PATH = os.path.join(CACHE_DIR, "games_dataset.csv")


# --------------------------------
# HELPERS
# --------------------------------
def safe_stem(name: str) -> str:
    """
    Make a Windows-safe, compact filename stem from a game title.
    Removes reserved characters and normalizes whitespace.
    """
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', name)
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-]', '', s)
    s = s.strip('._')
    return s[:80] or "game"


def get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET JSON with simple retry/backoff; auto-injects API key."""
    params = dict(params or {})
    params["key"] = API_KEY
    backoff = 0.5
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(backoff)
            backoff *= 2
    return {}


def _normalize_field(value, default=0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return value
    try:
        return float(value)
    except Exception:
        return default


def _to_name_list(objs: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not objs:
        return []
    return [o.get("name", "").strip() for o in objs if o.get("name")]


def _coerce_genres_list_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'genres_list' is a list for every row (handles CSV-loaded strings).
    """
    if "genres_list" not in df.columns:
        df["genres_list"] = [[] for _ in range(len(df))]
        return df

    def fix_cell(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            x = x.strip()
            # Try to parse "['Action', 'RPG']" or "[]"
            if (x.startswith("[") and x.endswith("]")) or (x.startswith("(") and x.endswith(")")):
                try:
                    val = ast.literal_eval(x)
                    return [str(s) for s in val] if isinstance(val, (list, tuple)) else []
                except Exception:
                    pass
            # Fallback: split by comma if looks like "Action, RPG"
            if "," in x:
                return [part.strip() for part in x.split(",") if part.strip()]
            if x == "" or x.lower() == "nan":
                return []
            return [x]  # single genre string
        return []
    df["genres_list"] = df["genres_list"].apply(fix_cell)
    return df


# --------------------------------
# DATA COLLECTION
# --------------------------------
def fetch_all_time_top(n: int = 250) -> List[Dict[str, Any]]:
    """Approximate RAWG 'All-time top' section by highest Metacritic, excluding DLC/additions."""
    results = []
    page = 1
    page_size = 40
    while len(results) < n:
        data = get_json(
            BASE_URL,
            {
                "page": page,
                "page_size": page_size,
                "ordering": "-metacritic",
                "exclude_additions": "true",
            },
        )
        batch = data.get("results", []) or []
        if not batch:
            break
        results.extend(batch)
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

    seen = set()
    deduped = []
    for g in results:
        gid = g.get("id")
        if gid and gid not in seen:
            deduped.append(g)
            seen.add(gid)
        if len(deduped) >= n:
            break
    return deduped


def fetch_popular_games(n: int = 200) -> List[Dict[str, Any]]:
    """Fetch well-rated/popular games by ordering -rating."""
    results = []
    page = 1
    page_size = 40
    while len(results) < n:
        data = get_json(
            BASE_URL,
            {
                "page": page,
                "page_size": page_size,
                "ordering": "-rating",
            },
        )
        batch = data.get("results", []) or []
        if not batch:
            break
        results.extend(batch)
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

    seen = set()
    deduped = []
    for g in results:
        gid = g.get("id")
        if gid and gid not in seen:
            deduped.append(g)
            seen.add(gid)
        if len(deduped) >= n:
            break
    return deduped


def fetch_custom(n: int, custom_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch games with custom RAWG query parameters."""
    results = []
    page = 1
    page_size = 40
    base_params = dict(custom_params or {})
    while len(results) < n:
        params = {"page": page, "page_size": page_size}
        params.update(base_params)
        data = get_json(BASE_URL, params)
        batch = data.get("results", []) or []
        if not batch:
            break
        results.extend(batch)
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

    seen = set()
    deduped = []
    for g in results:
        gid = g.get("id")
        if gid and gid not in seen:
            deduped.append(g)
            seen.add(gid)
        if len(deduped) >= n:
            break
    return deduped


def build_dataframe(raw_games: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for g in raw_games:
        rows.append(
            {
                "id": g.get("id"),
                "name": g.get("name", ""),
                "slug": g.get("slug", ""),
                "rating": _normalize_field(g.get("rating"), 0),
                "ratings_count": _normalize_field(g.get("ratings_count"), 0),
                "metacritic": _normalize_field(g.get("metacritic"), 0),
                "playtime": _normalize_field(g.get("playtime"), 0),
                "added": _normalize_field(g.get("added"), 0),
                "suggestions_count": _normalize_field(g.get("suggestions_count"), 0),
                "genres_list": _to_name_list(g.get("genres")),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["genres_list"] = df["genres_list"].apply(lambda x: x if isinstance(x, list) else [])
    return df


# --------------------------------
# FEATURE ENCODING
# --------------------------------
def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = [
        "rating",
        "ratings_count",
        "metacritic",
        "playtime",
        "added",
        "suggestions_count",
    ]

    mlb = MultiLabelBinarizer(sparse_output=False)
    genre_matrix = mlb.fit_transform(df["genres_list"])
    genre_cols = [f"genre::{g}" for g in mlb.classes_]

    scaler = StandardScaler()
    df_num_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    # Apply custom numeric weights to reduce popularity dominance
    weights = {
        "rating": 1.0,
        "ratings_count": 0.5,
        "metacritic": 1.0,
        "playtime": 1.0,
        "added": 0.4,
        "suggestions_count": 0.4,
    }
    for col, w in weights.items():
        if col in df_num_scaled.columns:
            df_num_scaled[col] *= w

    # Combine numeric + genres
    features = pd.concat(
        [df_num_scaled, pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)],
        axis=1,
    )

    #boost genre features a bit to emphasize design similarity
    features.loc[:, features.columns.str.startswith("genre::")] *= 1.5

    # Drop any all-zero columns (edge cases)
    nonzero_cols = features.columns[(features != 0).any(axis=0)]
    features = features[nonzero_cols]

    return features, numeric_cols, genre_cols



# --------------------------------
# SIMILARITY + RANKING
# --------------------------------
def compute_similarity(features: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(features.values)
    return pd.DataFrame(sim, index=features.index, columns=features.index)


def find_best_dataset_match(df: pd.DataFrame, query: str) -> Optional[int]:
    q = query.lower().strip()
    mask = df["name"].str.lower().str.contains(q, regex=False)
    candidates = df[mask]
    if candidates.empty:
        return None
    idx = candidates.sort_values(by=["rating", "ratings_count"], ascending=False).index[0]
    return int(idx)


def top_k_similar(df: pd.DataFrame, sim: pd.DataFrame, anchor_idx: int, k: int = 10) -> pd.DataFrame:
    scores = sim.iloc[anchor_idx].copy()
    scores.loc[anchor_idx] = -1.0  # exclude the query itself
    top_idx = scores.sort_values(ascending=False).head(k).index
    out = df.loc[top_idx, ["name", "rating", "metacritic", "playtime", "ratings_count", "genres_list"]].copy()
    out.insert(1, "similarity", [round(scores[i], 4) for i in top_idx])
    return out.reset_index(drop=True)


def make_query_row(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    row = df.loc[[idx], ["name", "rating", "metacritic", "playtime", "ratings_count", "genres_list"]].copy()
    row.insert(1, "similarity", 1.0)
    return row


# --------------------------------
# MAIN
# --------------------------------
def main():
    print(f"### Dataset size requested: {DATASET_SIZE}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Try to use cached dataset if sufficient
    use_cache = False
    if os.path.exists(CACHE_PATH):
        try:
            cached_df = pd.read_csv(CACHE_PATH)
            if len(cached_df) >= DATASET_SIZE:
                cached_df = cached_df.head(DATASET_SIZE).copy()
                cached_df = _coerce_genres_list_col(cached_df)
                df = cached_df.reset_index(drop=True)
                use_cache = True
                print(f"### Using cached dataset: {CACHE_PATH} (rows: {len(df)})")
            else:
                print(f"### Cache too small ({len(cached_df)} rows) < DATASET_SIZE; will fetch from RAWG.")
        except Exception as e:
            print(f"X Failed to read cache '{CACHE_PATH}': {e}. Will fetch from RAWG.")

    # Fetch from RAWG only if no adequate cache
    if not use_cache:
        print(f"### Fetching dataset from RAWG ({DATASET_SOURCE})...")
        if DATASET_SOURCE == "all_time_top":
            raw_games = fetch_all_time_top(DATASET_SIZE)
        elif DATASET_SOURCE == "popular":
            raw_games = fetch_popular_games(DATASET_SIZE)
        elif DATASET_SOURCE == "custom":
            raw_games = fetch_custom(DATASET_SIZE, CUSTOM_PARAMS)
        else:
            raise ValueError("Invalid DATASET_SOURCE. Use 'all_time_top', 'popular', or 'custom'.")

        df = build_dataframe(raw_games)
        if df.empty:
            raise RuntimeError("No games fetched. Check your API key, network, or dataset params.")

        print(f"### Collected {len(df)} games.")
        # Save/overwrite the cache to reflect this dataset
        df.to_csv(CACHE_PATH, index=False)
        print(f"### Saved dataset cache: {CACHE_PATH}")

    print("### Building feature matrix...")
    features, numeric_cols, genre_cols = build_feature_matrix(df)
    print(f"   Numeric features: {numeric_cols}")
    print(f"   Genre features: {len(genre_cols)} columns")

    print("### Computing cosine similarity...")
    sim = compute_similarity(features)

    # Align df indices with features
    df = df.loc[features.index].reset_index(drop=True)
    sim.index = df.index
    sim.columns = df.index

    # Also save features (optional cache of features)
    features_path = os.path.join(CACHE_DIR, "features_matrix.csv")
    features.to_csv(features_path, index=False)
    print(f"### Saved features to {features_path}")

    for query in QUERY_TITLES:
        best_idx = find_best_dataset_match(df, query)
        print("\n" + "—" * 80)
        if best_idx is None:
            search_hits = get_json(BASE_URL, {"search": query, "page_size": 5}).get("results", []) or []
            suggestions = ", ".join([h.get("name", "") for h in search_hits]) or "None"
            print(f"X '{query}' not found in dataset of {len(df)} games.")
            print(f"   Suggestions: {suggestions}")
            continue

        anchor_name = df.loc[best_idx, "name"]

        query_row = make_query_row(df, best_idx)[ROW_VIEW_COLS]
        print(f"### Query game info: {anchor_name}")
        with pd.option_context("display.max_colwidth", 80):
            print(query_row.to_string(index=False))

        top10 = top_k_similar(df, sim, best_idx, k=10)[ROW_VIEW_COLS]

        print(f"\n### Top 10 most similar to: {anchor_name}")
        with pd.option_context("display.max_colwidth", 80):
            print(top10.to_string(index=False))

        stem = safe_stem(anchor_name)
        path_top10 = os.path.join(CACHE_DIR, f"top10_{stem}.csv")

        try:
            if top10.empty:
                print(f"### No neighbors to save for '{anchor_name}'")
            else:
                top10.to_csv(path_top10, index=False)
                print(f"### Saved: {path_top10}")
        except OSError as e:
            print(f"X Could not write {path_top10}: {e}")

    print("\n### Done. See CSVs in the outputs/ folder.")


if __name__ == "__main__":
    main()
