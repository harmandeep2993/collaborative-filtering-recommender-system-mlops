"""
build_matrix.py

Purpose:
- Turn raw ratings into sparse user–item matrices for training/validation.

What it does:
1) Load the largest file matching data/interim/ratings_sample_*.csv
2) Keep positive feedback (rating >= POS_THRESH)
3) Drop users with < MIN_EVENTS positives
4) Map raw ids -> integer indices (uid, iid)
5) Time split per user: last interaction -> validation, rest -> train
6) Save CSR matrices + id maps + metadata to data/processed/

Outputs:
- data/processed/user_item_train.npz
- data/processed/user_item_valid.npz
- data/processed/mappings/user_map.csv
- data/processed/mappings/item_map.csv
- data/processed/meta.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# -------- config --------
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
MIN_EVENTS = 5       # min positives per user to keep
POS_THRESH = 3.5     # rating threshold for a "positive"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------- pick input file --------
# Choose the ratings_sample_*.csv with the largest numeric suffix.
samples = [f for f in os.listdir(INTERIM_DIR) if f.startswith("ratings_sample_")]
assert samples, "No sample found. Run make_sample.py first."

def suffix_num(name: str) -> int:
    # ratings_sample_123.csv -> 123
    return int(name.split("_")[-1].split(".")[0])

sample_file = max(samples, key=suffix_num)
src_path = os.path.join(INTERIM_DIR, sample_file)
print("Load:", src_path)

# -------- load and filter --------
use_cols = ["userId", "movieId", "rating", "timestamp"]
df = pd.read_csv(src_path, usecols=use_cols)

# Keep only implicit positives
df = df[df["rating"] >= POS_THRESH].copy()

# Drop users with too few events
user_counts = df["userId"].value_counts()
good_users = user_counts[user_counts >= MIN_EVENTS].index
df = df[df["userId"].isin(good_users)].copy()

# -------- index encode --------
# Map raw ids to 0..N-1 to build compact matrices.
df["uid"] = df["userId"].astype("category").cat.codes
df["iid"] = df["movieId"].astype("category").cat.codes
n_users = int(df["uid"].max() + 1)
n_items = int(df["iid"].max() + 1)
print(f"Users={n_users} Items={n_items} Interactions={len(df)}")

# -------- time-based split per user --------
# For each user, send the last interaction to validation.
df = df.sort_values(["uid", "timestamp"])
valid_idx = df.groupby("uid").tail(1).index
valid_df = df.loc[valid_idx]
train_df = df.drop(valid_idx)

def to_csr(d: pd.DataFrame) -> csr_matrix:
    """
    Build a (n_users x n_items) CSR matrix with 1.0 for each observed interaction.
    """
    rows = d["uid"].to_numpy()
    cols = d["iid"].to_numpy()
    data = np.ones(len(d), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

# Build matrices
ui_train = to_csr(train_df)
ui_valid = to_csr(valid_df)

# -------- save outputs --------
# Matrices
save_npz(os.path.join(PROCESSED_DIR, "user_item_train.npz"), ui_train)
save_npz(os.path.join(PROCESSED_DIR, "user_item_valid.npz"), ui_valid)

# ID mappings for serving/training
maps_dir = os.path.join(PROCESSED_DIR, "mappings")
os.makedirs(maps_dir, exist_ok=True)

(df[["userId", "uid"]]
 .drop_duplicates()
 .sort_values("uid")
 .to_csv(os.path.join(maps_dir, "user_map.csv"), index=False))

(df[["movieId", "iid"]]
 .drop_duplicates()
 .sort_values("iid")
 .to_csv(os.path.join(maps_dir, "item_map.csv"), index=False))

# Metadata for traceability
meta = {
    "n_users": n_users,
    "n_items": n_items,
    "n_train": int(ui_train.nnz),  # nonzeros = interactions
    "n_valid": int(ui_valid.nnz),
    "pos_threshold": POS_THRESH,
    "min_events": MIN_EVENTS,
    "source_file": src_path,
}
with open(os.path.join(PROCESSED_DIR, "meta.json"), "w") as f:
    json.dump(meta, f)

print("Saved to data/processed")
print(meta)