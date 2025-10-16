import os, pandas as pd, numpy as np, joblib
from scipy.sparse import load_npz

PROC = "data/processed"
RAW = "data/raw/ml-32m"

UI_TRAIN = load_npz(os.path.join(PROC, "user_item_train.npz")).tocsr()
ITEM_MATRIX = UI_TRAIN.T.tocsr()
KNN = joblib.load(os.path.join("models", "itemknn.joblib"))

user_map = pd.read_csv(os.path.join(PROC, "mappings", "user_map.csv"))
item_map = pd.read_csv(os.path.join(PROC, "mappings", "item_map.csv"))
uid_by_userId = dict(zip(user_map.userId, user_map.uid))
movieId_by_iid = dict(zip(item_map.iid, item_map.movieId))

title_by_movieId = {}
movies_csv = os.path.join(RAW, "movies.csv")
if os.path.exists(movies_csv):
    mdf = pd.read_csv(movies_csv, usecols=["movieId","title"])
    title_by_movieId = dict(zip(mdf.movieId, mdf.title))

pop_scores = np.asarray(UI_TRAIN.sum(axis=0)).ravel()
popular_iids = np.argsort(-pop_scores)

def _seen(uid:int):
    s,e = UI_TRAIN.indptr[uid], UI_TRAIN.indptr[uid+1]
    return set(UI_TRAIN.indices[s:e])

def recommend_for_user(user_id:int, k:int=10):
    if user_id not in uid_by_userId:
        rec_iids = popular_iids[:k].tolist()
    else:
        uid = int(uid_by_userId[user_id])
        seen = _seen(uid)
        scores = {}
        dists, idxs = KNN.kneighbors(ITEM_MATRIX[list(seen)], n_neighbors=50, return_distance=True)
        sims = 1.0 - dists
        for row_idxs, row_sims in zip(idxs, sims):
            for j, s in zip(row_idxs, row_sims):
                if j in seen: 
                    continue
                scores[j] = scores.get(j, 0.0) + float(s)
        rec_iids = [j for j,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:k]]
        if not rec_iids:
            rec_iids = popular_iids[:k].tolist()

    out = []
    for iid in rec_iids:
        mid = int(movieId_by_iid[int(iid)])
        out.append({"movieId": mid, "title": title_by_movieId.get(mid)})
    return out

def sample_user_id() -> int:
    return int(user_map.userId.sample(1).iloc[0])