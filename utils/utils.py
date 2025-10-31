from typing import Tuple
import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)




def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def scale_fit_transform(train, val, test, method="standard"):
    if method == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(train)))
    else:
        scaler = StandardScaler()
    tr = scaler.fit_transform(train)
    return tr, scaler.transform(val), scaler.transform(test)


def _mutual_pairs(indices: np.ndarray) -> np.ndarray:
    N, k = indices.shape
    neigh_sets = [set(indices[i].tolist()) for i in range(N)]
    rows = np.repeat(np.arange(N), k)
    cols = indices.reshape(-1)
    mutual = np.fromiter((rows[t] in neigh_sets[cols[t]] for t in range(rows.shape[0])),
                         dtype=bool, count=rows.shape[0])
    return mutual


def build_knn_edges_cpu_with_cosine(x_np: np.ndarray, k: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safe cosine KNN graph with mutual edges + fallback"""
    N = x_np.shape[0]
    if N <= 1:
        edge_index = torch.arange(N, device=device).repeat(2, 1)
        edge_attr = torch.ones((N, 1), device=device)
        return edge_index, edge_attr

    k_eff = max(1, min(k, N - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="cosine", n_jobs=-1)
    nbrs.fit(x_np)
    dists, indices = nbrs.kneighbors(x_np, return_distance=True)

    # --- partial mutual trick (70% mutual + 30% random)
    mask_full = np.ones_like(indices.reshape(-1), dtype=bool)
    mask_mutual = _mutual_pairs(indices)
    mutual_mask = np.logical_or(mask_mutual, np.random.rand(len(mask_full)) < 0.3)

    rows = np.repeat(np.arange(N), k_eff)
    cols = indices.reshape(-1)
    dists_flat = dists.reshape(-1)
    rows, cols, dists_flat = rows[mutual_mask], cols[mutual_mask], dists_flat[mutual_mask]
    if len(rows) == 0:  # fallback if empty
        rows = np.repeat(np.arange(N), k_eff)
        cols = indices.reshape(-1)
        dists_flat = dists.reshape(-1)

    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long, device=device)
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    sim = 1.0 - torch.tensor(np.concatenate([dists_flat, dists_flat]), dtype=torch.float32, device=device)

    edge_index, _ = remove_self_loops(edge_index)
    num_existing = sim.shape[0]
    edge_index, _ = add_self_loops(edge_index)
    num_new = edge_index.shape[1] - num_existing

    if num_new > 0:
        sim = torch.cat([sim, torch.ones((num_new,), device=device)], dim=0)
    elif num_new < 0:
        sim = sim[:edge_index.shape[1]]

    edge_attr = sim.unsqueeze(1)
    return edge_index, edge_attr


def find_best_threshold(probs, y_true, mode="acc"):
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.2, 0.8, 61):
        p = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, p).ravel()
        prec = precision_score(y_true, p, zero_division=0)
        rec = recall_score(y_true, p, zero_division=0)
        acc = accuracy_score(y_true, p)
        spec = tn / (tn + fp + 1e-6)
        if mode == "acc": score = acc
        elif mode == "f05":
            b2 = 0.5**2
            score = (1+b2)*prec*rec/(b2*prec + rec + 1e-12)
        else: score = rec + spec - 1
        if score > best_s: best_s, best_t = score, t
    return best_t



def mixup(xs, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(y.size(0))
    mixed_xs = [lam * x + (1 - lam) * x[idx] for x in xs]
    y_a, y_b = y, y[idx]
    return mixed_xs, y_a, y_b, lam