import os, argparse, warnings, random
import numpy as np
from typing import List, Tuple
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops

from utils.utils import find_best_threshold, mixup
# ---------------- Loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=1.5, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        targets = targets.float()
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * (1 - p_t).pow(self.gamma) * ce
        return loss.mean()


# ---------------- Model ----------------
class SingleViewGAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.g2 = GATv2Conv(hidden * heads, out_dim, heads=1, dropout=dropout, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act, self.dp = nn.PReLU(), nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.g1(x, edge_index, edge_attr); h = self.bn1(h); h = self.act(h); h = self.dp(h)
        h = self.g2(h, edge_index, edge_attr); h = self.bn2(h); h = self.act(h); h = self.dp(h)
        return h + self.skip(x)


class CrossViewAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)
        self.ln = nn.LayerNorm(dim)
    def forward(self, hs: List[torch.Tensor]):
        H = torch.stack(hs, dim=1)  # [N,V,D]
        Ht = H.transpose(0,1)
        out, _ = self.mha(Ht, Ht, Ht)
        return self.ln(out.transpose(0,1)).mean(dim=1)


class MultiViewGAT(nn.Module):
    def __init__(self, in_dims, enc_dim=64, hidden=64, fusion_dim=128, heads=4, dropout=0.2):
        super().__init__()
        self.views = len(in_dims)
        self.encoders = nn.ModuleList([
            SingleViewGAT(in_dims[i], hidden, enc_dim, heads=heads, dropout=dropout)
            for i in range(self.views)
        ])
        self.cross = CrossViewAttention(enc_dim, heads=heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(enc_dim * self.views, enc_dim * self.views), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim * self.views, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(fusion_dim + enc_dim, 1)

    def forward(self, xs, es, ews):
        hs = [self.encoders[i](xs[i], es[i], ews[i]) for i in range(self.views)]
        h_cat = torch.cat(hs, dim=1)
        h_gate = h_cat * self.gate(h_cat)
        h_fuse = self.fuse(h_gate)
        h_cross = self.cross(hs)
        return self.head(torch.cat([h_fuse, h_cross], dim=1)).squeeze(1)



@torch.no_grad()
def evaluate(model, data, mask, thr=0.5, amp=False, mc_passes=10):
    """Monte Carlo Dropout evaluation"""
    model.train()
    probs_mc = []
    for _ in range(mc_passes):
        with autocast(enabled=amp):
            logits = model(data.xs, data.es, data.ews)
            probs_mc.append(torch.sigmoid(logits[mask]).cpu().numpy())
    probs = np.mean(probs_mc, axis=0)
    y = data.y[mask].cpu().numpy().astype(int)
    preds = (probs >= thr).astype(int)
    return accuracy_score(y, preds), f1_score(y, preds), probs, y


def train(model, data, masks, args):
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, epochs=args.epochs,
                                                steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")
    scaler = GradScaler(enabled=args.amp)
    best_acc, best_thr, best_state, patience = -1, 0.5, None, args.patience

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)

        # --- Trick 3: Mixup ---
        xs_mix, y_a, y_b, lam = mixup(data.xs, data.y, alpha=0.3)
        with autocast(enabled=args.amp):
            logits = model(xs_mix, data.es, data.ews)
            loss = lam * criterion(logits[masks["train"]], y_a[masks["train"]]) + \
                   (1 - lam) * criterion(logits[masks["train"]], y_b[masks["train"]])

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()

        # --- Trick 1: anneal threshold ---
        thr_dynamic = 0.3 + 0.2 * min(ep / args.epochs, 1.0)
        acc_v, f1_v, probs_v, yv = evaluate(model, data, masks["val"], thr=thr_dynamic, amp=args.amp)
        thr = find_best_threshold(probs_v, yv, mode=args.target_metric)
        acc_val = accuracy_score(yv, (probs_v >= thr).astype(int))

        if acc_val > best_acc:
            best_acc, best_thr = acc_val, thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1

        if ep % args.log_every == 0 or ep == 1:
            print(f"[{ep:03d}] loss={loss:.4f} | VAL acc={acc_val:.3f} thr={thr:.2f} | patience={patience}")
        if patience <= 0: break

    if best_state: model.load_state_dict(best_state)
    return model, best_thr


