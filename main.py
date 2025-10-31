from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import torch
import argparse
import os
from utils.utils import set_seed, scale_fit_transform, build_knn_edges_cpu_with_cosine
from model.model import MultiViewGAT
from model.model import FocalLoss
from model.model import evaluate, train



def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    base = args.data_dir
    y = np.load(os.path.join(base, "labels_class.npy")).astype(np.float32)
    X1 = np.load(os.path.join(base, "view1_phonation.npy"))
    X2 = np.load(os.path.join(base, "view2_articulation.npy"))
    X3 = np.load(os.path.join(base, "view3_prosody.npy"))

    N = len(y)
    idx = np.arange(N)
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=args.seed)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=args.seed)

    method = "quantile" if args.quantile_scaler else "standard"
    def fit_transform(X):
        Xt, Xv, Xe = scale_fit_transform(X[idx_tr], X[idx_va], X[idx_te], method)
        Xs = np.zeros_like(X)
        Xs[idx_tr], Xs[idx_va], Xs[idx_te] = Xt, Xv, Xe
        return Xs

    X1s, X2s, X3s = fit_transform(X1), fit_transform(X2), fit_transform(X3)
    xs = [torch.from_numpy(X1s).float().to(device),
          torch.from_numpy(X2s).float().to(device),
          torch.from_numpy(X3s).float().to(device)]

    es, ews = [], []
    for Xs_np in [X1s, X2s, X3s]:
        edge_index, edge_attr = build_knn_edges_cpu_with_cosine(Xs_np, args.knn, device)
        es.append(edge_index)
        ews.append(edge_attr)

    class Pack: pass
    data = Pack()
    data.xs, data.es, data.ews = xs, es, ews
    data.y = torch.from_numpy(y).float().to(device)

    mask_tr = torch.zeros((N,), dtype=torch.bool, device=device)
    mask_va = torch.zeros_like(mask_tr)
    mask_te = torch.zeros_like(mask_tr)
    mask_tr[idx_tr], mask_va[idx_va], mask_te[idx_te] = True, True, True
    masks = {"train": mask_tr, "val": mask_va, "test": mask_te}

    model = MultiViewGAT(
        in_dims=[X1.shape[1], X2.shape[1], X3.shape[1]],
        enc_dim=args.enc_dim, hidden=args.hidden,
        fusion_dim=args.fusion_dim, heads=args.heads,
        dropout=args.dropout
    ).to(device)

    model, best_thr = train(model, data, masks, args)
    print("\n=== TEST ===")
    best_acc, best_thr_final = 0, 0
    for thr in np.linspace(0.2, 0.8, 61):
        acc_t, f1_t, probs_t, y_t = evaluate(model, data, masks["test"], thr=thr, amp=args.amp)
        if acc_t > best_acc:
            best_acc, best_thr_final, best_probs, best_y = acc_t, thr, probs_t, y_t

    preds = (best_probs >= best_thr_final).astype(int)
    cm = confusion_matrix(best_y, preds)
    prec = precision_score(best_y, preds, zero_division=0)
    rec = recall_score(best_y, preds, zero_division=0)
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-6)
    auc = roc_auc_score(best_y, best_probs)
    f1 = f1_score(best_y, preds)

    print(f"BEST thr={best_thr_final:.2f} | ACC={best_acc:.3f} | F1={f1:.3f} | "
          f"PREC={prec:.3f} | REC={rec:.3f} | SPEC={spec:.3f} | AUROC={auc:.3f}")
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"best_epoch_{args.seed}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "best_threshold": best_thr_final,
        "args": vars(args),
        "metrics": {
            "acc": best_acc, "f1": f1, "precision": prec,
            "recall": rec, "specificity": spec, "auc": auc
        }
    }, ckpt_path)

    print(f"\nModel checkpoint saved to: {ckpt_path}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="dataset/multiview_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--quantile_scaler", action="store_true")

    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--enc_dim", type=int, default=96)
    ap.add_argument("--fusion_dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--target_metric", type=str, default="acc", choices=["acc","f05","youden"])

    args = ap.parse_args()
    main(args)
