import numpy as np
import pandas as pd
import json
from pathlib import Path

# -----------------
# Settings
# -----------------
CSV_PATH = "cleaned_data.csv"  # <-- change this
OUT_DIR = "final_data"

T = 30
STRIDE = 5
CONF_THR = 0.30

# features
USE_CONF = True
USE_VEL = True

# gaps already computed? if not, compute segment_id from frame gaps:
COMPUTE_SEGMENTS = False
MAX_GAP = 2

K = 17
X_cols = [f"kpt{i}_x" for i in range(K)]
Y_cols = [f"kpt{i}_y" for i in range(K)]
C_cols = [f"kpt{i}_conf" for i in range(K)]

# COCO indices for YOLO pose
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12


def normalize_xy(xy):
    """xy: (T, K, 2) in [0..1] -> root-center + scale normalize"""
    root = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    xy = xy - root[:, None, :]

    mid_sh = (xy[:, L_SH] + xy[:, R_SH]) / 2.0
    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    scale = np.linalg.norm(mid_sh - mid_hip, axis=-1)
    scale = np.clip(scale, 1e-4, None)
    return xy / scale[:, None, None]


def make_windows(g: pd.DataFrame):
    g = g.sort_values("frame")

    xs = g[X_cols].to_numpy(np.float32)
    ys = g[Y_cols].to_numpy(np.float32)
    cs = g[C_cols].to_numpy(np.float32)

    xy = np.stack([xs, ys], axis=-1)  # (N, K, 2)

    windows = []
    for start in range(0, len(g) - T + 1, STRIDE):
        w_xy = xy[start : start + T]
        w_c = cs[start : start + T]

        w_xy = normalize_xy(w_xy)

        parts = [w_xy.reshape(T, -1)]  # 34 dims
        if USE_CONF:
            parts.append(w_c.reshape(T, -1))  # +17
        if USE_VEL:
            vel = np.zeros_like(w_xy)
            vel[1:] = w_xy[1:] - w_xy[:-1]
            parts.append(vel.reshape(T, -1))  # +34

        feat = np.concatenate(parts, axis=1)  # (T, F)
        windows.append(feat)

    return windows


def _split_counts(n, val_ratio, test_ratio):
    """Pick split counts while preserving at least one train sample when possible."""
    if n <= 1:
        return 0, 0
    if n == 2:
        return 0, 1 if test_ratio > 0 else 0

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    if val_ratio > 0 and n_val == 0:
        n_val = 1

    # keep at least one video in train
    while n_test + n_val > n - 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def split_by_video_stratified(df, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split videos per class so each class can appear in train/val/test (when enough videos exist).
    Returns sets of (class, video) pairs.
    """
    rng = np.random.default_rng(seed)
    train_pairs, val_pairs, test_pairs = set(), set(), set()

    for cls, g in df.groupby("class", sort=True):
        vids = g["video"].astype(str).unique().tolist()
        rng.shuffle(vids)
        n = len(vids)
        n_val, n_test = _split_counts(n, val_ratio, test_ratio)

        test_v = vids[:n_test]
        val_v = vids[n_test : n_test + n_val]
        train_v = vids[n_test + n_val :]

        train_pairs.update((cls, v) for v in train_v)
        val_pairs.update((cls, v) for v in val_v)
        test_pairs.update((cls, v) for v in test_v)

    return train_pairs, val_pairs, test_pairs


def build_split(df, pair_set, label2id):
    X_list, y_list, meta_list = [], [], []
    if not pair_set:
        return (
            np.empty((0, T, 0), np.float32),
            np.empty((0,), np.int64),
            np.empty((0, 5), object),
        )

    pair_df = pd.DataFrame(list(pair_set), columns=["class", "video"])
    d = df.merge(pair_df, on=["class", "video"], how="inner")

    for (cls, vid, tid, sid), g in d.groupby(
        ["class", "video", "track_id", "segment_id"], sort=False
    ):
        ws = make_windows(g)
        if not ws:
            continue
        y = label2id[cls]
        for w in ws:
            X_list.append(w)
            y_list.append(y)
            meta_list.append((cls, vid, int(tid), int(sid), int(g["frame"].min())))

    if not X_list:
        return (
            np.empty((0, T, 0), np.float32),
            np.empty((0,), np.int64),
            np.empty((0, 5), object),
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, np.int64),
        np.array(meta_list, dtype=object),
    )


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Basic filters
    df = df[df["frame_valid"] == 1].copy()
    print("After frame_valid filter:", df["class"].value_counts().to_dict())

    # If you still have -1 tracks, drop them
    if (df["track_id"] == -1).any():
        df = df[df["track_id"] != -1].copy()
    print("After track_id filter:", df["class"].value_counts().to_dict())

    # confidence filter
    df["mean_conf"] = df[C_cols].mean(axis=1)
    df = df[df["mean_conf"] >= CONF_THR].copy()
    print("After confidence filter:", df["class"].value_counts().to_dict())

    # ensure segment_id exists
    if COMPUTE_SEGMENTS or "segment_id" not in df.columns:
        df = df.sort_values(["video", "track_id", "frame"])
        df["gap"] = df.groupby(["video", "track_id"])["frame"].diff().fillna(1)
        df["segment_id"] = df.groupby(["video", "track_id"])["gap"].transform(
            lambda s: (s > MAX_GAP).cumsum()
        )

    # keep only segments long enough
    seg_sizes = (
        df.groupby(["class", "video", "track_id", "segment_id"])
        .size()
        .reset_index(name="n")
    )
    df = df.merge(
        seg_sizes[seg_sizes["n"] >= T][["class", "video", "track_id", "segment_id"]],
        on=["class", "video", "track_id", "segment_id"],
        how="inner",
    )
    print("After min segment length filter:", df["class"].value_counts().to_dict())

    # label mapping
    labels = sorted(df["class"].unique())
    label2id = {c: i for i, c in enumerate(labels)}
    with open(out_dir / "labels.json", "w") as f:
        json.dump(label2id, f, indent=2)

    # split by video
    train_v, val_v, test_v = split_by_video_stratified(df)

    X_tr, y_tr, m_tr = build_split(df, train_v, label2id)
    X_va, y_va, m_va = build_split(df, val_v, label2id)
    X_te, y_te, m_te = build_split(df, test_v, label2id)

    np.savez(out_dir / "train.npz", X=X_tr, y=y_tr, meta=m_tr)
    np.savez(out_dir / "val.npz", X=X_va, y=y_va, meta=m_va)
    np.savez(out_dir / "test.npz", X=X_te, y=y_te, meta=m_te)

    print("Saved to:", out_dir)
    print("Train:", X_tr.shape, y_tr.shape)
    print("Val:  ", X_va.shape, y_va.shape)
    print("Test: ", X_te.shape, y_te.shape)
    print("Feature dim F:", X_tr.shape[-1] if X_tr.size else None)


if __name__ == "__main__":
    main()
