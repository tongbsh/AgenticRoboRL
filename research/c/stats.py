"""Statistics of plan C (numpy only): paired cluster bootstrap / sign-flip tests, Holm, calibration metrics of the
self-assessments, and the semi-synthetic prediction-powered variance study of A0.

Pairs are episodes of the same (task, init, attempt) under two conditions; the cluster is the task (its 2-10 inits are
not independent), so both the bootstrap and the sign-flip test resample or flip whole tasks.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# paired differences
# ---------------------------------------------------------------------------

def _by_cluster(clusters: Sequence, diffs: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """(per-cluster sums, per-cluster counts) in a fixed cluster order."""
    acc: dict = defaultdict(lambda: [0.0, 0])
    for c, d in zip(clusters, diffs):
        acc[c][0] += float(d)
        acc[c][1] += 1
    keys = sorted(acc, key=str)
    return np.array([acc[k][0] for k in keys]), np.array([acc[k][1] for k in keys], dtype=float)


def paired_effect(clusters: Sequence, diffs: Sequence[float], *, n_boot: int = 10000, n_perm: int = 10000,
                  seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean paired difference with a task-cluster bootstrap CI and a two-sided cluster sign-flip p-value."""
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "n_clusters": 0, "mean": None, "ci": [None, None], "p": None}
    sums, counts = _by_cluster(clusters, diffs)
    k = len(sums)
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, k))
    perm = (signs * sums).sum(1) / counts.sum()
    p = float((np.sum(np.abs(perm) >= abs(mean) - 1e-12) + 1) / (n_perm + 1))
    return {"n": n, "n_clusters": k, "mean": mean, "ci": [float(lo), float(hi)], "p": p}


def bootstrap_diff_of_diffs(clusters: Sequence, d1: Sequence[float], d2: Sequence[float], *, n_boot: int = 10000,
                            seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of (d1 - d2) over aligned pairs (difference in differences, e.g. RL vs SFT effect), task-cluster bootstrap
    CI and a bootstrap two-sided p (twice the smaller tail mass at 0)."""
    dd = [float(a) - float(b) for a, b in zip(d1, d2)]
    if not dd:
        return {"n": 0, "mean": None, "ci": [None, None], "p": None}
    sums, counts = _by_cluster(clusters, dd)
    k = len(sums)
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return {"n": len(dd), "n_clusters": k, "mean": mean, "ci": [float(lo), float(hi)], "p": p}


def holm(pvals: dict) -> dict:
    """Holm-adjusted p-values (keys preserved; None stays None)."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m, out, running = len(items), {k: None for k in pvals}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    ph = successes / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# calibration of a probability forecast p for a binary outcome y
# ---------------------------------------------------------------------------

def auroc(p: Sequence[float], y: Sequence[int]) -> float | None:
    """Mann-Whitney AUROC with ties counted half; None when one class is missing."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rpos = r[: len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration(p: Sequence[float], y: Sequence[int], n_bins: int = 10) -> dict:
    """Brier score, ECE (equal-width bins) and the Murphy decomposition Brier = reliability - resolution + uncertainty
    (binned, so the identity holds up to the within-bin variance of p)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(p)
    if n == 0:
        return {"n": 0}
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = rel = res = 0.0
    table = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        pb, yb, wb = float(p[m].mean()), float(y[m].mean()), float(m.mean())
        ece += wb * abs(pb - yb)
        rel += wb * (pb - yb) ** 2
        res += wb * (yb - base) ** 2
        table.append({"bin": b, "n": int(m.sum()), "mean_p": pb, "freq": yb})
    return {"n": n, "brier": brier, "ece": float(ece), "reliability": float(rel), "resolution": float(res),
            "uncertainty": base * (1 - base), "base_rate": base, "auroc": auroc(p, y), "bins": table}


# ---------------------------------------------------------------------------
# A0: prediction-powered estimates on a fixed set of episodes (semi-synthetic)
# ---------------------------------------------------------------------------

def _lambda(r_lab: np.ndarray, f_lab: np.ndarray, f_all: np.ndarray) -> float:
    """PPI++ power-tuned coefficient Cov(r, f) / ((1 + m / n) Var(f)): labelled covariance, variance over all
    episodes (m labelled of n); 0 when it cannot be estimated. An uninformative f gets lambda ~ 0."""
    m, n = len(r_lab), len(f_all)
    if m < 3:
        return 0.0
    var_f = float(np.var(f_all))
    if var_f <= 0:
        return 0.0
    cov = float(np.mean((r_lab - r_lab.mean()) * (f_lab - f_lab.mean())))
    return cov / ((1 + m / n) * var_f)


def pp_resample(r: Sequence[float], f: Sequence[float], clusters: Sequence, q: float, *, n_rep: int = 2000,
                seed: int = 0) -> dict:
    """Bernoulli(q) label masks drawn again and again over the SAME episodes (the policy does not move -- this is the
    semi-synthetic part, it says nothing about how the policy reacts to the predictor). Per mask, the success rate is
    estimated by
        label-only   the mean of r over the labelled episodes (Hajek)
        PP           mean_all(lam * f) + mean_labelled(r - lam * f)        (PPI++, lambda cross-fitted over two task
                     folds: each fold's estimate uses the other fold's labels for lambda)
    var_ratio = Var(PP) / Var(label-only) over the masks: < 1 means fewer labels for the same precision.

    What A's policy gradient sees is the per-episode reward r~ = lam f + xi / q (r - lam f) after the group baseline,
    i.e. its centred label noise (1 - q) / q * E[((r - mean r) - lam (f - mean f))^2]. noise_ratio compares that at the
    best lambda with lambda = 0; it equals 1 - corr(r, f)^2, the share of label noise the predictor cannot remove."""
    r, f = np.asarray(r, float), np.asarray(f, float)
    n = len(r)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    keys = sorted(set(clusters), key=str)
    key_fold = {keys[j]: pos % 2 for pos, j in enumerate(rng.permutation(len(keys)))}
    fold = np.array([key_fold[c] for c in clusters])
    est_lab, est_pp = [], []
    for _ in range(n_rep):
        lab = rng.random(n) < q
        if lab.sum() < 2:
            continue
        est_lab.append(float(r[lab].mean()))
        parts = []
        for k in (0, 1):
            here, other = fold == k, fold != k
            lam = _lambda(r[other & lab], f[other & lab], f[other])
            lab_here = here & lab
            if not lab_here.any():
                break
            parts.append((here.sum(), float(np.mean(lam * f[here]) + np.mean(r[lab_here] - lam * f[lab_here]))))
        if len(parts) == 2:
            est_pp.append(sum(w * e for w, e in parts) / n)
    v_lab = float(np.var(est_lab)) if est_lab else None
    v_pp = float(np.var(est_pp)) if est_pp else None
    rc, fc = r - r.mean(), f - f.mean()
    var_f = float(np.mean(fc ** 2))
    lam_best = float(np.mean(rc * fc) / var_f) if var_f > 0 else 0.0
    noise_lab = (1 - q) / q * float(np.mean(rc ** 2))
    noise_pp = (1 - q) / q * float(np.mean((rc - lam_best * fc) ** 2))
    corr = float(np.mean(rc * fc) / np.sqrt(np.mean(rc ** 2) * var_f)) if var_f > 0 and np.mean(rc ** 2) > 0 else None
    return {"n": n, "q": q, "true_rate": float(r.mean()), "var_label_only": v_lab, "var_pp": v_pp,
            "var_ratio": (v_pp / v_lab) if v_lab and v_pp is not None else None,
            "mean_pp": float(np.mean(est_pp)) if est_pp else None, "corr": corr, "lambda_best": lam_best,
            "noise_label_only": noise_lab, "noise_pp": noise_pp,
            "noise_ratio": (noise_pp / noise_lab) if noise_lab > 0 else None,
            "brier_of_f": float(np.mean((r - f) ** 2))}
