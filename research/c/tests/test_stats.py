"""research/c/stats.py."""
import numpy as np
import pytest

import stats as S


def test_paired_effect_detects_a_real_effect_and_not_noise():
    rng = np.random.default_rng(1)
    clusters = [f"t{i // 4}" for i in range(160)]
    big = (rng.random(160) < 0.3).astype(float)                    # 30% of pairs drop by 1
    res = S.paired_effect(clusters, big, n_boot=2000, n_perm=2000)
    assert res["mean"] == pytest.approx(big.mean())
    assert res["ci"][0] > 0 and res["p"] < 0.01
    noise = rng.choice([-1.0, 0.0, 1.0], size=160, p=[0.15, 0.7, 0.15])
    res = S.paired_effect(clusters, noise, n_boot=2000, n_perm=2000)
    assert res["ci"][0] < 0 < res["ci"][1] and res["p"] > 0.05


def test_paired_effect_empty():
    assert S.paired_effect([], [])["mean"] is None


def test_holm_is_monotone_and_keeps_none():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["d"] is None
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_diff_of_diffs():
    cl = [f"t{i % 10}" for i in range(100)]
    res = S.bootstrap_diff_of_diffs(cl, [1.0] * 100, [0.0] * 50 + [1.0] * 50, n_boot=1000)
    assert res["mean"] == pytest.approx(0.5) and res["p"] < 0.05


def test_auroc_and_calibration():
    assert S.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert S.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert S.auroc([0.3, 0.4], [1, 1]) is None
    perfect = S.calibration([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
    assert perfect["brier"] == 0 and perfect["ece"] == 0 and perfect["auroc"] == 1.0
    over = S.calibration([0.9] * 10, [1] * 5 + [0] * 5)
    assert over["ece"] == pytest.approx(0.4) and over["reliability"] == pytest.approx(0.16)
    assert over["uncertainty"] == pytest.approx(0.25)


def test_pp_resample_informative_vs_useless_predictor():
    rng = np.random.default_rng(0)
    r = (rng.random(400) < 0.5).astype(float)
    good = np.clip(r * 0.8 + 0.1 + rng.normal(0, 0.05, 400), 0, 1)
    useless = rng.random(400)
    cl = [f"t{i // 5}" for i in range(400)]
    g = S.pp_resample(r, good, cl, 0.1, n_rep=400)
    u = S.pp_resample(r, useless, cl, 0.1, n_rep=400)
    assert g["var_ratio"] < 0.5 and g["noise_ratio"] < 0.2      # an informative predictor saves most labels
    assert u["var_ratio"] > 0.8 and u["noise_ratio"] > 0.95     # an unrelated one saves nothing, whatever its mean
    assert abs(g["mean_pp"] - r.mean()) < 0.02                    # unbiased on average
    assert g["noise_ratio"] == pytest.approx(1 - g["corr"] ** 2)
