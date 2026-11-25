"""Statistical tests from the approval form (Section 3).

* McNemar's test for paired yes/no outcomes (accuracy, validity), odds ratio effect size
* Wilcoxon signed-rank test for paired continuous measures, rank-biserial r effect size
* Shapiro-Wilk normality check on paired differences
* Holm-Bonferroni correction over the whole family of tests
* Bootstrap 95% confidence intervals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((~a & b).sum())   # A wrong, B right
    n10 = int((a & ~b).sum())   # A right, B wrong
    table = [[int((a & b).sum()), n10], [n01, int((~a & ~b).sum())]]
    exact = (n01 + n10) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    odds = (n10 + 0.5) / (n01 + 0.5)    # Haldane correction avoids division by zero
    return {"test": "McNemar", "statistic": float(res.statistic), "p": float(res.pvalue),
            "effect": float(odds), "effect_name": "odds_ratio", "n10": n10, "n01": n01}


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014). Positive: x > y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    shapiro_p = float(stats.shapiro(d[:5000]).pvalue) if len(d) >= 3 and d.std() > 0 else np.nan
    if np.allclose(d, 0):
        return {"test": "Wilcoxon", "statistic": 0.0, "p": 1.0, "effect": 0.0,
                "effect_name": "rank_biserial_r", "shapiro_p": shapiro_p}
    res = stats.wilcoxon(x, y, zero_method="wilcox")
    return {"test": "Wilcoxon", "statistic": float(res.statistic), "p": float(res.pvalue),
            "effect": rank_biserial(x, y), "effect_name": "rank_biserial_r",
            "shapiro_p": shapiro_p, "median_x": float(np.median(x)), "median_y": float(np.median(y))}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn=rank_biserial, n: int = 2000,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        vals.append(fn(x[i], y[i]))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def holm_correction(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    out = results.copy()
    reject, p_adj, _, _ = multipletests(out["p"].values, alpha=alpha, method="holm")
    out["p_holm"] = p_adj
    out["significant"] = reject
    return out
