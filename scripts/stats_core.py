#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stats_core.py — 本仓库统一规范的统计核心（供新分析复用）

与历史脚本的关系（重要）
------------------------
`01`–`06` 为**逐位复现**而保留了各自内联的统计函数，其中三处实现细节并不相同：

====================  ==========================  ==========================
函数                  03 / 05                    06
====================  ==========================  ==========================
delong_ci             返回 4 元组 / 3 元组，      返回 3 元组（np.var）
                      np.cov 与 np.var 混用
calib_slope           C=1e6 的强正则 LR            penalty=None 的无罚 LR
paired bootstrap      普通重抽样                  按阳/阴性分层重抽样
====================  ==========================  ==========================

这些差异**不影响各自论文表格的数值**，但不宜在新分析中继续沿用。
本模块给出单一规范实现：

* ``delong_ci`` 一律返回 ``(auc, lo, hi, se)``，方差用 ``np.var``；
* ``calib_slope`` 一律用**无罚项 logistic 回归**拟合 logit(p) → y。
  切勿用 OLS：本项目实测真值 1.0 时 OLS 给 0.095，系统性低估近 10 倍；
* 配对 bootstrap 提供普通版 ``paired_boot`` 与分层版 ``stratified_paired_boot``，
  调用方需显式选择并在方法中注明。

依赖仅 numpy / pandas / scipy / scikit-learn，不访问数据库。
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42


# ---------------------------------------------------------------------------
# 判别力与置信区间
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    """并列值取平均秩（DeLong 需要的 midrank）。"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_ci(y_true, y_score, alpha=0.05):
    """DeLong 法 AUROC 及其 95% 置信区间，返回 ``(auc, lo, hi, se)``。"""
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos, neg = y_score[y_true == 1], y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    se = float(np.sqrt(np.var(v01) / m + np.var(v10) / n))
    z = norm.ppf(1 - alpha / 2)
    return float(auc), float(auc - z * se), float(auc + z * se), se


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=SEED):
    """非参数 bootstrap 的 AUROC 百分位区间，返回 ``(lo, hi)``。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        vals.append(roc_auc_score(y_true[b], y_score[b]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot(y, pa, pb, n_boot=2000, seed=SEED):
    """普通配对 bootstrap，ΔAUROC = AUC(pb) − AUC(pa)。

    返回 ``(mean_delta, lo, hi, p_two_sided)``。对应 05 脚本的口径。
    """
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    idx = np.arange(len(y))
    d = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        d.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def stratified_paired_boot(y, pa, pb, n_boot=2000, seed=SEED):
    """按阳/阴性分层重抽样的配对 bootstrap（保持每次重抽样的事件数）。

    返回 ``(point_delta, lo, hi, p_two_sided)``，其中点估计为全样本 ΔAUROC。
    对应 06 脚本 ``boot_delta`` 的口径；事件数少时比普通版更稳定。
    """
    rng = np.random.default_rng(seed)
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan, np.nan, np.nan
    d0 = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        ip = rng.choice(pos, len(pos), replace=True)
        ing = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, ing])
        ds[b] = roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(p, 1.0))


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------
def _unpenalized_lr():
    """兼容不同 sklearn 版本的无罚项 logistic 回归。"""
    try:
        return LogisticRegression(penalty=None, max_iter=2000)
    except TypeError:  # sklearn < 1.2
        return LogisticRegression(penalty="none", max_iter=2000)


def calib_slope(y, p):
    """校准斜率：logit(p) → y 的 logistic 斜率，1.0 为理想。

    **不要用 OLS 拟合二分类结局**：真值 1.0 时 OLS 约给 0.095。
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    if np.std(lp) < 1e-9:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _unpenalized_lr().fit(lp.reshape(-1, 1), np.asarray(y).astype(int))
    return float(m.coef_[0, 0])


def calib_slope_intercept(y, p):
    """返回 ``(slope, intercept)``。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _unpenalized_lr().fit(lp.reshape(-1, 1), np.asarray(y).astype(int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def hosmer_lemeshow(y, p, g=10):
    """Hosmer–Lemeshow 检验，返回 ``(chi2, p_value)``。"""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    a = df.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    stat = (((a["obs"] - a["exp"]) ** 2) / (a["exp"] * (1 - a["exp"] / a["n"]))).sum()
    return float(stat), float(1 - chi2.cdf(stat, g - 2))


def oe_ratio(y, p):
    """O:E 比（观察事件数 / 预测事件数）。"""
    return float(np.sum(y) / np.sum(p))


# ---------------------------------------------------------------------------
# 临床效用
# ---------------------------------------------------------------------------
def net_benefit(y, p, pt):
    """单一阈值概率下的净获益。"""
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    tp = ((p >= pt) & (y == 1)).sum()
    fp = ((p >= pt) & (y == 0)).sum()
    return tp / n - fp / n * (pt / (1 - pt))


def dca(y, p, thresholds=np.arange(0.01, 0.51, 0.01)):
    """决策曲线分析表：模型 / 全治疗 / 全不治疗三条净获益曲线。"""
    y, p = np.asarray(y), np.asarray(p)
    n, prev = len(y), y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        rows.append({
            "threshold": pt,
            "nb_model": tp / n - fp / n * (pt / (1 - pt)),
            "nb_treat_all": prev - (1 - prev) * (pt / (1 - pt)),
            "nb_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


__all__ = [
    "SEED", "delong_ci", "bootstrap_auc", "paired_boot", "stratified_paired_boot",
    "calib_slope", "calib_slope_intercept", "hosmer_lemeshow", "oe_ratio",
    "net_benefit", "dca",
]
