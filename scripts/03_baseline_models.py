#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_baseline_models.py
=====================
ARDS 前瞻预测 —— 基线模型（开发集内部验证）

设计
----
数据：ards_analysis_cohort.csv（02 产出的前瞻式偶发 ARDS 队列）
  n=3,723，主标签 y_ards72h（t0=入科24h 后 48h 内发病），事件率 14.2%
特征：首个 24h 全部可得信息（人口学 + SOFA + APSIII + Charlson + 首日化验/生命体征
      + 基线呼吸生理 min/last P/F、PEEP、FiO2、动脉血气条数），严格早于 t0。

三个模型（头对头）
  M1 临床简约模型：8 个床旁可得变量 + LR
  M2 全变量 LR（L2，带缺失指示）
  M3 梯度提升（HistGradientBoosting）

评估
  区分度：AUROC（DeLong 95%CI）+ 1000 次 bootstrap CI、AUPRC
  校准：校准斜率/截距、Hosmer-Lemeshow、Brier、十分位校准表
  临床效用：DCA 净获益（0.01–0.50）
  重要性：LR 标准化系数 + 置换重要性
  敏感性：y_ards48h（24h 窗）、y_ards7d（7d 窗）、仅随访完整者

注意
  MIMIC-IV 日期按患者个体平移，**不能**做日历年份的时间验证（已排除该方案）；
  外部验证需 eICU / 928 医院队列。
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

T0 = time.time()
def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED = 42
np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------------------
# 0. DeLong AUROC 方差
# ---------------------------------------------------------------------------
def _compute_midrank(x):
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
    """返回 (auc, lo, hi, se)"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos)
        ty[r] = _compute_midrank(neg)
        tz[r] = _compute_midrank(np.concatenate([pos, neg]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    se = np.sqrt(sx / m + sy / n)
    auc = aucs[0]
    se = float(np.sqrt(se)) if np.ndim(se) == 0 else float(np.ravel(se)[0])
    z = 1.959963984540054
    return auc, auc - z * se, auc + z * se, se

def bootstrap_auc(y_true, y_score, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
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

def hosmer_lemeshow(y, p, g=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    agg = df.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = (((agg["obs"] - agg["exp"]) ** 2) / (agg["exp"] * (1 - agg["exp"] / agg["n"]))).sum()
    from scipy.stats import chi2
    pval = 1 - chi2.cdf(hl, g - 2)
    return float(hl), float(pval), agg

def calib_slope_intercept(y, p):
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def dca(y, p, thresholds=np.arange(0.01, 0.51, 0.01)):
    y = np.asarray(y); p = np.asarray(p)
    n = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        nb = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append({"threshold": pt, "nb_model": nb, "nb_treat_all": nb_all, "nb_treat_none": 0.0})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 1. 读数据
# ---------------------------------------------------------------------------
log("1/7 读入分析队列 ...")
d_all = pd.read_csv(os.path.join(DATA, "ards_analysis_cohort.csv"), low_memory=False)
log(f"    02 队列 n={len(d_all):,}  主标签阳性 {int(d_all['y_ards72h'].sum()):,} "
    f"({d_all['y_ards72h'].mean()*100:.1f}%)")

# 主队列收紧：基线窗内**全程** P/F > 300（无歧义 ARDS-free）
# 理由：02 的 ARDS 判定要求 P/F<=300 且 PEEP>=5；若 P/F<=300 但 PEEP 未记录，
#       既不能判为 ARDS 也不能判为 ARDS-free（197 例基线 P/F<=200 者 48h 发病率 36%，
#       极可能基线已患病）。主分析排除该歧义层，敏感性分析中纳入。
ambiguous = d_all["min_pf_base"] <= 300
log(f"    基线 P/F<=300 但未记录 PEEP 的歧义层：{int(ambiguous.sum()):,} 例"
    f"（48h 发病率 {d_all.loc[ambiguous,'y_ards72h'].mean()*100:.1f}%），主分析剔除")
d = d_all[~ambiguous].copy()
log(f"    主分析队列 n={len(d):,}  事件 {int(d['y_ards72h'].sum()):,} "
    f"({d['y_ards72h'].mean()*100:.1f}%)")

DROP = set([
    # 标识 / 时间
    "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "icu_intime", "icu_outtime",
    "dod", "deathtime", "icu_outtime", "index_time", "last_pf_time", "first_icu_stay",
    # 结局 / 雨后信息
    "ards", "ards_onset", "onset_pf", "onset_peep", "onset_severity", "n_qualifying_bg",
    "min_pf_24h", "worst_severity_24h", "onset_h", "onset_from_index_h",
    "y_ards72h", "y_ards48h", "y_ards7d",
    "los_icu", "los_hospital", "hospital_expire_flag", "discharge_location",
    "follow_h", "obs_complete", "any_mv", "t_end", "apsiii_prob", "n_bg_ards",
])

feat_cols = [c for c in d.columns if c not in DROP]
# 防御：任何日期时间列一律剔除
feat_cols = [c for c in feat_cols if not pd.api.types.is_datetime64_any_dtype(d[c])]

# 只保留数值列 + 明确指定的分类列（其余 object/时间戳字符串全部剔除）
CAT_WHITELIST = [c for c in ["gender", "race", "admission_type"] if c in feat_cols]
feat_cols = [c for c in feat_cols
             if c in CAT_WHITELIST or pd.api.types.is_numeric_dtype(d[c])]

# 去掉全缺失/常量列
drop_const = []
for c in feat_cols:
    s = d[c]
    if s.notna().sum() == 0 or (s.nunique(dropna=True) <= 1 and s.dtype != object):
        drop_const.append(c)
feat_cols = [c for c in feat_cols if c not in drop_const]
# 去掉缺失 > 70%
mi = d[feat_cols].isna().mean()
drop_sparse = mi[mi > 0.70].index.tolist()
feat_cols = [c for c in feat_cols if c not in drop_sparse]
log(f"    特征 {len(feat_cols)} 个（剔除常量 {len(drop_const)}、缺失>70% {len(drop_sparse)}）")

cat_cols = [c for c in ["gender", "race", "admission_type"] if c in feat_cols]
num_cols = [c for c in feat_cols if c not in cat_cols]

# 临床简约模型变量
CLIN = ["admission_age", "sofa", "apsiii", "charlson_comorbidity_index",
        "min_pf_base", "max_peep_base", "n_art_bg", "gender"]
CLIN = [c for c in CLIN if c in feat_cols]
log(f"    M1 临床变量：{CLIN}")

# ---------------------------------------------------------------------------
# 2. 划分
# ---------------------------------------------------------------------------
y = d["y_ards72h"].values
X = d[feat_cols].copy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
log(f"2/7 划分：训练 {len(Xtr):,}（事件 {int(ytr.sum())}）/ 测试 {len(Xte):,}（事件 {int(yte.sum())}）")

def make_pre(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]), cat),
    ])

# ---------------------------------------------------------------------------
# 3. 拟合三个模型
# ---------------------------------------------------------------------------
log("3/7 拟合模型 ...")
P = {}   # 名称 -> (test 预测概率, 说明)

# M1 临床简约
n1 = [c for c in CLIN if c not in cat_cols]; c1 = [c for c in CLIN if c in cat_cols]
m1 = Pipeline([("pre", make_pre(n1, c1)),
               ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))])
m1.fit(Xtr[CLIN], ytr)
P["M1 临床简约LR"] = m1.predict_proba(Xte[CLIN])[:, 1]

# M2 全变量 LR（5 折 CV 选 C）
best_c, best_auc = None, -1
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    pipe = Pipeline([("pre", make_pre(num_cols, cat_cols)),
                     ("clf", LogisticRegression(C=C, max_iter=3000, solver="lbfgs"))])
    a = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    log(f"    M2 C={C:<6} CV-AUROC={a:.4f}")
    if a > best_auc:
        best_auc, best_c = a, C
log(f"    M2 最优 C={best_c}（CV-AUROC {best_auc:.4f}）")

m2 = Pipeline([("pre", make_pre(num_cols, cat_cols)),
               ("clf", LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs"))])
m2.fit(Xtr, ytr)
P["M2 全变量LR"] = m2.predict_proba(Xte)[:, 1]
auc_cv_m2 = best_auc

# M3 梯度提升
m3 = Pipeline([("pre", make_pre(num_cols, cat_cols)),
               ("clf", HistGradientBoostingClassifier(
                   max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                   min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.15,
                   random_state=SEED))])
m3.fit(Xtr, ytr)
P["M3 梯度提升"] = m3.predict_proba(Xte)[:, 1]
auc_cv_m3 = cross_val_score(m3, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

# 参照：SOFA 单变量 / P/F 单变量
P["参照 SOFA"] = Xte["sofa"].fillna(Xtr["sofa"].median()).values
P["参照 基线P/F(反向)"] = -Xte["min_pf_base"].fillna(Xtr["min_pf_base"].median()).values

# ---------------------------------------------------------------------------
# 4. 性能
# ---------------------------------------------------------------------------
log("4/7 计算性能指标 ...")
rows, calib_rows, dca_all, roc_store, calib_store = [], [], [], {}, {}
for name, p in P.items():
    auc, lo, hi, se = delong_ci(yte, p)
    blo, bhi = bootstrap_auc(yte, p)
    ap = average_precision_score(yte, p)
    is_prob = name.startswith("M")
    if is_prob:
        brier = brier_score_loss(yte, p)
        slope, intercept = calib_slope_intercept(yte, p)
        hl, hlp, agg = hosmer_lemeshow(yte, p)
    else:
        brier = slope = intercept = hl = hlp = np.nan
        agg = None
    fpr, tpr, thr = roc_curve(yte, p)
    j = np.argmax(tpr - fpr)
    sens, spec = tpr[j], 1 - fpr[j]
    ppv = np.nan; npv = np.nan
    if is_prob:
        cut = thr[j]
        pred = p >= cut
        tp = np.sum(pred & (yte == 1)); fp = np.sum(pred & (yte == 0))
        fn = np.sum((~pred) & (yte == 1)); tn = np.sum((~pred) & (yte == 0))
        ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    rows.append({"model": name, "n": len(yte), "events": int(yte.sum()),
                 "auroc": auc, "auroc_lo_delong": lo, "auroc_hi_delong": hi,
                 "auroc_lo_boot": blo, "auroc_hi_boot": bhi,
                 "auprc": ap, "brier": brier, "calib_slope": slope,
                 "calib_intercept": intercept, "hosmer_lemeshow_chi2": hl,
                 "hosmer_lemeshow_p": hlp,
                 "sens_youden": sens, "spec_youden": spec, "ppv_youden": ppv, "npv_youden": npv})
    if is_prob:
        roc_store[name] = (fpr, tpr, auc)
        calib_store[name] = agg.reset_index()
        dca_all.append(dca(yte, p).assign(model=name))

perf = pd.DataFrame(rows).sort_values("auroc", ascending=False)
perf.to_csv(os.path.join(RES, "model_performance.csv"), index=False, encoding="utf-8-sig")
print("\n=== 测试集性能（n=%d，事件 %d）===" % (len(yte), int(yte.sum())))
print(perf[["model", "auroc", "auroc_lo_delong", "auroc_hi_delong", "auprc",
            "brier", "calib_slope", "hosmer_lemeshow_p"]].round(3).to_string(index=False))

pd.concat(dca_all).to_csv(os.path.join(RES, "dca_table.csv"), index=False, encoding="utf-8-sig")
pd.concat([c.assign(model=k) for k, c in calib_store.items()]).to_csv(
    os.path.join(RES, "calibration_table.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 5. 重要性与系数
# ---------------------------------------------------------------------------
log("5/7 变量重要性 ...")
from sklearn.inspection import permutation_importance
pi = permutation_importance(m3, Xte, yte, scoring="roc_auc", n_repeats=10,
                            random_state=SEED, n_jobs=1)
imp = (pd.DataFrame({"feature": feat_cols, "importance_mean": pi.importances_mean,
                     "importance_sd": pi.importances_std})
       .sort_values("importance_mean", ascending=False))
imp.to_csv(os.path.join(RES, "permutation_importance_M3.csv"), index=False, encoding="utf-8-sig")
print("\n=== M3 置换重要性 Top 12 ===")
print(imp.head(12).round(4).to_string(index=False))

# LR 标准化系数（M2）
pre = m2.named_steps["pre"]
names = pre.get_feature_names_out()
coefs = m2.named_steps["clf"].coef_[0]
coef_df = (pd.DataFrame({"feature": names, "coef_std": coefs,
                         "odds_ratio_per_SD": np.exp(coefs)})
           .assign(abs_coef=lambda x: x["coef_std"].abs())
           .sort_values("abs_coef", ascending=False))
coef_df.to_csv(os.path.join(RES, "coefficients_M2.csv"), index=False, encoding="utf-8-sig")
print("\n=== M2 标准化系数 Top 10（OR/SD）===")
print(coef_df.head(10)[["feature", "coef_std", "odds_ratio_per_SD"]].round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. 图
# ---------------------------------------------------------------------------
log("6/7 作图 ...")
EN = {"M1 临床简约LR": "M1 Clinical LR", "M2 全变量LR": "M2 Full LR", "M3 梯度提升": "M3 HistGB"}
# ROC
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for name, (fpr, tpr, auc) in roc_store.items():
    ax.plot(fpr, tpr, lw=1.6, label=f"{EN.get(name, name)}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC — incident ARDS within 48 h of t0", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_roc.png")); plt.close(fig)

# 校准
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Ideal")
for name, agg in calib_store.items():
    obs = agg["obs"] / agg["n"]
    exp = agg["exp"] / agg["n"]
    ax.plot(exp, obs, "o-", ms=3, lw=1.2, label=EN.get(name, name))
ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed proportion")
ax.set_title("Calibration (deciles, test set)", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_calibration.png")); plt.close(fig)

# DCA
fig, ax = plt.subplots(figsize=(4.6, 3.8))
dd = pd.concat(dca_all)
for name, g in dd.groupby("model"):
    ax.plot(g["threshold"], g["nb_model"], lw=1.5, label=EN.get(name, name))
g0 = dd[dd["model"] == dd["model"].iloc[0]]
ax.plot(g0["threshold"], g0["nb_treat_all"], "k:", lw=1.0, label="Treat all")
ax.plot(g0["threshold"], g0["nb_treat_none"], "k-", lw=0.8, label="Treat none")
ax.set_xlim(0.0, 0.4); ax.set_ylim(-0.02, max(0.2, g0["nb_treat_all"].max() * 1.1))
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis (test set)", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_dca.png")); plt.close(fig)

# 重要性
fig, ax = plt.subplots(figsize=(5.0, 4.4))
t = imp.head(15).iloc[::-1]
ax.barh(t["feature"], t["importance_mean"], xerr=t["importance_sd"], height=0.7,
        color="#4C72B0", ecolor="#999999", capsize=2)
ax.set_xlabel("Drop in AUROC (permutation)")
ax.set_title("M3 permutation importance (top 15)", fontsize=9)
ax.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_importance.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# 7. 敏感性分析
# ---------------------------------------------------------------------------
log("7/7 敏感性分析 ...")
sens_rows = []
def run_variant(tag, ycol, df=None, mask=None):
    src = d if df is None else df
    if mask is not None:
        src = src[mask]
    yy = src[ycol].values
    XX = src[feat_cols].copy()
    if len(np.unique(yy)) < 2 or len(src) < 100:
        return
    Xa, Xb, ya, yb = train_test_split(XX, yy, test_size=0.30, stratify=yy, random_state=SEED)
    pipe = Pipeline([("pre", make_pre(num_cols, cat_cols)),
                     ("clf", LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs"))])
    pipe.fit(Xa, ya)
    pp = pipe.predict_proba(Xb)[:, 1]
    auc, lo, hi, _ = delong_ci(yb, pp)
    sens_rows.append({"variant": tag, "n": len(src), "events": int(yy.sum()),
                      "event_rate": yy.mean(), "auroc": auc, "lo": lo, "hi": hi})

run_variant("主分析：24–72h 窗（y_ards72h）", "y_ards72h")
run_variant("敏感性1：24–48h 窗（y_ards48h）", "y_ards48h")
run_variant("敏感性2：24–168h 窗（y_ards7d）", "y_ards7d")
run_variant("敏感性3：仅随访完整者", "y_ards72h",
            mask=d["obs_complete"] == 1)
run_variant("敏感性4：仅基线有 PEEP>=5 记录者", "y_ards72h",
            mask=d["n_bg_peep5"].fillna(0) >= 1)
run_variant("敏感性5：纳入基线 P/F<=300 歧义层", "y_ards72h", df=d_all)

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(RES, "sensitivity_analysis.csv"), index=False, encoding="utf-8-sig")
print("\n=== 敏感性分析（M2 全变量 LR）===")
print(sens_df.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. 报告
# ---------------------------------------------------------------------------
meta = {
    "n_total": int(len(d)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    "events_total": int(y.sum()), "event_rate": float(y.mean()),
    "n_features": len(feat_cols), "best_C_M2": best_c,
    "cv_auroc_M2": float(auc_cv_m2), "cv_auroc_M3": float(auc_cv_m3),
    "seed": SEED,
}
with open(os.path.join(RES, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

with open(os.path.join(RES, "baseline_report.md"), "w", encoding="utf-8") as f:
    f.write("# ARDS 前瞻预测 —— 基线模型结果\n\n")
    f.write(f"- 分析队列：**{meta['n_total']:,}** 例（训练 {meta['n_train']:,} / 测试 {meta['n_test']:,}）\n")
    f.write(f"- 主标签：t0（入 ICU 24h）后 48h 内发病，事件 **{meta['events_total']:,}**（{meta['event_rate']*100:.1f}%）\n")
    f.write(f"- 特征：{meta['n_features']} 个（首个 24h，严格早于 t0）\n")
    f.write(f"- M2 最优正则 C={best_c}，训练集 5 折 CV-AUROC={auc_cv_m2:.3f}；M3 CV-AUROC={auc_cv_m3:.3f}\n\n")
    f.write("## 测试集性能\n\n")
    f.write(perf.round(3).to_markdown(index=False))
    f.write("\n\n## 敏感性分析\n\n")
    f.write(sens_df.round(3).to_markdown(index=False))
    f.write("\n\n## 判读要点\n\n")
    f.write("1. 参照臂（SOFA 单变量、基线 P/F 单变量）给出\"不用模型\"的下限；M1/M2/M3 的相对增益即多变量建模的价值。\n")
    f.write("2. 校准斜率<1 提示过拟合/分布偏移；>1 提示欠拟合。\n")
    f.write("3. DCA 在 5%–25% 阈值区间净获益高于\"全治\"与\"不治\"才有临床意义。\n")
    f.write("4. 本结果为开发集内部验证，**不构成外部验证证据**；MIMIC-IV 日期个体化平移，无法做日历年时间验证。\n")

print("\n输出目录：", RES)
print(sorted(os.listdir(RES)))
log("完成")
