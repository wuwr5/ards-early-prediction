#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_incremental_value.py
=======================
多模态特征的增量判别价值（同一训练/测试划分，配对比较）

基线特征集（BASE）= 03 的基础特征（首日化验/生命体征、SOFA、APSIII、Charlson、
                   人口学、基线 P/F 与 PEEP）
逐层加入：
  +RESP  呼吸力学（平台压、驱动压、Vt/PBW、分钟通气量、静态顺应性）
  +GAS   气体交换扩展（PaCO₂、A-a 梯度、血气乳酸）
  +CIRC  循环灌注（血管活性药、去甲当量、24h 尿量、RRT）
  +INF   感染/炎症/凝血（感染可疑、24h 抗生素、纤维蛋白原、CK、化验乳酸）

统计：配对 bootstrap（2000 次）检验 ΔAUROC，同时给出 DeLong 95%CI、AUPRC、
      Brier、校准斜率、Hosmer-Lemeshow。
另给一个"临床可用简约多模态模型"（MM-parsimonious），便于床旁落地。
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
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------------- 统计工具 ---------------------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    se = np.sqrt(np.var(v01) / m + np.var(v10) / n)
    z = 1.959963984540054
    return auc, auc - z * se, auc + z * se


def hosmer_lemeshow(y, p, g=10):
    from scipy.stats import chi2
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    a = df.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = (((a["obs"] - a["exp"]) ** 2) / (a["exp"] * (1 - a["exp"] / a["n"]))).sum()
    return float(hl), float(1 - chi2.cdf(hl, g - 2))


def calib_slope(y, p):
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def paired_boot(y, pa, pb, n_boot=2000, seed=SEED):
    """ΔAUROC = AUC(pb) - AUC(pa)，配对 bootstrap"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); idx = np.arange(len(y))
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


# --------------------------- 特征分组 ---------------------------
RESP = ["pplat_max", "pplat_mean", "dp_max", "dp_mean", "vt_obs_mean", "vt_obs_max",
        "vt_per_pbw_mean", "vt_per_pbw_max", "mv_mean", "mv_max",
        "rr_total_mean", "rr_total_max", "cstat_mean", "n_vent_setting",
        "dp_over_peep", "peep_24_max", "peep_24_mean"]
GAS = ["paco2_max", "paco2_min", "aa_grad_max", "lact_bg_max"]
CIRC = ["vaso_any", "vaso_pressor", "norepi_max", "uo_24h", "uo_rate6_min", "rrt_any"]
INF = ["susp_infection_24h", "abx_24h", "fibrinogen_max", "ck_max", "lact_lab_max"]
AUX = ["height_cm", "pbw_kg", "height_missing"]          # 派生用，不进模型
MM_ALL = RESP + GAS + CIRC + INF

log("读入多模态队列 ...")
d = pd.read_csv(os.path.join(DATA, "ards_cohort_mm.csv"), low_memory=False)
log(f"    n={len(d):,}  事件 {int(d['y_ards72h'].sum()):,} ({d['y_ards72h'].mean()*100:.1f}%)")

# 与 03 完全一致的排除规则，保证 BASE 可复现
DROP = set([
    "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "icu_intime", "icu_outtime",
    "dod", "deathtime", "index_time", "last_pf_time", "first_icu_stay",
    "ards", "ards_onset", "onset_pf", "onset_peep", "onset_severity", "n_qualifying_bg",
    "min_pf_24h", "worst_severity_24h", "onset_h", "onset_from_index_h",
    "y_ards72h", "y_ards48h", "y_ards7d",
    "los_icu", "los_hospital", "hospital_expire_flag", "discharge_location",
    "follow_h", "obs_complete", "any_mv", "t_end", "apsiii_prob", "n_bg_ards",
])
base_all = [c for c in d.columns if c not in DROP and c not in MM_ALL and c not in AUX]
base_all = [c for c in base_all if not pd.api.types.is_datetime64_any_dtype(d[c])]
CAT = [c for c in ["gender", "race", "admission_type"] if c in base_all]
base_all = [c for c in base_all if c in CAT or pd.api.types.is_numeric_dtype(d[c])]

# 常量与高缺失（与 03 同阈值 70%，仅作用于 BASE）
keep = []
for c in base_all:
    s = d[c]
    if s.notna().sum() == 0:
        continue
    if s.nunique(dropna=True) <= 1 and s.dtype != object:
        continue
    if s.isna().mean() > 0.70:
        continue
    keep.append(c)
BASE = keep
BASE_NUM = [c for c in BASE if c not in CAT]
log(f"    BASE {len(BASE)} 个（数值 {len(BASE_NUM)} / 分类 {len(CAT)}）")

# 多模态特征：只剔除覆盖率 <15% 的
MM = [c for c in MM_ALL if c in d.columns and d[c].notna().mean() >= 0.15]
dropped_mm = [c for c in MM_ALL if c not in MM]
if dropped_mm:
    log(f"    多模态特征剔除过低覆盖：{dropped_mm}")

y = d["y_ards72h"].values

# 与 03 相同的划分（seed 42, 30%），保证可比
idx_tr, idx_te = train_test_split(np.arange(len(d)), test_size=0.30,
                                  stratify=y, random_state=SEED)
ytr, yte = y[idx_tr], y[idx_te]
log(f"    训练 {len(idx_tr):,} / 测试 {len(idx_te):,}（事件 {int(yte.sum())}）")


def make_pipe(num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]), cat),
    ])
    return pre


def fit_eval(name, feats, c_grid=(0.003, 0.01, 0.03, 0.1, 0.3)):
    num = [c for c in feats if c not in CAT]
    cat = [c for c in feats if c in CAT]
    Xtr = d.iloc[idx_tr][feats]; Xte = d.iloc[idx_te][feats]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best_c, best = None, -1
    for C in c_grid:
        pipe = Pipeline([("pre", make_pipe(num, cat)),
                         ("clf", LogisticRegression(C=C, max_iter=3000, solver="lbfgs"))])
        a = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        if a > best:
            best_c, best = C, a
    pipe = Pipeline([("pre", make_pipe(num, cat)),
                     ("clf", LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs"))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc, lo, hi = delong_ci(yte, p)
    hl, hlp = hosmer_lemeshow(yte, p)
    return {"model": name, "n_features": len(feats), "best_C": best_c,
            "cv_auroc": best, "auroc": auc, "lo": lo, "hi": hi,
            "auprc": average_precision_score(yte, p),
            "brier": brier_score_loss(yte, p),
            "calib_slope": calib_slope(yte, p), "hl_p": hlp}, p, pipe


log("逐层拟合 ...")
specs = [
    ("S0 BASE", BASE),
    ("S1 +RESP 呼吸力学", BASE + RESP),
    ("S2 +GAS 气体交换", BASE + RESP + GAS),
    ("S3 +CIRC 循环灌注", BASE + RESP + GAS + CIRC),
    ("S4 +INF 感染炎症", BASE + RESP + GAS + CIRC + INF),
    ("MM-parsimonious 简约多模态",
     ["admission_age", "sofa", "apsiii", "charlson_comorbidity_index", "gender",
      "min_pf_base", "max_peep_base", "n_art_bg"] +
     ["dp_max", "vt_per_pbw_mean", "cstat_mean", "aa_grad_max", "lact_bg_max",
      "norepi_max", "uo_24h", "fibrinogen_max"]),
]

rows, preds = [], {}
for name, feats in specs:
    feats = [c for c in feats if c in d.columns]
    r, p, pipe = fit_eval(name, feats)
    rows.append(r); preds[name] = p
    log(f"    {name:26s} k={r['n_features']:3d}  AUROC={r['auroc']:.3f} "
        f"({r['lo']:.3f}-{r['hi']:.3f})  C={r['best_C']}")

perf = pd.DataFrame(rows)
perf.to_csv(os.path.join(RES, "mm_incremental_performance.csv"), index=False, encoding="utf-8-sig")

print("\n=== 多模态增量性能（测试集）===")
print(perf[["model", "n_features", "auroc", "lo", "hi", "auprc", "brier",
            "calib_slope", "hl_p"]].round(3).to_string(index=False))

# --------------------------- 配对 ΔAUROC ---------------------------
log("配对 bootstrap 检验 ΔAUROC ...")
delta_rows = []
base_name = "S0 BASE"
for name in preds:
    if name == base_name:
        continue
    md, lo, hi, pval = paired_boot(yte, preds[base_name], preds[name])
    delta_rows.append({"vs": base_name, "model": name,
                       "delta_auroc": md, "lo": lo, "hi": hi, "p_value": pval})
delta = pd.DataFrame(delta_rows)
delta.to_csv(os.path.join(RES, "mm_delta_auroc.csv"), index=False, encoding="utf-8-sig")
print("\n=== 相对 BASE 的 ΔAUROC（配对 bootstrap 2000 次）===")
print(delta.round(3).to_string(index=False))

# --------------------------- 图 ---------------------------
log("作图 ...")
# 森林图：ΔAUROC
fig, ax = plt.subplots(figsize=(5.2, 2.8))
dd = delta.iloc[::-1]
ypos = np.arange(len(dd))
ax.errorbar(dd["delta_auroc"], ypos,
            xerr=[dd["delta_auroc"] - dd["lo"], dd["hi"] - dd["delta_auroc"]],
            fmt="o", color="#C44E52", ms=4, lw=1.2, capsize=2)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(ypos)
EN_MM = {"S0 BASE": "S0 BASE", "S1 +RESP 呼吸力学": "S1 +RESP",
         "S2 +GAS 气体交换": "S2 +GAS", "S3 +CIRC 循环灌注": "S3 +CIRC",
         "S4 +INF 感染炎症": "S4 +INF",
         "MM-parsimonious 简约多模态": "MM parsimonious"}
ax.set_yticklabels([EN_MM.get(m, m) for m in dd["model"]], fontsize=7.5)
ax.set_xlabel("ΔAUROC vs BASE (95% bootstrap CI)")
ax.set_title("Incremental value of multimodal blocks", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_mm_delta.png")); plt.close(fig)

# ROC：BASE vs S4 vs 简约
fig, ax = plt.subplots(figsize=(4.2, 4.2))
show = [("S0 BASE", "BASE"), ("S4 +INF 感染炎症", "BASE+RESP+GAS+CIRC+INF"),
        ("MM-parsimonious 简约多模态", "MM parsimonious")]
for k, lab in show:
    if k not in preds:
        continue
    fpr, tpr, _ = roc_curve(yte, preds[k])
    ax.plot(fpr, tpr, lw=1.5, label=f"{lab} ({roc_auc_score(yte, preds[k]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC: base vs multimodal", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_mm_roc.png")); plt.close(fig)

# 新增特征的单变量效应（OR/SD，full 模型）
log("提取多模态变量效应 ...")
feats_full = [c for c in BASE + RESP + GAS + CIRC + INF if c in d.columns]
num = [c for c in feats_full if c not in CAT]
cat = [c for c in feats_full if c in CAT]
pipe = Pipeline([("pre", make_pipe(num, cat)),
                 ("clf", LogisticRegression(C=0.01, max_iter=3000, solver="lbfgs"))])
pipe.fit(d.iloc[idx_tr][feats_full], ytr)
names = pipe.named_steps["pre"].get_feature_names_out()
coef = pipe.named_steps["clf"].coef_[0]
co = pd.DataFrame({"feature": names, "coef": coef, "OR_per_SD": np.exp(coef)})
co = co[co["feature"].str.replace("num__", "", regex=False).isin(MM)]
co = co.assign(feature=co["feature"].str.replace("num__", "", regex=False))
co.sort_values("coef", key=lambda s: s.abs(), ascending=False) \
  .to_csv(os.path.join(RES, "mm_coefficients.csv"), index=False, encoding="utf-8-sig")
print("\n=== 多模态新特征效应（OR/SD，全模型）===")
print(co.reindex(co["coef"].abs().sort_values(ascending=False).index)
      .head(14)[["feature", "coef", "OR_per_SD"]].round(3).to_string(index=False))

with open(os.path.join(RES, "mm_report.md"), "w", encoding="utf-8") as f:
    f.write("# 多模态/生物标志物特征的增量价值\n\n")
    f.write(f"- 队列 n={len(d):,}，事件 {int(y.sum()):,}（{y.mean()*100:.1f}%），"
            f"测试集 n={len(yte):,}（事件 {int(yte.sum())}）\n")
    f.write("- 同一 train/test 划分（seed 42），逐层加入模态，配对 bootstrap 2000 次\n\n")
    f.write("## 各层性能\n\n")
    f.write(perf.round(3).to_markdown(index=False))
    f.write("\n\n## 相对 BASE 的增量\n\n")
    f.write(delta.round(3).to_markdown(index=False))
    f.write("\n\n## 本机可得性限制\n\n")
    f.write("- **胸部影像报告不可用**：`mimiciv_note.radiology` 为空表（0 行），"
            "MIMIC-CXR-JPG 未下载 → 柏林定义的影像学要件无法落地\n")
    f.write("- hs-CRP 0%、铁蛋白 3.1%、D-二聚体 0.7% → 炎症标志物模态不可用\n")
    f.write("- 因此'多模态'限于结构化多模态：呼吸力学 / 气体交换 / 循环灌注 / 感染炎症\n")

print("\n输出：", RES)
log("完成")
