#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_dynamic_prediction.py
========================
ARDS 动态（landmark）预测：在 24 / 48 / 72 h 三个 landmark 上，用累积到该时点的
纵向轨迹更新风险，固定 24 h 预测视野，与"冻结在 24 h 的静态模型"配对比较。

设计要点
--------
1) 逆时序泄漏控制：landmark L 的特征只取 [0, L] 的数据；结局窗 (L, L+24h]。
2) 风险集：L 时点仍在 ICU 且 L 前未发生 ARDS 者。L 前已发病 / 已离科者剔除。
3) 删失处理：主分析要求完整 24h 观察（L+24 前离科或死亡者剔除）；
   敏感性 A 将删失记为阴性；敏感性 B 视野改为 48h。
4) 模型分两层：
   主分析（预设简约集，受 EPV 约束）—— P_STATIC（首日核心特征，冻结）/ P_DYN（+ 至 L 的轨迹摘要）/ P_TRAJ（仅轨迹）
   次要分析（全变量集）—— F_STATIC（BASE 157）/ F_DYN（BASE+轨迹 239），用于验证简约集结论是否稳健
5) 正则强度由训练集内层 CV（负对数损失）选定，避免手设 C 导致的过拟合与失校准。
5) 按患者划分训练/测试，同一患者不会跨集；配对 bootstrap（2000 次）检验 ΔAUROC。

输入：data/ards_cohort_mm.csv（3,418 例，min_pf_base>300）、MIMIC-IV PostgreSQL
输出：data/ards_long_traj.csv、data/ards_landmark.csv、results/dyn_*
"""
import os
from ards_config import db_config
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
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
LANDMARKS = [24, 48, 72]
HORIZON = 24
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# ============================ 统计工具 ============================
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


def calib_slope(y, p):
    """校准斜率：logit(y) ~ logit(p) 的 logistic 斜率（1.0 = 完美）

    注意：不能用 OLS 拟合二分类结局——实测真值 1.0 时 OLS 给 0.095，
    系统性低估近 10 倍（线性概率模型对 sigmoid 的错设所致）。
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    if np.std(lp) < 1e-9:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(penalty=None, max_iter=2000)
        m.fit(lp.reshape(-1, 1), np.asarray(y).astype(int))
    return float(m.coef_[0, 0])


def hosmer_lemeshow(y, p, g=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    grp = df.groupby("bin").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))
    chi = (((grp.o - grp.e) ** 2) / (grp.e * (1 - grp.e / grp.n))).sum()
    from scipy.stats import chi2
    return chi, 1 - chi2.cdf(chi, g - 2)


def boot_delta(y, pa, pb, n_boot=2000, seed=SEED):
    """配对 bootstrap：ΔAUROC = AUC(b) - AUC(a)，返回点估计与 95%CI 与双侧 p"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb)
    n = len(y)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    d0 = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        ip = rng.choice(pos, len(pos), replace=True)
        iN = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, iN])
        yy = y[idx]
        ds[b] = roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # 双侧 p：bootstrap 分布中越过 0 的比例（两倍单侧）
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, lo, hi, min(p, 1.0)


def net_benefit(y, p, pt):
    n = len(y)
    tp = ((p >= pt) & (y == 1)).sum()
    fp = ((p >= pt) & (y == 0)).sum()
    return tp / n - fp / n * (pt / (1 - pt))


# ============================ A. 纵向轨迹抽取 ============================
def extract_long():
    """从 MIMIC-IV 抽取 [0,96h] 纵向轨迹，长表落盘"""
    import psycopg2
    fp = os.path.join(DATA, "ards_long_traj.csv")
    if os.path.exists(fp):
        log("A/3 纵向轨迹已存在，跳过抽取")
        return pd.read_csv(fp)

    log("A/3 抽取纵向轨迹 ...")
    conn = psycopg2.connect(connect_timeout=10, **db_config())
    conn.autocommit = True
    cur = conn.cursor()

    d = pd.read_csv(os.path.join(DATA, "ards_cohort_mm.csv"), low_memory=False)
    ids = d["stay_id"].astype(int).tolist()
    cur.execute("CREATE TEMP TABLE _c (stay_id BIGINT PRIMARY KEY)")
    cur.executemany("INSERT INTO _c VALUES (%s)", [(i,) for i in ids])
    log(f"    队列 {len(ids):,} 例")

    frames = []

    # --- 动脉血气 ---
    cur.execute("""
        SELECT i.stay_id,
               EXTRACT(EPOCH FROM (b.charttime - i.intime))/3600.0 AS t_h,
               b.pao2fio2ratio AS pf, b.pco2, b.ph, b.lactate
        FROM mimiciv_derived.bg b
        JOIN mimiciv_icu.icustays i ON i.hadm_id = b.hadm_id
             AND b.charttime BETWEEN i.intime AND i.outtime
        JOIN _c c ON c.stay_id = i.stay_id
        WHERE b.specimen LIKE 'ART%%'
          AND b.charttime <= i.intime + interval '96 hour'
    """)
    bgx = pd.DataFrame(cur.fetchall(), columns=["stay_id", "t_h", "pf", "paco2", "ph", "lactate_bg"])
    log(f"    血气 {len(bgx):,} 行 / {bgx.stay_id.nunique():,} 例")
    for v in ["pf", "paco2", "ph", "lactate_bg"]:
        t = bgx[["stay_id", "t_h", v]].rename(columns={v: "value"}).dropna()
        t["var"] = v
        frames.append(t)

    # --- 呼吸机参数 ---
    cur.execute("""
        SELECT v.stay_id,
               EXTRACT(EPOCH FROM (v.charttime - i.intime))/3600.0 AS t_h,
               v.peep, v.plateau_pressure, v.tidal_volume_observed,
               v.minute_volume, v.respiratory_rate_total
        FROM mimiciv_derived.ventilator_setting v
        JOIN _c c ON c.stay_id = v.stay_id
        JOIN mimiciv_icu.icustays i ON i.stay_id = v.stay_id
        WHERE v.charttime <= i.intime + interval '96 hour'
    """)
    vx = pd.DataFrame(cur.fetchall(), columns=["stay_id", "t_h", "peep", "pplat",
                                               "vt_obs", "mv", "rr_total"])
    log(f"    呼吸机 {len(vx):,} 行 / {vx.stay_id.nunique():,} 例")
    for v in ["peep", "pplat", "vt_obs", "mv", "rr_total"]:
        t = vx[["stay_id", "t_h", v]].rename(columns={v: "value"}).dropna()
        t["var"] = v
        frames.append(t)
    # 驱动压 = 平台压 - PEEP
    dp = vx[["stay_id", "t_h", "pplat", "peep"]].dropna()
    dp["value"] = dp["pplat"] - dp["peep"]
    dp = dp[dp.value.between(-5, 60)]
    dp["var"] = "dp"
    frames.append(dp[["stay_id", "t_h", "value", "var"]])

    # --- 生命体征 ---
    cur.execute("""
        SELECT v.stay_id,
               EXTRACT(EPOCH FROM (v.charttime - i.intime))/3600.0 AS t_h,
               v.heart_rate, v.mbp, v.resp_rate, v.temperature, v.spo2
        FROM mimiciv_derived.vitalsign v
        JOIN _c c ON c.stay_id = v.stay_id
        JOIN mimiciv_icu.icustays i ON i.stay_id = v.stay_id
        WHERE v.charttime BETWEEN i.intime AND i.intime + interval '96 hour'
    """)
    sx = pd.DataFrame(cur.fetchall(), columns=["stay_id", "t_h", "hr", "mbp", "rr", "temp", "spo2"])
    log(f"    生命体征 {len(sx):,} 行 / {sx.stay_id.nunique():,} 例")
    for v in ["hr", "mbp", "rr", "temp", "spo2"]:
        t = sx[["stay_id", "t_h", v]].rename(columns={v: "value"}).dropna()
        t["var"] = v
        frames.append(t)

    # --- 化验 ---
    ITEMS = {50813: "lactate", 51265: "platelet", 51301: "wbc", 50912: "creat",
             50885: "bili", 51222: "hgb"}
    cur.execute("""
        SELECT i.stay_id,
               EXTRACT(EPOCH FROM (l.charttime - i.intime))/3600.0 AS t_h,
               l.itemid, l.valuenum
        FROM mimiciv_hosp.labevents l
        JOIN mimiciv_icu.icustays i ON i.hadm_id = l.hadm_id
        JOIN _c c ON c.stay_id = i.stay_id
        WHERE l.itemid IN (50813, 51265, 51301, 50912, 50885, 51222)
          AND l.valuenum IS NOT NULL AND l.valuenum > 0
          AND l.charttime BETWEEN i.intime AND i.intime + interval '96 hour'
    """)
    lx = pd.DataFrame(cur.fetchall(), columns=["stay_id", "t_h", "itemid", "value"])
    lx["var"] = lx["itemid"].map(ITEMS)
    log(f"    化验 {len(lx):,} 行 / {lx.stay_id.nunique():,} 例")
    frames.append(lx[["stay_id", "t_h", "value", "var"]])

    long = pd.concat(frames, ignore_index=True)
    long = long[(long.t_h >= -2) & long.t_h.notna()]
    long["value"] = long["value"].astype(float)
    long["t_h"] = long["t_h"].astype(float)
    long.to_csv(fp, index=False, float_format="%.4f")
    log(f"    长表落盘：{len(long):,} 行 × {long['var'].nunique()} 变量 -> {fp}")
    conn.close()
    return long


# ============================ B. landmark 数据集 ============================
TRAJ_VARS = ["pf", "peep", "pplat", "dp", "vt_obs", "rr_total", "paco2",
             "lactate", "platelet", "wbc", "creat", "bili",
             "hr", "mbp", "rr", "spo2"]


def traj_at(long, L, stays):
    """计算 landmark L 时点的轨迹特征（只用 t_h<=L 的数据）

    实现要点：一律用 `Series.reindex(Index(stays))` 做**显式**标签映射后转 numpy，
    不使用 `DataFrame[col] = Series` 的隐式对齐——后者在重复调用时会偶发整列变 NaN
    （实测同一进程内 rep2 才崩，属 pandas 对齐的不确定行为）。
    """
    stays = np.asarray(stays, dtype=np.int64)
    idx = pd.Index(stays)
    sub = long[(long.t_h <= L) & (long.t_h >= -2)]
    feats = {}
    for v in TRAJ_VARS:
        s = sub[sub["var"] == v]
        if len(s) == 0:
            for k in ["mean", "min", "max", "last", "slope"]:
                feats[f"T_{v}_{k}"] = np.full(len(stays), np.nan)
            if v in ("pf", "lactate"):
                feats[f"T_{v}_n"] = np.zeros(len(stays))
            continue
        g = s.groupby("stay_id")["value"]
        for k, ser in [("mean", g.mean()), ("min", g.min()), ("max", g.max())]:
            feats[f"T_{v}_{k}"] = np.asarray(ser.reindex(idx), dtype=float)
        if v in ("pf", "lactate"):
            feats[f"T_{v}_n"] = np.asarray(g.size().reindex(idx).fillna(0), dtype=float)
        # 最近 12h 末次值
        s2 = s[s.t_h > L - 12].sort_values("t_h")
        feats[f"T_{v}_last"] = np.asarray(
            s2.groupby("stay_id")["value"].last().reindex(idx), dtype=float)
        # 近 24h 斜率（单位/h）
        s3 = s[s.t_h > L - 24]
        if len(s3):
            t = s3["t_h"].to_numpy(dtype=float)
            val = s3["value"].to_numpy(dtype=float)
            a = (pd.DataFrame({"stay_id": s3["stay_id"].to_numpy(), "n": 1.0,
                               "st": t, "sv": val, "stt": t ** 2, "stv": t * val})
                 .groupby("stay_id").sum())
            den = a["n"] * a["stt"] - a["st"] ** 2
            sl = (a["n"] * a["stv"] - a["st"] * a["sv"]) / den.replace(0, np.nan)
            sl[a["n"] < 2] = np.nan
            feats[f"T_{v}_slope"] = np.asarray(sl.reindex(idx), dtype=float)
        else:
            feats[f"T_{v}_slope"] = np.full(len(stays), np.nan)
    return pd.DataFrame(feats, index=idx)


def build_landmarks():
    log("B/3 构建 landmark 数据集 ...")
    d = pd.read_csv(os.path.join(DATA, "ards_cohort_mm.csv"), low_memory=False)
    d["idx"] = np.arange(len(d))
    long = pd.read_csv(os.path.join(DATA, "ards_long_traj.csv"), low_memory=False)

    t_ards = np.where(d["ards"] == 1, d["onset_h"].astype(float), np.inf)
    t_cens = d["los_icu"].astype(float) * 24.0
    d["t_ards"] = t_ards
    d["t_cens"] = t_cens

    rows = []
    for L in LANDMARKS:
        at_risk = (t_cens > L) & (t_ards > L)
        sub = d[at_risk].copy()
        if len(sub) == 0:
            continue
        # 结局判定
        hi = L + HORIZON
        y = np.where(sub.t_ards <= np.minimum(sub.t_cens, hi), 1, 0)
        censored = (y == 0) & (sub.t_cens <= hi)      # 窗口内离科/死亡未发病
        sub["y"] = y
        sub["censored"] = censored.astype(int)
        sub["landmark"] = L
        # 轨迹特征
        tr = traj_at(long, L, sub["stay_id"].values)
        # 防御：整列为空说明上游映射出错，必须立刻失败而不是静默产出空特征
        bad = [c for c in tr.columns if tr[c].notna().sum() == 0]
        if bad:
            raise RuntimeError(f"L={L}h 轨迹特征整列为空，拒绝继续：{bad[:5]}")
        sub = pd.concat([sub.reset_index(drop=True), tr.reset_index(drop=True)], axis=1)
        rows.append(sub)
        log(f"    L={L:2d}h  风险集 {len(sub):,}  事件 {int(sub.y.sum()):,} "
            f"({sub.y.mean()*100:.1f}%)  窗口内删失 {int(censored.sum()):,}")

    lm = pd.concat(rows, ignore_index=True)
    lm.to_csv(os.path.join(DATA, "ards_landmark.csv"), index=False, float_format="%.4f")
    log(f"    landmark 长表：{len(lm):,} 行 × {lm.shape[1]} 列")
    return lm


# ============================ C. 建模与评价 ============================
DROP = set([
    "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "icu_intime", "icu_outtime",
    "dod", "deathtime", "index_time", "last_pf_time", "first_icu_stay",
    "ards", "ards_onset", "onset_pf", "onset_peep", "onset_severity", "n_qualifying_bg",
    "min_pf_24h", "worst_severity_24h", "onset_h", "onset_from_index_h",
    "y_ards72h", "y_ards48h", "y_ards7d",
    "los_icu", "los_hospital", "hospital_expire_flag", "discharge_location",
    "follow_h", "obs_complete", "any_mv", "t_end", "apsiii_prob", "n_bg_ards",
    "t_ards", "t_cens", "y", "censored", "landmark", "idx",
])


# 与 05 一致：多模态层与派生辅助列不属于"仅首日基础特征"
MM_ALL = ["pplat_max", "pplat_mean", "dp_max", "dp_mean", "vt_obs_mean", "vt_obs_max",
          "vt_per_pbw_mean", "vt_per_pbw_max", "mv_mean", "mv_max",
          "rr_total_mean", "rr_total_max", "cstat_mean", "n_vent_setting",
          "dp_over_peep", "peep_24_max", "peep_24_mean",
          "paco2_max", "paco2_min", "aa_grad_max", "lact_bg_max",
          "vaso_any", "vaso_pressor", "norepi_max", "uo_24h", "uo_rate6_min", "rrt_any",
          "susp_infection_24h", "abx_24h", "fibrinogen_max", "ck_max", "lact_lab_max"]
AUX = ["height_cm", "pbw_kg", "height_missing"]


def build_base_cols(d):
    """复用 03/05 的 BASE 规则（仅首日特征，排除多模态层）"""
    cand = [c for c in d.columns
            if c not in DROP and c not in MM_ALL and c not in AUX and not c.startswith("T_")]
    cand = [c for c in cand if not pd.api.types.is_datetime64_any_dtype(d[c])]
    CAT = [c for c in ["gender", "race", "admission_type"] if c in cand]
    cand = [c for c in cand if c in CAT or pd.api.types.is_numeric_dtype(d[c])]
    keep = []
    for c in cand:
        s = d[c]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1 and s.dtype != object:
            continue
        if s.isna().mean() > 0.70:
            continue
        keep.append(c)
    return keep, CAT


def make_pre(num_cols, cat_cols):
    tn = [("num", SimpleImputer(strategy="median"), num_cols)]
    tc = [("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]),
           cat_cols)]
    return ColumnTransformer(tn + (tc if cat_cols else []), remainder="drop")


C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def tune_C(num, cat, X, y, grid=C_GRID, folds=4):
    """内层 CV 选 C（负对数损失 = 兼顾判别与校准），仅在训练集内进行"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    best, best_ll = grid[0], np.inf
    for c in grid:
        lls = []
        for itr, iva in skf.split(X, y):
            p = Pipeline([("pre", make_pre(num, cat)),
                          ("clf", LogisticRegression(C=c, max_iter=4000, solver="lbfgs"))])
            p.fit(X.iloc[itr], y[itr])
            pp = np.clip(p.predict_proba(X.iloc[iva])[:, 1], 1e-6, 1 - 1e-6)
            yy = y[iva]
            lls.append(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
        ll = float(np.mean(lls))
        if ll < best_ll:
            best, best_ll = c, ll
    return best, best_ll


def safe_auroc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def eval_pred(y, p):
    auc, lo, hi = delong_ci(y, p)
    chi, hl_p = hosmer_lemeshow(y, p)
    obs = float(np.mean(y)); exp = float(np.mean(p))
    return dict(n=len(y), events=int(np.sum(y)), event_rate=obs,
                auroc=auc, lo=lo, hi=hi,
                auprc=average_precision_score(y, p),
                brier=brier_score_loss(y, p),
                calib_slope=calib_slope(y, p), hl_p=hl_p,
                mean_pred=exp, OE=exp / obs if obs > 0 else np.nan)


def main():
    lm = build_landmarks()

    log("C/3 建模 ...")
    # 主分析：仅完整窗口
    m0 = lm[lm.censored == 0].copy()
    log(f"    主分析（完整窗口）n={len(m0):,}")

    BASE, CAT = build_base_cols(m0)
    TRAJ = [c for c in m0.columns if c.startswith("T_")]
    log(f"    BASE {len(BASE)} 个 / TRAJ {len(TRAJ)} 个")

    # ---- 预设简约集：主分析用。特征数受事件数约束（EPV），避免 157/239 特征
    #      配 322 事件的噪声主导，否则"静态 vs 动态"的比较无信息量。
    CORE_STATIC = [c for c in [
        "admission_age", "gender", "race", "admission_type",
        "sofa", "apsiii", "charlson_comorbidity_index",
        "min_pf_base", "mean_pf_base", "max_peep_base",
        "lab_creatinine_max", "lab_platelets_min", "lab_wbc_max",
        "lab_bilirubin_total_max", "lab_hemoglobin_min",
        "vital_heart_rate_mean", "vital_mbp_min", "vital_resp_rate_mean",
        "vital_temperature_max", "vital_spo2_min",
        "lact_bg_max", "dp_mean", "vt_per_pbw_mean",
    ] if c in m0.columns]
    CORE_TRAJ_VARS = ["pf", "peep", "lactate", "platelet", "creat", "spo2"]
    CORE_TRAJ = [f"T_{v}_{k}" for v in CORE_TRAJ_VARS
                 for k in ["mean", "min", "last", "slope"] if f"T_{v}_{k}" in m0.columns]
    log(f"    简约集：CORE_STATIC {len(CORE_STATIC)} + CORE_TRAJ {len(CORE_TRAJ)}")

    # 按患者划分（用 24h 队列的 y_ards72h 分层，保证稳定）
    pat = m0.drop_duplicates("stay_id")[["stay_id", "y_ards72h"]]
    tr_pat, te_pat = train_test_split(pat, test_size=0.30, stratify=pat["y_ards72h"],
                                      random_state=SEED)
    tr_ids = set(tr_pat.stay_id); te_ids = set(te_pat.stay_id)
    log(f"    训练患者 {len(tr_ids):,} / 测试患者 {len(te_ids):,}")

    C = 0.03  # 兜底值；实际由内层 CV 选定
    csel = {}  # (landmark, model) -> C

    specs = []
    for L in LANDMARKS:
        tr = m0[(m0.landmark == L) & (m0.stay_id.isin(tr_ids))]
        te = m0[(m0.landmark == L) & (m0.stay_id.isin(te_ids))]
        if len(tr) < 100 or tr.y.sum() < 10 or len(te) < 50 or te.y.sum() < 5:
            log(f"    L={L}h 样本不足，跳过")
            continue
        for tag, cols in [("P_STATIC", CORE_STATIC),
                          ("P_DYN", CORE_STATIC + CORE_TRAJ),
                          ("P_TRAJ", CORE_TRAJ),
                          ("F_STATIC", BASE),
                          ("F_DYN", BASE + TRAJ)]:
            num = [c for c in cols if c not in CAT]
            cat = [c for c in cols if c in CAT]
            Cb, ll = tune_C(num, cat, tr[cols], tr.y.values)
            csel[(L, tag)] = Cb
            pipe = Pipeline([("pre", make_pre(num, cat)),
                             ("clf", LogisticRegression(C=Cb, max_iter=4000, solver="lbfgs"))])
            pipe.fit(tr[cols], tr.y)
            p_tr = pipe.predict_proba(tr[cols])[:, 1]
            p_te = pipe.predict_proba(te[cols])[:, 1]
            specs.append(dict(landmark=L, model=tag, k=len(cols), C=Cb,
                              ytr=tr.y.values, ptr=p_tr, yte=te.y.values, pte=p_te,
                              stay_te=te.stay_id.values, pipe=pipe, cols=cols))
            log(f"    L={L:2d}h {tag:9s} k={len(cols):3d} C={Cb:<6g} "
                f"AUROC={roc_auc_score(te.y, p_te):.3f}  (train {roc_auc_score(tr.y, p_tr):.3f})")

    # --- 性能表 ---
    rows = []
    for s in specs:
        e = eval_pred(s["yte"], s["pte"])
        rows.append(dict(landmark=s["landmark"], model=s["model"], k=s["k"], C=s["C"], **e))
    perf = pd.DataFrame(rows)
    perf.to_csv(os.path.join(RES, "dyn_performance.csv"), index=False, float_format="%.4f")
    print("\n=== 动态预测性能（测试集，24h 视野）===")
    print(perf[["landmark", "model", "k", "C", "n", "events", "auroc", "lo", "hi",
                "auprc", "brier", "calib_slope", "OE", "hl_p"]].round(3).to_string(index=False))

    # 存测试集预测，便于后续只重画图 / 重算指标而不必重跑拟合
    pd.concat([pd.DataFrame(dict(landmark=s["landmark"], model=s["model"],
                                 stay_id=s["stay_te"], y=s["yte"], p=s["pte"]))
               for s in specs], ignore_index=True).to_csv(
        os.path.join(RES, "dyn_test_predictions.csv"), index=False, float_format="%.6f")

    # --- ARDS 发病的时间结构（解释后段 landmark 为何信息枯竭） ---
    dc = pd.read_csv(os.path.join(DATA, "ards_cohort_mm.csv"), low_memory=False)
    ta = dc.loc[dc["ards"] == 1, "onset_h"].astype(float).values
    cap = []
    for L in LANDMARKS:
        cap.append(dict(landmark=L, n_onset_by_L=int((ta <= L).sum()),
                        pct_of_all_ards=round(float((ta <= L).mean() * 100), 1),
                        n_still_ards_free=int((ta > L).sum())))
    capdf = pd.DataFrame(cap)
    capdf.to_csv(os.path.join(RES, "dyn_incidence_structure.csv"), index=False)
    print("\n=== ARDS 发病时间结构（队列 n=3,418，事件 442）===")
    print(capdf.to_string(index=False))

    # --- 配对 ΔAUROC：P_DYN vs P_STATIC ---
    drows = []
    for L in LANDMARKS:
        a = [s for s in specs if s["landmark"] == L and s["model"] == "P_STATIC"]
        b = [s for s in specs if s["landmark"] == L and s["model"] == "P_DYN"]
        c = [s for s in specs if s["landmark"] == L and s["model"] == "P_TRAJ"]
        if not a or not b:
            continue
        assert np.array_equal(a[0]["yte"], b[0]["yte"])
        d0, lo, hi, p = boot_delta(a[0]["yte"], a[0]["pte"], b[0]["pte"])
        drows.append(dict(landmark=L, comparison="P_DYN - P_STATIC",
                          delta=d0, lo=lo, hi=hi, p=p))
        if c:
            d1, l1, h1, p1 = boot_delta(c[0]["yte"], c[0]["pte"], b[0]["pte"])
            drows.append(dict(landmark=L, comparison="P_DYN - P_TRAJ",
                              delta=d1, lo=l1, hi=h1, p=p1))
    dfa = pd.DataFrame(drows)
    dfa.to_csv(os.path.join(RES, "dyn_delta_auroc.csv"), index=False, float_format="%.4f")
    print("\n=== 配对 ΔAUROC（bootstrap 2000 次）===")
    print(dfa.round(3).to_string(index=False))

    # --- 主表：静态 vs 动态 的 AUROC 随 landmark 变化 ---
    piv = perf.pivot_table(index="landmark", columns="model", values="auroc")
    piv_lo = perf.pivot_table(index="landmark", columns="model", values="lo")
    piv_hi = perf.pivot_table(index="landmark", columns="model", values="hi")

    # --- 图 1：AUROC vs landmark ---
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    style = {"P_STATIC": ("#8c8c8c", "o", "Static (frozen at 24 h)"),
             "P_DYN": ("#c0392b", "s", "Dynamic (updated at landmark)"),
             "P_TRAJ": ("#2c7fb8", "^", "Trajectory only")}
    for m in ["P_STATIC", "P_DYN", "P_TRAJ"]:
        if m not in piv.columns:
            continue
        col, mk, lab = style[m]
        ax.errorbar(piv.index, piv[m], yerr=[piv[m] - piv_lo[m], piv_hi[m] - piv[m]],
                    marker=mk, color=col, lw=1.5, ms=4, capsize=2.5, label=lab)
    ax.set_xlabel("Landmark (hours since ICU admission)")
    ax.set_ylabel("AUROC (95% CI)")
    ax.set_xticks(LANDMARKS)
    ax.set_ylim(0.52, 0.80)
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", ncol=1)
    ax.set_title("Dynamic vs static ARDS prediction", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_dyn_auroc.png"))
    plt.close(fig)

    # --- 图 2：24h landmark 的 ROC 与校准 ---
    for L in [48]:
        if L not in piv.index:
            L = LANDMARKS[0]
        a = [s for s in specs if s["landmark"] == L and s["model"] == "P_STATIC"]
        b = [s for s in specs if s["landmark"] == L and s["model"] == "P_DYN"]
        if not a or not b:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
        ax = axes[0]
        for s, col, lab in [(a[0], "#8c8c8c", "Static"), (b[0], "#c0392b", "Dynamic")]:
            fpr, tpr, _ = roc_curve(s["yte"], s["pte"])
            ax.plot(fpr, tpr, color=col, lw=1.7,
                    label=f"{lab}  AUC={roc_auc_score(s['yte'], s['pte']):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax.set_title(f"ROC at {L} h", fontsize=9)

        ax = axes[1]
        for s, col, lab in [(a[0], "#8c8c8c", "Static"), (b[0], "#c0392b", "Dynamic")]:
            df = pd.DataFrame({"y": s["yte"], "p": s["pte"]})
            df["bin"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
            g = df.groupby("bin").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))
            ax.plot(g.e / g.n, g.o / g.n, "o-", color=col, ms=3.2, lw=1.2, label=lab)
        ax.plot([0, 0.6], [0, 0.6], "--", color="#bbbbbb", lw=1)
        ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed risk")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax.set_title(f"Calibration at {L} h", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_dyn_roc_cal.png"))
        plt.close(fig)

    # --- 图 3：风险轨迹（按是否在 (24,48] 发病分层） ---
    b48 = [s for s in specs if s["landmark"] == 48 and s["model"] == "P_DYN"]
    b24 = [s for s in specs if s["landmark"] == 24 and s["model"] == "P_DYN"]
    if b48 and b24:
        ev = pd.Series(b48[0]["yte"], index=b48[0]["stay_te"])
        r24 = pd.Series(b24[0]["pte"], index=b24[0]["stay_te"])
        r48 = pd.Series(b48[0]["pte"], index=b48[0]["stay_te"])
        common = r24.index.intersection(r48.index).intersection(ev.index)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        for flag, col, lab in [(1, "#c0392b", "ARDS in (48, 72] h"), (0, "#2c7fb8", "No ARDS in (48, 72]")]:
            ids = [i for i in common if ev[i] == flag]
            if not ids:
                continue
            m24 = r24[ids].mean(); m48 = r48[ids].mean()
            s24 = r24[ids].std() / np.sqrt(len(ids)); s48 = r48[ids].std() / np.sqrt(len(ids))
            ax.errorbar([24, 48], [m24, m48], yerr=[1.96 * s24, 1.96 * s48],
                        marker="o", color=col, lw=1.6, capsize=3,
                        label=f"{lab} (n={len(ids)})")
        ax.set_xlabel("Landmark (h)"); ax.set_ylabel("Mean predicted risk")
        ax.set_xticks([24, 48]); ax.legend(frameon=False, fontsize=7.5)
        ax.set_title("Predicted-risk trajectory by outcome", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_dyn_risk_traj.png"))
        plt.close(fig)

    # --- 图 4：48h landmark 的 DCA ---
    if b48:
        a = [s for s in specs if s["landmark"] == 48 and s["model"] == "P_STATIC"]
        pts = np.arange(0.02, 0.41, 0.01)
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        y48 = b48[0]["yte"]
        ax.plot(pts, [net_benefit(y48, b48[0]["pte"], t) for t in pts],
                color="#c0392b", lw=1.6, label="Dynamic")
        if a:
            ax.plot(pts, [net_benefit(y48, a[0]["pte"], t) for t in pts],
                    color="#8c8c8c", lw=1.5, label="Static")
        prev = y48.mean()
        ax.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts],
                "--", color="#333333", lw=1.1, label="Treat all")
        ax.axhline(0, color="#999999", lw=0.9, ls=":")
        ax.set_ylim(-0.05, max(prev * 0.6, 0.05))
        ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
        ax.legend(frameon=False, fontsize=7.5)
        ax.set_title("Decision curve at 48 h", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_dyn_dca.png"))
        plt.close(fig)

    # --- 风险重分类（24h → 48h） ---
    recl = None
    if b48 and b24:
        ev = pd.Series(b48[0]["yte"], index=b48[0]["stay_te"])
        r24 = pd.Series(b24[0]["pte"], index=b24[0]["stay_te"])
        r48 = pd.Series(b48[0]["pte"], index=b48[0]["stay_te"])
        common = r24.index.intersection(r48.index)
        if len(common) > 50:
            up = ((r24[common] < 0.15) & (r48[common] >= 0.15)).sum()
            dn = ((r24[common] >= 0.15) & (r48[common] < 0.15)).sum()
            rr = (r48[common] / r24[common]).replace([np.inf, -np.inf], np.nan).median()
            recl = dict(n_common=int(len(common)), n_up=int(up), n_down=int(dn),
                        up_rate=float(up / len(common)), down_rate=float(dn / len(common)),
                        median_risk_ratio=float(rr),
                        events_up=int(ev.reindex(common)[(r24[common] < 0.15) & (r48[common] >= 0.15)].sum()))

    # --- 敏感性分析 ---
    log("C/3 敏感性分析 ...")
    sens = []
    for tag, mask, hor in [("主分析（完整窗口，24h 视野）", m0.censored == 0, 24),
                           ("敏感性A：删失记为阴性", pd.Series(True, index=m0.index), 24),
                           ("敏感性B：48h 视野", m0.censored == 0, 48)]:
        sub = m0[mask]
        if hor == 48:
            # 重算结局：48h 视野
            sub = sub.copy()
            y48v = np.where(sub.t_ards <= np.minimum(sub.t_cens, sub.landmark + 48), 1, 0)
            ok = ~((y48v == 0) & (sub.t_cens <= sub.landmark + 48))
            sub = sub[ok].copy()
            sub["y"] = np.where(sub.t_ards <= np.minimum(sub.t_cens, sub.landmark + 48), 1, 0)
        for L in LANDMARKS:
            tr = sub[(sub.landmark == L) & (sub.stay_id.isin(tr_ids))]
            te = sub[(sub.landmark == L) & (sub.stay_id.isin(te_ids))]
            if len(tr) < 100 or tr.y.sum() < 10 or len(te) < 50 or te.y.sum() < 5:
                continue
            res = {}
            for mtag, cols in [("P_STATIC", CORE_STATIC), ("P_DYN", CORE_STATIC + CORE_TRAJ)]:
                num = [c for c in cols if c not in CAT]
                cat = [c for c in cols if c in CAT]
                Cb = csel.get((L, mtag), C)
                pipe = Pipeline([("pre", make_pre(num, cat)),
                                 ("clf", LogisticRegression(C=Cb, max_iter=4000, solver="lbfgs"))])
                pipe.fit(tr[cols], tr.y)
                p = pipe.predict_proba(te[cols])[:, 1]
                res[mtag] = (roc_auc_score(te.y, p), p)
            auc_s, _ = res["P_STATIC"]; auc_d, pd_d = res["P_DYN"]
            sens.append(dict(scenario=tag, landmark=L, n=len(te),
                             events=int(te.y.sum()), event_rate=te.y.mean(),
                             auroc_static=auc_s, auroc_dynamic=auc_d,
                             delta=auc_d - auc_s))
    sdf = pd.DataFrame(sens)
    sdf.to_csv(os.path.join(RES, "dyn_sensitivity.csv"), index=False, float_format="%.4f")
    print("\n=== 敏感性分析 ===")
    print(sdf.round(3).to_string(index=False))

    # --- 轨迹变量的效应（全动态模型，48h） ---
    eff = None
    if b48:
        pipe = b48[0]["pipe"]; cols = b48[0]["cols"]
        names = list(pipe.named_steps["pre"].get_feature_names_out(cols))
        co = pipe.named_steps["clf"].coef_[0]
        ct = pd.DataFrame({"feature": names, "coef": co})
        ct = ct[ct.feature.str.contains("num__T_")]
        ct["feature"] = ct["feature"].str.replace("num__", "", regex=False)
        ct["OR_per_SD"] = np.exp(ct.coef)
        ct["abs"] = ct.coef.abs()
        eff = ct.sort_values("abs", ascending=False).head(20).drop(columns="abs")
        eff.to_csv(os.path.join(RES, "dyn_traj_effects.csv"), index=False, float_format="%.4f")
        print("\n=== 轨迹特征效应（动态模型 @48h，前 12 位）===")
        print(eff.head(12)[["feature", "coef", "OR_per_SD"]].round(3).to_string(index=False))

    # --- 报告 ---
    with open(os.path.join(RES, "dyn_report.md"), "w", encoding="utf-8") as f:
        f.write("# ARDS 动态（landmark）预测报告\n\n")
        f.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 设计\n\n")
        f.write("- 队列：`ards_cohort_mm.csv`（n=3,418，基线 24h 内无 ARDS 且 min P/F>300）\n")
        f.write(f"- Landmark：{LANDMARKS} h；固定预测视野 {HORIZON} h\n")
        f.write("- 风险集：L 时点仍在 ICU 且 L 前未发病；L+24h 前离科/死亡者主分析剔除（敏感性 A 记为阴性）\n")
        f.write("- 主分析（预设简约集）：P_STATIC 仅首日核心特征；P_DYN 加至 L 时点轨迹摘要；P_TRAJ 仅轨迹\n")
        f.write(f"- 次要分析（全变量集）：F_STATIC（首日 BASE {len(BASE)}）/ F_DYN（BASE+轨迹 {len(BASE)+len(TRAJ)}）\n")
        f.write("- 说明：核心集含 3 个多模态变量（dp_mean / vt_per_pbw_mean / lact_bg_max），"
                "因其为最强的基线预测因子；F_* 集已排除多模态层，故与 03/05 的 BASE 同源\n")
        f.write("- 轨迹摘要：累计 mean/min/max、近 12h 末次值、近 24h OLS 斜率\n\n")
        f.write("## 性能（测试集）\n\n")
        f.write(perf.round(3).to_markdown(index=False))
        f.write("\n\n## ARDS 发病的时间结构\n\n")
        f.write(capdf.to_markdown(index=False))
        f.write("\n\n## 配对 ΔAUROC\n\n")
        f.write(dfa.round(3).to_markdown(index=False))
        f.write("\n\n## 敏感性\n\n")
        f.write(sdf.round(3).to_markdown(index=False))
        if recl:
            f.write("\n\n## 风险重分类（24h → 48h，15% 阈值）\n\n")
            f.write(f"- 共同患者 {recl['n_common']:,}；升级 {recl['n_up']}，降级 {recl['n_down']}\n")
            f.write(f"- 升级者中在 (48,72] 发病 {recl['events_up']} 例\n")
            f.write(f"- 风险比中位数（48h/24h）：{recl['median_risk_ratio']:.2f}\n")
        if eff is not None:
            f.write("\n\n## 轨迹特征效应（动态模型 @48h）\n\n")
            f.write(eff.round(3).to_markdown(index=False))

    meta = dict(landmarks=LANDMARKS, horizon=HORIZON, C=C,
                n_main=int(len(m0)), base_k=len(BASE), traj_k=len(TRAJ),
                reclass=recl, elapsed_s=round(time.time() - T0, 1))
    with open(os.path.join(RES, "dyn_run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"C/3 完成，输出 -> {RES}")


if __name__ == "__main__":
    extract_long()
    main()
