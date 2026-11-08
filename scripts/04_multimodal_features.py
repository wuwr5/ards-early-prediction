#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_multimodal_features.py
=========================
多模态 / 生物标志物特征层（首个 24 h，严格早于索引时点 t0）

本机可得性实测（主队列 n=3,418）
--------------------------------
| 模态 | 覆盖率 | 结论 |
|---|---|---|
| 呼吸力学（平台压/潮气量/分钟通气量/RR） | 84–99% | ✅ 主增量来源 |
| 动脉血气（PaCO₂、P/F、乳酸） | 100%（入组要求） | ✅ |
| 乳酸（化验） | 83% | ✅ |
| 纤维蛋白原 | 37% | ⚠️ 部分可用 |
| 肌酸激酶 | 31% | ⚠️ |
| 血管活性药 | 49% | ✅ |
| 尿量 / RRT | 100% | ✅ |
| 感染可疑 / 抗生素 | 83% | ✅ |
| **胸部影像报告** | **0%（mimiciv_note.radiology 空表）** | ❌ 本机不可用 |
| **hs-CRP / 铁蛋白 / D-二聚体** | 0% / 3.1% / 0.7% | ❌ 未测或过稀 |

因此"多模态"在本机只能是**结构化多模态**：呼吸力学 + 血气 + 循环灌注 + 感染/炎症 + 凝血。
影像模态需另下 MIMIC-CXR-JPG（约数百 GB）或换用 928 医院院内影像，本机不可行。

新增特征
--------
A 呼吸力学：平台压、驱动压(ΔP=Pplat−PEEP)、潮气量/PBW、分钟通气量、静态顺应性
B 血气扩展：PaCO₂、肺泡-动脉氧分压差 A-a gradient、血气乳酸
C 循环灌注：血管活性药、去甲肾上腺素当量、24h 尿量、尿量 ml/kg/h、RRT
D 感染炎症：感染可疑、24h 内启动抗生素、纤维蛋白原、CK、化验乳酸
E 体型：身高、预测体重 PBW（ARDSnet 公式，用于 Vt/kg）

输出：data/ards_cohort_mm.csv
"""
import os
from ards_config import db_config
import sys
import time
import warnings
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

DB = db_config()

T0 = time.time()
def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

conn = psycopg2.connect(connect_timeout=10, **DB)
conn.set_session(readonly=False, autocommit=True)
log("连接成功")

d = pd.read_csv(os.path.join(DATA, "ards_analysis_cohort.csv"), low_memory=False)
ids = d.loc[d["min_pf_base"] > 300, "stay_id"].astype(int).tolist()
log(f"主队列 n={len(ids):,}")

cur = conn.cursor()
cur.execute("CREATE TEMP TABLE _c (stay_id BIGINT PRIMARY KEY)")
cur.executemany("INSERT INTO _c VALUES (%s)", [(i,) for i in ids])

def rd(sql):
    return pd.read_sql(sql, conn)

# ---------------------------------------------------------------------------
# A. 呼吸力学（首个 24h）
# ---------------------------------------------------------------------------
log("A 呼吸力学 ...")
vs = rd("""
SELECT vs.stay_id,
       max(vs.plateau_pressure)                          AS pplat_max,
       avg(vs.plateau_pressure)                          AS pplat_mean,
       max(vs.peep)                                      AS peep_24_max,
       avg(vs.peep)                                      AS peep_24_mean,
       max(vs.plateau_pressure - vs.peep)                AS dp_max,
       avg(NULLIF(vs.plateau_pressure - vs.peep, 0))     AS dp_mean,
       avg(vs.tidal_volume_observed)                     AS vt_obs_mean,
       max(vs.tidal_volume_observed)                     AS vt_obs_max,
       avg(vs.minute_volume)                             AS mv_mean,
       max(vs.minute_volume)                             AS mv_max,
       avg(vs.respiratory_rate_total)                    AS rr_total_mean,
       max(vs.respiratory_rate_total)                    AS rr_total_max,
       count(*)                                          AS n_vent_setting
FROM mimiciv_derived.ventilator_setting vs
JOIN _c c ON c.stay_id = vs.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = vs.stay_id
WHERE vs.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
GROUP BY vs.stay_id
""")
log(f"    呼吸力学 n={len(vs):,}")

# 静态顺应性 = Vt(mL) / ΔP
vs["cstat_mean"] = np.where((vs["dp_mean"].fillna(0) > 0) & vs["vt_obs_mean"].notna(),
                            vs["vt_obs_mean"] / vs["dp_mean"], np.nan)

# ---------------------------------------------------------------------------
# B. 血气扩展（首个 24h 动脉血）
# ---------------------------------------------------------------------------
log("B 血气扩展（PaCO2 / A-a gradient）...")
cur.execute("""
CREATE TEMP TABLE _bgx AS
SELECT i.stay_id, b.charttime, b.po2, b.pco2 AS paco2, b.fio2,
       b.pao2fio2ratio AS pf, b.lactate
FROM mimiciv_derived.bg b
JOIN mimiciv_icu.icustays i ON i.hadm_id = b.hadm_id
JOIN _c c ON c.stay_id = i.stay_id
WHERE b.specimen LIKE 'ART%'
  AND b.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
""")
bgx = rd("""
SELECT stay_id,
       max(paco2)   AS paco2_max,
       min(paco2)   AS paco2_min,
       max(lactate) AS lact_bg_max
FROM _bgx GROUP BY stay_id
""")
# A-a gradient：PAO2 = FiO2*(760-47) - PaCO2/0.8
aa = rd("""
SELECT stay_id,
       max( (fio2/100.0)*(760-47) - paco2/0.8 - po2 ) AS aa_grad_max
FROM _bgx
WHERE fio2 IS NOT NULL AND paco2 IS NOT NULL AND po2 IS NOT NULL
GROUP BY stay_id
""")
bgx_agg = bgx.merge(aa, on="stay_id", how="left")
log(f"    血气 n={len(bgx_agg):,}")

# ---------------------------------------------------------------------------
# C. 循环灌注
# ---------------------------------------------------------------------------
log("C 循环灌注 ...")
vaso = rd("""
SELECT v.stay_id,
       max(1) AS vaso_any,
       max(CASE WHEN v.norepinephrine IS NOT NULL OR v.epinephrine IS NOT NULL
                 OR v.dopamine IS NOT NULL OR v.phenylephrine IS NOT NULL
                 OR v.vasopressin IS NOT NULL THEN 1 ELSE 0 END) AS vaso_pressor
FROM mimiciv_derived.vasoactive_agent v
JOIN _c c ON c.stay_id = v.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = v.stay_id
WHERE v.starttime < i.intime + interval '24 hour' AND v.endtime > i.intime
GROUP BY v.stay_id
""")
norepi = rd("""
SELECT n.stay_id, max(n.norepinephrine_equivalent_dose) AS norepi_max
FROM mimiciv_derived.norepinephrine_equivalent_dose n
JOIN _c c ON c.stay_id = n.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = n.stay_id
WHERE n.starttime < i.intime + interval '24 hour' AND n.endtime > i.intime
GROUP BY n.stay_id
""")
uo = rd("""
SELECT u.stay_id,
       max(u.urineoutput_24hr)     AS uo_24h,
       min(u.uo_mlkghr_24hr)       AS uo_rate_min,
       min(u.uo_mlkghr_6hr)        AS uo_rate6_min
FROM mimiciv_derived.urine_output_rate u
JOIN _c c ON c.stay_id = u.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = u.stay_id
WHERE u.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
GROUP BY u.stay_id
""")
rrt = rd("""
SELECT r.stay_id, max(CASE WHEN r.dialysis_present = 1 THEN 1 ELSE 0 END) AS rrt_any
FROM mimiciv_derived.first_day_rrt r
JOIN _c c ON c.stay_id = r.stay_id
GROUP BY r.stay_id
""")
log(f"    血管活性药 {len(vaso):,} / 去甲当量 {len(norepi):,} / 尿量 {len(uo):,} / RRT {len(rrt):,}")

# ---------------------------------------------------------------------------
# D. 感染 / 炎症 / 凝血
# ---------------------------------------------------------------------------
log("D 感染炎症凝血 ...")
inf = rd("""
SELECT x.stay_id,
       max(CASE WHEN x.suspected_infection_time <= i.intime + interval '24 hour'
                 AND x.suspected_infection_time >= i.intime - interval '24 hour'
                THEN 1 ELSE 0 END) AS susp_infection_24h
FROM mimiciv_derived.suspicion_of_infection x
JOIN _c c ON c.stay_id = x.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = x.stay_id
GROUP BY x.stay_id
""")
abx = rd("""
SELECT a.stay_id, max(1) AS abx_24h
FROM mimiciv_derived.antibiotic a
JOIN _c c ON c.stay_id = a.stay_id
JOIN mimiciv_icu.icustays i ON i.stay_id = a.stay_id
WHERE a.starttime BETWEEN i.intime AND i.intime + interval '24 hour'
GROUP BY a.stay_id
""")
bio = rd("""
SELECT i.stay_id,
       max(CASE WHEN l.itemid IN (51623,52116,51214,52117) THEN l.valuenum END) AS fibrinogen_max,
       max(CASE WHEN l.itemid IN (50910,50911,51594,51595,51596) THEN l.valuenum END) AS ck_max,
       max(CASE WHEN l.itemid IN (53154,52442,50813) THEN l.valuenum END) AS lact_lab_max
FROM _c c
JOIN mimiciv_icu.icustays i ON i.stay_id = c.stay_id
JOIN mimiciv_hosp.labevents l ON l.hadm_id = i.hadm_id
  AND l.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
WHERE l.itemid IN (51623,52116,51214,52117,50910,50911,51594,51595,51596,53154,52442,50813)
GROUP BY i.stay_id
""")
log(f"    感染可疑 {len(inf):,} / 抗生素 {len(abx):,} / 标志物 {len(bio):,}")

# ---------------------------------------------------------------------------
# E. 体型 / PBW
# ---------------------------------------------------------------------------
log("E 身高与预测体重 ...")
ht = rd("""
SELECT h.stay_id, avg(h.height) AS height_cm
FROM mimiciv_derived.height h
JOIN _c c ON c.stay_id = h.stay_id
GROUP BY h.stay_id
""")
log(f"    身高 n={len(ht):,} ({len(ht)/len(ids)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------
log("合并特征 ...")
mm = pd.DataFrame({"stay_id": ids})
for part in [vs, bgx_agg, vaso, norepi, uo, rrt, inf, abx, bio, ht]:
    mm = mm.merge(part, on="stay_id", how="left")

for c in ["vaso_any", "vaso_pressor", "rrt_any",
          "susp_infection_24h", "abx_24h"]:
    if c in mm:
        mm[c] = mm[c].fillna(0).astype(int)

out = d[d["min_pf_base"] > 300].merge(mm, on="stay_id", how="left")

# PBW（ARDSnet）：男 50 + 0.91×(H−152.4)；女 45.5 + 0.91×(H−152.4)
sex = out["gender"].map({"M": 1, "F": 0})
h = out["height_cm"].copy()
h_missing = h.isna().astype(int)
h = h.fillna(pd.Series(np.where(sex == 1, 170.0, 160.0), index=out.index))  # 性别特异中位数填补
pbw = np.where(sex == 1, 50 + 0.91 * (h - 152.4), 45.5 + 0.91 * (h - 152.4))
out["height_missing"] = h_missing
out["pbw_kg"] = pbw
out["vt_per_pbw_mean"] = out["vt_obs_mean"] / out["pbw_kg"]
out["vt_per_pbw_max"] = out["vt_obs_max"] / out["pbw_kg"]

# 驱动压缺失时用 "平台压-PEEP" 的 max 近似（dp_max 已算）
out["dp_over_peep"] = out["dp_max"] / out["peep_24_max"].replace(0, np.nan)

out.to_csv(os.path.join(DATA, "ards_cohort_mm.csv"), index=False, encoding="utf-8-sig")
log(f"写出 ards_cohort_mm.csv：{out.shape[0]:,} 行 × {out.shape[1]:,} 列")

# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------
new_cols = [c for c in mm.columns if c != "stay_id"]
print("\n=== 新增特征覆盖率与分布 ===")
rows = []
for c in new_cols:
    s = out[c]
    rows.append({"feature": c, "coverage_%": round(s.notna().mean() * 100, 1),
                 "median": round(s.median(), 2) if s.notna().any() else np.nan,
                 "p25": round(s.quantile(.25), 2) if s.notna().any() else np.nan,
                 "p75": round(s.quantile(.75), 2) if s.notna().any() else np.nan})
cov = pd.DataFrame(rows).sort_values("coverage_%", ascending=False)
print(cov.to_string(index=False))

print("\n=== 关键新特征按 48h ARDS 分组（中位数）===")
key = ["dp_max", "vt_per_pbw_mean", "cstat_mean", "mv_max", "aa_grad_max",
       "lact_bg_max", "fibrinogen_max", "norepi_max", "uo_rate_min"]
key = [c for c in key if c in out.columns]
print(out.groupby("y_ards72h")[key].median().T.round(2).to_string())

conn.close()
log("完成")
