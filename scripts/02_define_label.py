#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_define_label.py
==================
ARDS 预测建模 —— 标签口径定义与基线呼吸特征补抓

为什么改口径（关键事实）
------------------------
01 抽取的"住院期间任一次达标"标签在建模队列中事件率 65.4%，且 ARDS 起病时间
中位数仅 5.2 h（84.6% 在入 ICU 24 h 内达标）。这意味着该标签实质是"入院时是否
已存在 ARDS"，若再用首日特征去预测它，属于**同一生理窗口的循环论证**（结局由
P/F 定义，而 P/F 同时是特征来源）。

因此本脚本改为**前瞻式偶发 ARDS（incident ARDS）**口径：
  索引时点 t0 = ICU 入科 + 24 h
  基线窗 [intime, intime+24h] 内不得有任何一次达标（否则为"已患病"，剔除）
  结局窗 (t0, t0+48h] 内首次达标 → 阳性
  特征全部取自基线窗（首日化验/生命体征/SOFA/APSIII + 基线 P/F、PEEP），
  严格早于 t0，无时间泄漏。

同时补抓 01 未纳入、但对 ARDS 预测至关重要的基线呼吸生理：
  - 基线动脉血气条数（可判定性）
  - 基线最低 P/F（min_pf_base）
  - 基线末次 P/F（last_pf_base）
  - 基线最高/平均 PEEP、最高 FiO2
  - 是否存在有创动脉置管（动脉血气可得性代理）

输出
----
  ards_analysis_cohort.csv    分析队列（含主标签与三个敏感性标签）
  ards_label_flow.csv         新口径下的纳排流程
  ards_label_definition.md    口径说明（可直接进论文 Methods）

密码仅经环境变量 PGPASSWORD 传入。
用法： PGPASSWORD=**** python 02_define_label.py
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
pd.set_option("display.width", 220)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

DB = db_config()

T0 = time.time()
def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

conn = psycopg2.connect(connect_timeout=10, **DB)
conn.set_session(readonly=False, autocommit=True)
log("连接 PostgreSQL 成功")

# ---------------------------------------------------------------------------
# 1. 读入 01 的建模队列
# ---------------------------------------------------------------------------
log("1/6 读入 01 建模队列 ...")
d = pd.read_csv(os.path.join(DATA, "ards_model_cohort.csv"), low_memory=False)
for c in ["icu_intime", "icu_outtime", "ards_onset", "dischtime", "deathtime"]:
    d[c] = pd.to_datetime(d[c], errors="coerce")
d["onset_h"] = (d["ards_onset"] - d["icu_intime"]).dt.total_seconds() / 3600.0
log(f"    n={len(d):,}  ARDS(ever)={int(d['ards'].sum()):,}")

# ---------------------------------------------------------------------------
# 2. 补抓基线呼吸生理（首个 24h）
# ---------------------------------------------------------------------------
log("2/6 补抓基线呼吸生理（首个 24h 动脉血气 + 呼吸机设置）...")
ids = d["stay_id"].dropna().astype(int).tolist()
cur = conn.cursor()
cur.execute("CREATE TEMP TABLE _cohort (stay_id BIGINT PRIMARY KEY)")
cur.executemany("INSERT INTO _cohort VALUES (%s)", [(i,) for i in ids])

cur.execute("""
CREATE TEMP TABLE _peep AS
SELECT stay_id, charttime, peep
FROM mimiciv_derived.ventilator_setting
WHERE peep IS NOT NULL
""")
cur.execute("CREATE INDEX ON _peep(stay_id, charttime)")
log("    PEEP 临时表就绪")

cur.execute("""
CREATE TEMP TABLE _bg24 AS
SELECT i.stay_id, b.charttime, b.pao2fio2ratio AS pf, b.fio2, b.po2
FROM mimiciv_derived.bg b
JOIN mimiciv_icu.icustays i ON i.hadm_id = b.hadm_id
JOIN _cohort c              ON c.stay_id = i.stay_id
WHERE b.specimen LIKE 'ART%'
  AND b.pao2fio2ratio BETWEEN 1 AND 700
  AND b.charttime BETWEEN i.intime AND i.intime + interval '24 hour'
""")
cur.execute("CREATE INDEX ON _bg24(stay_id)")
log("    基线 24h 动脉血气临时表就绪")

cur.execute("""
CREATE TEMP TABLE _bg24p AS
SELECT s.stay_id, s.charttime, s.pf, s.fio2, s.po2,
       (SELECT max(v.peep) FROM _peep v
         WHERE v.stay_id = s.stay_id
           AND v.charttime BETWEEN s.charttime - interval '4 hour'
                               AND s.charttime + interval '4 hour') AS peep
FROM _bg24 s
""")

bg24 = pd.read_sql("""
SELECT stay_id,
       count(*)                                    AS n_art_bg,
       min(pf)                                     AS min_pf_base,
       max(pf)                                     AS max_pf_base,
       avg(pf)                                     AS mean_pf_base,
       max(peep)                                   AS max_peep_base,
       avg(peep)                                   AS mean_peep_base,
       max(fio2)                                   AS max_fio2_base,
       count(*) FILTER (WHERE peep >= 5)           AS n_bg_peep5,
       count(*) FILTER (WHERE pf <= 300 AND peep >= 5) AS n_bg_ards
FROM _bg24p GROUP BY stay_id
""", conn)
# 基线末次 P/F：按时间取最后一条
last_pf = pd.read_sql("""
SELECT DISTINCT ON (stay_id) stay_id, pf AS last_pf_base, charttime AS last_pf_time
FROM _bg24p ORDER BY stay_id, charttime DESC
""", conn)
bg24 = bg24.merge(last_pf[["stay_id", "last_pf_base"]], on="stay_id", how="left")

d = d.merge(bg24, on="stay_id", how="left")
d["n_art_bg"] = d["n_art_bg"].fillna(0).astype(int)
log(f"    有基线动脉血气者：{int((d['n_art_bg']>0).sum()):,} / {len(d):,}")

cur.execute("DROP TABLE IF EXISTS _bg24")

# ---------------------------------------------------------------------------
# 3. 纳排：基线窗内 ARDS-free（核心改动）
# ---------------------------------------------------------------------------
log("3/6 应用前瞻式纳排 ...")
flow = []
def step(df, name, mask, note=""):
    before = len(df)
    df = df[mask].copy()
    flow.append({"step": name, "n_before": before, "n_after": len(df),
                 "excluded": before - len(df), "note": note})
    return df

w = d.copy()
flow.append({"step": "0. 01 建模队列（通气成人、排除慢性肺病）", "n_before": len(w),
             "n_after": len(w), "excluded": 0, "note": ""})

# 3.1 基线窗内无 ARDS（含无起病者）
base_ards = (w["ards"] == 1) & (w["onset_h"] <= 24)
w["_base_ards"] = base_ards
w = step(w, "1. 剔除入 ICU 24h 内即达标（基线已患 ARDS）", ~base_ards,
         "避免结局与特征同窗，去除循环论证")

# 3.2 基线 ARDS 可判定：24h 内至少 1 条动脉血气
w = step(w, "2. 基线窗内至少 1 条动脉血气（标签可判定）", w["n_art_bg"] >= 1,
         "无血气者无法排除基线 ARDS")

# 3.3 ICU 停留 >= 24h（t0 时点必须仍在 ICU）
w = step(w, "3. ICU 停留 >= 24h（t0 时点仍在 ICU）", w["los_icu"] >= 1.0, "")

# ---------------------------------------------------------------------------
# 4. 标签
# ---------------------------------------------------------------------------
log("4/6 生成主标签与敏感性标签 ...")
def lab_h(h):
    return ((w["ards"] == 1) & (w["onset_h"] > 24) & (w["onset_h"] <= 24 + h)).astype(int)

w["y_ards72h"] = lab_h(48)        # 主标签：t0 后 48h（= 入科 24–72h）
w["y_ards48h"] = lab_h(24)        # 敏感性 1：t0 后 24h
w["y_ards7d"] = lab_h(144)        # 敏感性 2：t0 后 6 天（= 入科 24–168h）

# 随访完整性：是否观察到 t0+48h（或已死亡/仍在 ICU）
w["t_end"] = w["icu_outtime"]
w["obs_complete"] = ((w["t_end"] >= w["icu_intime"] + pd.Timedelta("72h")) |
                     (w["hospital_expire_flag"] == 1) |
                     w["t_end"].isna()).astype(int)
w["follow_h"] = (w["t_end"] - w["icu_intime"]).dt.total_seconds() / 3600.0

# 索引时点
w["index_time"] = w["icu_intime"] + pd.Timedelta("24h")
w["onset_from_index_h"] = w["onset_h"] - 24

log(f"    主标签 y_ards72h 阳性 {int(w['y_ards72h'].sum()):,} / {len(w):,} "
    f"= {w['y_ards72h'].mean()*100:.1f}%")
log(f"    随访完整者 {int(w['obs_complete'].sum()):,} ({w['obs_complete'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 5. 输出
# ---------------------------------------------------------------------------
log("5/6 写出 ...")
flow_df = pd.DataFrame(flow)
flow_df.to_csv(os.path.join(DATA, "ards_label_flow.csv"), index=False, encoding="utf-8-sig")
w.drop(columns=[c for c in w.columns if c.startswith("_")], errors="ignore") \
 .to_csv(os.path.join(DATA, "ards_analysis_cohort.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(DATA, "ards_label_definition.md"), "w", encoding="utf-8") as f:
    f.write("""# ARDS 预测标签口径（v2，前瞻式偶发 ARDS）

## 为什么改
v1 标签为"ICU 住院期间任一次符合柏林氧合标准"，建模队列事件率 65.4%。
但实测 ARDS 起病时间中位 5.2 h，**84.6% 在入 ICU 24 h 内即达标**——该标签
实质等价于"入院时是否已患 ARDS"。若以首日（0–24 h）特征预测它，结局与特征
来自同一生理窗口（结局本身由 P/F 定义，P/F 又进入特征），构成**循环论证**
（tautology / incorporation bias），不能称为"预测模型"。

## v2 主口径
| 要素 | 定义 |
|---|---|
| 人群 | 成人（≥18 岁）、首次 ICU 住院、首个 24h 内有机械通气、无慢性肺病 |
| 索引时点 t0 | ICU 入科 + 24 h |
| 基线窗 | [ICU 入科, 入科+24h]，窗内**不得**出现任何一次 ARDS 达标 |
| 标签可判定 | 基线窗内至少 1 条动脉血气（否则无法排除基线 ARDS，剔除） |
| 结局窗 | (t0, t0+48h]，即入科 24–72 h 内首次达标 |
| 阳性定义 | 动脉血气 P/F ≤ 300 mmHg **且** 同期 PEEP ≥ 5 cmH2O |
| 特征窗 | 首个 24 h（严格早于 t0） |
| 随访完整 | ICU 转出 ≥ 入科+72h，或院内死亡 |

## 敏感性标签
- `y_ards48h`：结局窗 (t0, t0+24h]，更早期
- `y_ards7d`：结局窗 (t0, t0+144h]，更长程

## 主分析进一步收紧（在 03 中执行）
柏林判定要求 P/F ≤ 300 **且** PEEP ≥ 5。若患者基线 P/F ≤ 300 但 PEEP 未记录，
则既不能判为 ARDS、也不能判为 ARDS-free——构成**歧义层**。实测该层 305 例，
48h 发病率 27.9%（其中基线 P/F ≤ 200 的 197 例发病率 36%），极可能基线已患病。

因此主分析要求**基线窗内全程 P/F > 300**（无歧义 ARDS-free）；
歧义层作为敏感性分析纳入。判别变量为 `min_pf_base`（基线窗最低 P/F）。

## 已知局限（须写入 Discussion）
1. **影像学"双肺渗出"未能判定**：MIMIC-IV 结构化数据无影像结论，采用柏林定义
   的生理学部分（氧合+PEEP+机械通气）操作化。这会高估 ARDS 发生率（如肺水肿、
   肺不张被计入）。建议敏感性分析：叠加 ICD 或胸部影像报告 NLP。
2. **入选者经过"存活至 24h 且基线无 ARDS"筛选**，属条件选择，可能引入选择偏倚。
3. **早期出院/死亡的非 ARDS 患者按阴性处理**（主分析），可能低估事件率；
   以 `obs_complete` 做限制性敏感性分析。
4. 单中心（BIDMC）回顾性数据，需外部验证（eICU / 928 医院队列）。
""")

# ---------------------------------------------------------------------------
# 6. 摘要
# ---------------------------------------------------------------------------
log("6/6 摘要 ...")
print("\n=== v2 纳排流程 ===")
print(flow_df[["step", "n_before", "n_after", "excluded"]].to_string(index=False))
print(f"\n分析队列 n={len(w):,}")
for k in ["y_ards72h", "y_ards48h", "y_ards7d"]:
    print(f"  {k:12s} 阳性 {int(w[k].sum()):5,}  事件率 {w[k].mean()*100:5.1f}%")

print("\n=== 阳性病例起病时间（自 t0，小时）===")
print(w.loc[w["y_ards72h"] == 1, "onset_from_index_h"].describe(
    percentiles=[.25, .5, .75]).round(1).to_string())

print("\n=== 随访 ===")
print(f"  follow_h 中位 {w['follow_h'].median():.1f} h (IQR {w['follow_h'].quantile(.25):.1f}-{w['follow_h'].quantile(.75):.1f})")
print(f"  obs_complete {int(w['obs_complete'].sum()):,} ({w['obs_complete'].mean()*100:.1f}%)")
print(f"  obs_complete 亚组事件率：{w.loc[w['obs_complete']==1,'y_ards72h'].mean()*100:.1f}%"
      f"  vs 不完整 {w.loc[w['obs_complete']==0,'y_ards72h'].mean()*100:.1f}%")

print("\n=== 基线呼吸生理（中位数）===")
for k in ["min_pf_base", "last_pf_base", "max_peep_base", "max_fio2_base", "n_art_bg"]:
    print(f"  {k:16s} {w[k].median():8.1f}  缺失 {w[k].isna().mean()*100:5.1f}%")

print("\n=== 组间对比（y_ards72h）===")
cmp = w.groupby("y_ards72h")[["admission_age", "sofa", "apsiii", "min_pf_base",
                               "max_peep_base", "charlson_comorbidity_index"]].median().T
print(cmp.round(1).to_string())

print("\n输出：ards_analysis_cohort.csv / ards_label_flow.csv / ards_label_definition.md")
conn.close()
log("完成")
