#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_extract_ards.py
==================
ARDS 临床预测模型 —— MIMIC-IV 队列抽取

口径严格对齐《第九二八医院人才培育项目申报书》：
  纳入：符合柏林定义 ARDS 的 ICU 患者（MIMIC-IV 2012-2019，首次 ICU 住院）
  柏林氧合：P/F <= 300 mmHg 且 PEEP >= 5 cmH2O（影象学"双肺渗出"无法由结构化数据可靠判定，
            以氧合+通气+PEEP 生理学口径操作化，写入数据字典与局限）。
  排除：年龄<18；慢性肺病（COPD/哮喘/肺癌/肺纤维化/结核）致慢性呼衰；
        未行有创/无创机械通气；诊断 ARDS 后 24h 内死亡；诊断后 24h 内自动出院。

输出（落到 ards_mimic/data/）：
  ards_label.csv          每个首次 ICU 住院的 ARDS 标签与起病信息
  ards_cases.csv          ARDS=1 病例集（= 院内部 ARDS 专病数据库的 MIMIC 侧雏形）
  ards_model_cohort.csv   预测建模队列（病例+通气非 ARDS 对照，宽表含特征与标签）
  ards_cohort_flow.csv    纳排流程计数（可作 Figure 1 流程图底稿）
  data_dictionary.md      字段字典

密码：仅经环境变量 PGPASSWORD 传入，绝不写入本文件。
用法：
  PGPASSWORD=**** python 01_extract_ards.py
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
os.makedirs(DATA, exist_ok=True)

DB = db_config()

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

def q(sql):
    return pd.read_sql(sql, conn)

log("连接 PostgreSQL ...")
conn = psycopg2.connect(connect_timeout=10, **DB)
conn.set_session(readonly=False, autocommit=True)
log("连接成功")

# 呼吸机 PEEP 临时表 + 索引（近邻时间窗查询用；仅建临时表，不改动原库结构）
log("    构建 PEEP 临时表 + 索引 ...")
cur = conn.cursor()
cur.execute("""CREATE TEMP TABLE _peep AS
               SELECT stay_id, charttime, peep
               FROM mimiciv_derived.ventilator_setting WHERE peep IS NOT NULL""")
cur.execute("CREATE INDEX ON _peep(stay_id, charttime)")

# ----------------------------------------------------------------------------
# 1. 基础队列：首次 ICU 住院
# ----------------------------------------------------------------------------
log("1/8 拉取首次 ICU 住院 ...")
base = q("""
    SELECT d.subject_id, d.hadm_id, d.stay_id, d.gender, d.admission_age, d.race,
           d.admittime, d.dischtime, d.icu_intime, d.icu_outtime,
           d.los_icu, d.los_hospital, d.hospital_expire_flag, d.dod, d.first_icu_stay,
           a.deathtime, a.discharge_location, a.admission_type
    FROM mimiciv_derived.icustay_detail d
    JOIN mimiciv_hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.first_icu_stay = TRUE
""")
log(f"    首次 ICU 住院：{len(base):,} 例")
base["icu_intime"] = pd.to_datetime(base["icu_intime"])
base["icu_outtime"] = pd.to_datetime(base["icu_outtime"])
base["dischtime"] = pd.to_datetime(base["dischtime"])
base["deathtime"] = pd.to_datetime(base["deathtime"])

# ----------------------------------------------------------------------------
# 2. 动脉血气（P/F 来源）
# ----------------------------------------------------------------------------
log("2/8 拉取动脉血气 + 近邻 PEEP（±4h）...")
bg = q("""
    SELECT b.stay_id, b.charttime, b.po2, b.fio2, b.pao2fio2ratio, b.lactate, pv.peep
    FROM (
        SELECT i.stay_id, b.charttime, b.po2, b.fio2, b.pao2fio2ratio, b.lactate
        FROM mimiciv_derived.bg b
        JOIN mimiciv_icu.icustays i
          ON b.hadm_id = i.hadm_id AND b.charttime BETWEEN i.intime AND i.outtime
        JOIN mimiciv_derived.icustay_detail d
          ON i.stay_id = d.stay_id AND d.first_icu_stay = TRUE
        WHERE b.specimen LIKE 'ART%' AND b.pao2fio2ratio BETWEEN 1 AND 500
    ) b
    LEFT JOIN LATERAL (
        SELECT max(v.peep) AS peep FROM _peep v
        WHERE v.stay_id = b.stay_id
          AND v.charttime BETWEEN b.charttime - interval '4 hour' AND b.charttime + interval '4 hour'
    ) pv ON TRUE
""")
bg["charttime"] = pd.to_datetime(bg["charttime"])
log(f"    动脉血气（首次 ICU，含 PEEP）：{len(bg):,} 条 / {bg['stay_id'].nunique():,} 例")
bg["pf"] = bg["pao2fio2ratio"]

# ARDS 事件（生理学柏林口径）
bg["ards_event"] = (bg["pf"] <= 300) & (bg["peep"] >= 5)
def severity(pf):
    if pd.isna(pf): return np.nan
    if pf <= 100: return "severe"
    if pf <= 200: return "moderate"
    if pf <= 300: return "mild"
    return None
bg["severity"] = bg["pf"].apply(severity)

# ----------------------------------------------------------------------------
# 5. 按 stay 汇总起病信息
# ----------------------------------------------------------------------------
log("5/8 汇总 ARDS 起病 ...")
ev = bg[bg["ards_event"]].copy()
onset = (ev.sort_values("charttime")
           .groupby("stay_id")
           .agg(ards_onset=("charttime", "first"),
                onset_pf=("pf", "first"),
                onset_peep=("peep", "first"),
                onset_severity=("severity", "first"),
                n_qualifying_bg=("pf", "size"))
           .reset_index())
# 起病后 24h 内最差 P/F（向量化，避免逐行扫描）
tmp = bg.merge(onset[["stay_id", "ards_onset"]], on="stay_id", how="inner")
tmp = tmp[(tmp["charttime"] >= tmp["ards_onset"]) &
          (tmp["charttime"] <= tmp["ards_onset"] + pd.Timedelta("24h"))]
worst = tmp.groupby("stay_id")["pf"].min().rename("min_pf_24h")
onset = onset.merge(worst, on="stay_id", how="left")
onset["worst_severity_24h"] = onset["min_pf_24h"].apply(severity)

# 标签表：所有首次 ICU 住院
lab = base.merge(onset, on="stay_id", how="left")
lab["ards"] = lab["ards_onset"].notna().astype(int)

# ----------------------------------------------------------------------------
# 6. 机械通气状态（用于纳排 + 通气类型特征）
# ----------------------------------------------------------------------------
log("6/8 拉取通气状态 ...")
vent = q("""
    SELECT v.stay_id,
           MAX(CASE WHEN v.ventilation_status IN ('InvasiveVent','Tracheostomy') THEN 1 ELSE 0 END) AS invasive_vent,
           MAX(CASE WHEN v.ventilation_status = 'NonInvasiveVent' THEN 1 ELSE 0 END) AS niv,
           MAX(CASE WHEN v.ventilation_status IN ('InvasiveVent','Tracheostomy','NonInvasiveVent') THEN 1 ELSE 0 END) AS any_mv,
           MIN(v.starttime) AS first_vent_start
    FROM mimiciv_derived.ventilation v
    JOIN mimiciv_derived.icustay_detail d ON v.stay_id = d.stay_id AND d.first_icu_stay = TRUE
    GROUP BY v.stay_id
""")
lab = lab.merge(vent, on="stay_id", how="left").fillna({"invasive_vent": 0, "niv": 0, "any_mv": 0})

# ----------------------------------------------------------------------------
# 7. 慢性肺病（ICD 排除） + 特征表
# ----------------------------------------------------------------------------
log("7/8 慢性肺病 ICD 标记 + 特征合并 ...")
dx = q("""
    SELECT d.hadm_id, d.icd_code, d.icd_version
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN mimiciv_derived.icustay_detail i
      ON d.hadm_id = i.hadm_id AND i.first_icu_stay = TRUE
""")
dx["icd_code"] = dx["icd_code"].astype(str).str.strip()
def is_chronic_lung(code, ver):
    c = code.upper().replace(".", "")
    if ver == 9:
        return (c.startswith(("491", "492", "496")) or c.startswith("493") or
                c.startswith("162") or c.startswith("516") or
                ("011" <= c[:3] <= "018"))
    else:
        return (c.startswith(("J43", "J44", "J42", "J45", "C34", "J84")) or
                c.startswith(("A15", "A16", "A17", "A18", "A19")))
dx["chronic_lung"] = dx.apply(lambda r: is_chronic_lung(r["icd_code"], r["icd_version"]), axis=1)
chronic = dx.groupby("hadm_id")["chronic_lung"].max().rename("chronic_lung")
lab = lab.merge(chronic, on="hadm_id", how="left").fillna({"chronic_lung": False})

# 特征表（注意：charlson 无 stay_id，按 hadm 级合并；lab/vital 仅留 stay_id 避免列冲突）
feats = lab[["subject_id", "hadm_id", "stay_id"]].copy()

log("    合并 first_day_sofa ...")
sofa_df = q("""SELECT stay_id, sofa, respiration, coagulation, liver, cardiovascular, cns, renal
               FROM mimiciv_derived.first_day_sofa""")
feats = feats.merge(sofa_df, on="stay_id", how="left")

log("    合并 apsiii ...")
aps_df = q("SELECT stay_id, apsiii, apsiii_prob FROM mimiciv_derived.apsiii")
feats = feats.merge(aps_df, on="stay_id", how="left")

log("    合并 charlson（hadm 级）...")
char_cols = ["charlson_comorbidity_index", "myocardial_infarct", "congestive_heart_failure",
             "peripheral_vascular_disease", "cerebrovascular_disease", "dementia",
             "chronic_pulmonary_disease", "rheumatic_disease", "peptic_ulcer_disease",
             "mild_liver_disease", "diabetes_without_cc", "diabetes_with_cc", "paraplegia",
             "renal_disease", "malignant_cancer", "severe_liver_disease", "metastatic_solid_tumor", "aids"]
char_df = q("SELECT subject_id, hadm_id, " + ", ".join(char_cols) + " FROM mimiciv_derived.charlson")
feats = feats.merge(char_df, on=["subject_id", "hadm_id"], how="left")

log("    合并 first_day_lab ...")
labfeat = q("SELECT * FROM mimiciv_derived.first_day_lab")
labfeat = labfeat.drop(columns=[c for c in ["subject_id", "hadm_id"] if c in labfeat.columns])
labfeat = labfeat.add_prefix("lab_").rename(columns={"lab_stay_id": "stay_id"})
feats = feats.merge(labfeat, on="stay_id", how="left")

log("    合并 first_day_vitalsign ...")
vitfeat = q("SELECT * FROM mimiciv_derived.first_day_vitalsign")
vitfeat = vitfeat.drop(columns=[c for c in ["subject_id", "hadm_id"] if c in vitfeat.columns])
vitfeat = vitfeat.add_prefix("vital_").rename(columns={"vital_stay_id": "stay_id"})
feats = feats.merge(vitfeat, on="stay_id", how="left")

log(f"    特征宽表：{feats.shape[0]:,} 行 × {feats.shape[1]:,} 列")

full = lab.merge(feats, on=["subject_id", "hadm_id", "stay_id"], how="left")

# ----------------------------------------------------------------------------
# 8. 纳排流程
# ----------------------------------------------------------------------------
log("8/8 应用纳排标准 ...")
flow = []
def step(df, name, mask):
    before = len(df)
    df = df[mask].copy()
    flow.append({"step": name, "n_before": before, "n_after": len(df), "excluded": before - len(df)})
    return df

# 起始：首次 ICU 住院
work = full.copy()
flow.append({"step": "0. 首次 ICU 住院", "n_before": len(work), "n_after": len(work), "excluded": 0})

# 年龄 >= 18
work = step(work, "1. 年龄 >= 18", work["admission_age"] >= 18)

# 有机械通气（有创/无创）
work = step(work, "2. 行有创/无创机械通气", work["any_mv"] == 1)

# 无慢性肺病
work = step(work, "3. 排除慢性肺病（COPD/哮喘/肺癌/肺纤维化/结核）", ~work["chronic_lung"].fillna(False))

# 记录 ARDS 标签分布（此时为建模队列骨架）
# 早死/早出院排除（仅对 ARDS 病例；对照保留其存活状态）
# 诊断 ARDS 后 24h 内死亡
def early_death(row):
    if row["ards"] != 1 or pd.isna(row["ards_onset"]) or pd.isna(row["deathtime"]):
        return False
    return (row["deathtime"] - row["ards_onset"]) < pd.Timedelta("24h")
work["_early_death"] = work.apply(early_death, axis=1)
work = step(work, "4. 排除 ARDS 诊断后 24h 内死亡", ~work["_early_death"])

# 诊断 ARDS 后 24h 内自动出院（近似：存活且出院时间距起病 <24h）
def early_discharge(row):
    if row["ards"] != 1 or pd.isna(row["ards_onset"]) or pd.isna(row["dischtime"]):
        return False
    if row["hospital_expire_flag"] == 1:
        return False
    return (row["dischtime"] - row["ards_onset"]) < pd.Timedelta("24h")
work["_early_disch"] = work.apply(early_discharge, axis=1)
work = step(work, "5. 排除 ARDS 后 24h 内自动出院", ~work["_early_disch"])

# 数据完整：关键字段非空
work = step(work, "6. 排除关键数据缺失", work["admission_age"].notna() & work["gender"].notna() & work["hospital_expire_flag"].notna())

# 对照（无 ARDS）也需 ICU 停留 >= 24h 以有 first-day 特征
def short_stay_ctrl(row):
    if row["ards"] == 1:
        return False
    return pd.notna(row["los_icu"]) and row["los_icu"] < 1.0
work["_short_ctrl"] = work.apply(short_stay_ctrl, axis=1)
work = step(work, "7. 对照剔除 ICU 停留 <24h", ~work["_short_ctrl"])

# 清理临时列
work = work.drop(columns=[c for c in work.columns if c.startswith("_")])

# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------
log("写出文件 ...")
cases = work[work["ards"] == 1].copy()
flow_df = pd.DataFrame(flow)
flow_df.to_csv(os.path.join(DATA, "ards_cohort_flow.csv"), index=False, encoding="utf-8-sig")
lab.to_csv(os.path.join(DATA, "ards_label.csv"), index=False, encoding="utf-8-sig")
cases.to_csv(os.path.join(DATA, "ards_cases.csv"), index=False, encoding="utf-8-sig")
work.to_csv(os.path.join(DATA, "ards_model_cohort.csv"), index=False, encoding="utf-8-sig")

# 数据字典
with open(os.path.join(DATA, "data_dictionary.md"), "w", encoding="utf-8") as f:
    f.write("# ARDS MIMIC-IV 抽取数据字典\n\n")
    f.write("口径：柏林定义生理学操作化（P/F<=300 且 PEEP>=5，动脉血气，首次 ICU 住院）。\n")
    f.write("局限：影像学\"双肺渗出\"无法由结构化数据判定，未强制；后续可结合 radiology 报告 NLP 或 ICD 做敏感性分析。\n\n")
    f.write("## 文件\n")
    f.write("- `ards_label.csv`：每个首次 ICU 住院一行，含 ards 标签与起病信息。\n")
    f.write("- `ards_cases.csv`：ARDS=1 病例集（专病库雏形）。\n")
    f.write("- `ards_model_cohort.csv`：建模队列（病例+通气非 ARDS 对照），宽表。\n")
    f.write("- `ards_cohort_flow.csv`：纳排流程计数。\n\n")
    f.write("## 关键字段\n")
    f.write("| 字段 | 含义 |\n|---|---|\n")
    for k, v in [
        ("ards", "ARDS 标签（1=符合，0=通气非 ARDS 对照）"),
        ("ards_onset", "ARDS 首次符合生理学标准的时间"),
        ("onset_pf", "起病时 P/F (mmHg)"),
        ("onset_peep", "起病时 PEEP (cmH2O)"),
        ("onset_severity", "起病严重度 (mild/moderate/severe)"),
        ("min_pf_24h", "起病后 24h 内最低 P/F"),
        ("worst_severity_24h", "起病后 24h 内最差严重度"),
        ("admission_age", "入 ICU 年龄"),
        ("hospital_expire_flag", "院内死亡（结局）"),
        ("los_icu", "ICU 停留天数"),
        ("sofa", "首日 SOFA"),
        ("apsiii", "首日 APS III"),
        ("charlson_comorbidity_index", "Charlson 合并症指数"),
        ("lab_*", "首日实验室（min/max）"),
        ("vital_*", "首日生命体征（min/max/mean）"),
    ]:
        f.write(f"| {k} | {v} |\n")

# ----------------------------------------------------------------------------
# 摘要
# ----------------------------------------------------------------------------
log("=" * 60)
print("\n=== 纳排流程 ===")
print(flow_df.to_string(index=False))
print(f"\n最终建模队列：{len(work):,} 例；ARDS 病例：{len(cases):,}；对照：{len(work)-len(cases):,}；事件率 {len(cases)/len(work)*100:.1f}%")
if len(cases):
    print("\n=== ARDS 严重度分布（起病 24h 内最差）===")
    print(cases["worst_severity_24h"].value_counts(dropna=False).to_string())
    print("\n=== 结局（ARDS 病例）===")
    print(cases["hospital_expire_flag"].value_counts(dropna=False).to_string())
print("\n=== 关键字段缺失率（建模队列）===")
key = ["admission_age", "sofa", "apsiii", "charlson_comorbidity_index"]
for k in key:
    if k in work:
        print(f"  {k:32s} 缺失 {work[k].isna().mean()*100:5.1f}%")
print("\n输出目录：", DATA)
log("完成")
conn.close()
