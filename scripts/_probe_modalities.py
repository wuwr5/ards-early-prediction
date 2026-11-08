#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探查多模态/生物标志物在主分析队列中的可得性（一次性诊断脚本）"""
import os
from ards_config import db_config
import pandas as pd
import psycopg2

conn = psycopg2.connect(connect_timeout=10, **db_config())
conn.autocommit = True
cur = conn.cursor()

d = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "ards_analysis_cohort.csv"), low_memory=False)
ids = d.loc[d.min_pf_base > 300, "stay_id"].astype(int).tolist()
print(f"主队列 n = {len(ids)}")

cur.execute("CREATE TEMP TABLE _c (stay_id BIGINT PRIMARY KEY)")
cur.executemany("INSERT INTO _c VALUES (%s)", [(i,) for i in ids])


def cov(sql, label):
    cur.execute(sql)
    r = cur.fetchone()
    print(f"  {label:34s} {r[0]:6d} / {len(ids)}  ({r[0]/len(ids)*100:5.1f}%)")


LAB_24H = ("FROM _c s JOIN mimiciv_icu.icustays i ON i.stay_id=s.stay_id "
           "JOIN mimiciv_hosp.labevents l ON l.hadm_id=i.hadm_id "
           "AND l.charttime BETWEEN i.intime AND i.intime+interval '24 hour' ")

print("\n=== 首个 24h 模态覆盖率 ===")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid=51652", "hs-CRP")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid=50924", "Ferritin")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid IN (50915,51196,52551)", "D-Dimer")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid IN (51623,52116,51214,52117)", "Fibrinogen")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid IN (53154,52442,50813)", "Lactate (lab)")
cov("SELECT count(DISTINCT s.stay_id) " + LAB_24H + "WHERE l.itemid IN (50910,50911,51594,51595,51596)", "CK")

VS_24H = ("FROM mimiciv_derived.ventilator_setting vs JOIN _c s ON s.stay_id=vs.stay_id "
          "JOIN mimiciv_icu.icustays i ON i.stay_id=vs.stay_id "
          "WHERE vs.charttime BETWEEN i.intime AND i.intime+interval '24 hour' AND ")
cov("SELECT count(DISTINCT s.stay_id) " + VS_24H + "vs.plateau_pressure IS NOT NULL", "Plateau pressure")
cov("SELECT count(DISTINCT s.stay_id) " + VS_24H + "vs.tidal_volume_observed IS NOT NULL", "Tidal volume")
cov("SELECT count(DISTINCT s.stay_id) " + VS_24H + "vs.minute_volume IS NOT NULL", "Minute volume")
cov("SELECT count(DISTINCT s.stay_id) " + VS_24H + "vs.respiratory_rate_total IS NOT NULL", "RR total")

cov("""SELECT count(DISTINCT s.stay_id) FROM _c s
       JOIN mimiciv_derived.vasoactive_agent v ON v.stay_id=s.stay_id
       JOIN mimiciv_icu.icustays i ON i.stay_id=s.stay_id
       WHERE v.starttime < i.intime + interval '24 hour' AND v.endtime > i.intime""", "Vasoactive agent")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.suspicion_of_infection x USING(stay_id)",
    "Suspicion of infection")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.inflammation x USING(stay_id)",
    "inflammation 表")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.height h USING(stay_id)", "Height(PBW)")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.first_day_weight w USING(stay_id)",
    "first_day_weight")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.first_day_urine_output u USING(stay_id)",
    "first_day_urine_output")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.blood_differential b USING(stay_id)",
    "blood_differential")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.oxygen_delivery o USING(stay_id)",
    "oxygen_delivery")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.cardiac_marker c2 USING(stay_id)",
    "cardiac_marker")
cov("SELECT count(DISTINCT s.stay_id) FROM _c s JOIN mimiciv_derived.coagulation c3 USING(stay_id)",
    "coagulation")

print("\n=== inflammation 表结构 ===")
cur.execute("""SELECT string_agg(column_name, ', ' ORDER BY ordinal_position)
               FROM information_schema.columns
               WHERE table_schema='mimiciv_derived' AND table_name='inflammation'""")
print("  ", cur.fetchone()[0])

print("\n=== 影像报告（radiology）总量 ===")
cur.execute("SELECT count(*) FROM mimiciv_note.radiology")
print("   mimiciv_note.radiology rows =", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM mimiciv_note.radiology_detail")
print("   mimiciv_note.radiology_detail rows =", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM mimiciv_note.discharge")
print("   mimiciv_note.discharge rows =", cur.fetchone()[0])

conn.close()
