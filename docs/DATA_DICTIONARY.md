# ARDS MIMIC-IV 抽取数据字典

口径：柏林定义生理学操作化（P/F<=300 且 PEEP>=5，动脉血气，首次 ICU 住院）。
局限：影像学"双肺渗出"无法由结构化数据判定，未强制；后续可结合 radiology 报告 NLP 或 ICD 做敏感性分析。

## 文件
- `ards_label.csv`：每个首次 ICU 住院一行，含 ards 标签与起病信息。
- `ards_cases.csv`：ARDS=1 病例集（专病库雏形）。
- `ards_model_cohort.csv`：建模队列（病例+通气非 ARDS 对照），宽表。
- `ards_cohort_flow.csv`：纳排流程计数。

## 关键字段
| 字段 | 含义 |
|---|---|
| ards | ARDS 标签（1=符合，0=通气非 ARDS 对照） |
| ards_onset | ARDS 首次符合生理学标准的时间 |
| onset_pf | 起病时 P/F (mmHg) |
| onset_peep | 起病时 PEEP (cmH2O) |
| onset_severity | 起病严重度 (mild/moderate/severe) |
| min_pf_24h | 起病后 24h 内最低 P/F |
| worst_severity_24h | 起病后 24h 内最差严重度 |
| admission_age | 入 ICU 年龄 |
| hospital_expire_flag | 院内死亡（结局） |
| los_icu | ICU 停留天数 |
| sofa | 首日 SOFA |
| apsiii | 首日 APS III |
| charlson_comorbidity_index | Charlson 合并症指数 |
| lab_* | 首日实验室（min/max） |
| vital_* | 首日生命体征（min/max/mean） |
