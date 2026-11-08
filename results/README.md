# results/ — 已提交的示例输出

本目录收录**聚合统计结果与图件**，便于在不接触 MIMIC-IV 数据的情况下核对论文表格、
检查代码是否跑出一致数值。所有文件均为行列汇总或曲线，不含 `subject_id` / `hadm_id` /
`stay_id` 等标识，也不含个体预测值。

被 `.gitignore` 排除的只有一类：`*predictions*.csv`（如 `dyn_test_predictions.csv`，
含 `stay_id` 与个体预测概率），这类文件属于派生患者级数据，不入库。

## 汇总表

| 文件 | 来源 | 内容 |
|---|---|---|
| `model_performance.csv` | 03 | M1/M2/M3 与两个参照臂的 AUROC、AUPRC、Brier、校准、HL |
| `sensitivity_analysis.csv` | 03 | 结局窗 / 随访完整性等敏感性分析 |
| `calibration_table.csv` | 03 | 校准分组表 |
| `dca_table.csv` | 03 | 决策曲线净获益 |
| `coefficients_M2.csv` | 03 | M2 回归系数 |
| `permutation_importance_M3.csv` | 03 | 置换重要性 |
| `mm_incremental_performance.csv` | 05 | S0–S4 与简约多模态模型性能 |
| `mm_delta_auroc.csv` | 05 | 逐层 ΔAUROC 及配对 bootstrap CI / P |
| `mm_coefficients.csv` | 05 | 多模态模型系数 |
| `dyn_performance.csv` | 06 | 三个 landmark × 各模型的性能与校准 |
| `dyn_delta_auroc.csv` | 06 | 动态 − 静态配对 ΔAUROC |
| `dyn_sensitivity.csv` | 06 | 删失与视野敏感性 |
| `dyn_incidence_structure.csv` | 06 | 各 landmark 风险集与事件结构 |
| `dyn_traj_effects.csv` | 06 | 轨迹变量效应量 |

## 图件

`fig_roc.png`、`fig_calibration.png`、`fig_dca.png`、`fig_importance.png`（03）；
`fig_mm_roc.png`、`fig_mm_delta.png`（05）；
`fig_dyn_auroc.png`、`fig_dyn_roc_cal.png`、`fig_dyn_dca.png`、`fig_dyn_risk_traj.png`（06）。

## 报告与运行元数据

`baseline_report.md`、`mm_report.md`、`dyn_report.md` 为脚本自动生成的文字报告；
`run_meta.json`、`dyn_run_meta.json` 记录运行参数（随机种子、划分比例等）。

论文级的完整结果汇总见 [`docs/RESULTS.md`](../docs/RESULTS.md)。
图中文字使用英文，以避免 matplotlib 在无中文字体环境下出现方框。
