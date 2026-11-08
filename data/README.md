# data/ — 运行期数据目录（内容不入库）

本目录只放**脚本运行后生成的中间数据**。仓库中 `data/*.csv` 已被 `.gitignore` 排除，
任何患者级数据都**不得**提交：MIMIC-IV 受 PhysioNet Credentialed Health Data License
与 Data Use Agreement 约束，禁止再分发。

## 由哪个脚本生成

| 文件 | 生成脚本 | 内容 | 患者级 |
|---|---|---|---|
| `ards_label.csv` | 01 | 首次 ICU 住院的柏林口径 ARDS 标签与起病时间 | 是 |
| `ards_cases.csv` | 01 | ARDS 阳性病例集（专病库雏形） | 是 |
| `ards_model_cohort.csv` | 01 | 建模宽表（病例 + 通气非 ARDS 对照） | 是 |
| `ards_cohort_flow.csv` | 01 | 纳排流程计数（聚合） | 否 |
| `ards_label_flow.csv` | 02 | 前瞻式标签纳排计数（聚合） | 否 |
| `ards_analysis_cohort.csv` | 02 | **主分析队列**，含 `y_ards72h` 与基线呼吸生理 | 是 |
| `ards_cohort_mm.csv` | 04 | 主分析队列 + 四层结构化多模态特征 | 是 |
| `ards_landmark.csv` | 06 | 24 / 48 / 72 h landmark 风险集与轨迹摘要 | 是 |
| `ards_long_traj.csv` | 06 | 纵向轨迹长表 | 是 |
| `data_dictionary.md` | 01 | 字段字典（与本目录同时生成） | 否 |

主分析队列：**n = 3,418，事件 442（12.9%）**，判定条件 `min_pf_base > 300`。
字段含义见 [`docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md)，
标签口径见 [`docs/LABEL_DEFINITION.md`](../docs/LABEL_DEFINITION.md)。

## 复现方式

```bash
python run_all.py --from 1 --to 2
```

需要先在本机装载 MIMIC-IV 并完成 PhysioNet 认证；配置见 `.env.example`。
本仓库不提供、也不代为分发任何源数据。
