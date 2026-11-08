# MIMIC-IV 中前瞻式偶发 ARDS 的早期预测

**Incident ARDS Early Prediction in MIMIC-IV: Label Definition, Structured
Multimodal Increment, and Landmark Dynamic Models**

> 一套可复现的 ARDS 早期预测代码：从 MIMIC-IV 重建队列、定义**不与特征共窗**的
> 前瞻式标签，到基线模型、结构化多模态增量与 landmark 动态预测。
> 代码 MIT 许可；**不含任何患者级数据**。

---

## 核心结论（先看这里）

| 项目 | 结果 |
|---|---|
| 主分析队列 | **3,418 例 / 442 事件（12.9%）** |
| 基线最佳模型 | M2 全变量 logistic，**AUROC 0.738（0.690–0.787）** |
| 参照 | SOFA 0.645；基线 P/F 0.581（range restriction 所致） |
| 结构化多模态 | S0 0.738 → **S4 0.759**，Δ +0.021（0.004–0.038），**P = 0.016**（四层中唯一显著） |
| 动态预测 | 24 / 48 / 72 h 方向一致为正（+0.010 / +0.020 / +0.019），**均不显著** |
| 简约 vs 全量 | 24 h：47 变量 0.724、校准斜率 1.145 **优于** 205 变量 0.680、0.491 |

**这三个结论都是"阴性为主"，仓库照实保留**：结构化数据已接近判别力上限，
变量更多反而让校准崩坏；动态更新的价值被"ARDS 是前移型事件"这一结构限制在 24 h 时点。

---

## 为什么要公开这套代码

**最有复用价值的不是模型，是标签口径。** 初版按"住院期间任一次达标"抽出的队列
（17,664 例、事件率 65.4%）看起来很漂亮，但实测 **84.6% 的 ARDS 在入 ICU 24 h 内
就已达标**——用首日特征去预测它，等于用 P/F 预测"由 P/F 定义的结局"，是循环论证
（incorporation bias）。本仓库把它完整重建为 t0 = 入科 + 24 h 的**前瞻式偶发 ARDS**
标签，并把 305 例"基线 P/F ≤ 300 但无 PEEP 记录"的歧义层单独处理。

如果你也在做 ICU 事件的"早期预测"，建议先跑一遍
[`docs/METHODS.md`](docs/METHODS.md) 第 3 节的自查，再谈模型。

---

## 目录结构

```
ards-early-prediction/
├── README.md                  本文件
├── LICENSE                   MIT（仅代码；数据权利另见下方合规声明）
├── CITATION.cff               引用信息
├── requirements.txt           Python 依赖
├── Makefile / run_all.py      一键复现入口
├── .env.example               配置模板（不含密码）
├── scripts/
│   ├── ards_config.py         统一配置：路径、schema、PostgreSQL 连接
│   ├── stats_core.py          规范统计核心（DeLong / bootstrap / 校准 / DCA）
│   ├── 01_extract_ards.py     MIMIC-IV 队列抽取（柏林生理学口径）
│   ├── 02_define_label.py     前瞻式偶发 ARDS 标签 + 基线呼吸生理
│   ├── 03_baseline_models.py  M1/M2/M3 + 参照臂 + 全套评估
│   ├── 04_multimodal_features.py  四层结构化多模态特征
│   ├── 05_incremental_value.py    逐层增量 + 配对 bootstrap
│   ├── 06_dynamic_prediction.py   24/48/72 h landmark 动态预测
│   └── _probe_modalities.py  可选：特征可得性诊断
├── tests/test_stats_core.py   合成数据单测（无需数据库）
├── docs/
│   ├── METHODS.md             口径、泄漏控制、统计方法、局限
│   ├── REPRODUCE.md           逐步复现 + 10 个实测踩坑 + 预期数值
│   ├── RESULTS.md             论文级结果汇总
│   ├── LABEL_DEFINITION.md    标签口径 v2（可直接进 Methods）
│   └── DATA_DICTIONARY.md     字段字典
├── data/README.md             运行期数据目录（内容不入库）
└── results/                   已提交的聚合结果与图件（无患者标识）
```

---

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v     # 不需要数据库，先验证统计核心
cp .env.example .env                        # 填写本机 PostgreSQL 参数
python scripts/ards_config.py               # 打印生效配置（不显示密码）
PGPASSWORD='****' python run_all.py         # 01→06 全流程
```

完整前置条件（PhysioNet 认证、MIMIC-IV 装载、schema 映射、常见报错）
见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

---

## 结果一览

### 基线模型（测试集 n = 1,026 / 133 事件）

| 模型 | AUROC (95%CI) | AUPRC | Brier | 校准斜率 |
|---|---|---|---|---|
| **M2 全变量 LR** | **0.738 (0.690–0.787)** | 0.313 | 0.103 | 1.09 |
| M3 梯度提升 | 0.720 (0.673–0.767) | 0.290 | 0.104 | 0.99 |
| M1 简约 LR（8 变量） | 0.698 (0.646–0.750) | 0.326 | 0.103 | 1.12 |
| SOFA 单变量 | 0.645 (0.590–0.701) | 0.230 | — | — |
| 基线 P/F 单变量 | 0.581 (0.530–0.632) | 0.179 | — | — |

全变量 LR 优于梯度提升：442 个事件不足以支撑高容量模型。DCA 显示 M2 在
**10%–25%** 阈值区间净获益最高。

### 逐层增量（同一划分，配对 bootstrap 2000 次）

| 层 | 特征数 | AUROC | ΔAUROC vs BASE | P |
|---|---|---|---|---|
| S0 BASE | 121 | 0.738 | — | — |
| S1 +RESP | 138 | 0.750 | +0.012 | 0.106 |
| S2 +GAS | 142 | 0.749 | +0.011 | 0.149 |
| S3 +CIRC | 148 | 0.752 | +0.014 | 0.088 |
| **S4 +INF** | 153 | **0.759** | **+0.021** | **0.016** |

四层累计只换来 +0.021；校准却从斜率 1.087 改善到 0.862。
16 变量简约多模态仅 0.705，提示信息集中而非分散。

### 动态 landmark（固定 24 h 视野）

| Landmark | P_STATIC | **P_DYN** | Δ | P |
|---|---|---|---|---|
| 24 h | 0.714 | **0.724** | +0.010 | 0.39 |
| 48 h | 0.634 | **0.654** | +0.020 | 0.30 |
| 72 h | 0.606 | **0.625** | +0.019 | 0.75 |

方向一致为正但无一显著；后两档测试集事件仅 36 / 26 例，CI 极宽。
原因：3,418 例中曾达 ARDS 者 **48 h 前 50.2%、72 h 前 69.0%**，而队列已剔除
基线 24 h 内发作者，越往后风险集越小越同质——动态更新天然只能在 24 h 起作用。

完整表格、敏感性分析与局限见 [`docs/RESULTS.md`](docs/RESULTS.md)。

---

## 数据合规与伦理（重要）

- MIMIC-IV 受 **PhysioNet Credentialed Health Data License / Data Use Agreement** 约束，
  **禁止再分发**。本仓库只包含 SQL 抽取逻辑与聚合统计，不含 `subject_id`、
  `hadm_id`、`stay_id` 或任何个体预测值；`data/*.csv` 与 `results/*predictions*.csv`
  已被 `.gitignore` 排除。
- 使用者须**自行**完成 PhysioNet 认证、CITI 培训与 DUA 签署，并遵守所在机构的
  伦理/数据管理规定。本仓库不提供任何绕过认证的方法。
- 数据库密码只从 `PGPASSWORD` 环境变量或本地 `.env` 读取，代码中无默认值、
  无硬编码、无日志输出；`.env` 不入库。
- MIT 许可仅覆盖**代码与文档**，不授予任何源数据权利。

引用本仓库请见 [`CITATION.cff`](CITATION.cff)；使用 MIMIC-IV 还须同时引用
Johnson AEW, et al. *MIMIC-IV, a freely accessible electronic health record dataset*.
Sci Data. 2023;10:1.

---

## 已知局限

影像学"双肺渗出"要件无法由结构化数据判定（会高估发病率）；尚无外部验证；
后段 landmark 功效不足；入选需"存活至 24 h 且基线无 ARDS"带来选择偏倚；
`bg.fio2` 缺失约 25%；主队列剔除 P/F ≤ 300 者造成 range restriction。
逐条讨论见 [`docs/METHODS.md`](docs/METHODS.md) 第 10 节。

---

## 状态

代码与结果对应中文论著《MIMIC-IV 中前瞻式偶发 ARDS 的早期预测：标签口径的
决定性作用、结构化多模态增量与 landmark 动态模型》（**撰写中 / 未投稿**）。
外部验证（院内专病队列）数据尚在录入，未纳入本版本；`results/` 中的数值均为
**内部验证**结果，不应解读为模型已具备临床可用性。

欢迎就标签口径、泄漏控制或复现差异提 issue。
