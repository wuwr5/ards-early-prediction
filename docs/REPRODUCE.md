# 复现指南

## 0. 前置条件（缺一不可）

1. **PhysioNet 认证**：完成 CITI「Data or Specimens Only Research」培训，签署
   MIMIC-IV Data Use Agreement，取得 credentialed 访问资格。
   数据须自行从 PhysioNet 下载；**本仓库不分发任何源数据**。
2. **MIMIC-IV 已装载到本地 PostgreSQL**（推荐 v2.2 或 v3.1），schema 名为
   `mimiciv_hosp` / `mimiciv_icu` / `mimiciv_derived` / `mimiciv_note`。
3. **Python ≥ 3.9**，建议 3.10/3.11；PostgreSQL ≥ 14。

## 1. 安装

```bash
git clone https://github.com/wuwr5/ards-early-prediction.git
cd ards-early-prediction
python -m pip install -r requirements.txt
```

## 2. 配置

```bash
cp .env.example .env      # Windows Git Bash；PowerShell 用 Copy-Item
```

编辑 `.env`，填入本机端口、库名、schema（默认值对应标准安装）：

| 变量 | 默认 | 本项目原始环境 |
|---|---|---|
| `PGHOST` | `localhost` | `localhost` |
| `PGPORT` | `5432` | **`2023`** |
| `PGDATABASE` | `mimiciv` | **`postgres`** |
| `PGUSER` | `postgres` | `postgres` |
| `PGPASSWORD` | — | **只放本地 `.env`，切勿提交** |

密码也可以不写进文件，临时用环境变量传入：

```bash
# Git Bash / Linux / macOS
PGPASSWORD='****' python run_all.py

# Windows PowerShell
$env:PGPASSWORD="****"; python run_all.py
```

`.env` 已被 `.gitignore` 排除。校验当前生效配置（**不显示密码**）：

```bash
python scripts/ards_config.py
```

## 3. 运行

```bash
python run_all.py                 # 01→06 全流程
python run_all.py --from 3       # 01/02 产物已存在时，只跑建模部分
python run_all.py --only 5       # 只跑增量分析
python run_all.py --dry-run      # 只打印命令
```

`run_all.py` 每步结束后检查预期产物是否落盘，失败即中止，不会静默继续。
也可用 `make all` / `make figures`（从 03 开始）。

| 步 | 脚本 | 需要数据库 | 前置 | 主要产物 | 本机参考耗时 |
|---|---|---|---|---|---|
| 01 | `01_extract_ards.py` | 是 | — | `data/ards_label.csv` 等 4 个 | ~1 min |
| 02 | `02_define_label.py` | 是 | 01 | `data/ards_analysis_cohort.csv` | ~1 min |
| 03 | `03_baseline_models.py` | 否 | 02 | `results/model_performance.csv` + 4 图 | ~2 min |
| 04 | `04_multimodal_features.py` | 是 | 02 | `data/ards_cohort_mm.csv` | ~2 min |
| 05 | `05_incremental_value.py` | 否 | 03/04 | `results/mm_*.csv` + 2 图 | ~3 min |
| 06 | `06_dynamic_prediction.py` | 是 | 02/04 | `results/dyn_*.csv` + 4 图 | ~5 min |

> 03 / 05 只读本地 CSV，可在没有数据库的机器上重跑模型与统计部分。
> 04 可以跳过：若已有等价的多模态宽表，直接放进 `data/ards_cohort_mm.csv` 即可。

## 4. 单元测试（不需要数据库）

```bash
python -m unittest discover -s tests -v
# 或 make test
```

测试覆盖 DeLong CI、bootstrap、两种配对 bootstrap、
**校准斜率（防止退回 OLS 的系统性低估）**、HL、DCA。全部使用合成数据。

## 5. 预期数值（用于判断是否复现一致）

| 检查点 | 预期 |
|---|---|
| 01 纳排终点 | 17,664 例，ARDS 11,555（65.4%） |
| 02 管线队列 | 3,723 例 / 527 事件 |
| 02 主分析队列 | **3,418 例 / 442 事件（12.9%）** |
| 03 测试集 | n = 1,026 / 133 事件 |
| 03 M2 AUROC | **0.738（0.690–0.787）** |
| 05 S4 AUROC | **0.759**，Δ +0.021（0.004–0.038），P = 0.016 |
| 06 24 h P_DYN | 0.724，校准斜率 1.145 |

若 01 起就对不上，先核对 MIMIC-IV 版本与 `mimiciv_derived.bg` 的表结构。

## 6. 本机踩过的坑（复现时最可能卡住的地方）

1. **`mimiciv_derived.bg` 没有 `stay_id`**：必须经 `mimiciv_icu.icustays` 按
   `subject_id` + `charttime` 落在住院区间内映射，否则时间窗 join 会错配。
2. **`specimen` 值是 `'ART.'`（带句点）**：用 `LIKE 'ART%'`，不要写 `= 'ART'`。
3. **`bg.fio2` 是百分数**（60 表示 60%），不是 0–1 小数；直接当小数会把 P/F 算高 100 倍。
4. **时间窗关联子查询无索引会挂死**：先建带索引的临时表（`_peep` / `_c`），
   再做 `LATERAL` 时间窗 join；本仓库 01 已这样实现，约 33 s 完成。
5. **`mimiciv_note.radiology` 是空表（0 行）**：影像文本不可用，别在脚本里假设有数据。
6. **校准斜率必须用 logistic**，不能用 OLS（真值 1.0 时 OLS 给 0.095）。
7. **pandas 空表布尔索引陷阱**：空表上 `d[mask]` 若 mask 是空 object Series
   会被当成列名列表；统一写 `d.loc[mask.fillna(False).to_numpy(dtype=bool)]`。
8. **pandas 隐式对齐**：给 DataFrame 赋 Series 时偶发整列变 NaN，模型会静默退化；
   统一用 `Series.reindex(df.index)` 并加断言。
9. **matplotlib 中文方框**：图内文字一律英文；中文字体缺 Unicode 上下标时改用 mathtext。
10. **`merge_asof` 排序要求**：跨表时间匹配在 pandas 里极易报排序错，
    本项目改在 PostgreSQL 侧完成。

## 7. 允许出现差异的地方

- MIMIC-IV **版本不同**（v2.2 vs v3.1）会改变纳排计数；
- `sklearn` 版本影响 `HistGradientBoosting` 与 `LogisticRegression(penalty=None)` 的细节；
- 随机种子固定为 42，但并行 BLAS 线程数可能让末位小数略有浮动；
- 若把 `fio2` 缺失值用呼吸机设置补算，队列会多约 2.8% 例数（本项目**未**采用）。

发现系统性偏差时，请把 MIMIC-IV 版本、`python -m pip freeze` 与
`results/*_meta.json` 一并贴到 issue 中，便于定位。
