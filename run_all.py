#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_all.py — 顺序复现 01–06 全流程（不接触任何密钥）

用法
----
    python run_all.py                 # 全部六步
    python run_all.py --from 3       # 从 03 开始（01/02 的产物已就绪时）
    python run_all.py --only 5       # 只跑 05
    python run_all.py --dry-run      # 仅打印将执行的命令

数据库密码由各脚本自行从 PGPASSWORD / .env 读取；本脚本不接收、不打印密码。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

STEPS = [
    (1, "01_extract_ards.py", "队列抽取", ["data/ards_label.csv", "data/ards_model_cohort.csv"], True),
    (2, "02_define_label.py", "前瞻式标签定义", ["data/ards_analysis_cohort.csv"], True),
    (3, "03_baseline_models.py", "基线模型与评估", ["results/model_performance.csv"], False),
    (4, "04_multimodal_features.py", "结构化多模态特征", ["data/ards_cohort_mm.csv"], True),
    (5, "05_incremental_value.py", "逐层增量价值", ["results/mm_incremental_performance.csv"], False),
    (6, "06_dynamic_prediction.py", "landmark 动态预测", ["results/dyn_performance.csv"], True),
]


def main():
    ap = argparse.ArgumentParser(description="顺序复现 ARDS 早期预测全流程")
    ap.add_argument("--from", dest="start", type=int, default=1, help="从第几步开始")
    ap.add_argument("--to", dest="end", type=int, default=6, help="到第几步结束")
    ap.add_argument("--only", type=int, default=None, help="只运行指定一步")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    todo = [s for s in STEPS
            if (args.only is None and args.start <= s[0] <= args.end)
            or (args.only == s[0])]
    if not todo:
        sys.exit("没有匹配的步骤，请检查 --from / --to / --only")

    print("仓库根目录:", ROOT)
    print("待执行:", ", ".join("%02d %s" % (s[0], s[2]) for s in todo))
    if any(s[4] for s in todo):
        print("注意：含需要 PostgreSQL 的步骤，请确保已配置 .env / PGPASSWORD")
    print("-" * 66)

    for num, script, label, outputs, needs_db in todo:
        path = os.path.join(SCRIPTS, script)
        cmd = [sys.executable, path]
        print("[%02d] %s  ->  %s" % (num, label, " ".join(cmd)), flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            sys.exit("[%02d] 失败（exit=%d），中止后续步骤" % (num, rc))
        missing = [o for o in outputs if not os.path.exists(os.path.join(ROOT, o))]
        if missing:
            sys.exit("[%02d] 未生成预期产物：%s" % (num, ", ".join(missing)))
        print("[%02d] 完成，用时 %.1f s" % (num, time.time() - t0), flush=True)

    if not args.dry_run:
        print("-" * 66)
        print("全流程完成。结果见 results/，方法说明见 docs/。")


if __name__ == "__main__":
    main()
