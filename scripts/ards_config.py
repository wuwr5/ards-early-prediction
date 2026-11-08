#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ards_config.py — 全仓库统一的配置入口（无密钥、无绝对路径）

设计原则
--------
1. 所有连接参数、路径均由环境变量 / 仓库根目录的 .env 文件提供；
   **密码绝不写入任何版本控制文件**。
2. 路径一律相对本文件位置推导，克隆到任意目录都能运行。
3. 标准库实现 .env 解析，不引入 python-dotenv 这类额外依赖。

优先级：进程环境变量 > 仓库根目录 .env > 内置默认值。
"""
import os

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
DOCS_DIR = os.path.join(ROOT, "docs")

for _d in (DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# MIMIC-IV v2.2 / v3.1 的 schema 名；如自建库改名，用 MIMIC_* 环境变量覆盖
SCHEMAS = {
    "hosp": os.environ.get("MIMIC_SCHEMA_HOSP", "mimiciv_hosp"),
    "icu": os.environ.get("MIMIC_SCHEMA_ICU", "mimiciv_icu"),
    "derived": os.environ.get("MIMIC_SCHEMA_DERIVED", "mimiciv_derived"),
    "note": os.environ.get("MIMIC_SCHEMA_NOTE", "mimiciv_note"),
}


def _load_dotenv():
    """读取仓库根目录 .env（KEY=VALUE），不覆盖已有的进程环境变量。"""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
def db_config(require_password=True):
    """返回 psycopg2.connect 可直接展开的关键字参数字典。

    默认值对应标准 PostgreSQL 安装；若使用本项目的原始环境
    （端口 2023、数据库名 postgres），在 .env 里设置 PGPORT / PGDATABASE 即可。
    """
    cfg = {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "mimiciv"),
        "user": os.environ.get("PGUSER", "postgres"),
    }
    if require_password:
        pwd = os.environ.get("PGPASSWORD")
        if not pwd:
            raise SystemExit(
                "ERROR: 未提供数据库密码。请设置环境变量 PGPASSWORD，"
                "或在仓库根目录的 .env 中填写（.env 已被 .gitignore 排除）。\n"
                "  例：PGPASSWORD=**** python scripts/01_extract_ards.py"
            )
        cfg["password"] = pwd
    return cfg


def describe():
    """打印当前生效配置（不含密码），用于复现记录。"""
    cfg = db_config(require_password=False)
    return ("host={host} port={port} dbname={dbname} user={user} "
            "schemas={s}".format(s=",".join(SCHEMAS.values()), **cfg))


if __name__ == "__main__":
    print("ROOT =", ROOT)
    print(describe())
