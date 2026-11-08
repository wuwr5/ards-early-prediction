.PHONY: help install test all figures clean

PY ?= python

help:
	@echo "make install   安装依赖"
	@echo "make test      运行统计核心单元测试（不需要数据库）"
	@echo "make all       顺序复现 01-06（需要 MIMIC-IV PostgreSQL）"
	@echo "make test-conf 打印当前生效的数据库配置（不显示密码）"
	@echo "make clean     删除 __pycache__"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m unittest discover -s tests -v

test-conf:
	$(PY) scripts/ards_config.py

all:
	$(PY) run_all.py

figures:
	$(PY) run_all.py --from 3

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"
