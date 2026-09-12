"""Build Colab Enterprise notebooks from the human-edited stage notebooks."""

from __future__ import annotations

import argparse
import ast
import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGES_DIR = REPO_ROOT / "notebooks" / "stages"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "colab_enterprise"
DEFAULT_MERGED_OUTPUT = DEFAULT_OUTPUT_DIR / "ZIP_RC_experiment_all_in_one.ipynb"
EXPECTED_STAGES = tuple(f"{index:02d}" for index in range(9))

CELL_TITLES = {
    "00": [
        "初始化运行环境",
        "实验进度与下一步",
        "检查 RAM、磁盘和 GPU",
        "编辑并保存实验配置",
        "Step 0 验收",
    ],
    "01": ["生成 pilot rollouts", "可视化生成长度与截断率"],
    "02": ["运行 pilot grader", "可视化质量并作出 Go / No-Go"],
    "03": ["生成并评分正式训练数据", "检查长度、标签与题目难度"],
    "04": ["按 prompt 创建 train / validation / test", "检查分布稳定性与数据泄漏"],
    "05": [
        "Stage 1：训练 correctness head",
        "生成中间 value",
        "Stage 2：训练最终 predictor",
        "检查 loss 与 value 分离",
    ],
    "06": ["准备 held-out 指标", "运行 predictor evaluation", "可视化校准与长度误差"],
    "07": ["生成、评分并预测 controller rollouts", "构造固定预算策略", "比较 Pareto 曲线"],
    "08": ["汇总全部 stage reports", "作出是否升级到 1.7B 的决定"],
}

INTRO = """
# ZIP-RC 0.6B 完整实验

这是由 `notebooks/stages/00–08` 自动合并的 Colab Enterprise 单文件版本。请编辑分步源文件，不要直接维护本文件。

## 怎么避免“又长又乱”

- 使用左侧 **目录 / Outline** 跳到对应 Step；
- 每个代码块都有标题并默认折叠；
- 重跑 `Step 00.2 — 实验进度与下一步` 查看当前完成度；
- 一次只运行当前 Step，不必 `Run all`；
- 已有产物时，可关闭对应的 `RUN_STAGE` / `RUN_EVALUATION`，只重新运行检查和画图。

## 实验顺序

`0 环境 → 1 Pilot 生成 → 2 Pilot 质量 → 3 正式数据 → 4 Split → 5 训练 → 6 Predictor 评估 → 7 Controller → 8 决策`

在 Colab Enterprise 中选择 **My notebooks → Import → Your computer** 导入本文件，并让整份 Notebook 始终连接同一个 L4 GPU runtime。
"""

COLAB_SETUP = """
from pathlib import Path
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys

REPO = Path("/content/ZIP-RC-Colab")
ZIP_PY = Path("/content/mamba/envs/zip/bin/python")
REPO_URL = "https://github.com/wtree101/ZIP-RC-Colab.git"
REPO_BRANCH = "main"
SYNC_REPO = True

if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)], check=True)
elif SYNC_REPO:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "origin", REPO_BRANCH], check=True)

if not ZIP_PY.exists():
    raise FileNotFoundError(
        f"未找到 {ZIP_PY}。请先建立 ZIP-RC 的 mamba 环境，再重新运行本 Notebook。"
    )

kernel_required = ["torch", "numpy", "pandas", "pyarrow", "matplotlib", "sklearn", "psutil"]
kernel_missing = [name for name in kernel_required if importlib.util.find_spec(name) is None]
if kernel_missing:
    raise ModuleNotFoundError(f"Colab kernel 缺少可视化依赖: {kernel_missing}")

env_check = subprocess.run(
    [
        str(ZIP_PY),
        "-c",
        (
            "import importlib.util, json; "
            "mods=['torch','vllm','transformers','datasets','pandas','pyarrow']; "
            "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
        ),
    ],
    check=True,
    capture_output=True,
    text=True,
)
env_missing = json.loads(env_check.stdout.strip())
if env_missing:
    raise ModuleNotFoundError(f"zip mamba 环境缺少依赖: {env_missing}")

os.environ["ZIPRC_PYTHON"] = str(ZIP_PY)

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
from IPython.display import display

sys.path.insert(0, str(REPO / "notebooks"))
from ziprc_notebook_utils import (
    gate,
    gate_frame,
    load_config,
    model_artifacts_exist,
    read_jsonl,
    require_columns,
    rolling_edges,
    run_repo,
    save_stage_report,
)

print("Repository:", REPO)
print("ZIP Python:", ZIP_PY)
"""

DASHBOARD = """
stage_names = {
    "00": "环境与配置",
    "01": "Pilot 生成",
    "02": "Pilot 质量",
    "03": "正式训练数据",
    "04": "Prompt split",
    "05": "Predictor 训练",
    "06": "Predictor 评估",
    "07": "Controller 对比",
    "08": "毕业决策",
}
report_directory = REPO / "artifacts" / "stage_reports"
progress_rows = []
for stage, name in stage_names.items():
    matches = sorted(report_directory.glob(f"{stage}_*.json")) if report_directory.exists() else []
    report = json.loads(matches[-1].read_text(encoding="utf-8")) if matches else None
    operational_ok = bool(report and report.get("operational_passed"))
    scientific_ok = bool(report and report.get("scientific_passed"))
    complete = operational_ok and scientific_ok
    status = "✅ 完成" if complete else "⚠️ 需检查" if report else "⬜ 未运行"
    progress_rows.append(
        {
            "Step": stage,
            "阶段": name,
            "状态": status,
            "operational": operational_ok if report else None,
            "scientific": scientific_ok if report else None,
            "report": matches[-1].name if matches else "—",
            "complete": complete,
        }
    )

progress = pd.DataFrame(progress_rows)
display(progress.drop(columns="complete"))
completed_count = int(progress["complete"].sum())
next_rows = progress[~progress["complete"]]
next_step = str(next_rows.iloc[0]["Step"]) if not next_rows.empty else None

fig, ax = plt.subplots(figsize=(9, 1.25))
ax.barh(["ZIP-RC"], [completed_count], color="#49beaa")
ax.barh(["ZIP-RC"], [len(progress) - completed_count], left=[completed_count], color="#e5e7eb")
ax.set(xlim=(0, len(progress)), xlabel="completed stages")
ax.text(completed_count / 2 if completed_count else 0.15, 0, f"{completed_count}/{len(progress)}", va="center")
plt.tight_layout()
plt.show()

if next_step is None:
    print("✅ Step 0–8 均已完成；查看 Step 08 的毕业结论。")
else:
    print(f"下一步：从左侧 Outline 跳到 Step {int(next_step)}。完成后重新运行本面板。")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages-dir", type=Path, default=DEFAULT_STAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-out", "--out", dest="merged_out", type=Path, default=None)
    return parser.parse_args()


def cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise TypeError("Notebook cell source must be a string or a list of strings.")


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip(),
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def read_stage_notebooks(stages_dir: Path) -> list[tuple[str, dict[str, object]]]:
    paths = sorted(stages_dir.glob("[0-9][0-9]_*.ipynb"))
    stages = tuple(path.name[:2] for path in paths)
    if stages != EXPECTED_STAGES:
        raise ValueError(f"Expected stages {EXPECTED_STAGES}; found {stages} in {stages_dir}.")

    notebooks = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
            raise TypeError(f"Invalid notebook structure: {path}")
        notebooks.append((path.name[:2], payload))
    return notebooks


def first_code_index(cells: list[dict[str, object]]) -> int:
    try:
        return next(index for index, cell in enumerate(cells) if cell.get("cell_type") == "code")
    except StopIteration as error:
        raise ValueError("A stage notebook has no code cells.") from error


def title_code_cells(
    cells: list[dict[str, object]],
    stage: str,
    *,
    standalone: bool = False,
) -> None:
    defaults = CELL_TITLES.get(stage, [])
    if standalone and stage != "00":
        defaults = ["初始化运行环境", *defaults]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    for index, cell in enumerate(code_cells, start=1):
        metadata = cell.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Notebook cell metadata must be an object.")
        custom_title = metadata.get("ziprc_title")
        title = (
            custom_title
            if isinstance(custom_title, str) and custom_title.strip()
            else defaults[index - 1]
            if index <= len(defaults)
            else f"代码块 {index}"
        )
        source = cell_source(cell)
        if source.startswith("# @title "):
            source = source.split("\n", 1)[1] if "\n" in source else ""
        cell["source"] = f"# @title Step {stage}.{index} — {title}\n{source}"
        metadata["cellView"] = "form"
        metadata["tags"] = [f"step-{stage}"]


def prepare_stage(stage: str, payload: dict[str, object]) -> list[dict[str, object]]:
    cells = deepcopy(payload["cells"])
    if not isinstance(cells, list):
        raise TypeError(f"Step {stage} cells must be a list.")

    markdown_index = next(
        (index for index, cell in enumerate(cells) if cell.get("cell_type") == "markdown"),
        None,
    )
    if markdown_index is None:
        raise ValueError(f"Step {stage} has no markdown heading.")
    cells[markdown_index]["source"] = cell_source(cells[markdown_index]).replace("# Step", "## Step", 1)

    bootstrap_index = first_code_index(cells)
    if stage == "00":
        cells[bootstrap_index]["source"] = dedent(COLAB_SETUP).strip()
        cells.insert(bootstrap_index + 1, code_cell(DASHBOARD))
        config_cells = [
            cell
            for cell in cells
            if cell.get("cell_type") == "code" and "config = {" in cell_source(cell)
        ]
        if len(config_cells) != 1:
            raise ValueError("Step 00 must contain exactly one experiment config cell.")
        config_cells[0]["source"] = f"{cell_source(config_cells[0])}\n\nCONFIG = config"
    else:
        bootstrap_source = cell_source(cells[bootstrap_index])
        if "ziprc_notebook_utils" not in bootstrap_source or "load_config(REPO)" not in bootstrap_source:
            raise ValueError(f"Step {stage} first code cell is not the expected bootstrap.")
        cells.pop(bootstrap_index)

    title_code_cells(cells, stage)
    for cell in cells:
        if cell.get("cell_type") == "code":
            ast.parse(cell_source(cell))
    return cells


def prepare_standalone_stage(stage: str, payload: dict[str, object]) -> list[dict[str, object]]:
    """Give one source stage a complete Colab bootstrap and collapsible cells."""
    cells = deepcopy(payload["cells"])
    if not isinstance(cells, list):
        raise TypeError(f"Step {stage} cells must be a list.")

    bootstrap_index = first_code_index(cells)
    setup = dedent(COLAB_SETUP).strip()
    if stage != "00":
        setup += dedent(
            """

            CONFIG = load_config(REPO)
            print("Experiment:", CONFIG["experiment_name"])
            """
        )
    cells[bootstrap_index]["source"] = setup

    if stage == "00":
        cells.insert(bootstrap_index + 1, code_cell(DASHBOARD))

    title_code_cells(cells, stage, standalone=True)
    for cell in cells:
        if cell.get("cell_type") == "code":
            ast.parse(cell_source(cell))
    return cells


def standalone_notebook(stage: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "cells": prepare_standalone_stage(stage, payload),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def merge_notebooks(stages: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    cells = [markdown_cell(INTRO)]
    for stage, payload in stages:
        cells.extend(prepare_stage(stage, payload))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    args = parse_args()
    stages_dir = args.stages_dir.resolve()
    stages = read_stage_notebooks(stages_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage, payload in stages:
        filename = next(path.name for path in stages_dir.glob(f"{stage}_*.ipynb"))
        output = output_dir / filename
        output.write_text(
            json.dumps(standalone_notebook(stage, payload), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Built Step {stage} -> {output}")

    merged_output = (
        args.merged_out.resolve()
        if args.merged_out is not None
        else output_dir / DEFAULT_MERGED_OUTPUT.name
    )
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged_output.write_text(
        json.dumps(merge_notebooks(stages), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Merged {len(stages)} stages -> {merged_output}")


if __name__ == "__main__":
    main()
