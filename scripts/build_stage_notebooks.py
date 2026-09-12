"""Build the stage-oriented ZIP-RC experiment notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip(),
    }


def notebook(*cells: dict[str, object]) -> dict[str, object]:
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOTSTRAP = """
from pathlib import Path
import os
import sys

candidates = [Path.cwd(), *Path.cwd().parents, Path("/content/ZIP-RC-Colab"), Path("/content/ZIP-RC")]
REPO = next(
    (path for path in candidates if (path / "notebooks" / "ziprc_notebook_utils.py").exists()),
    None,
)
if REPO is None:
    raise FileNotFoundError("找不到 ZIP-RC 仓库；请从仓库根目录或 notebooks/ 运行。")

colab_python = Path("/content/mamba/envs/zip/bin/python")
if colab_python.exists():
    os.environ["ZIPRC_PYTHON"] = str(colab_python)

sys.path.insert(0, str(REPO / "notebooks"))
from ziprc_notebook_utils import *

CONFIG = load_config(REPO)
print("Repository:", REPO)
print("Experiment:", CONFIG["experiment_name"])
"""


COLAB_BOOTSTRAP = """
from pathlib import Path
import os
import subprocess
import sys

REPO = Path("/content/ZIP-RC-Colab")
ZIP_PY = Path("/content/mamba/envs/zip/bin/python")

if not REPO.exists():
    raise FileNotFoundError("远端仓库不存在；请先运行 colab/00_memory_and_config.ipynb。")
if not ZIP_PY.exists():
    raise FileNotFoundError("ZIP Python 环境不存在；请先运行 colab/00_memory_and_config.ipynb。")

os.environ["ZIPRC_PYTHON"] = str(ZIP_PY)
sys.path.insert(0, str(REPO / "notebooks"))
from ziprc_notebook_utils import *

CONFIG = load_config(REPO)
print("Repository:", REPO)
print("ZIP Python:", ZIP_PY)
print("Experiment:", CONFIG["experiment_name"])
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

# Notebook kernel 只负责分析和画图；vLLM/transformers/datasets 在 ZIP_PY 子进程中检查。
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
from ziprc_notebook_utils import gate, gate_frame, save_stage_report

print("Repository:", REPO)
print("ZIP Python:", ZIP_PY)
"""


def build_00() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 0 — 内存、磁盘与实验配置

            对应 `reference/plan.md` 的 Step 0。本 Notebook 不删除任何缓存或模型，先回答：

            - 系统内存、GPU 显存、磁盘是否足以开始；
            - 是否有遗留进程占用 GPU；
            - 建立后续所有 Notebook 共用的 3090/L4 单卡配置。

            **通过标准：** CUDA 可用、GPU 显存约 24GB、磁盘至少预留 15GB、系统可用内存至少 12GB。
            """
        ),
        code(
            """
            from pathlib import Path
            import gc
            import importlib.util
            import json
            import shutil
            import subprocess
            import sys

            required = ["torch", "vllm", "transformers", "datasets", "pandas", "pyarrow", "matplotlib", "sklearn", "psutil"]
            missing = [name for name in required if importlib.util.find_spec(name) is None]
            if missing:
                raise ModuleNotFoundError(f"缺少依赖 {missing}；请先执行 conda env create -f environment.yml")

            import matplotlib.pyplot as plt
            import pandas as pd
            import psutil
            import torch
            from IPython.display import display

            candidates = [Path.cwd(), *Path.cwd().parents, Path("/content/ZIP-RC")]
            REPO = next((path for path in candidates if (path / "src" / "generate_ziprc_rollouts.py").exists()), None)
            if REPO is None:
                raise FileNotFoundError("找不到 ZIP-RC 仓库根目录。")
            sys.path.insert(0, str(REPO / "notebooks"))
            from ziprc_notebook_utils import gate, gate_frame, save_stage_report
            """
        ),
        code(
            """
            # 安全清理当前 Notebook 自己不再引用的 Python/CUDA 缓存；不会删除文件，也不会终止其他进程。
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vm = psutil.virtual_memory()
            disk = shutil.disk_usage(REPO)
            cuda_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "No CUDA GPU"
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda_ok else 0.0
            gpu_free_gb = torch.cuda.mem_get_info(0)[0] / 2**30 if cuda_ok else 0.0

            resources = pd.DataFrame(
                [
                    {"resource": "System RAM", "used_gb": (vm.total - vm.available) / 2**30, "free_gb": vm.available / 2**30},
                    {"resource": "Disk", "used_gb": (disk.total - disk.free) / 2**30, "free_gb": disk.free / 2**30},
                    {"resource": "GPU VRAM", "used_gb": gpu_total_gb - gpu_free_gb, "free_gb": gpu_free_gb},
                ]
            )
            display(resources.round(2))

            ax = resources.set_index("resource")[["used_gb", "free_gb"]].plot(
                kind="barh", stacked=True, figsize=(9, 3.5), color=["#ef767a", "#49beaa"]
            )
            ax.set_xlabel("GB")
            ax.set_title("开始实验前的资源占用")
            ax.legend(["已用", "可用"], loc="lower right")
            plt.tight_layout()
            plt.show()

            def directory_size_gb(path):
                if not path.exists():
                    return 0.0
                return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 2**30

            storage_roots = {
                "repo/data": REPO / "data",
                "repo/models": REPO / "models",
                "HF cache": Path.home() / ".cache" / "huggingface",
                "torch cache": Path.home() / ".cache" / "torch",
                "pip cache": Path.home() / ".cache" / "pip",
            }
            storage = pd.Series({name: directory_size_gb(path) for name, path in storage_roots.items()}).sort_values()
            display(storage.rename("GB").to_frame().round(2))
            storage.plot.barh(figsize=(9, 3.5), color="#f2cf5b", title="常见数据/模型缓存占用（只读检查）")
            plt.xlabel("GB")
            plt.tight_layout()
            plt.show()

            print({"gpu": gpu_name, "vram_gb": round(gpu_total_gb, 1), "bf16": torch.cuda.is_bf16_supported() if cuda_ok else False})
            try:
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                    check=False, capture_output=True, text=True,
                ).stdout.strip()
                print("GPU compute processes:\\n", apps or "无正在运行的 compute process")
            except FileNotFoundError:
                print("nvidia-smi 不可用；跳过进程列表。")
            """
        ),
        code(
            """
            # 单张 RTX 3090 / Colab L4 的默认正式小实验。后续只需修改这里并重新运行本 cell。
            config = {
                "experiment_name": "qwen3_0.6b_ziprc_small_validation",
                "model_id": "Qwen/Qwen3-0.6B",
                "grader_model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                "dataset": "rohinm/adaptivemath",
                "split": "train",
                "prompt_column": "problem",
                "answer_column": "answer",
                "dtype": "bfloat16",
                "distribution_token_id": 151669,
                "num_length_bins": 8,
                "reward_values": [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0],
                "generation_max_model_len": 4096,
                "max_output_tokens": 2048,
                "max_num_seqs": 2,
                "temperature": 1.0,
                "min_p": 0.1,
                "pilot_prompts": 200,
                "pilot_rollouts_per_prompt": 2,
                "training_prompts": 2000,
                "training_rollouts_per_prompt": 2,
                "train_fraction": 0.8,
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "stage1_steps": 500,
                "stage2_steps": 800,
                "num_epochs": 3,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "stage1_learning_rate": 1e-4,
                "stage2_learning_rate": 5e-5,
                "train_max_length": 4096,
                "grader_max_model_len": 4096,
                "gpu_memory_utilization": 0.85,
                "controller_rollouts_per_prompt": 4,
                "paths": {
                    "pilot": "data/pilot_rollouts.parquet",
                    "full": "data/experiment_rollouts.parquet",
                    "train": "data/splits/train.parquet",
                    "validation": "data/splits/validation.parquet",
                    "test": "data/splits/test.parquet",
                    "train_value": "data/splits/train_with_value.parquet",
                    "intermediate_model": "models/experiment_joint_correct",
                    "final_model": "models/experiment_ziprc_final",
                    "stage1_metrics": "artifacts/metrics/stage1.jsonl",
                    "stage2_metrics": "artifacts/metrics/stage2.jsonl",
                    "predictor_positions": "artifacts/predictor_positions.parquet",
                    "predictor_metrics": "artifacts/predictor_metrics.json",
                    "controller_rollouts": "data/controller_rollouts.parquet",
                    "controller_scored": "data/controller_scored.parquet",
                },
            }

            for directory in [REPO / "data" / "splits", REPO / "models", REPO / "artifacts" / "metrics", REPO / "artifacts" / "stage_reports"]:
                directory.mkdir(parents=True, exist_ok=True)
            config_path = REPO / "artifacts" / "experiment_config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            display(pd.DataFrame([config]).T.rename(columns={0: "value"}))
            print("Saved:", config_path)
            """
        ),
        code(
            """
            checks = [
                gate("CUDA 可用", cuda_ok, gpu_name),
                gate("24GB 级 GPU", gpu_total_gb >= 20, f"检测到 {gpu_total_gb:.1f} GB；目标为 RTX 3090/L4"),
                gate("系统可用内存", vm.available / 2**30 >= 12, f"可用 {vm.available / 2**30:.1f} GB"),
                gate("磁盘空间", disk.free / 2**30 >= 15, f"可用 {disk.free / 2**30:.1f} GB"),
                gate("BF16 可用", bool(cuda_ok and torch.cuda.is_bf16_supported()), "配置固定使用 BF16"),
                gate("配置已保存", config_path.exists(), str(config_path)),
            ]
            display(gate_frame(checks))
            report = save_stage_report(
                REPO,
                "00_memory_and_config",
                checks,
                {"gpu": gpu_name, "gpu_total_gb": gpu_total_gb, "ram_available_gb": vm.available / 2**30, "disk_free_gb": disk.free / 2**30},
            )
            print("Stage report:", report)
            """
        ),
    )


def build_01() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 1 — Pilot rollout 与生成长度验收

            生成 `200 prompts × 2 rollouts = 400 trajectories`。本阶段只解决生成长度与截断问题，不训练。

            **Go/No-Go：** finished rate ≥95%、撞到 2048-token 上限的比例 <5%、字段和样本数完整。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            RUN_STAGE = True
            output_path = REPO / CONFIG["paths"]["pilot"]
            if RUN_STAGE:
                run_repo(
                    REPO,
                    "python3", "src/generate_ziprc_rollouts.py",
                    "--model", CONFIG["model_id"],
                    "--dataset", CONFIG["dataset"],
                    "--split", CONFIG["split"],
                    "--prompt-column", CONFIG["prompt_column"],
                    "--answer-column", CONFIG["answer_column"],
                    "--out", output_path,
                    "--max-num-prompts", CONFIG["pilot_prompts"],
                    "--thinking-samples", 0,
                    "--non-thinking-samples", CONFIG["pilot_rollouts_per_prompt"],
                    "--temperature", CONFIG["temperature"],
                    "--min-p", CONFIG["min_p"],
                    "--max-model-len", CONFIG["generation_max_model_len"],
                    "--max-new-tokens", CONFIG["max_output_tokens"],
                    "--max-num-seqs", CONFIG["max_num_seqs"],
                    "--dtype", CONFIG["dtype"],
                    "--dp-size", 1, "--tp-size", 1,
                )
            print("Artifact:", output_path)
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            df = pd.read_parquet(output_path)
            required = ["prompt_idx", "prompt", "answer", "response", "length", "finished", "reasoning_enabled", "input_ids", "label_positions"]
            missing = require_columns(df, required)
            if missing:
                raise ValueError(f"缺少列: {missing}")

            expected_rows = int(CONFIG["pilot_prompts"]) * int(CONFIG["pilot_rollouts_per_prompt"])
            finished_rate = float(df["finished"].mean())
            cap_rate = float((df["length"] >= int(CONFIG["max_output_tokens"])).mean())

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(df["length"], bins=30, color="#4c78a8", edgecolor="white")
            axes[0].axvline(CONFIG["max_output_tokens"], color="#e45756", linestyle="--", label="output cap")
            axes[0].set(title="Pilot response length", xlabel="output tokens", ylabel="rollouts")
            axes[0].legend()
            rates = pd.Series({"finished": finished_rate, "hit output cap": cap_rate})
            rates.plot.bar(ax=axes[1], color=["#49beaa", "#ef767a"], ylim=(0, 1))
            axes[1].axhline(0.95, color="#49beaa", linestyle=":")
            axes[1].axhline(0.05, color="#ef767a", linestyle=":")
            axes[1].set(title="Completion / truncation gates", ylabel="fraction")
            plt.tight_layout()
            plt.show()

            display(df["length"].describe(percentiles=[.5, .9, .95, .99]).to_frame("tokens").round(2))
            display(df[["prompt", "response", "length", "finished"]].head(5))

            checks = [
                gate("字段完整", not missing, f"missing={missing}"),
                gate("样本数完整", len(df) == expected_rows, f"{len(df)}/{expected_rows}"),
                gate("Prompt 数完整", df["prompt_idx"].nunique() == CONFIG["pilot_prompts"], f"{df['prompt_idx'].nunique()} prompts"),
                gate("Finished rate ≥95%", finished_rate >= 0.95, f"{finished_rate:.1%}", kind="scientific"),
                gate("撞输出上限 <5%", cap_rate < 0.05, f"{cap_rate:.1%}", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "01_pilot_generation", checks, {"rows": len(df), "finished_rate": finished_rate, "cap_rate": cap_rate, "p95_length": float(df['length'].quantile(.95))})
            """
        ),
    )


def build_02() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 2 — Pilot 标签与数据质量

            用数学 grader 写入 `correct`，检查完成率、正确率、正负样本平衡和长度关系。

            **Go/No-Go：** 正确率最好落在 30%–70%；至少要求两类都有、少数类 ≥10%。若不满足，应调整题目难度，而不是直接扩采。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            RUN_STAGE = True
            pilot_path = REPO / CONFIG["paths"]["pilot"]
            grader_metrics = REPO / "artifacts/metrics/pilot_grader.json"
            if RUN_STAGE:
                run_repo(
                    REPO,
                    "python3", "src/evaluate_and_label_rollouts.py",
                    "--data", pilot_path,
                    "--model", CONFIG["grader_model_id"],
                    "--tensor-parallel-size", 1,
                    "--gpu-memory-utilization", CONFIG["gpu_memory_utilization"],
                    "--max-model-len", CONFIG["grader_max_model_len"],
                    "--max-num-seqs", CONFIG["max_num_seqs"],
                    "--dtype", CONFIG["dtype"],
                    "--output-json", grader_metrics,
                    "--show-examples",
                )
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            df = pd.read_parquet(pilot_path)
            missing = require_columns(df, ["finished", "correct", "length"])
            if missing:
                raise ValueError(f"缺少列: {missing}")
            accuracy = float(df["correct"].mean())
            class_share = df["correct"].value_counts(normalize=True)
            minority_share = float(class_share.min()) if len(class_share) == 2 else 0.0

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            pd.crosstab(df["finished"], df["correct"]).plot.bar(stacked=True, ax=axes[0], color=["#e45756", "#49beaa"])
            axes[0].set_title("Finished × Correct")
            df["correct"].value_counts().sort_index().plot.bar(ax=axes[1], color=["#e45756", "#49beaa"])
            axes[1].set_title(f"Correctness balance ({accuracy:.1%} correct)")
            for label, group in df.groupby("correct"):
                axes[2].hist(group["length"], bins=25, alpha=.55, label=f"correct={label}")
            axes[2].set(title="Length by correctness", xlabel="output tokens")
            axes[2].legend()
            plt.tight_layout()
            plt.show()

            display(df.groupby("correct")["length"].describe().round(1))
            checks = [
                gate("Correct 标签完整", not missing and df["correct"].notna().all(), f"null={int(df['correct'].isna().sum())}"),
                gate("正负样本同时存在", df["correct"].nunique() == 2, str(df["correct"].value_counts().to_dict()), kind="scientific"),
                gate("少数类 ≥10%", minority_share >= 0.10, f"minority={minority_share:.1%}", kind="scientific"),
                gate("理想正确率 30%–70%", 0.30 <= accuracy <= 0.70, f"accuracy={accuracy:.1%}", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "02_pilot_quality", checks, {"accuracy": accuracy, "minority_share": minority_share, "finished_rate": float(df['finished'].mean())})
            """
        ),
    )


def build_03() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 3 — 正式训练数据生成与标注

            在 Pilot 通过后生成 `2,000 prompts × 2 = 4,000 trajectories`，并写入 correctness 标签。

            本阶段再次检查截断和标签平衡，防止规模扩大后数据分布发生变化。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            RUN_STAGE = True
            full_path = REPO / CONFIG["paths"]["full"]
            full_grader_metrics = REPO / "artifacts/metrics/full_grader.json"
            if RUN_STAGE:
                run_repo(
                    REPO,
                    "python3", "src/generate_ziprc_rollouts.py",
                    "--model", CONFIG["model_id"], "--dataset", CONFIG["dataset"],
                    "--split", CONFIG["split"], "--prompt-column", CONFIG["prompt_column"],
                    "--answer-column", CONFIG["answer_column"], "--out", full_path,
                    "--max-num-prompts", CONFIG["training_prompts"],
                    "--thinking-samples", 0, "--non-thinking-samples", CONFIG["training_rollouts_per_prompt"],
                    "--temperature", CONFIG["temperature"], "--min-p", CONFIG["min_p"],
                    "--max-model-len", CONFIG["generation_max_model_len"],
                    "--max-new-tokens", CONFIG["max_output_tokens"],
                    "--max-num-seqs", CONFIG["max_num_seqs"], "--dtype", CONFIG["dtype"],
                    "--dp-size", 1, "--tp-size", 1,
                )
                run_repo(
                    REPO,
                    "python3", "src/evaluate_and_label_rollouts.py",
                    "--data", full_path, "--model", CONFIG["grader_model_id"],
                    "--tensor-parallel-size", 1, "--gpu-memory-utilization", CONFIG["gpu_memory_utilization"],
                    "--max-model-len", CONFIG["grader_max_model_len"], "--max-num-seqs", CONFIG["max_num_seqs"],
                    "--dtype", CONFIG["dtype"], "--output-json", full_grader_metrics,
                )
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            df = pd.read_parquet(full_path)
            expected = int(CONFIG["training_prompts"]) * int(CONFIG["training_rollouts_per_prompt"])
            accuracy = float(df["correct"].mean())
            finished_rate = float(df["finished"].mean())
            cap_rate = float((df["length"] >= CONFIG["max_output_tokens"]).mean())

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(df["length"], bins=40, color="#4c78a8")
            axes[0].axvline(CONFIG["max_output_tokens"], color="#e45756", linestyle="--")
            axes[0].set(title="Full-data response length", xlabel="tokens")
            df["correct"].value_counts().sort_index().plot.bar(ax=axes[1], color=["#e45756", "#49beaa"])
            axes[1].set_title("Correct / incorrect samples")
            per_prompt = df.groupby("prompt_idx").agg(correct_rate=("correct", "mean"), mean_length=("length", "mean"))
            axes[2].scatter(per_prompt["mean_length"], per_prompt["correct_rate"], s=10, alpha=.35)
            axes[2].set(title="Per-prompt difficulty", xlabel="mean output tokens", ylabel="correct rate")
            plt.tight_layout()
            plt.show()

            checks = [
                gate("样本数完整", len(df) == expected, f"{len(df)}/{expected}"),
                gate("Prompt 数完整", df["prompt_idx"].nunique() == CONFIG["training_prompts"], f"{df['prompt_idx'].nunique()} prompts"),
                gate("Finished rate ≥95%", finished_rate >= .95, f"{finished_rate:.1%}", kind="scientific"),
                gate("截断 <5%", cap_rate < .05, f"{cap_rate:.1%}", kind="scientific"),
                gate("正负标签可学习", df["correct"].nunique() == 2 and df["correct"].value_counts(normalize=True).min() >= .10, f"accuracy={accuracy:.1%}", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "03_training_data", checks, {"rows": len(df), "prompts": int(df['prompt_idx'].nunique()), "accuracy": accuracy, "finished_rate": finished_rate, "cap_rate": cap_rate})
            """
        ),
    )


def build_04() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 4 — 按 prompt 切分 train / validation / test

            同一题的所有 rollout 必须属于同一个 split，避免 trajectory-level leakage。

            默认使用 80% / 10% / 10%，并验证三个集合的 prompt 完全不重叠。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import numpy as np
            import pandas as pd
            from IPython.display import display

            source_path = REPO / CONFIG["paths"]["full"]
            df = pd.read_parquet(source_path)
            prompt_ids = np.array(sorted(df["prompt_idx"].unique()))
            rng = np.random.default_rng(42)
            rng.shuffle(prompt_ids)

            n_prompts = len(prompt_ids)
            train_end = int(n_prompts * CONFIG["train_fraction"])
            validation_end = train_end + int(n_prompts * CONFIG["validation_fraction"])
            assignments = {
                "train": set(prompt_ids[:train_end].tolist()),
                "validation": set(prompt_ids[train_end:validation_end].tolist()),
                "test": set(prompt_ids[validation_end:].tolist()),
            }

            split_frames = {}
            for name, ids in assignments.items():
                split_frame = df[df["prompt_idx"].isin(ids)].copy().reset_index(drop=True)
                split_frame["data_split"] = name
                path = REPO / CONFIG["paths"][name]
                path.parent.mkdir(parents=True, exist_ok=True)
                split_frame.to_parquet(path, index=False)
                split_frames[name] = split_frame
                print(name, path, len(split_frame))
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            summary = pd.DataFrame([
                {
                    "split": name,
                    "prompts": frame["prompt_idx"].nunique(),
                    "rollouts": len(frame),
                    "accuracy": frame["correct"].mean(),
                    "finished_rate": frame["finished"].mean(),
                    "median_length": frame["length"].median(),
                }
                for name, frame in split_frames.items()
            ])
            display(summary.round(3))

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            summary.set_index("split")[["prompts", "rollouts"]].plot.bar(ax=axes[0])
            axes[0].set_title("Split sizes")
            summary.set_index("split")["accuracy"].plot.bar(ax=axes[1], color="#49beaa", ylim=(0, 1))
            axes[1].set_title("Correctness stability")
            for name, frame in split_frames.items():
                axes[2].hist(frame["length"], bins=30, histtype="step", density=True, label=name)
            axes[2].set(title="Length distributions", xlabel="tokens")
            axes[2].legend()
            plt.tight_layout()
            plt.show()

            overlaps = {
                "train∩validation": len(assignments["train"] & assignments["validation"]),
                "train∩test": len(assignments["train"] & assignments["test"]),
                "validation∩test": len(assignments["validation"] & assignments["test"]),
            }
            checks = [
                gate("无 prompt 泄漏", sum(overlaps.values()) == 0, str(overlaps)),
                gate("所有数据均已分配", sum(len(frame) for frame in split_frames.values()) == len(df), f"{sum(len(frame) for frame in split_frames.values())}/{len(df)}"),
                gate("每个 split 正负标签都有", all(frame["correct"].nunique() == 2 for frame in split_frames.values()), str({name: frame['correct'].value_counts().to_dict() for name, frame in split_frames.items()}), kind="scientific"),
                gate("Split accuracy 漂移 ≤10pp", summary["accuracy"].max() - summary["accuracy"].min() <= .10, f"range={summary['accuracy'].max() - summary['accuracy'].min():.1%}", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "04_prompt_split", checks, {"summary": summary.to_dict(orient="records"), "overlaps": overlaps})
            """
        ),
    )


def build_05() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 5 — 两阶段 ZIP-RC predictor 训练

            1. Stage 1：用 `correct`、KL=0 训练 intermediate predictor；
            2. 用 intermediate model 给 train split 写入 denoised `value`；
            3. Stage 2：用 `value`、KL=10 训练最终 ZIP-RC。

            训练脚本会把每个 optimizer step 的 loss 写到 JSONL，本 Notebook 绘制 loss、KL 和学习率曲线。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            RUN_STAGE_1 = True
            train_path = REPO / CONFIG["paths"]["train"]
            intermediate_model = REPO / CONFIG["paths"]["intermediate_model"]
            stage1_metrics = REPO / CONFIG["paths"]["stage1_metrics"]
            if RUN_STAGE_1:
                run_repo(
                    REPO,
                    "python3", "src/train_ziprc_joint_head.py",
                    "--model-id", CONFIG["model_id"], "--data-path", train_path,
                    "--weights-path", intermediate_model,
                    "--distribution-token-id", CONFIG["distribution_token_id"],
                    "--label-column", "correct", "--reward-values", 0.0, 1.0,
                    "--kl-coefficient", 0.0,
                    "--batch-size", CONFIG["batch_size"],
                    "--gradient-accumulation-steps", CONFIG["gradient_accumulation_steps"],
                    "--num-epochs", CONFIG["num_epochs"], "--max-steps", CONFIG["stage1_steps"],
                    "--learning-rate", CONFIG["stage1_learning_rate"],
                    "--max-length", CONFIG["train_max_length"], "--dtype", CONFIG["dtype"],
                    "--metrics-path", stage1_metrics, "--log-every", 10, "--visualization-freq", 50,
                )
            """
        ),
        code(
            """
            RUN_VALUE_SCORING = True
            train_value_path = REPO / CONFIG["paths"]["train_value"]
            if RUN_VALUE_SCORING:
                run_repo(
                    REPO,
                    "python3", "src/score_with_ziprc_joint_head.py",
                    "--model", intermediate_model, "--in-parquet", train_path,
                    "--out-parquet", train_value_path,
                    "--distribution-token-id", CONFIG["distribution_token_id"],
                    "--num-length-bins", CONFIG["num_length_bins"],
                    "--reward-values", 0.0, 1.0, "--last-k", 64,
                    "--max-length", CONFIG["train_max_length"], "--batch-size", 1,
                    "--num-workers", 0, "--dtype", CONFIG["dtype"],
                )
            """
        ),
        code(
            """
            RUN_STAGE_2 = True
            final_model = REPO / CONFIG["paths"]["final_model"]
            stage2_metrics = REPO / CONFIG["paths"]["stage2_metrics"]
            if RUN_STAGE_2:
                run_repo(
                    REPO,
                    "python3", "src/train_ziprc_joint_head.py",
                    "--model-id", CONFIG["model_id"], "--data-path", train_value_path,
                    "--weights-path", final_model,
                    "--distribution-token-id", CONFIG["distribution_token_id"],
                    "--label-column", "value", "--reward-values", *CONFIG["reward_values"],
                    "--kl-coefficient", 10.0,
                    "--batch-size", CONFIG["batch_size"],
                    "--gradient-accumulation-steps", CONFIG["gradient_accumulation_steps"],
                    "--num-epochs", CONFIG["num_epochs"], "--max-steps", CONFIG["stage2_steps"],
                    "--learning-rate", CONFIG["stage2_learning_rate"],
                    "--max-length", CONFIG["train_max_length"], "--dtype", CONFIG["dtype"],
                    "--metrics-path", stage2_metrics, "--log-every", 10, "--visualization-freq", 50,
                )
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            histories = {"stage1": read_jsonl(stage1_metrics), "stage2": read_jsonl(stage2_metrics)}
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for row, (name, history) in enumerate(histories.items()):
                smooth = history.set_index("step")[["total_loss", "distribution_loss", "kl_loss"]].rolling(15, min_periods=1).mean()
                smooth.plot(ax=axes[row, 0])
                axes[row, 0].set_title(f"{name}: smoothed losses")
                history.plot(x="step", y="learning_rate", ax=axes[row, 1], legend=False, color="#f58518")
                axes[row, 1].set_title(f"{name}: learning rate")
            plt.tight_layout()
            plt.show()

            stage1_start, stage1_end = rolling_edges(histories["stage1"]["distribution_loss"])
            stage2_start, stage2_end = rolling_edges(histories["stage2"]["total_loss"])
            value_df = pd.read_parquet(train_value_path)
            value_std = float(value_df["value"].std())

            fig, ax = plt.subplots(figsize=(7, 4))
            for label, group in value_df.groupby("correct"):
                ax.hist(group["value"].dropna(), bins=30, alpha=.55, density=True, label=f"correct={label}")
            ax.set(title="Intermediate value separation on train", xlabel="predicted value")
            ax.legend()
            plt.show()

            checks = [
                gate("Stage 1 模型已保存", model_artifacts_exist(intermediate_model), str(intermediate_model)),
                gate("Stage 2 模型已保存", model_artifacts_exist(final_model), str(final_model)),
                gate("训练 metrics 完整", len(histories["stage1"]) >= 50 and len(histories["stage2"]) >= 50, f"steps={len(histories['stage1'])}/{len(histories['stage2'])}"),
                gate("Loss 全部有限", all(np.isfinite(history[["total_loss", "distribution_loss", "kl_loss"]]).all().all() for history in histories.values()), "no NaN/Inf"),
                gate("Stage 1 distribution loss 下降", stage1_end < stage1_start, f"{stage1_start:.3f} → {stage1_end:.3f}", kind="scientific"),
                gate("Stage 2 total loss 下降", stage2_end < stage2_start, f"{stage2_start:.3f} → {stage2_end:.3f}", kind="scientific"),
                gate("Value 非常数", value_std >= .01, f"std={value_std:.4f}", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "05_predictor_training", checks, {"stage1_loss_start": stage1_start, "stage1_loss_end": stage1_end, "stage2_loss_start": stage2_start, "stage2_loss_end": stage2_end, "value_std": value_std})
            """
        ),
    )


def build_06() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 6 — Predictor 完整 held-out evaluation

            在 validation/test split 上抽取生成进度 25%、50%、75%、100% 四个位置的联合分布，评估：

            - Correctness：AUROC、AUPRC、accuracy、F1、incorrect recall、Brier、ECE；
            - Remaining length：MAE、median AE、预测值 vs 真值；
            - reliability diagram，以及越接近结尾是否越会判断正确性。

            本步骤默认评估最终 ZIP-RC 模型；模型从未用 validation/test trajectory 更新参数。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import json

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, recall_score, roc_auc_score

            final_model = REPO / CONFIG["paths"]["final_model"]
            eval_sources = {
                "validation": REPO / CONFIG["paths"]["validation"],
                "test": REPO / CONFIG["paths"]["test"],
            }
            positions_path = REPO / CONFIG["paths"]["predictor_positions"]
            metrics_path = REPO / CONFIG["paths"]["predictor_metrics"]

            def expected_calibration_error(labels, probabilities, bins=10):
                labels = np.asarray(labels, dtype=float)
                probabilities = np.asarray(probabilities, dtype=float)
                edges = np.linspace(0, 1, bins + 1)
                result = 0.0
                for left, right in zip(edges[:-1], edges[1:]):
                    mask = (probabilities >= left) & (probabilities < right if right < 1 else probabilities <= right)
                    if mask.any():
                        result += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
                return float(result)
            """
        ),
        code(
            """
            RUN_EVALUATION = True
            if RUN_EVALUATION:
                run_repo(
                    REPO,
                    "python3", "src/evaluate_ziprc_predictor.py",
                    "--model", final_model,
                    "--data", eval_sources["validation"], eval_sources["test"],
                    "--split-names", "validation", "test",
                    "--out-parquet", positions_path,
                    "--distribution-token-id", CONFIG["distribution_token_id"],
                    "--num-length-bins", CONFIG["num_length_bins"],
                    "--reward-values", *CONFIG["reward_values"],
                    "--progress-points", .25, .50, .75, 1.0,
                    "--max-length", CONFIG["train_max_length"],
                    "--dtype", CONFIG["dtype"],
                )
            position_df = pd.read_parquet(positions_path)
            print("Rows:", len(position_df), "saved to", positions_path)
            """
        ),
        code(
            """
            metric_rows = []
            for (split_name, progress), group in position_df.groupby(["eval_split", "progress"]):
                labels = group["correct"].astype(int)
                scores = group["predicted_reward"].clip(0, 1)
                predictions = scores >= .5
                errors = (group["predicted_remaining"] - group["true_remaining"]).abs()
                naive_remaining = float(group["true_remaining"].median())
                naive_errors = (group["true_remaining"] - naive_remaining).abs()
                metric_rows.append({
                    "eval_split": split_name,
                    "progress": progress,
                    "auroc": roc_auc_score(labels, scores) if labels.nunique() == 2 else np.nan,
                    "auprc": average_precision_score(labels, scores) if labels.nunique() == 2 else np.nan,
                    "accuracy": accuracy_score(labels, predictions),
                    "f1": f1_score(labels, predictions, zero_division=0),
                    "incorrect_recall": recall_score(labels, predictions, pos_label=0, zero_division=0),
                    "brier": brier_score_loss(labels, scores),
                    "ece": expected_calibration_error(labels, scores),
                    "length_mae": errors.mean(),
                    "length_median_ae": errors.median(),
                    "length_naive_mae": naive_errors.mean(),
                })
            metrics = pd.DataFrame(metric_rows).sort_values(["eval_split", "progress"])
            metrics_path.write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")
            display(metrics.round(4))

            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            test_metrics = metrics[metrics["eval_split"] == "test"]
            test_metrics.plot(x="progress", y=["auroc", "auprc"], marker="o", ax=axes[0, 0], ylim=(0, 1))
            axes[0, 0].axhline(.5, color="gray", linestyle="--")
            axes[0, 0].set_title("Correctness discrimination by progress")
            test_metrics.plot(x="progress", y=["incorrect_recall", "f1"], marker="o", ax=axes[0, 1], ylim=(0, 1))
            axes[0, 1].set_title("Error detection / F1")
            test_metrics.plot(x="progress", y=["length_mae", "length_median_ae", "length_naive_mae"], marker="o", ax=axes[1, 0])
            axes[1, 0].set_title("Remaining-length error")

            final = position_df[(position_df["eval_split"] == "test") & (position_df["progress"] == 1.0)].copy()
            final["calibration_bin"] = pd.cut(final["predicted_reward"], bins=np.linspace(0, 1, 11), include_lowest=True)
            calibration = final.groupby("calibration_bin", observed=False).agg(predicted=("predicted_reward", "mean"), observed=("correct", "mean"), count=("correct", "size")).dropna()
            axes[1, 1].plot([0, 1], [0, 1], color="gray", linestyle="--")
            axes[1, 1].plot(calibration["predicted"], calibration["observed"], marker="o")
            axes[1, 1].set(xlim=(0, 1), ylim=(0, 1), title="Reliability diagram @100%", xlabel="predicted", ylabel="observed")
            plt.tight_layout()
            plt.show()

            final_metrics = test_metrics.iloc[-1]
            early_length = test_metrics[test_metrics["progress"] < 1.0]
            length_has_signal = bool((early_length["length_mae"] < early_length["length_naive_mae"]).any())
            checks = [
                gate("Validation / test 均齐全", set(position_df["eval_split"]) == {"validation", "test"}, str(position_df['eval_split'].value_counts().to_dict())),
                gate("四个进度点齐全", set(position_df["progress"]) == {.25, .5, .75, 1.0}, str(sorted(position_df['progress'].unique()))),
                gate("预测值有效", position_df[["predicted_reward", "predicted_remaining"]].notna().all().all(), "no missing values"),
                gate("AUROC >0.55", final_metrics["auroc"] > .55, f"{final_metrics['auroc']:.3f}", kind="scientific"),
                gate("Incorrect recall ≥0.50", final_metrics["incorrect_recall"] >= .50, f"{final_metrics['incorrect_recall']:.3f}", kind="scientific"),
                gate("Remaining length 优于 naive", length_has_signal, "25%/50%/75% 至少一个进度点 MAE 更低", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "06_predictor_evaluation", checks, {"metrics": metrics.to_dict(orient="records")})
            """
        ),
    )


def build_07() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 7 — Adaptive compute / controller 对比

            使用训练数据之后的 fresh prompts，每题生成 4 个 rollout。比较：

            - Fixed N = 1 / 2 / 4；
            - confidence-based sequential stopping；
            - ZIP-RC value-per-token controller。

            这是基于完整 trajectory 的**离线反事实模拟**，用于在实现在线 adaptive sampler 前判断是否存在 Pareto signal。核心图是 accuracy–tokens。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            RUN_FRESH_ROLLOUTS = True
            controller_path = REPO / CONFIG["paths"]["controller_rollouts"]
            controller_scored = REPO / CONFIG["paths"]["controller_scored"]
            controller_prompts = 400
            offset = int(CONFIG["training_prompts"])
            if RUN_FRESH_ROLLOUTS:
                run_repo(
                    REPO,
                    "python3", "src/generate_ziprc_rollouts.py",
                    "--model", CONFIG["model_id"], "--dataset", CONFIG["dataset"],
                    "--split", CONFIG["split"], "--prompt-column", CONFIG["prompt_column"],
                    "--answer-column", CONFIG["answer_column"], "--out", controller_path,
                    "--skip-num-prompts", offset, "--max-num-prompts", controller_prompts,
                    "--thinking-samples", 0, "--non-thinking-samples", CONFIG["controller_rollouts_per_prompt"],
                    "--temperature", CONFIG["temperature"], "--min-p", CONFIG["min_p"],
                    "--max-model-len", CONFIG["generation_max_model_len"], "--max-new-tokens", CONFIG["max_output_tokens"],
                    "--max-num-seqs", CONFIG["max_num_seqs"], "--dtype", CONFIG["dtype"], "--dp-size", 1, "--tp-size", 1,
                )
                run_repo(
                    REPO,
                    "python3", "src/evaluate_and_label_rollouts.py",
                    "--data", controller_path, "--model", CONFIG["grader_model_id"],
                    "--tensor-parallel-size", 1, "--gpu-memory-utilization", CONFIG["gpu_memory_utilization"],
                    "--max-model-len", CONFIG["grader_max_model_len"], "--max-num-seqs", CONFIG["max_num_seqs"], "--dtype", CONFIG["dtype"],
                )
                run_repo(
                    REPO,
                    "python3", "src/score_with_ziprc_joint_head.py",
                    "--model", REPO / CONFIG["paths"]["final_model"],
                    "--in-parquet", controller_path, "--out-parquet", controller_scored,
                    "--distribution-token-id", CONFIG["distribution_token_id"],
                    "--num-length-bins", CONFIG["num_length_bins"], "--reward-values", *CONFIG["reward_values"],
                    "--last-k", 64, "--max-length", CONFIG["train_max_length"],
                    "--batch-size", 1, "--num-workers", 0, "--dtype", CONFIG["dtype"],
                )
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            df = pd.read_parquet(controller_scored).copy()
            df["sample_order"] = df.groupby("prompt_idx").cumcount() + 1

            def fixed_n_metrics(n):
                subset = df[df["sample_order"] <= n]
                selected = subset.loc[subset.groupby("prompt_idx")["value"].idxmax()]
                cost = subset.groupby("prompt_idx")["length"].sum().mean()
                latency_proxy = subset.groupby("prompt_idx")["length"].max().mean()
                return {"policy": f"Fixed N={n}", "avg_tokens": cost, "latency_tokens": latency_proxy, "accuracy": selected["correct"].mean()}

            rows = [fixed_n_metrics(n) for n in (1, 2, 4)]
            for threshold in (.55, .65, .75, .85):
                selected_rows, costs = [], []
                for _, group in df.groupby("prompt_idx"):
                    group = group.sort_values("sample_order")
                    seen = []
                    for _, row in group.iterrows():
                        seen.append(row)
                        if row["value"] >= threshold:
                            break
                    selected_rows.append(max(seen, key=lambda item: item["value"]))
                    costs.append(sum(item["length"] for item in seen))
                rows.append({"policy": f"Confidence ≥{threshold:.2f}", "avg_tokens": np.mean(costs), "latency_tokens": np.mean(costs), "accuracy": np.mean([item["correct"] for item in selected_rows])})

            max_tokens = float(CONFIG["max_output_tokens"])
            expected_next_cost = float(df["length"].median())
            for penalty in (0.00, 0.25, 0.50, 1.00):
                selected_rows, costs = [], []
                for _, group in df.groupby("prompt_idx"):
                    group = group.sort_values("sample_order")
                    seen = []
                    for _, row in group.iterrows():
                        seen.append(row)
                        best_value = max(item["value"] for item in seen)
                        expected_gain_upper_bound = 1.0 - best_value
                        cost_penalty = penalty * expected_next_cost / max_tokens
                        if expected_gain_upper_bound <= cost_penalty:
                            break
                    selected_rows.append(max(seen, key=lambda item: item["value"]))
                    costs.append(sum(item["length"] for item in seen))
                rows.append({"policy": f"ZIP-RC λ={penalty:.2f}", "avg_tokens": np.mean(costs), "latency_tokens": np.mean(costs), "accuracy": np.mean([item["correct"] for item in selected_rows])})

            results = pd.DataFrame(rows).sort_values("avg_tokens")
            display(results.round(3))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for family, group in results.assign(family=results["policy"].str.split().str[0]).groupby("family"):
                axes[0].plot(group["avg_tokens"], group["accuracy"], marker="o", label=family)
                axes[1].plot(group["latency_tokens"], group["accuracy"], marker="o", label=family)
                for _, row in group.iterrows():
                    axes[0].annotate(row["policy"], (row["avg_tokens"], row["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            axes[0].set(title="Accuracy vs average generated tokens", xlabel="average generated tokens / prompt", ylabel="accuracy", ylim=(0, 1))
            axes[1].set(title="Accuracy vs latency-token proxy", xlabel="parallel max / sequential sum tokens", ylabel="accuracy", ylim=(0, 1))
            for ax in axes:
                ax.grid(alpha=.25)
                ax.legend()
            plt.tight_layout()
            plt.show()

            fixed = results[results["policy"].str.startswith("Fixed")]
            adaptive = results[~results["policy"].str.startswith("Fixed")]
            pareto_signal = any(
                (adaptive_row["accuracy"] >= fixed_row["accuracy"] and adaptive_row["avg_tokens"] <= fixed_row["avg_tokens"])
                for _, adaptive_row in adaptive.iterrows()
                for _, fixed_row in fixed.iterrows()
            )
            checks = [
                gate("每题 4 个 fresh rollout", df.groupby("prompt_idx").size().eq(CONFIG["controller_rollouts_per_prompt"]).all(), str(df.groupby('prompt_idx').size().describe().to_dict())),
                gate("Value score 完整", df["value"].notna().all(), f"missing={int(df['value'].isna().sum())}"),
                gate("Fixed N=1/2/4 均已评估", len(fixed) == 3, str(fixed['policy'].tolist())),
                gate("出现 Pareto signal", pareto_signal, "至少一个 adaptive 点不低于某 fixed 点且 token 更少/相等", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "07_controller_comparison", checks, {"results": results.to_dict(orient="records"), "offline_proxy": True})
            """
        ),
    )


def build_08() -> dict[str, object]:
    return notebook(
        markdown(
            """
            # Step 8 — 0.6B 实验毕业决策

            汇总所有 stage report，给出是否值得升级到 1.7B 的明确结论。

            **毕业条件：** predictor discrimination、remaining-length signal、controller 至少部分 budget 不输 fixed baseline。任何 operational gate 失败都应先修流水线。
            """
        ),
        code(BOOTSTRAP),
        code(
            """
            import json
            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display, Markdown

            report_dir = REPO / "artifacts" / "stage_reports"
            expected = [
                "00_memory_and_config", "01_pilot_generation", "02_pilot_quality",
                "03_training_data", "04_prompt_split", "05_predictor_training",
                "06_predictor_evaluation", "07_controller_comparison",
            ]
            reports = {}
            for name in expected:
                path = report_dir / f"{name}.json"
                reports[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

            summary = pd.DataFrame([
                {
                    "stage": name,
                    "report_exists": report is not None,
                    "operational": report["operational_passed"] if report else False,
                    "scientific": report["scientific_passed"] if report else False,
                }
                for name, report in reports.items()
            ])
            display(summary)

            ax = summary.set_index("stage")[["operational", "scientific"]].astype(int).plot.barh(figsize=(10, 6), color=["#4c78a8", "#49beaa"])
            ax.set(xlim=(0, 1.05), title="ZIP-RC 0.6B stage gates", xlabel="pass (1) / review (0)")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            operational_ok = bool(summary["report_exists"].all() and summary["operational"].all())
            predictor_ok = bool(summary.loc[summary["stage"] == "06_predictor_evaluation", "scientific"].iloc[0]) if reports["06_predictor_evaluation"] else False
            controller_ok = bool(summary.loc[summary["stage"] == "07_controller_comparison", "scientific"].iloc[0]) if reports["07_controller_comparison"] else False
            graduate = operational_ok and predictor_ok and controller_ok

            if graduate:
                display(Markdown("## ✅ 建议升级到 Qwen3-1.7B\\n0.6B 已证明 predictor 和 controller 均有可用信号。"))
            elif not operational_ok:
                display(Markdown("## ⛔ 暂停扩模\\n至少一个流水线阶段未妥善完成，请先处理 operational gate。"))
            else:
                display(Markdown("## ⚠️ 保持 0.6B 调试\\n流水线正常，但 discrimination / length / controller 信号尚未全部达到毕业条件。"))

            failed_gates = []
            for stage, report in reports.items():
                if report:
                    failed_gates.extend({"stage": stage, **item} for item in report["gates"] if not item["passed"])
            display(pd.DataFrame(failed_gates) if failed_gates else pd.DataFrame([{"status": "all gates passed"}]))

            checks = [
                gate("所有阶段报告存在", summary["report_exists"].all(), f"{summary['report_exists'].sum()}/{len(summary)}"),
                gate("所有 operational gates 通过", operational_ok, "流水线完整性"),
                gate("Predictor 达标", predictor_ok, "discrimination + incorrect recall + length", kind="scientific"),
                gate("Controller 达标", controller_ok, "至少出现一个 Pareto signal", kind="scientific"),
            ]
            display(gate_frame(checks))
            save_stage_report(REPO, "08_graduation_decision", checks, {"graduate_to_1_7b": graduate})
            """
        ),
    )


def make_colab_notebook(
    payload: dict[str, object],
    *,
    filename: str,
) -> dict[str, object]:
    """Replace local bootstrap cells with the fixed Colab runtime contract."""
    result = json.loads(json.dumps(payload))
    cells = result["cells"]
    if filename.startswith("00_"):
        cells[0]["source"] = (
            f'<a href="https://colab.research.google.com/github/wtree101/ZIP-RC-Colab/blob/main/notebooks/colab/{filename}" '
            'target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Open In Colab"/></a>\n\n'
            + cells[0]["source"]
            + "\n\n此版本固定使用 `/content/ZIP-RC-Colab` 和 `/content/mamba/envs/zip/bin/python`。"
        )
        cells[1]["source"] = dedent(COLAB_SETUP).strip()
        return result

    generic_bootstrap = dedent(BOOTSTRAP).strip()
    replacement = dedent(COLAB_BOOTSTRAP).strip()
    replaced = False
    for cell in cells:
        if cell["cell_type"] == "code" and cell["source"] == generic_bootstrap:
            cell["source"] = replacement
            replaced = True
            break
    if not replaced:
        raise ValueError(f"Could not find bootstrap cell in {filename}")

    cells[0]["source"] = (
        f'<a href="https://colab.research.google.com/github/wtree101/ZIP-RC-Colab/blob/main/notebooks/colab/{filename}" '
        'target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open In Colab"/></a>\n\n'
        + cells[0]["source"]
        + "\n\n先运行 `colab/00_memory_and_config.ipynb`；本 Notebook 的训练命令会自动使用 ZIP mamba 环境。"
    )
    return result


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_memory_and_config.ipynb": build_00(),
        "01_pilot_generation.ipynb": build_01(),
        "02_pilot_quality.ipynb": build_02(),
        "03_training_data.ipynb": build_03(),
        "04_prompt_split.ipynb": build_04(),
        "05_predictor_training.ipynb": build_05(),
        "06_predictor_evaluation.ipynb": build_06(),
        "07_controller_comparison.ipynb": build_07(),
        "08_graduation_decision.ipynb": build_08(),
    }
    for filename, payload in notebooks.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(REPO_ROOT))

    colab_dir = NOTEBOOK_DIR / "colab"
    colab_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in notebooks.items():
        colab_payload = make_colab_notebook(payload, filename=filename)
        path = colab_dir / filename
        path.write_text(json.dumps(colab_payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
