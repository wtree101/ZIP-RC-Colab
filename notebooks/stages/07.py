# %% [markdown]
# # Step 7 — Adaptive compute / controller 对比
# 
# 使用训练数据之后的 fresh prompts，每题生成 4 个 rollout。比较：
# 
# - Fixed N = 1 / 2 / 4；
# - confidence-based sequential stopping；
# - ZIP-RC value-per-token controller。
# 
# 这是基于完整 trajectory 的**离线反事实模拟**，用于在实现在线 adaptive sampler 前判断是否存在 Pareto signal。核心图是 accuracy–tokens。

# %%
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

selected_gpu = select_best_cuda_device()

CONFIG = load_config(REPO)
print("Repository:", REPO)
print("Experiment:", CONFIG["experiment_name"])
print("Selected GPU:", selected_gpu or "unchanged")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))

# %%
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

# %%
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

# %%
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


