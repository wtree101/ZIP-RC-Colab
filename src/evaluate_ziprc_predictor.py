"""Evaluate ZIP-RC reward and remaining-length predictions at fixed progress points."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

LENGTH_EDGES = np.array([0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])


def as_int_list(value: object) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in ast.literal_eval(value)]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    raise TypeError(f"Expected an iterable of token IDs, got {type(value).__name__}.")


def load_model(model_path: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"flash_attention_2 unavailable ({error}); using default attention.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model.to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--split-names", nargs="+", required=True)
    parser.add_argument("--out-parquet", required=True)
    parser.add_argument("--distribution-token-id", type=int, default=151669)
    parser.add_argument("--num-length-bins", type=int, default=8)
    parser.add_argument("--reward-values", type=float, nargs="+", required=True)
    parser.add_argument("--progress-points", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.data) != len(args.split_names):
        raise ValueError("--data and --split-names must have the same number of values.")
    if len(LENGTH_EDGES) - 1 != args.num_length_bins:
        raise ValueError(
            f"This evaluator defines {len(LENGTH_EDGES) - 1} length bins, "
            f"but --num-length-bins={args.num_length_bins}."
        )
    if any(not 0 < point <= 1 for point in args.progress_points):
        raise ValueError("Each --progress-points value must be in (0, 1].")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for predictor evaluation.")

    frames: list[pd.DataFrame] = []
    for split_name, data_path in zip(args.split_names, args.data, strict=True):
        frame = pd.read_parquet(data_path).copy()
        frame["eval_split"] = split_name
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)

    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    model = load_model(args.model, dtype, device)
    lm_head = model.get_output_embeddings()

    reward_values = torch.tensor(args.reward_values, dtype=torch.float32, device=device)
    length_midpoints = torch.tensor(
        (LENGTH_EDGES[:-1] + LENGTH_EDGES[1:]) / 2,
        dtype=torch.float32,
        device=device,
    )
    num_reward_states = len(args.reward_values)
    num_bins = num_reward_states * args.num_length_bins
    start = args.distribution_token_id
    stop = start + num_bins
    if stop > lm_head.weight.size(0):
        raise ValueError(f"Distribution slice [{start}:{stop}] exceeds vocab size {lm_head.weight.size(0)}.")
    weight = lm_head.weight[start:stop]
    bias = lm_head.bias[start:stop] if getattr(lm_head, "bias", None) is not None else None

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for row_index, row in source.iterrows():
            input_ids = as_int_list(row["input_ids"])[:-1][: args.max_length]
            label_positions = [
                position - 1
                for position in as_int_list(row["label_positions"])
                if 0 <= position - 1 < len(input_ids)
            ]
            if not label_positions:
                continue

            inputs = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            hidden = model(
                input_ids=inputs,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1][0]

            for progress in args.progress_points:
                progress_index = min(
                    len(label_positions) - 1,
                    max(0, round(progress * len(label_positions)) - 1),
                )
                position = label_positions[progress_index]
                logits = F.linear(hidden[position], weight, bias).float()
                probabilities = F.softmax(logits, dim=-1).view(
                    num_reward_states,
                    args.num_length_bins,
                )
                reward_probabilities = probabilities.sum(dim=1)
                length_probabilities = probabilities.sum(dim=0)
                rows.append(
                    {
                        "row_idx": int(row_index),
                        "prompt_idx": int(row["prompt_idx"]),
                        "eval_split": str(row["eval_split"]),
                        "progress": float(progress),
                        "correct": bool(row["correct"]),
                        "predicted_reward": float(torch.dot(reward_probabilities, reward_values).item()),
                        "predicted_remaining": float(torch.dot(length_probabilities, length_midpoints).item()),
                        "true_remaining": int(len(input_ids) - position - 1),
                        "position": int(position),
                    }
                )

            if (row_index + 1) % max(1, args.log_every) == 0:
                print(f"Evaluated {row_index + 1}/{len(source)} trajectories", flush=True)

    output_path = Path(args.out_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Wrote {len(rows)} position predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
