#!/usr/bin/env python3
"""
Label and evaluate ZIP-generated responses for correctness.

This script can:
1. Label training data by checking correctness of ZIP-generated responses with a reference model
2. Evaluate responses to a test set and compute metrics including pass@n and best-of-n selection
3. Support consistency-based evaluation methods with answer extraction

Example usage:
    python src/evaluate_and_label_rollouts.py --data results/results.parquet
    python src/evaluate_and_label_rollouts.py --data results/results.parquet --use-consistency
"""

from __future__ import annotations
import argparse, json, os, random, time, warnings
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from ziprc_progress import (
    PersistentTqdm,
    configure_progress,
    default_progress_path,
    mark_progress_failed,
    persistent_vllm_progress,
)

os.environ["VLLM_USE_V1"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULTS = {
    "thinking_token": "</think>",
    "data_path": "results/results.parquet", 
    "eval_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "thinking_token_id": 151667,
}

GRADER_SYSTEM_PROMPT = (
    "You are a strict mathematical answer-equivalence grader. "
    "Compare the proposed solution with the verified answer. "
    "Output exactly `Yes.` if they are equivalent and `No.` otherwise. "
    "Do not explain your reasoning."
)


def apply_hf_chat(
    tokenizer,
    user_content: str,
    *,
    system_content: str | None = None,
) -> str:
    """Build a Harmony-format chat prompt via the tokenizer's chat template."""
    messages = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        if system_content is None:
            return user_content
        return f"{system_content}\n\n{user_content}"

def extract_response(response: str, token: str) -> str:
    """Extract response after thinking token if present."""
    return response[response.rfind(token) + len(token):].strip() if token in response else response.strip()

def get_eval_prompt(prompt: str, response: str, answer: str, thinking_token: str) -> str:
    """Craft the grading prompt for the eval model (correctness w/ gold answer)."""
    return (
        "Determine whether the proposed solution gives an answer equivalent "
        "to the verified answer.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Verified answer:\n{answer}\n\n"
        "Proposed solution:\n"
        f"{extract_response(response, thinking_token)}\n\n"
        "Output exactly one of:\n"
        "Yes.\n"
        "No."
    )


def parse_grader_output(output: str) -> bool:
    """Parse a grader decision, rejecting anything other than Yes or No."""
    text = output.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    raise ValueError(f"Unexpected grader output: {output!r}")


def resolve_grader_output(output: str) -> tuple[bool, bool]:
    """Return ``(correct, valid)`` without aborting on malformed output."""
    try:
        return parse_grader_output(output), True
    except ValueError:
        return "Yes" in output, False


def get_answer_extraction_prompt(question: str, response: str, thinking_token: str) -> str:
    """Craft a prompt to extract the final answer from a solution."""
    return (
        "Your task is to extract the *final answer* from the proposed solution. "
        "Start by identifying exactly what the question is asking. "
        "Next, determine the final answer from the proposed solution. "
        "Finally, extract the final answer from the proposed solution. "
        'Respond with ONLY that answer (e.g. "42", "A").\n\n'
        f'Question:\n\n"\n{question}\n"\n\n'
        f'Proposed Solution:\n\n"\n{extract_response(response, thinking_token)}\n"'
    )

def find_most_common_answer(answers: list[str]) -> str:
    """Find the most common answer; if tied, return the last occurrence."""
    if not answers: return ""
    counts = Counter(answers)
    max_count = max(counts.values())
    most_common = {ans for ans, count in counts.items() if count == max_count}
    return next((ans for ans in reversed(answers) if ans in most_common), answers[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULTS["data_path"], help="Parquet file to evaluate")
    p.add_argument("--model", default=DEFAULTS["eval_model"], help="Grading model ID")
    p.add_argument("--thinking-token", default=DEFAULTS["thinking_token"], help="Token marking the end of thinking section")
    p.add_argument("--show-examples", action="store_true", help="Write evaluation and consistency prompt/response pairs to a text file")
    p.add_argument("--use-consistency", action="store_true", help="Enable consistency-based methods (requires datasets with answers)")
    p.add_argument("--tensor-parallel-size", type=int, default=8, help="Number of GPUs to use via tensor parallelism")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Fraction of GPU memory that vLLM can use (0.0–1.0)")
    p.add_argument("--max-model-len", type=int, default=32_768, help="Maximum sequence length to allocate KV cache for")
    p.add_argument("--enforce-eager", action="store_true", default=False, help="Run model in eager mode (disable CUDA graphs)")
    p.add_argument("--max-num-seqs", type=int, default=8, help="Upper bound on concurrently scheduled sequences to reduce memory")
    p.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="vLLM compute dtype.",
    )
    p.add_argument("--output-json", type=str, help="Path to save evaluation results as JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_progress(
        default_progress_path("evaluate_and_label_rollouts.json"),
        job="grade rollout correctness",
    )
    metrics = {"data_file": args.data, "eval_model": args.model, "use_consistency": args.use_consistency}
    
    df = pq.read_table(args.data).to_pandas()
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    llm = LLM(model=args.model,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              dtype=dtype,
              trust_remote_code=True,
              enforce_eager=args.enforce_eager,
              max_num_seqs=args.max_num_seqs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    has_expected_reward = "expected_reward" in df.columns
    
    def has_thinking(token_ids: list[int]) -> bool:
        """Check if thinking token appears in token list."""
        if isinstance(token_ids, str):
            try: token_ids = eval(token_ids)
            except: return False
        return DEFAULTS["thinking_token_id"] in token_ids

    # Prefer per-try grouping when available; otherwise fall back to per-prompt
    group_col = "group_id" if "group_id" in df.columns else ("prompt_idx" if "prompt_idx" in df.columns else "prompt")

    def _precision(metric_name: str, selector_fn: callable) -> float:
        """Compute precision for a best-of-n selector."""
        answered_mask = df.groupby(group_col).apply(
            lambda g: not g[(g["finished"]) & (~g["pruned"] if "pruned" in g.columns else True)].empty)
        answered_prompts = answered_mask[answered_mask].index
        if answered_prompts.empty: return 0.0
        
        correct_flags = [bool(idx is not None and df.at[idx, "correct"]) 
                         for p in answered_prompts
                         if (idx := selector_fn(df[df[group_col] == p])) is not None or True]
        
        precision = sum(correct_flags) / len(answered_prompts) * 100
        metrics[f"precision_{metric_name}"] = float(precision)
        print(f"Precision ({metric_name.replace('_', ' ')}): {precision:.2f}%")
        return precision

    # Evaluate correctness for finished rows first so downstream metrics can use the 'correct' column
    df["correct"] = df["finished"]
    df["grader_output"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["grader_valid"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df_finished = df[df["finished"]].copy()
    invalid_grader_outputs: list[tuple[object, str]] = []

    inputs, row_indices = [], []
    for idx, row in PersistentTqdm(
        df_finished.iterrows(),
        total=len(df_finished),
        desc="Prepare grader prompts",
        unit="row",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}, remaining {remaining}]",
    ):
        prompt_text = get_eval_prompt(row["prompt"], row["response"], row.get("answer", ""), args.thinking_token)
        chat_input = apply_hf_chat(
            tokenizer,
            prompt_text,
            system_content=GRADER_SYSTEM_PROMPT,
        )

        prompt_length = len(tokenizer(chat_input).input_ids)
        if prompt_length > args.max_model_len:
            message = (
                f"Grader prompt for row {idx} has {prompt_length} tokens, "
                f"exceeding --max-model-len={args.max_model_len}."
            )
            df_finished.at[idx, "correct"] = False
            df_finished.at[idx, "grader_output"] = message
            df_finished.at[idx, "grader_valid"] = False
            invalid_grader_outputs.append((idx, message))
            continue

        inputs.append(chat_input)
        row_indices.append(idx)

    elapsed = 0.0
    if inputs:
        start = time.perf_counter()
        with persistent_vllm_progress():
            generations = llm.generate(
                inputs,
                SamplingParams(max_tokens=16, temperature=0.0, top_k=1),
                use_tqdm=True,
            )
        elapsed = time.perf_counter() - start

        if args.show_examples:
            examples_path = os.path.splitext(args.data)[0] + "_examples.txt"
            with open(examples_path, "w") as f:
                f.write("--- Evaluation prompt/response pairs ---\n")
                for i, (prompt, gen) in enumerate(
                    random.sample(
                        list(zip(inputs, generations, strict=True)),
                        min(len(inputs), 50),
                    )
                ):
                    f.write(
                        f"\nExample {i + 1}:\nPrompt to eval model:\n{prompt}\n"
                        f"Eval model response:\n{gen.outputs[0].text.strip()}\n"
                    )
                f.write("--- End of evaluation examples ---\n")
            print(f"Wrote evaluation examples to {examples_path}")

        graded_results = {}
        for idx, gen in zip(row_indices, generations, strict=True):
            output = gen.outputs[0].text
            df_finished.at[idx, "grader_output"] = output
            correct, valid = resolve_grader_output(output)
            graded_results[idx] = correct
            df_finished.at[idx, "grader_valid"] = valid
            if not valid:
                invalid_grader_outputs.append((idx, output))
        for idx, correct in graded_results.items():
            df_finished.at[idx, "correct"] = correct

    df.update(df_finished)
    pq.write_table(pa.Table.from_pandas(df), args.data)

    row_to_answer = {}
    if args.use_consistency:
        extraction_inputs, extraction_indices = [], []
        total_prompts = prompts_with_no_finished = 0
        
        grouped = df.groupby(group_col)
        for _, grp in PersistentTqdm(
            grouped,
            total=grouped.ngroups,
            desc="Prepare consistency prompts",
            unit="prompt",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}, remaining {remaining}]",
        ):
            total_prompts += 1
            finished = grp[grp["finished"]]
            if finished.empty:
                prompts_with_no_finished += 1
                continue
            
            q_text = finished.iloc[0]["prompt"]
            for idx, row in finished.iterrows():
                extraction_inputs.append(
                    apply_hf_chat(
                        tokenizer,
                        get_answer_extraction_prompt(q_text, row["response"], args.thinking_token)
                    )
                )
                extraction_indices.append(idx)
        
        if extraction_inputs:
            with persistent_vllm_progress():
                extracts = llm.generate(
                    extraction_inputs,
                    SamplingParams(max_tokens=8, temperature=0.0, top_k=1),
                    use_tqdm=True,
                )
            for idx, gen in zip(extraction_indices, extracts):
                row_to_answer[idx] = gen.outputs[0].text.strip().strip('"').strip()
        
        correct_flags = []
        for _, grp in df.groupby(group_col):
            finished = grp[grp["finished"]]
            if finished.empty: continue
            
            answers = [row_to_answer.get(idx, f"__NO_ANSWER_{idx}__") for idx in finished.index]
            most_common = find_most_common_answer(answers)
            chosen_idx = next((idx for ans, idx in zip(reversed(answers), reversed(finished.index.tolist())) 
                              if ans == most_common), finished.index[-1])
            correct_flags.append(bool(df.at[chosen_idx, "correct"]))
        
        total_for_accuracy = len(correct_flags) + prompts_with_no_finished
        consistency_acc = sum(correct_flags) / total_for_accuracy * 100 if total_for_accuracy else 0.0
        metrics["best_of_n_accuracy_consistency"] = float(consistency_acc)
        print(f"Best-of-n Accuracy (consistency / majority-vote): {consistency_acc:.2f}%")
        
        _precision("best_of_n_consistency", lambda g: (
            None if g[g["finished"]].empty else
            g.loc[g[g["finished"]].assign(extracted_answer=lambda x: x.index.map(lambda i: row_to_answer.get(i)))
                  .pipe(lambda x: x[x["extracted_answer"] == find_most_common_answer(x["extracted_answer"].tolist())])
                  .index[-1]].name))

    if args.show_examples and 'extraction_inputs' in locals() and extraction_inputs:
        examples_path = os.path.splitext(args.data)[0] + "_examples.txt"
        with open(examples_path, "w") as f:
            f.write("--- Answer extraction prompt/response pairs ---\n")
            for i, (prompt, gen) in enumerate(random.sample(list(zip(extraction_inputs, extracts)), 
                                                            min(len(extraction_inputs), 50))):
                f.write(f"\nExtraction Example {i+1}:\nPrompt to extraction model:\n{prompt}\n"
                       f"Extraction model response:\n{gen.outputs[0].text.strip()}\n")
            f.write("--- End of extraction examples ---\n")
        print(f"Wrote extraction examples to {examples_path}")
    
    
    total = len(df)
    finished_count = df['finished'].sum()
    finished_pct = df['finished'].mean() * 100
    
    metrics.update({
        "total_responses": int(total),
        "finished_responses": int(finished_count),
        "finished_percentage": float(finished_pct)
    })
    
    print(f"Finished responses: {finished_count}/{total} ({finished_pct:.2f}%)")
    
    if "pruned" in df.columns:
        pruned_count = df['pruned'].sum()
        pruned_pct = df['pruned'].mean() * 100
        metrics.update({
            "pruned_responses": int(pruned_count),
            "pruned_percentage": float(pruned_pct)
        })
        print(f"Pruned responses:   {pruned_count}/{total} ({pruned_pct:.2f}%)")

    mean_latency_per_prompt = df.groupby(group_col)["length"].mean()
    total_tokens_per_prompt = df.groupby(group_col)["length"].sum()
    avg_total_tokens = total_tokens_per_prompt.mean()
    n_latency_per_prompt = df.groupby(group_col)["length"].max()
    avg_n_latency = n_latency_per_prompt.mean()

    metrics.update({
        "average_total_tokens": int(avg_total_tokens),
        "average_n_latency": float(avg_n_latency)
    })
    
    print(f"Average total tokens:      {int(avg_total_tokens)}")
    print(f"Average n-latency:        {avg_n_latency:.2f}")

    if "pruned" in df.columns:
        unpruned_acc = df.loc[~df['pruned']].groupby(group_col)["correct"].mean().mean() * 100
        metrics["correctness_unpruned"] = float(unpruned_acc)
        print(f"Correctness (unpruned): {unpruned_acc:.2f}%")

    pass_at_n = df.groupby(group_col)["correct"].any().mean() * 100
    metrics["pass_at_n_accuracy"] = float(pass_at_n)
    print(f"Pass@n Accuracy:         {pass_at_n:.2f}%")

    if args.use_consistency and row_to_answer and has_expected_reward:
        def select_best_weighted_with_reasoning_preference(group):
            """Select best answer by summing expected_reward per extracted answer; prefer reasoning if tied."""
            finished = group[group['finished']]
            if finished.empty:
                return None

            finished = finished.copy()
            finished['used_reasoning'] = finished['output_token_ids'].apply(has_thinking)
            finished['extracted_answer'] = finished.index.map(lambda idx: row_to_answer.get(idx, f"__NO_ANSWER_{idx}__"))
            # Treat missing expected_reward as -inf to avoid selecting
            er = finished['expected_reward'].astype(float)
            er = er.fillna(float('-inf'))
            finished['__er__'] = er

            # Sum expected_reward by extracted answer
            answer_scores = finished.groupby('extracted_answer')['__er__'].sum()
            if answer_scores.empty:
                return None
            # Prefer answers that come from reasoning rows if tie on total score
            best_answers = answer_scores[answer_scores == answer_scores.max()].index.tolist()
            if len(best_answers) > 1:
                reasoning_answers = finished[finished['used_reasoning']].groupby('extracted_answer')['__er__'].sum()
                reasoning_answers = reasoning_answers[reasoning_answers.index.isin(best_answers)]
                if not reasoning_answers.empty:
                    best_answers = [reasoning_answers.idxmax()]
                else:
                    best_answers = [best_answers[0]]
            best_answer = best_answers[0]

            # Within best answer, pick the row with highest expected_reward; prefer reasoning in tie
            subset = finished[finished['extracted_answer'] == best_answer]
            if subset.empty:
                return None
            # Prefer reasoning subset first
            reasoning_subset = subset[subset['used_reasoning']]
            candidate_subset = reasoning_subset if not reasoning_subset.empty else subset
            return candidate_subset['__er__'].idxmax()

        best_of_n_weighted = df.groupby(group_col, group_keys=False).apply(
            lambda g: g.loc[select_best_weighted_with_reasoning_preference(g), 'correct']
            if select_best_weighted_with_reasoning_preference(g) is not None else False
        ).mean() * 100

        metrics["best_of_n_accuracy_expected_reward_consistency"] = float(best_of_n_weighted)
        print(f"Best-of-n Accuracy (expected_reward + consistency): {best_of_n_weighted:.2f}%")
        _precision("best_of_n_expected_reward_consistency", selector_fn=select_best_weighted_with_reasoning_preference)
    elif args.use_consistency and row_to_answer and not has_expected_reward:
        print("Skipping expected_reward + consistency metrics (no 'expected_reward' column).")

    if has_expected_reward:
        def select_best_with_reasoning_preference(group):
            """Select best sample by expected_reward, preferring reasoning samples."""
            finished = group[group['finished']]
            if finished.empty:
                return None

            finished = finished.copy()
            finished['used_reasoning'] = finished['output_token_ids'].apply(has_thinking)
            # Treat missing expected_reward as -inf
            er = finished['expected_reward'].astype(float).fillna(float('-inf'))
            finished['__er__'] = er
            reasoning = finished[finished['used_reasoning']]
            candidate_subset = reasoning if not reasoning.empty else finished
            return candidate_subset['__er__'].idxmax()

        best_of_n = df.groupby(group_col, group_keys=False).apply(
            lambda g: g.loc[select_best_with_reasoning_preference(g), 'correct']
            if select_best_with_reasoning_preference(g) is not None else False
        ).mean() * 100

        metrics["best_of_n_accuracy_expected_reward"] = float(best_of_n)
        print(f"Best-of-n Accuracy (expected_reward, reasoning-preferred): {best_of_n:.2f}%")
        _precision("best_of_n_expected_reward", selector_fn=select_best_with_reasoning_preference)
    else:
        print("Skipping expected_reward-based metrics (no 'expected_reward' column).")

    def select_shortest_with_reasoning_preference(group):
        """Return idx of the shortest sample, preferring reasoning samples."""
        finished = group[group['finished']]
        if finished.empty:
            return None
        finished = finished.copy()
        finished['used_reasoning'] = finished['output_token_ids'].apply(has_thinking)
        reasoning = finished[finished['used_reasoning']]
        return reasoning['length'].idxmin() if not reasoning.empty else finished['length'].idxmin()

    shortest_reasoning_baseline_acc = df.groupby(group_col, group_keys=False).apply(
        lambda g: g.loc[select_shortest_with_reasoning_preference(g), 'correct']
        if select_shortest_with_reasoning_preference(g) is not None else False
    ).mean() * 100

    metrics["best_of_n_accuracy_shortest_response"] = float(shortest_reasoning_baseline_acc)
    print(f"Best-of-n Accuracy (shortest-response, reasoning-preferred): {shortest_reasoning_baseline_acc:.2f}%")
    _precision("best_of_n_shortest_response", selector_fn=select_shortest_with_reasoning_preference)

    if has_expected_reward:
        # Analyze best-of-n picks
        chosen_idx = df.groupby(group_col, group_keys=False).apply(select_best_with_reasoning_preference).dropna().astype(int)
        chosen_rows = df.loc[chosen_idx]
        correct_rows = chosen_rows[chosen_rows["correct"]]

        best_correct_flag = pd.Series(False, index=mean_latency_per_prompt.index)
        if not chosen_idx.empty:
            best_correct_flag.loc[correct_rows[group_col]] = True

        correct_mask = best_correct_flag
        incorrect_mask = ~best_correct_flag

        correct_total_tokens = total_tokens_per_prompt[correct_mask].mean()
        incorrect_total_tokens = total_tokens_per_prompt[incorrect_mask].mean()
        correct_n_latency = n_latency_per_prompt[correct_mask].mean()
        incorrect_n_latency = n_latency_per_prompt[incorrect_mask].mean()

        metrics.update({
            "correct_best_of_n_avg_total_tokens": int(correct_total_tokens),
            "correct_best_of_n_avg_n_latency": float(correct_n_latency),
            "incorrect_best_of_n_avg_total_tokens": int(incorrect_total_tokens),
            "incorrect_best_of_n_avg_n_latency": float(incorrect_n_latency)
        })

        print(f"Average total tokens for correct best-of-n selections: {int(correct_total_tokens)}")
        print(f"Average n-latency for correct best-of-n selections:        {correct_n_latency:.2f}")
        print(f"Average total tokens for incorrect best-of-n selections: {int(incorrect_total_tokens)}")
        print(f"Average n-latency for incorrect best-of-n selections:        {incorrect_n_latency:.2f}")

        chosen_rows["used_reasoning"] = chosen_rows["output_token_ids"].apply(has_thinking)
        num_reasoning = chosen_rows["used_reasoning"].sum()
        total_chosen = len(chosen_rows)
        reasoning_pct = num_reasoning/total_chosen*100 if total_chosen > 0 else 0.0

        metrics.update({
            "best_of_n_selections_using_reasoning": int(num_reasoning),
            "best_of_n_selections_total": int(total_chosen),
            "best_of_n_selections_reasoning_percentage": float(reasoning_pct)
        })

        print(f"Best-of-n selections using reasoning: {num_reasoning}/{total_chosen} ({reasoning_pct:.2f}%)")

    metrics["evaluation_time_seconds"] = float(elapsed)
    metrics["invalid_grader_outputs"] = len(invalid_grader_outputs)
    print(f"Evaluation time: {elapsed:.2f}s")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved evaluation results to {args.output_json}")

    if invalid_grader_outputs:
        preview = ", ".join(
            f"row {idx}: {output.strip()!r}"
            for idx, output in invalid_grader_outputs[:5]
        )
        warnings.warn(
            f"{len(invalid_grader_outputs)} grader outputs were not valid Yes/No "
            f"decisions. Legacy contains-Yes fallback was used. Examples: {preview}",
            RuntimeWarning,
            stacklevel=1,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        mark_progress_failed(f"{type(error).__name__}: {error}")
        raise
