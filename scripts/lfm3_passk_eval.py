#!/usr/bin/env python3
"""Estimate coder pass@1 and pass@8 from eight sampled solutions per task."""
import argparse
import json
import math
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm3_dual_baseline import ROOT, MODEL, extract_code, isolated_run


def pass_at_k(n, correct, k):
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod(range(n - correct - k + 1, n - correct + 1)) / math.prod(
        range(n - k + 1, n + 1)
    )


def eligible_rows(tasks):
    rows = pq.read_table(
        ROOT / "Code-A1/code/rl/val_data/validation_data_mix.parquet"
    ).to_pylist()
    rows = [
        row for row in rows
        if row.get("test") and row.get("prompt")
        and row.get("extra_info", {}).get("function_name")
    ]
    random.Random(2026).shuffle(rows)
    rows = rows[:tasks]
    assert len(rows) == tasks, "Not enough held-out examples"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "trained"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/lfm3-dual-resumable")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.cuda.device_count() >= 1

    path = MODEL if args.model == "base" else args.checkpoint / "coder-step-128"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    rows = eligible_rows(args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {"pass_at_1": 0.0, "pass_at_8": 0.0, "correct_samples": 0}

    with args.output.open("w") as output_file:
        for index, row in enumerate(rows, 1):
            prompt = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to("cuda:0")
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            info = row["extra_info"]
            passed = []
            for output in outputs:
                text = tokenizer.decode(
                    output[input_ids.shape[1]:], skip_special_tokens=True
                )
                code = extract_code(text)
                program = (
                    code + "\n\ncandidate = " + info["function_name"]
                    + "\n" + row["test"]
                )
                passed.append(bool(isolated_run(program)[0]))
            correct = sum(passed)
            p1 = pass_at_k(args.samples, correct, 1)
            p8 = pass_at_k(args.samples, correct, min(8, args.samples))
            totals["pass_at_1"] += p1
            totals["pass_at_8"] += p8
            totals["correct_samples"] += correct
            output_file.write(json.dumps({
                "question_id": info.get("question_id"),
                "data_source": row.get("data_source"),
                "correct": correct,
                "samples": args.samples,
                "pass_at_1": p1,
                "pass_at_8": p8,
            }) + "\n")
            output_file.flush()
            print(f"{args.model}: evaluated {index}/{args.tasks}", flush=True)

    summary = {
        "model": args.model,
        "model_path": str(path),
        "tasks": args.tasks,
        "samples_per_task": args.samples,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "pass_at_1": totals["pass_at_1"] / args.tasks,
        "pass_at_8": totals["pass_at_8"] / args.tasks,
        "correct_samples": totals["correct_samples"],
        "total_samples": args.tasks * args.samples,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
