#!/usr/bin/env python3
"""Held-out KodCode comparison for the original and trained LFM3 policies."""
import argparse
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from lfm3_dual_baseline import ROOT, MODEL, extract_asserts, extract_code, isolated_run, valid_tests


def greedy(model, tokenizer, messages, device, max_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--tasks", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert torch.cuda.device_count() >= 2
    rows = pq.read_table(ROOT / "Code-A1/code/rl/val_data/validation_data_mix.parquet").to_pylist()
    rows = [r for r in rows if r.get("GT_code") and r.get("GT_test") and r.get("prompt") and not r.get("is_OJ")]
    random.Random(2026).shuffle(rows)
    rows = rows[:args.tasks]
    assert len(rows) == args.tasks, "Not enough held-out examples"
    set_seed(2026)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    paths = [MODEL, MODEL] if args.checkpoint is None else [args.checkpoint / "coder-step-128", args.checkpoint / "tester-step-128"]
    models = []
    for path, device in zip(paths, ("cuda:0", "cuda:1")):
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        model.eval()
        models.append(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"tasks": args.tasks, "model_paths": [str(x) for x in paths],
               "coder_passed": 0, "coder_test_total": 0, "coder_test_passed": 0,
               "tester_assertions": 0, "tester_valid_assertions": 0}
    with args.output.open("w") as f:
        for index, row in enumerate(rows, 1):
            code = extract_code(greedy(models[0], tokenizer, row["prompt"], "cuda:0", args.max_tokens))
            tests = row["GT_test"][:4]
            passed = [isolated_run(code + "\n" + test)[0] for test in tests]
            tester_prompt = [{"role": "system", "content": "Write valid Python assert statements only."},
                {"role": "user", "content": "Problem:\n" + row["problem"] +
                 "\nCorrect solution:\n```python\n" + row["GT_code"] +
                 "\n```\nWrite up to eight diverse assert statements for the function."}]
            assertions = extract_asserts(greedy(models[1], tokenizer, tester_prompt, "cuda:1", args.max_tokens))
            valid = valid_tests(row["GT_code"], assertions)
            metrics["coder_passed"] += int(bool(passed) and all(passed))
            metrics["coder_test_total"] += len(tests)
            metrics["coder_test_passed"] += sum(passed)
            metrics["tester_assertions"] += len(assertions)
            metrics["tester_valid_assertions"] += len(valid)
            f.write(json.dumps({"question_id": row["extra_info"].get("question_id"),
                                "coder_passed": sum(passed), "coder_total": len(tests),
                                "tester_assertions": len(assertions), "tester_valid": len(valid)}) + "\n")
            f.flush()
            print(f"evaluated {index}/{args.tasks}", flush=True)
    metrics["coder_pass_at_1"] = metrics["coder_passed"] / args.tasks
    metrics["tester_valid_per_query"] = metrics["tester_valid_assertions"] / args.tasks
    args.output.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
