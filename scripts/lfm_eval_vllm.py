#!/usr/bin/env python3
"""vLLM coder evaluation for LFM3 checkpoints, matching Jack's held-out protocol.

held-out: the same 64 validation_data_mix tasks as lfm3_dual_eval.py / lfm3_passk_eval.py
          (seed-2026 shuffle), scored with the full official harness in Bubblewrap.
          greedy pass@1, plus 8 samples (T=0.7, top-p 0.95) for sampled pass@1 / pass@8.
train:    the first N training problems, scored with Code-A1's acg_gt (all GT tests pass).
"""
import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from lfm3_dual_baseline import extract_code, isolated_run
from lfm_grpo_gt import A1, MODEL, ROOT, TRAIN, acg_gt_reward
from lfm3_passk_eval import pass_at_k


def heldout_rows(n=64):
    rows = pq.read_table(ROOT / "Code-A1/code/rl/val_data/validation_data_mix.parquet").to_pylist()
    rows = [r for r in rows if r.get("GT_code") and r.get("test") and r.get("prompt")
            and r.get("extra_info", {}).get("function_name") and r.get("extra_info", {}).get("function_declaration")]
    random.Random(2026).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(MODEL))
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train-problems", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    out = args.output_dir / args.label
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.checkpoint, tokenizer=str(MODEL), dtype="bfloat16", seed=2026,
              max_model_len=4096, gpu_memory_utilization=0.85, trust_remote_code=True)
    tok = llm.get_tokenizer()

    def prompts(rows):
        return [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False) for r in rows]

    greedy = SamplingParams(temperature=0.0, max_tokens=1024)
    sampled = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, n=args.samples, seed=2026)
    summary = {"checkpoint": args.checkpoint, "label": args.label}

    held = heldout_rows()
    def held_pass(row, text):
        info = row["extra_info"]
        program = extract_code(text) + "\n\ncandidate = " + info["function_name"] + "\n" + row["test"]
        return isolated_run(program)[0]
    g = llm.generate(prompts(held), greedy)
    s = llm.generate(prompts(held), sampled)
    recs, p1, p8, c = [], 0.0, 0.0, 0
    for row, go, so in zip(held, g, s):
        gp = held_pass(row, go.outputs[0].text)
        correct = sum(held_pass(row, o.text) for o in so.outputs)
        p1 += pass_at_k(args.samples, correct, 1); p8 += pass_at_k(args.samples, correct, 8); c += correct
        recs.append({"task_id": row["extra_info"].get("question_id"), "greedy_pass": int(gp), "sampled_correct": correct,
                     "greedy_truncated": go.outputs[0].finish_reason == "length"})
    n = len(held)
    summary["heldout"] = {"tasks": n, "greedy_pass_at_1": sum(r["greedy_pass"] for r in recs) / n,
                          "sampled_pass_at_1": p1 / n, "pass_at_8": p8 / n, "correct_samples": c,
                          "greedy_truncated": sum(r["greedy_truncated"] for r in recs)}
    (out / "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    train = pq.read_table(TRAIN).slice(0, args.train_problems).to_pylist()
    g = llm.generate(prompts(train), greedy)
    s = llm.generate(prompts(train), sampled)
    def score(row, texts):
        return acg_gt_reward(None, texts, [row["GT_code"]] * len(texts), [list(row["GT_test"])] * len(texts))
    recs = []
    for row, go, so in zip(train, g, s):
        gs = score(row, [go.outputs[0].text])[0]
        ss = score(row, [o.text for o in so.outputs])
        recs.append({"question_id": row["extra_info"].get("question_id"), "greedy_reward": gs,
                     "sampled_mean_reward": sum(ss) / len(ss), "sampled_all_pass": sum(x >= 1 for x in ss)})
    n = len(train)
    summary["train"] = {"problems": n, "greedy_mean_reward": sum(r["greedy_reward"] for r in recs) / n,
                        "greedy_all_pass": sum(r["greedy_reward"] >= 1 for r in recs) / n,
                        "sampled_mean_reward": sum(r["sampled_mean_reward"] for r in recs) / n,
                        "sampled_pass_at_8": sum(r["sampled_all_pass"] > 0 for r in recs) / n}
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
