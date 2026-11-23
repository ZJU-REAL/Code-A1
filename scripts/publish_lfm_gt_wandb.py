#!/usr/bin/env python3
"""Publish an LFM3 GT-reward GRPO run (numeric metrics only) to Weights & Biases.

Uploads: per-step numeric training metrics from metrics.jsonl (logged at their true step),
the frozen protocol as config, and aggregate eval summaries. No prompts, completions,
per-example records, data or weights are uploaded.

Usage: .venv-rl/bin/python scripts/publish_lfm_gt_wandb.py --run artifacts/lfm-gt-coder128 \
         --eval artifacts/lfm-gt-coder128-eval --name lfm3-gt-coder128
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--entity", default="6257-final-proj")
    ap.add_argument("--project", default="6257-adversarial-co-training")
    args = ap.parse_args()

    from dotenv import dotenv_values
    os.environ["WANDB_API_KEY"] = dotenv_values(ROOT / ".env")["WANDB_API_KEY"]
    os.environ.setdefault("WANDB_DIR", str(ROOT / "artifacts/wandb"))
    os.environ["WANDB_MODE"] = "online"
    import wandb

    protocol = json.loads((args.run / "protocol.json").read_text())
    protocol.pop("question_ids", None)
    records = [json.loads(l) for l in (args.run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    steps = {}
    for r in records:
        if "event" in r and r["event"] == "step_end":
            steps.setdefault(r["step"], {}).update({"perf/step_seconds": r["seconds"], "perf/max_mem_gib": r["max_mem_gib"]})
        elif "event" not in r and "loss" in r:
            steps.setdefault(r["step"], {}).update({k: v for k, v in r.items()
                                                    if k not in ("step", "time") and isinstance(v, (int, float))})
    jobs = sorted({r["slurm_job_id"] for r in records if r.get("event") == "job_start"})
    final = max(steps) if steps else 0

    run = wandb.init(entity=args.entity, project=args.project, name=args.name, id=args.name,
                     resume="allow", job_type="train",
                     config={**protocol, "framework": "TRL 1.18.2 GRPOTrainer + vLLM 0.34.0 colocate",
                             "slurm_jobs": jobs, "eval": "Jack held-out 64 (MBPP+ 36 / HumanEval+ 28), 1024 tokens"},
                     tags=["lfm3", "gt-reward", "coder-only", "grpo"])
    run.define_metric("train_step")
    run.define_metric("*", step_metric="train_step")
    already = run.summary.get("train_step", 0) or 0
    for s in sorted(steps):
        if s > already:
            run.log({"train_step": s, **steps[s]})

    for label in sorted(p.name for p in args.eval.iterdir() if (p / "summary.json").exists()):
        summ = json.loads((args.eval / label / "summary.json").read_text())
        for split in ("heldout", "train"):
            for k, v in summ[split].items():
                if isinstance(v, (int, float)):
                    run.summary[f"eval_{label}/{split}_{k}"] = v
    run.summary["training/completed_steps"] = final
    run.finish()
    print(f"published {len(steps)} steps (final {final}) to {args.entity}/{args.project}/{args.name}")


if __name__ == "__main__":
    main()
