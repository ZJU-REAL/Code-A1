#!/usr/bin/env python3
"""Backfill the completed LFM3 baseline run into Weights & Biases."""
import argparse
import json
import statistics
from pathlib import Path

import wandb


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="6257-final-proj")
    parser.add_argument("--project", default="6257-adversarial-co-training")
    parser.add_argument("--steps", type=Path, default=Path("artifacts/lfm3-dual-resumable/steps.jsonl"))
    args = parser.parse_args()

    latest = {}
    with args.steps.open() as handle:
        for line in handle:
            row = json.loads(line)
            latest[int(row["step"])] = row

    config = {
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "training_examples": 512,
        "optimizer_steps": 128,
        "batch_size": 4,
        "samples_per_problem": 2,
        "max_new_tokens": 1024,
        "seed": 2026,
        "evaluation_tasks": 64,
        "evaluation_samples_per_task": 8,
        "evaluation_temperature": 0.7,
        "evaluation_top_p": 0.95,
        "historical_backfill": True,
        "loss_logged": False,
        "kl_logged": False,
    }
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id="jack-lfm3-dual-128steps-seed2026",
        name="Jack LFM3 dual-policy baseline (128 steps)",
        resume="allow",
        tags=["lfm3", "dual-policy", "baseline", "backfill", "seed-2026"],
        config=config,
    )

    for step in sorted(latest):
        row = latest[step]
        samples = row.get("samples", [])
        gt_total = sum(s.get("gt_total", 0) for s in samples)
        metrics = {
            "train/coder_reward": mean([s.get("coder_reward", 0.0) for s in samples]),
            "train/tester_reward": mean([s.get("tester_reward", 0.0) for s in samples]),
            "train/valid_tests_per_sample": mean([s.get("valid_tests", 0.0) for s in samples]),
            "train/coder_grad_norm": row.get("coder_grad_norm", 0.0),
            "train/tester_grad_norm": row.get("tester_grad_norm", 0.0),
            "train/mistake_book_size": row.get("mistake_book_size", 0),
        }
        if gt_total:
            metrics["train/ground_tests_pass_rate"] = sum(s.get("gt_passed", 0) for s in samples) / gt_total
        run.log(metrics, step=step)

    run.summary.update({
        "eval/greedy_base_coder_pass_at_1": 0.5625,
        "eval/greedy_trained_coder_pass_at_1": 0.53125,
        "eval/base_tester_valid_tests_per_query": 1.265625,
        "eval/trained_tester_valid_tests_per_query": 1.75,
        "eval/sampled_base_pass_at_1": 0.55859375,
        "eval/sampled_trained_pass_at_1": 0.556640625,
        "eval/base_pass_at_8": 0.828125,
        "eval/trained_pass_at_8": 0.78125,
        "eval/base_correct_samples": 286,
        "eval/trained_correct_samples": 285,
        "eval/pass_at_8_base_only_tasks": 3,
        "eval/pass_at_8_trained_only_tasks": 0,
        "eval/pass_at_8_both_tasks": 50,
        "eval/pass_at_8_neither_tasks": 11,
        "notes/loss": "Unavailable: the historical trainer did not log loss.",
        "notes/kl": "Unavailable: the historical trainer did not log KL divergence.",
    })
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
