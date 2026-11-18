# LFM3 Dual-Policy Training Result

## Scope

This run evaluates a resource-limited adversarial coder/tester training baseline using `LiquidAI/LFM3-1.7B`. It is separate from the earlier Code-A1 smoke run, which used `Qwen2.5-Coder-1.5B-Instruct`, and from the published LFM3 inference-only baseline in `report/lfm3/summary.json` (`training_steps: 0`).

## Training configuration

- Branch: `jack/lfm3-baseline`
- Model: `LiquidAI/LFM3-1.7B` for both coder and tester
- Training problems: 512 non-OJ examples
- Optimizer steps: 128
- Batch size: 4
- Samples per problem: 2
- Maximum response length: 1024 tokens
- Seed: 2026
- Checkpoint interval: 10 steps, with an atomic resumable latest checkpoint
- Final checkpoints: `coder-step-128` and `tester-step-128`
- Training job: Slurm `7150967` (completed successfully)

## Held-out evaluation

The original and trained policies were evaluated on the same 64 examples selected from `validation_data_mix.parquet` with seed 2026. Decoding was greedy. Coder outputs were executed against each task's complete official test harness. Tester outputs were measured by the number of generated assertions that execute successfully against the reference solution.

Evaluation job: Slurm `7155161` (completed successfully in 59 minutes on two H200 GPUs).

| Metric | Original LFM3 | Step-128 policies | Change |
| --- | ---: | ---: | ---: |
| Coder pass@1 | 36/64 (56.25%) | 34/64 (53.13%) | -3.12 pp |
| Tester assertions generated | 229 | 308 | +34.5% |
| Valid tester assertions | 81 | 112 | +38.3% |
| Valid assertions per query | 1.266 | 1.750 | +38.3% |
| Assertion validity fraction | 35.4% | 36.4% | +1.0 pp |

Paired task comparison:

- Coder: 30 tasks passed under both policies, 6 passed only with the original policy, and 4 passed only after training.
- Tester valid-assertion count: improved on 8 tasks, was unchanged on 53, and decreased on 3.

## Interpretation

Under this fixed 64-task evaluation, co-training increased the quantity of valid tester feedback but did not improve coder pass@1. The two-task coder difference is too small to support a strong regression claim from a single 64-task sample and one training seed.

The earlier LFM3 report on `main` is an inference-only benchmark evaluation over HumanEval, BigCodeBench, and ULT Lite. Its numbers are useful as a model baseline but are not directly comparable with this held-out KodCode-style evaluation because the datasets, decoding settings, and tester metrics differ.

## Reproduction

Run the paired evaluation after the step-128 checkpoints are available:

```bash
sbatch scripts/lfm3_dual_eval.sbatch
```

The ignored output directory contains per-task JSONL records and aggregate summaries:

- `artifacts/lfm3-dual-resumable/heldout-base.jsonl`
- `artifacts/lfm3-dual-resumable/heldout-base.summary.json`
- `artifacts/lfm3-dual-resumable/heldout-trained.jsonl`
- `artifacts/lfm3-dual-resumable/heldout-trained.summary.json`
