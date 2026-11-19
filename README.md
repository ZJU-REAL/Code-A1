# ECE6257: Adversarial Co-Training for Agentic Coding

Reproduce Code-A1's adversarial coder–tester co-training and evaluate how branch coverage and selected runtime values affect test quality and code generation.
- [LFM3 dual-policy training and held-out results](LFM3_TRAINING_RESULTS.md) (128 steps; tester valid assertions +38%; coder pass@1 -3.1 pp)

- [LFM3-1.7B inference-only results](LFM3_EVALUATION.md) (128-task feasibility and full benchmarks completed; W&B disabled)
- [Progress and seven-week plan](progress.md)
- [Study results and final report](FINAL_REPORT.md) (resource-limited, 32 steps, one seed)
- [Environment, data, Slurm, and operating instructions](ENVIRONMENT.md)
- [W&B training and evaluation metrics](https://wandb.ai/6257-final-proj/6257-adversarial-co-training/workspace) (private team project)

Repository documentation and generated reports are maintained in English.

The inference-only follow-up is complete: LFM3 scores 106/164 (64.63%) on HumanEval and 290/1140 (25.44%) on BigCodeBench Instruct. The ULT Lite bounded adaptation reports 21.74% mean branch coverage and 22.99% mutation detection, with 197/200 functions measured. Historical Qwen weights/checkpoints were removed at the user's request; their numeric results and reports remain available.

Use `source scripts/activate.sh lfm` for current model inference. Sandbox scoring continues to use the existing evaluation environment; setup and submission commands are in the [environment guide](ENVIRONMENT.md).

This team repository retains the original Code-A1 source in `Code-A1/` and project website in `docs/`. See [UPSTREAM_README.md](UPSTREAM_README.md) for the original setup, training, and paper results. The PACE study scripts use a separately pinned checkout at `external/Code-A1` for reproducibility; prepare source checkouts as described in [ENVIRONMENT.md](ENVIRONMENT.md).

Activate the training environment from the project root:

```bash
source scripts/activate.sh rl
python scripts/check_quota.py
```

Submit the weeks 1–2 smoke checks through Slurm:

```bash
sbatch scripts/gpu_smoke.sbatch       # coe-gpu: single-GPU inference
sbatch scripts/sandbox_smoke.sbatch   # ice-cpu: Python sandbox and isolation checks
sbatch scripts/benchmark_smoke.sbatch # coe-gpu: one example from each of three benchmarks
sbatch scripts/week12_smoke.sbatch    # coe-gpu: two GPUs, one co-training update
```

Scripts default to `account=ece` and `qos=coe-ice`; submit them from the project root. Logs and check reports are stored in `artifacts/setup/`. `scripts/prepare_week12.py` writes local training configurations to `artifacts/week12/`. The sandbox requires `.venv-sandbox` and the pinned upstream code; rebuild it with `bash scripts/setup_sandbox.sh`.

These smoke checks validate execution paths; they do not establish benchmark scores or complete the baseline reproduction. The 300 GiB and one-million-file Scratch quotas are shared across all of the user's Scratch directories. Check quota before new jobs; training smoke checks do not save checkpoints.

The fixed-budget study is complete: E0–E3 each finished 32 steps of dual-model training, followed by evaluation of five model variants, ablation statistics, figures, and W&B readback verification. The main paired intervals for the feedback variants all include zero; there is no evidence of a stable improvement in this study. See the [experiment protocol](EXPERIMENT_PROTOCOL.md), [Code-A1 implementation review](CODE_A1_METHOD.md), and [final report](FINAL_REPORT.md) for scope and deviations. Actual job IDs are recorded in `artifacts/study/jobs.json`.

In a fresh experiment workspace with the environment and data already prepared, run `python scripts/submit_study.py` to submit the dependency graph. The script refuses to submit again if `jobs.json` exists; archive the current study before rerunning. W&B monitoring instructions are in the [environment guide](ENVIRONMENT.md).
