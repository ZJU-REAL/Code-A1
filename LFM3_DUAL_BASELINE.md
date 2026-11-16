# LFM3 dual-model baseline (Jack branch)

This branch implements a **separate, resource-limited dual-policy baseline** for `LiquidAI/LFM3-1.7B`. It does not invoke the upstream Code-A1 verl/vLLM trainer. The old vLLM/Transformers stack is incompatible with the pinned LFM3 checkpoint, so this runner uses Hugging Face Transformers 5.22.2 and two GPUs. It never loads Qwen weights.

## Training loop

- Read the bundled KodCode Parquet. Shuffle with seed 42 and select 512 non-OJ problems. There are 128 steps, each with 4 problems and 2 coder/tester samples per problem: 1,024 coder and 1,024 tester generations.
- Initialize independent coder and tester policies from the same pinned LFM3 base checkpoint. Coder generates a solution; tester receives the problem and candidate and generates assertions.
- Execute generated code and assertions in a rootless Bubblewrap namespace with no network or project/home mount, CPU/memory/file limits, and a timeout. Keep tester assertions that pass on the reference code. Replay previous valid assertions that found bugs through a per-question Mistake Book.
- Coder reward is pass fraction on valid generated and replayed assertions. If neither exists, use the original KodCode tests. Tester reward combines validity and attack rate with `alpha=0.5`. Both policies use grouped mean-centered policy-gradient updates with gradient clipping and AdamW, `lr=1e-6`.
- Save JSONL per-step metrics and final coder/tester HF checkpoints to Scratch via the repository's `artifacts` symlink. No checkpoint is written to the GitHub branch.

This is **not an exact Code-A1 replication**: it uses REINFORCE instead of upstream GRPO/PPO clipping, no reference-policy KL, only up to four original tests for diagnostics, and a 1,024-token output cap. It does not include branch-coverage or runtime-value feedback. Compare its results as an LFM3 implementation baseline, not against the paper's absolute scores without matching training and evaluation conditions.

## PACE usage

From `/home/hice1/jli3596/ECE6257-Adversarial-Co-Learning` on `jack/lfm3-baseline`, after the pinned LFM3 checkpoint and `.venv-lfm` are installed:

```bash
mkdir -p artifacts/lfm3-dual-baseline
sbatch -p ice-cpu -A ece --qos=coe-ice -c 2 --mem=8G -t 00:10:00 \
  -o artifacts/lfm3-dual-baseline/data-%j.log \
  --wrap='source scripts/activate.sh lfm; python -u scripts/lfm3_dual_baseline.py --data-check --steps 1 --batch-size 1 --problems 1'
sbatch --time=03:00:00 scripts/lfm3_dual_baseline.sbatch \
  --steps 1 --batch-size 16 --problems 16 --max-tokens 1024 \
  --require-both-gradients --output artifacts/lfm3-dual-gate
sbatch scripts/lfm3_dual_baseline.sbatch \
  --steps 128 --batch-size 4 --problems 512 --samples 2 --max-tokens 1024 \
  --output artifacts/lfm3-dual-baseline
```

Only launch the 512-problem job after the 16-problem gate completes with finite, nonzero coder and tester gradient norms. The full run's maximum wall time is 16 hours on `coe-gpu`; if it does not finish within that limit, reduce the scale and record the actual number of completed steps from `steps.jsonl`.
