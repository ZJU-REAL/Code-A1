# LFM3-1.7B coder-only GRPO with ground-truth reward (acg_gt): 128 problems, 1 epoch

Date: 2026-11-23. Training job 7255935 (15 min, 1x H100). Eval jobs 7255936 (base), 7255937 (step 32).
W&B: https://wandb.ai/6257-final-proj/6257-adversarial-co-training/runs/lfm3-gt-coder128

## Result

| Held-out 64 (36 MBPP+, 28 HumanEval+) | base | step 32 | Δ |
|---|---:|---:|---:|
| greedy pass@1 | 59.4% (38/64) | **84.4%** (54/64) | +25.0 |
| sampled pass@1 (8 samples) | 58.8% | **83.2%** | +24.4 |
| pass@8 | 78.1% | **90.6%** | +12.5 |
| correct samples | 301/512 | 426/512 | +125 |
| greedy answers cut off at 1024 tokens | 30/64 | 6/64 | −24 |

| Train 128 (KodCode, scored with acg_gt) | base | step 32 |
|---|---:|---:|
| greedy mean reward (GT pass fraction) | 0.236 | 0.629 |
| greedy all GT tests pass | 19.5% | 51.6% |
| sampled mean reward | 0.235 | 0.590 |
| pass@8 (≥1 of 8 passes all tests) | 42.2% | 69.5% |

The trained model is better on every task or tied with base. Per task, its sampled correct count went up on 35 of 64 tasks, stayed the same on 29 and went down on none. A paired sign-flip test gives p < 5e-5.

### Where the gain comes from: answers that fit in 1024 tokens

| | tasks | base greedy pass | step-32 greedy pass |
|---|---:|---:|---:|
| base answer **not** cut off | 34 | 32 | 31 |
| base answer cut off at 1024 | 30 | 6 | 23 |

- All 17 tasks that became greedy passes were tasks where base ran out of tokens before finishing its answer.
- On the 34 tasks where base finished its answer, base and step 32 are tied (32 vs 31).
- During training, the mean completion length fell from 704 to about 585 tokens, and the share of completions cut off fell from 50% to about 33%.

So under the 1024-token budget (same as Jack's eval) the improvement is real. But it mostly comes from writing shorter answers that finish inside the budget, with the code still inside a `` ```python `` block. There is no sign yet that the model solves problems it could not solve before.

## Training parameters

| | |
|---|---|
| Model | `models/LFM3-1.7B` (loaded with `trust_remote_code`, SDPA attention) |
| Data | `artifacts/study/train.parquet` (the same 128 problems as E0): from Code-A1's `data/Code-A1/train_data/kodcode_hard_dual_model_training_data_mix.parquet` (9,688 rows, all `kodcode_light`), keep the 9,684 prompts of at most 1024 tokens, apply `random.Random(42).shuffle`, take the first 128 (`scripts/prepare_experiments.py`). Each problem has 1–10 GT tests (pytest). |
| Reward | Code-A1 `acg_gt`: fraction of KodCode GT tests passed; 0 if the code is invalid. No tester and no Mistake Book. |
| Algorithm | GRPO (TRL 1.18.2 `GRPOTrainer`), advantages normalised within each problem's group (`scale_rewards="group"`) |
| Loss | `loss_type="dapo"` (token-mean); clip ε 0.2 low / 0.28 high; KL β = 0.001 against the base model |
| Batch | 4 problems × 8 rollouts = 32 completions per optimizer step (micro-batch 8 × grad-accum 4); 1 PPO iteration per batch |
| Schedule | 32 steps = 1 epoch over 128 problems, no shuffling (dataset order) |
| Optimizer | AdamW, lr 1e-6, cosine decay, 0 warmup, weight decay 0.1, grad clip 1.0 |
| Precision | fp32 weights and AdamW moments (verified `torch.float32` at job end), bf16 autocast compute, gradient checkpointing |
| Seeds | `seed=42`, `data_seed=42` |
| Rollout engine | vLLM 0.34.0, running on the same GPU as training (memory 0.3, sleep mode, max model length 4096) |
| Checkpoints | every 8 steps, last 2 kept (`ckpt/checkpoint-24`, `ckpt/checkpoint-32`, 20 GB each). Resume is automatic. |
| Cost | 19.7 s per step on average, peak memory 41.9 GiB, 14 min 47 s of training (15 min wall-clock job) |

Code: `scripts/lfm_grpo_gt.py`, `scripts/lfm_grpo_gt.sbatch`. Environment: `envs/lfm-grpo-env.tar.zst` (vllm 0.34.0, torch 2.14.0, transformers 5.22.2, trl 1.18.2).

## Decoding parameters

| | Training rollouts | Eval: greedy | Eval: sampled |
|---|---|---|---|
| temperature | 1.0 | 0 | 0.7 |
| top-p | 1.0 | — | 0.95 |
| samples per prompt | 8 | 1 | 8 |
| max new tokens | 1024 | 1024 | 1024 |
| seed | trainer seed 42 | vLLM seed 2026 | vLLM seed 2026 |
| chat template | `enable_thinking=False` | same | same |
| engine | vLLM 0.34 (bf16) | vLLM 0.34 (bf16) | vLLM 0.34 (bf16) |

## Training curve (averages over 8 steps)

| steps | reward | cut-off ratio | mean length | KL | problems with no learning signal | grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 1–8 | 0.431 | 0.504 | 704 | 0.001 | 0.53 | 1.17 |
| 9–16 | 0.520 | 0.293 | 566 | 0.004 | 0.41 | 1.13 |
| 17–24 | 0.514 | 0.336 | 598 | 0.006 | 0.25 | 1.50 |
| 25–32 | 0.505 | 0.332 | 585 | 0.008 | 0.22 | 1.48 |

"No learning signal" means all 8 rollouts for a problem got the same reward, so that problem contributes nothing to the update. Per-step values are in `metrics.jsonl` and on W&B.

## Evaluation protocol

- **Held-out:** Jack's 64 tasks, the same ones used by `lfm3_dual_eval.py` and `lfm3_passk_eval.py`. They come from `validation_data_mix.parquet` with a `random.Random(2026)` shuffle, first 64 (36 MBPP+, 28 HumanEval+), so they are out of distribution relative to KodCode training.
- **Scoring:** take the first `` ```python `` block (Jack's `extract_code`), then run `code + "candidate = <fn>" + official test` in Bubblewrap (`isolated_run`: no network, 2 GB, 5 s). Exit code 0 counts as a pass.
- **pass@k:** unbiased estimator (`pass_at_k` from `lfm3_passk_eval.py`).
- **Train set:** the 128 training problems scored with `acg_gt`, as in training.
- **Backend:** the only difference from Jack is the backend, vLLM instead of HF `generate`. Base on vLLM gets 59.4 / 58.8 / 78.1; Jack's HF base got 56.3 / 55.9 / 82.8. Compare runs only within the same backend.
- Code: `scripts/lfm_eval_vllm.py`, `scripts/lfm_eval_vllm.sbatch`. Outputs: `artifacts/lfm-gt-coder128-eval/{base,step32}/`.

## Takeaways and next steps

1. Training the coder alone on the GT reward does improve held-out pass@1 under the 1024-token budget, by +25 points greedy and +24 sampled, while KL stays small (< 0.01).
2. The gain is almost entirely on tasks where base ran out of tokens. Re-evaluating base and step 32 at 2048 or 4096 tokens would show how much is ability and how much is concision. We are keeping 1024 to match Jack for now.
3. The earlier E0 setup (dual-policy, `acg_llm` reward from the tester and Mistake Book) did not show a gain. This run differs in four ways: GT reward, n = 8 instead of 2, group-normalised GRPO, and LFM3 instead of Qwen. The GT reward is the cleanest signal, but the other three changes also differ, so this run does not single it out as the cause.
