#!/usr/bin/env python3
"""Coder-only GRPO for LiquidAI/LFM3-1.7B with the Code-A1 ground-truth reward (acg_gt).

Stack: TRL GRPOTrainer + vLLM 0.34 colocated rollout (vLLM >= 0.34 is the first release with
Lfm3ForCausalLM). Hyperparameters mirror the E0 study (artifacts/study/E0/config.yaml): GRPO with
group-normalised advantages, token-mean loss, clip 0.2/0.28, k3 KL 0.001, lr 1e-6 cosine,
weight decay 0.1, grad clip 1.0, 1,024-token responses, 4 problems x n samples per optimizer step.

Precision: master weights and AdamW moments in fp32 (model loaded in fp32), bf16 autocast compute.
Resume: newest *complete* checkpoint in OUTPUT/ckpt is resumed automatically (model, optimizer,
LR scheduler, RNG, trainer state). --stop-after-step ends a job cleanly with a checkpoint while the
LR schedule still spans --max-steps.
"""
import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/LFM3-1.7B"
TRAIN = ROOT / "artifacts/study/train.parquet"  # random.Random(42).shuffle(eligible)[:128]


def load_code_a1_reward():
    """Import Code-A1's reward.py without verl: it only needs DataProto/register for class bodies."""
    verl = types.ModuleType("verl")
    verl.DataProto = object
    workers = types.ModuleType("verl.workers")
    manager = types.ModuleType("verl.workers.reward_manager")
    manager.register = lambda name: (lambda cls: cls)
    sys.modules.update({"verl": verl, "verl.workers": workers, "verl.workers.reward_manager": manager})
    sys.path.insert(0, str(ROOT / "Code-A1/code/rl"))
    import reward
    return reward


A1 = load_code_a1_reward()


def acg_gt_reward(prompts, completions, GT_code, GT_test, **kwargs):
    """Exactly the acg_gt coder reward: fraction of KodCode GT tests passed, 0 if code is invalid."""
    texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
    codes, info = A1.extract_code_from_response(texts)
    result = A1.evaluate(codes, [], list(GT_code), [list(t) for t in GT_test])
    valid, gt_passed, gt_total, gt_failed = result[0], result[4], result[5], result[6]
    scores = []
    for i in range(len(codes)):
        out = A1.compute_dual_model_reward(
            is_code_valid=valid[i], gt_passed=gt_passed[i], gt_total=gt_total[i],
            gt_failed_msgs=gt_failed[i], valid_generated_tests=[], invalid_count=0,
            total_generated_count=0, gen_passed=0, gen_failed_msgs=[], test_phase="comprehensive",
            code_nonrelated_ratio=info["nonrelated_ratio"][i])
        scores.append(float(out["reward_a"]))
    return scores


def complete_checkpoint(ckpt_dir: Path):
    """Newest checkpoint with every file needed for an exact resume (a killed save is skipped)."""
    need = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    found = []
    for p in ckpt_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and all((p / f).exists() for f in need) and any(p.glob("*.safetensors")):
            found.append((int(m.group(1)), p))
    return str(max(found)[1]) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--problems", type=int, default=16)
    ap.add_argument("--rollout-n", type=int, default=8)
    ap.add_argument("--problems-per-step", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=2)
    ap.add_argument("--stop-after-step", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    rows = pq.read_table(TRAIN).slice(0, args.problems).to_pylist()
    assert len(rows) == args.problems
    data = Dataset.from_list([{"prompt": r["prompt"], "GT_code": r["GT_code"], "GT_test": list(r["GT_test"]),
                               "question_id": str(r["extra_info"].get("question_id", i))}
                              for i, r in enumerate(rows)])
    out = args.output
    ckpt_dir = out / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": str(MODEL), "train_file": str(TRAIN), "problems": args.problems,
        "question_ids": list(data["question_id"]), "rollout_n": args.rollout_n,
        "problems_per_step": args.problems_per_step, "max_steps": args.max_steps,
        "epochs": args.max_steps * args.problems_per_step / args.problems, "lr": args.lr,
        "reward": "Code-A1 acg_gt (GT pass fraction)", "precision": "fp32 master+AdamW, bf16 autocast",
    }
    proto_path = out / "protocol.json"
    if proto_path.exists():
        assert json.loads(proto_path.read_text()) == protocol, "protocol changed; refusing to resume"
    else:
        proto_path.write_text(json.dumps(protocol, indent=2) + "\n")

    micro = 8
    config = GRPOConfig(
        output_dir=str(ckpt_dir), seed=42, data_seed=42, shuffle_dataset=False,
        max_steps=args.max_steps, per_device_train_batch_size=micro,
        gradient_accumulation_steps=args.problems_per_step * args.rollout_n // micro,
        num_generations=args.rollout_n, num_iterations=1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0, weight_decay=0.1,
        max_grad_norm=1.0, loss_type="dapo", scale_rewards="group",
        epsilon=0.2, epsilon_high=0.28, beta=0.001,
        max_completion_length=args.max_completion_length, temperature=1.0, top_p=1.0,
        chat_template_kwargs={"enable_thinking": False},
        bf16=True, gradient_checkpointing=True,
        model_init_kwargs={"dtype": torch.float32, "trust_remote_code": True, "attn_implementation": "sdpa"},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, vllm_max_model_length=4096,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2, save_only_model=False,
        logging_steps=1, report_to="none", log_completions=False,
    )

    metrics = open(out / "metrics.jsonl", "a", buffering=1)
    metrics.write(json.dumps({"event": "job_start", "time": time.time(),
                              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                              "stop_after_step": args.stop_after_step}) + "\n")

    class Monitor(TrainerCallback):
        def on_step_begin(self, a, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, a, state, control, **kw):
            metrics.write(json.dumps({"event": "step_end", "step": state.global_step,
                                      "seconds": time.time() - self.t0, "time": time.time(),
                                      "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
            if args.stop_after_step is not None and state.global_step >= args.stop_after_step:
                control.should_save = True
                control.should_training_stop = True

        def on_save(self, a, state, control, **kw):
            metrics.write(json.dumps({"event": "checkpoint", "step": state.global_step, "time": time.time()}) + "\n")

        def on_log(self, a, state, control, logs=None, **kw):
            metrics.write(json.dumps({"step": state.global_step, "time": time.time(), **(logs or {})}) + "\n")

    trainer = GRPOTrainer(model=str(MODEL), reward_funcs=acg_gt_reward, args=config,
                          train_dataset=data, callbacks=[Monitor()])
    p = next(trainer.model.parameters())
    metrics.write(json.dumps({"event": "precision", "param_dtype": str(p.dtype)}) + "\n")
    resume = complete_checkpoint(ckpt_dir)
    metrics.write(json.dumps({"event": "resume_from", "path": resume, "time": time.time()}) + "\n")
    trainer.train(resume_from_checkpoint=resume)
    state = trainer.optimizer.state
    dtypes = sorted({str(v.dtype) for s in state.values() for v in s.values() if torch.is_tensor(v) and v.ndim})
    metrics.write(json.dumps({"event": "job_end", "time": time.time(), "global_step": trainer.state.global_step,
                              "optimizer_state_dtypes": dtypes}) + "\n")


if __name__ == "__main__":
    main()
