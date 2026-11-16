#!/usr/bin/env python3
"""Check that the pinned LFM3 checkpoint can take one optimizer step.

This is an architecture and environment check, not Code-A1 training.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/LFM3-1.7B"
OUT = ROOT / "artifacts/lfm3-training-smoke"
OUT.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), "Run within a GPU allocation"
assert (MODEL / "config.json").is_file(), "Pinned LFM3 checkpoint is missing"

set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda",
)
model.train()
model.config.use_cache = False

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python function add(a, b) that returns their sum."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
text = prompt + "def add(a, b):\n    return a + b"
batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
labels = batch["input_ids"].clone()
prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
labels[:, :prompt_length] = -100
assert (labels != -100).sum().item() > 0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad(set_to_none=True)
loss = model(**batch, labels=labels, use_cache=False).loss
assert torch.isfinite(loss).item(), f"Nonfinite loss: {loss.item()}"
loss.backward()
gradients = [p.grad for p in model.parameters() if p.grad is not None]
assert gradients, "No parameter gradients"
assert all(torch.isfinite(g).all().item() for g in gradients), "Nonfinite gradient"
gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
assert gradient_norm.item() > 0, "Zero gradient norm"
optimizer.step()

result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "kind": "single_model_gradient_smoke_not_code_a1",
    "model": "LiquidAI/LFM3-1.7B",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "prompt_tokens": prompt_length,
    "target_tokens": int((labels != -100).sum().item()),
    "loss": float(loss.item()),
    "gradient_norm_before_clip": float(gradient_norm.item()),
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    "optimizer_step_completed": True,
}
(OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
