#!/usr/bin/env python3
"""Resource-limited LFM3 coder/tester policy-gradient baseline.

This is a separate HF implementation, not the upstream verl/GRPO trainer.
Generated code runs only inside an unprivileged Bubblewrap namespace.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/LFM3-1.7B"
DATA = ROOT / "Code-A1/code/rl/train_data/kodcode_hard_dual_model_training_data_mix.parquet"


def isolated_run(source, timeout=5):
    if len(source.encode()) > 131072:
        return False, "source_too_large"
    with tempfile.TemporaryDirectory(prefix="lfm3-eval-") as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(source)
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--ro-bind", tmp, "/input", "--chdir", "/tmp",
               "--setenv", "PATH", "/usr/bin", "/usr/bin/prlimit", "--as=2147483648",
               "--fsize=1048576", "--nofile=128", f"--cpu={timeout+1}", "--",
               "/usr/bin/python3", "-I", "-B", "/input/main.py"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout + 2, check=False)
            return result.returncode == 0, (result.stdout + result.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    return (match.group(1) if match else text).strip()


def extract_asserts(text):
    code = extract_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [ast.unparse(node) for node in tree.body if isinstance(node, ast.Assert)][:8]


def valid_tests(reference, generated):
    accepted = []
    for assertion in dict.fromkeys(generated):
        ok, _ = isolated_run(reference + "\n" + assertion + "\n")
        if ok:
            accepted.append(assertion)
    return accepted


def score(code, reference, ground_tests, assertions, history):
    syntax_ok = True
    try:
        ast.parse(code)
    except SyntaxError:
        syntax_ok = False
    gt_results = [isolated_run(code + "\n" + t)[0] for t in ground_tests[:4]] if syntax_ok else []
    valid = valid_tests(reference, assertions)
    active = list(dict.fromkeys(valid + history))
    candidate_results = [isolated_run(code + "\n" + t)[0] for t in active] if syntax_ok else []
    coder_reward = sum(candidate_results) / len(active) if active else (sum(gt_results) / len(gt_results) if gt_results else 0.0)
    validity = max(0.0, min(1.0, len(valid) / 8.0, 1.0 - (len(assertions) - len(valid)) / 8.0))
    attack = sum(not passed for passed in candidate_results[:len(valid)]) / 8.0
    tester_reward = 0.5 * validity + 0.5 * attack if syntax_ok else 0.0
    mistakes = [t for t, passed in zip(active, candidate_results) if not passed and t in valid]
    return coder_reward, tester_reward, valid, mistakes, sum(gt_results), len(gt_results)


def generate(model, tokenizer, messages, device, max_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.shape[1] > 2048:
        raise ValueError(f"Prompt too long: {ids.shape[1]} tokens")
    model.eval()
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=max_tokens, do_sample=True,
                              temperature=1.0, top_p=0.95,
                              pad_token_id=tokenizer.eos_token_id)
    completion = full[0, ids.shape[1]:].detach().cpu()
    return ids[0].detach().cpu(), completion, tokenizer.decode(completion, skip_special_tokens=True)


def update(model, optimizer, samples, device):
    rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    advantage = rewards - rewards.mean()
    if len(samples) > 1:
        advantage /= rewards.std(unbiased=False).clamp(min=1e-4)
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    for (prompt, completion, _), weight in zip(samples, advantage):
        if completion.numel() == 0:
            continue
        ids = torch.cat((prompt, completion)).unsqueeze(0).to(device)
        logits = model(ids, use_cache=False).logits[:, prompt.numel()-1:-1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, completion.to(device)[None, :, None]).squeeze(-1)
        loss = -weight.to(device) * selected.mean() / len(samples)
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(norm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--problems", type=int, default=512)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--require-both-gradients", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/lfm3-dual-baseline")
    p.add_argument("--data-check", action="store_true")
    args = p.parse_args()
    if args.samples < 2 or args.problems < args.steps * args.batch_size:
        p.error("samples must be >= 2 and problems >= steps * batch-size")
    rows = pq.read_table(DATA).to_pylist()
    rows = [r for r in rows if r.get("GT_code") and r.get("GT_test") and r.get("prompt") and not r.get("is_OJ")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.problems]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "steps.jsonl").exists():
        raise FileExistsError(f"Existing run in {args.output}; choose a new output directory")
    (args.output / "config.json").write_text(json.dumps({**vars(args), "output": str(args.output),
        "model": "LiquidAI/LFM3-1.7B", "kind": "hf_two_policy_reinforce_not_official_code_a1_grpo"}, indent=2))
    print(f"selected {len(rows)} non-OJ tasks from {pq.ParquetFile(DATA).metadata.num_rows}", flush=True)
    if args.data_check:
        assert rows and rows[0]["prompt"][0]["role"] == "system"
        assert isolated_run("print('sandbox-ok')")[0], "Bubblewrap execution failed"
        print("data and sandbox check passed", flush=True)
        return
    assert torch.cuda.device_count() >= 2, "Two GPUs required, one per policy"
    assert MODEL.joinpath("config.json").is_file(), "LFM3 checkpoint missing"
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    models, optimizers = [], []
    for device in ("cuda:0", "cuda:1"):
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        model.gradient_checkpointing_enable()
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.1))
    mistake_book = {}
    log_path = args.output / "steps.jsonl"
    for step in range(args.steps):
        coder_samples, tester_samples, details = [], [], []
        for row in rows[step * args.batch_size:(step + 1) * args.batch_size]:
            key = str(row["extra_info"].get("question_id", step))
            for _ in range(args.samples):
                ca, cb, code_text = generate(models[0], tokenizer, row["prompt"], "cuda:0", args.max_tokens)
                code = extract_code(code_text)
                tester_prompt = [{"role": "system", "content": "Write valid Python assert statements only."},
                    {"role": "user", "content": "Problem:\n" + row["problem"] +
                     "\nCandidate solution:\n```python\n" + code +
                     "\n```\nWrite up to eight assert statements that pass on the correct solution and reveal bugs in the candidate."}]
                ta, tb, test_text = generate(models[1], tokenizer, tester_prompt, "cuda:1", args.max_tokens)
                cr, tr, valid, mistakes, gt_passed, gt_total = score(code, row["GT_code"], row["GT_test"],
                    extract_asserts(test_text), mistake_book.get(key, []))
                coder_samples.append((ca, cb, cr))
                tester_samples.append((ta, tb, tr))
                details.append({"question_id": key, "coder_reward": cr, "tester_reward": tr,
                                "valid_tests": len(valid), "gt_passed": gt_passed, "gt_total": gt_total})
                mistake_book.setdefault(key, [])
                mistake_book[key] = list(dict.fromkeys(mistake_book[key] + mistakes))[:64]
        coder_norm = update(models[0], optimizers[0], coder_samples, "cuda:0")
        tester_norm = update(models[1], optimizers[1], tester_samples, "cuda:1")
        record = {"step": step + 1, "samples": details,
                  "coder_grad_norm": coder_norm, "tester_grad_norm": tester_norm,
                  "mistake_book_size": sum(map(len, mistake_book.values()))}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.require_both_gradients and step + 1 == args.steps:
            assert any(json.loads(line)["coder_grad_norm"] > 0 for line in log_path.read_text().splitlines())
            assert any(json.loads(line)["tester_grad_norm"] > 0 for line in log_path.read_text().splitlines())
        if step + 1 == args.steps:
            for name, model in zip(("coder", "tester"), models):
                model.save_pretrained(args.output / f"{name}-step-{step+1}", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "tokenizer")
            (args.output / "mistake_book.json").write_text(json.dumps(mistake_book))


if __name__ == "__main__":
    main()
