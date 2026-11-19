# LFM3 training feasibility check

On 2026-11-16, Jack's PACE-ICE clone used the `jack/lfm3-baseline` branch and
the pinned `LiquidAI/LFM3-1.7B` revision
`1da06bec6a819da10b54c67e5c90972b46d81871`. The environment used
Transformers 5.22.2 and PyTorch 2.6.0+cu124. Model and environment files were
stored in Jack's Scratch, separate from the Git checkout.

`sbatch scripts/lfm3_gradient_smoke.sbatch` completed as Slurm job 7119976
with exit code 0 on one H100. It trained on one short synthetic coding example,
with 25 prompt tokens and 11 target tokens. The loss was 1.09615, the gradient
norm before clipping was 270.0 and finite, and one AdamW step completed. Peak
allocated GPU memory was 16.16 GiB. The job took 24 seconds. The raw result is
`artifacts/lfm3-training-smoke/result.json` on Jack's PACE workspace.

This verifies only that this LFM3 checkpoint can load and take a single
optimizer step in the dedicated environment. It is **not** Code-A1 baseline
training: it does not generate coder/tester rollouts, use KodCode, execute tests,
compute adversarial rewards, update two models, or evaluate a checkpoint. No
weights were saved. A larger baseline requires a compatible dual-model
Code-A1/verl rollout stack and a frozen training and evaluation protocol.

The initial smoke attempt, job 7119931, stopped before loading the model
because `download_lfm_model.py` had generated broken relative file links when
the model directory was symlinked to Scratch. The downloader was corrected and
the successful retry used the repaired links.
