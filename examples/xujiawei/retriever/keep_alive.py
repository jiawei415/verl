"""GPU keep-alive: encode a dummy query every N seconds so nvidia-smi sees the
retriever GPU as active. Prevents mlx from culling the retriever worker for
low utilization (2h @ <30% rule)."""

from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--interval_s", type=float, default=30.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--burst_iters", type=int, default=32, help="Encoder forward passes per tick.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[keep_alive] no CUDA, exiting")
        return

    device = torch.device("cuda")
    print(f"[keep_alive] loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device).eval()

    dummy_text = ["query: keep the GPU busy so mlx does not cull the worker"] * args.batch
    enc = tokenizer(
        dummy_text,
        max_length=args.seq_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    tick = 0
    # Continuous busy loop: keep GPU util > 30% average so mlx doesn't cull.
    # Adjust `sleep_between_bursts` to trade util for wasted compute.
    sleep_between_bursts = 0.0  # 0 = flat-out (~90%+ util); 0.5 = ~50%; etc.
    while True:
        with torch.no_grad():
            for _ in range(args.burst_iters):
                _ = model(**enc)
        torch.cuda.synchronize()
        tick += 1
        if tick % 20 == 0:
            print(f"[keep_alive] tick={tick}", flush=True)
        if sleep_between_bursts > 0:
            time.sleep(sleep_between_bursts)


if __name__ == "__main__":
    main()
