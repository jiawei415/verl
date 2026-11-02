"""Stable-RL binary DPPO: rollout anchor, directional mask and detached TIS."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
from verl.workers.config import ActorConfig


@pytest.mark.parametrize("mode", ["dppo_tv", "dppo_kl", "dppo_binary_tv", "dppo_binary_kl"])
def test_stable_rl_loss_and_gradient(mode):
    # Both advantage signs, beneficial/harmful updates, tiny probabilities,
    # truncated IS, zero advantage and padding.
    old = torch.tensor([[0.2, 0.2, 0.8, 0.8, 0.001, 0.3, 0.4]], dtype=torch.float64)
    prob = torch.tensor([[0.6, 0.6, 0.4, 0.4, 0.05, 0.7, 0.9]], dtype=torch.float64)
    adv = torch.tensor([[1., -1., -1., 1., 2., 0., 1.]])
    mask = torch.tensor([[1., 1., 1., 1., 1., 1., 0.]])
    log_prob = prob.log().requires_grad_()
    batch = DataProto.from_dict({"old_log_probs": old.log() + 0.1, "rollout_log_probs": old.log()})
    policy = OmegaConf.create({"loss_mode": mode})
    apply_bypass_mode(batch, OmegaConf.create({"bypass_mode": True}), policy)
    assert policy.loss_mode == mode
    torch.testing.assert_close(batch.batch["old_log_probs"], old.log())
    config = ActorConfig(
        strategy="fsdp", rollout_n=8, use_dynamic_bsz=True,
        clip_ratio_low=0.1, clip_ratio_high=0.15, clip_ratio_c=10.,
    )
    loss, metrics = get_policy_loss_fn(policy.loss_mode)(
        batch.batch["old_log_probs"], log_prob, adv, mask, config=config
    )
    # Independent Stable-RL equations, including the rollout-based IS cap.
    if mode.endswith("tv"):
        invalid = torch.where(adv > 0, prob - old > 0.15, prob - old < -0.1)
    else:
        kl = old * (old.log() - prob.log()) + (1 - old) * torch.log(
            (1 - old + 1e-8) / (1 - prob + 1e-8)
        )
        invalid = torch.where(adv > 0, (kl > 0.15) & (prob > old), (kl > 0.1) & (prob < old))
    expected_grad = -adv * (prob / old).clamp(max=10) * (~invalid) * mask / mask.sum()
    torch.testing.assert_close(loss, (expected_grad * prob.log()).sum())
    loss.backward()
    torch.testing.assert_close(log_prob.grad, expected_grad)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_existing_bypass_loss_is_preserved():
    batch = DataProto.from_dict({"rollout_log_probs": torch.zeros(1, 2)})
    policy = OmegaConf.create({"loss_mode": "vanilla"})
    apply_bypass_mode(batch, OmegaConf.create({"bypass_mode": True}), policy)
    assert policy.loss_mode == "bypass_mode"


@pytest.mark.parametrize("mode,threshold", [("dppo_tv", "0.15"), ("dppo_binary_kl", "0.05")])
def test_math_script_wires_driver_and_worker(mode, threshold, tmp_path):
    # Capture launcher arguments without loading models or starting training.
    stub = tmp_path / "python3"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ROLLOUT_", "CLIP_RATIO_", "LOSS_MODE"))}
    env.update(PATH=f"{tmp_path}:{env['PATH']}", LOSS_MODE=mode)
    result = subprocess.run(
        ["bash", str(root / "examples/xujiawei/run_grpo_math.sh")],
        env=env, capture_output=True, text=True, check=True,
    )
    args = result.stdout.splitlines()
    assert f"actor_rollout_ref.actor.policy_loss.loss_mode={mode}" in args
    assert "algorithm.rollout_correction.bypass_mode=True" in args
    assert "actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True" in args
    assert f"actor_rollout_ref.actor.clip_ratio_low={threshold}" in args
    assert f"actor_rollout_ref.actor.clip_ratio_high={threshold}" in args
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in args
    assert "algorithm.adv_estimator=grpo" in args
