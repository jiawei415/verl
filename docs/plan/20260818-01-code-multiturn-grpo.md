# Code Multiturn GRPO 训练计划

- 日期: 2026-08-18
- 作者: xujiawei.415
- 目标: 基于 LiveCodeBench + `open-r1/codeforces` (verifiable) 搭建 rule-based verifiable 的多轮 GRPO 训练管线。
- 流程: 下载 → 处理 + 统计 → sandbox 决策 → 写 reward + tool + 训练脚本。

## 1. 数据集选择 (rule-based verifiable)

| 用途 | 数据集 | HF repo | config | 规模 | 验证方式 |
|---|---|---|---|---|---|
| Train | Codeforces | `open-r1/codeforces` | **`verifiable`** | 8338 train / 422 test | `official_tests` + 部分 `generated_checker` (SJ) |
| Test  | LiveCodeBench | `livecodebench/code_generation_lite` | `release_latest` | 1055 (post-2024-08 cutoff = 454) | `public_test_cases` + `private_test_cases` |

- CF `verifiable`: `official_tests_complete=True` 全部, 19.7% 有 SJ, 99.6% stdio, 100% executable。
- LCB: 60% stdio (AtCoder+CF) + 40% functional (LeetCode w/ `starter_code`)。
- 弃 `deepmind/code_contests` (源不纯)。

## 2. 路径布局

```
/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/hf_datasets/
├── codeforces/                    ✅ HF snapshot 已下载
├── livecodebench/                 ✅ HF snapshot 已下载
└── code_multiturn/
    ├── train.parquet              ✅ 5560 rows
    ├── test.parquet               ✅ 454 rows
    └── stats.json                 ✅ 分布报告
```

## 3. 已确定决策

| # | 决策 | 值 |
|---|---|---|
| Q1 | CF 数据 config | `verifiable` (8338 train / 422 test) |
| Q2 | Sandbox executor | ByteIntl seed-sandbox FaaS (URL: `https://seed-sandbox.byteintl.net/faas/sandbox/`), env var `SANDBOX_ENDPOINT` |
| Q3 | 模型输出语言 | Python-only |
| Q4 | LCB 版本 | `release_latest`, `release_date >= 2024-08-01` cutoff |
| Q5 | rating 上限 | ≤ 2500 (训练池 ~6903, 现实际 5560 因还叠加 executable/≥3 tests/stdio) |
| Q5b | Python TL 系数 | **3.0x** (对齐 CF 官方 Python multiplier), env var `CODE_REWARD_TL_MULT` |
| Q7 | Special Judge | 保留, 跑 `generated_checker`, reward 内二段调用 sandbox |
| Q8 | LCB functional | 保留, reward 内 import + call `fn_name`, 对比返回值 |
| Q9 | Reward shape | Binary all-pass (对齐 LCB Pass@1) |
| Q10 | Max turns | 4 (默认, 训练时可调 `MAX_TURNS`) |
| Q11 | 训练超参 | 抄 `run_grpo_math_multiturn.sh` + `common.sh`, 训练前手改 |
| — | Base model | Qwen3-8B-Base (与你 math_multiturn 一致) |

## 4. 数据处理

### 下载脚本 (已建)

- `verl/examples/data_preprocess/download_code_datasets.py` — snapshot → 本地 `/tmp/hf_cache` → rsync → `/mnt/hdfs/.../hf_datasets/`
- `verl/examples/data_preprocess/download_code_datasets.sh` — shell wrapper, 设置 proxy + env

### 预处理脚本 (已建)

- `verl/examples/data_preprocess/code_multiturn.py`
- 每行 verl 多轮 schema:
  ```python
  {
    "data_source": "codeforces" | "livecodebench",
    "prompt": [{"role": "user", "content": problem_prompt}],
    "ability": "code",
    "reward_model": {"style": "rule", "ground_truth": {
        "official_tests": [{"input": ..., "output": ..., "testtype"?: ...}],
        "input_mode": "stdio" | "file",
        "generated_checker": None | str,
        "time_limit_s": float, "memory_limit_mb": float,
        "fn_name": None | str, "language": "python",
    }},
    "extra_info": {"split", "index", "source_id", "rating", "tags", ...},
  }
  ```
- CF 过滤: rating 800-2500, executable, ≥3 tests, stdio (drop 37 file-mode 题)。5560 kept。
- LCB 过滤: `release_date >= 2024-08-01`。454 kept。

## 5. Reward function (已建)

`verl/examples/xujiawei/code_multiturn_reward.py` — 三分支 rule-based, sandbox 后端同 tool:

| Mode | 判定 | 触发条件 |
|---|---|---|
| **stdio_diff** | 跑 user code w/ stdin, 比对 stdout (normalized) | `fn_name is None and generated_checker is None` |
| **special_judge** | 跑 user code → 再跑 `generated_checker` w/ (input, expected, user_out), 解析末 token `1` | `generated_checker` 非空 |
| **functional** | `exec` user code, 找 `fn_name` (支持 `Solution.fn_name`), 调用 → 对比返回值 (JSON parse) | `fn_name` 非空 |

- 每题最多 `CODE_REWARD_MAX_TESTS=40` 个 test, per-problem 内 `PER_PROBLEM_CONCURRENCY=8` 并发跑 sandbox。
- `time_limit_s * TL_MULT` (3.0) 传给 sandbox `run_timeout`。
- Score = **binary all-pass** (Q9 决定); LCB val 直接得 Pass@1 数字。
- 返回 dict `{score, acc, pred, n_passed, n_total, mode}`, 供 verl reward manager 用。

**Smoke test 通过**: stdio branch 打分正确 (n=3 → 6 双测试 pass, mode=stdio_diff)。

## 6. Tool (已建)

`verl/examples/xujiawei/code_multiturn_tool.py`:

- `@function_tool("execute_python")` — 模型多轮 rollout 中调用。
- 参数: `code: str, stdin: str = ""`。
- 返回: 单字符串 `[status ...] [stdout ...] [stderr ...]`, 截断到 `SANDBOX_MAX_OUTPUT=1024` 字符。
- 后端同 reward: seed-sandbox FaaS (env `SANDBOX_ENDPOINT`)。

## 7. 训练脚本 (已建)

`verl/examples/xujiawei/run_grpo_code_multiturn.sh`:

- Source `common.sh`, 覆盖数据/prompt/response length。
- `algorithm.adv_estimator=grpo`, `kl_coef=0`, `kl_loss_coef=0` (纯 GRPO, 无 KL 惩罚)。
- `actor_rollout_ref.rollout.multi_turn.enable=True`, `max_user/assistant_turns=4`。
- `function_tool_path=$SCRIPT_DIR/code_multiturn_tool.py`。
- `custom_reward_function.path=$SCRIPT_DIR/code_multiturn_reward.py`, `.name=compute_score`。
- 其余超参 (batch, lr, N rollouts, tp size, memory util) 全走 `common.sh` 默认, 训前 shell env 变量覆盖。

## 8. 目录总览

```
verl/
├── examples/
│   ├── data_preprocess/
│   │   ├── code_multiturn.py                       ✅
│   │   ├── download_code_datasets.py               ✅
│   │   └── download_code_datasets.sh               ✅
│   └── xujiawei/
│       ├── code_multiturn_tool.py                  ✅ (execute_python + stdin)
│       ├── code_multiturn_reward.py                ✅ (3 branch rule-based)
│       └── run_grpo_code_multiturn.sh              ✅
└── docs/plan/20260818-01-code-multiturn-grpo.md    ✅ (本文)
```

## 9. 遗留 + 首跑清单

- [ ] 集群调度 (mlx worker) 起单机 8 GPU dry-run 一步, 验证:
  - vLLM async rollout + function_tool 加载正确
  - `execute_python` 沙盒调用成功
  - `code_multiturn_reward.compute_score` 被 verl reward manager 拾取, 三分支跑通
- [ ] LCB val 首次跑 (`trainer.val_before_train=True`), 记 baseline Pass@1
- [ ] 全量训练, 定期 val, 收敛后归档
- [ ] 若 SJ 分支 fail rate 异常高, 检查 `generated_checker` 输出格式假设
- [ ] 若 functional 分支 fail rate 高, LCB LeetCode 输入格式可能非 JSON, 加 fallback 解析
