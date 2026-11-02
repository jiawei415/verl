# 新 trial 的 W&B 启动流程

此目录的训练脚本在加载模型前，默认先向实际配置的 W&B project 创建一个带 `preflight` 标签的测试 run，写入并读回指标，确认 run 正常结束后再启动训练。预检最多运行 180 秒，失败时不启动训练。每次预检会保留一条测试 run，可在 W&B 中按 `jobType` 或标签过滤。

已验证 W&B `0.23.1`。部分训练节点直连 W&B 返回 403，而当前开发服务器可以直连。直连失败的节点需要单独建立 SSH 转发；仅设置 `no_proxy` 不够。

## 使用修改后的代码

以下命令均以 `verl` 仓库根目录为当前目录。需要使用包含这些脚本的代码版本，重新 checkout 旧 commit 不会包含本流程。可以直接使用共享工作目录：

```bash
cd /mnt/bn/jiawei-llm2/mlx/users/xujiawei.415/playground/VERL/verl
```

不要覆盖正在运行的 trial 的启动脚本或代理配置。新 trial 使用下述流程。

## 1. 在可访问 W&B 的中转服务器建立隧道

此处的中转服务器是开发服务器，不是 Mac，也不是网络受限的训练节点。

为新 trial 填入其 job ID 和 trial ID，例如从 Seed 页面 URL 获取。脚本针对单节点 trial；不自动切换到同一 job 下另一个 trial。

```bash
export JOB_ID='填写新job的ID'
export TRIAL_ID='填写新trial的数字ID'
```

准备一个有权连接该 pod 的 SSH 私钥。如果已有密钥，直接指定它。若选择导出 Merlin 的账号级私钥，以下命令会将它保存在用户目录，供断线重连使用；它权限为 `0600`，不要加入代码仓库，也不要分享：

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
export MERLIN_SSH_KEY="$HOME/.ssh/merlin_wandb_id_rsa"
/home/tiger/.merlin-cli/bin/merlin-cli --control-plane cn-seed \
  job-v2 runs export-ssh-key \
  --json "{\"output\":\"${MERLIN_SSH_KEY}\"}"
```

密钥已存在时不需要重新导出，直接 `export MERLIN_SSH_KEY=实际路径`。

保持下面的管理脚本运行。它会通过 Merlin 查询该 trial 的 SSH 地址、建立只监听训练节点 `127.0.0.1:18881` 的转发，并在 SSH 断开后每 10 秒重试。代理仅允许 W&B 子域和 `storage.googleapis.com` 的 HTTPS CONNECT，不开放公网端口。

```bash
python3 examples/xujiawei/wandb_relay.py \
  --job-run-id "$JOB_ID" \
  --trial-id "$TRIAL_ID" \
  --ssh-key "$MERLIN_SSH_KEY"
```

建议在 tmux 中运行，避免终端断开后退出：

```bash
tmux new-session -d -s "wandb-${TRIAL_ID}" \
  "cd '$PWD' && python3 examples/xujiawei/wandb_relay.py --job-run-id '$JOB_ID' --trial-id '$TRIAL_ID' --ssh-key '$MERLIN_SSH_KEY' >> '/tmp/wandb-${TRIAL_ID}.log' 2>&1"
tail -f "/tmp/wandb-${TRIAL_ID}.log"
```

看到 `W&B tunnel ready` 才表示 SSH 转发已建立；`Connecting trial ...` 仅表示正在尝试连接。相同 trial 重复启动管理脚本会被锁阻止。若远端端口已占用，用 `--remote-port 18882` 等空闲端口，并同步修改下一节的代理地址。

## 2. 在新 trial 内启动训练

首先进入修改后的 verl 根目录。加载隔离 W&B 环境，仅安装 W&B 和 Sentry，不升级基础镜像的 Pydantic、FastAPI、protobuf 或 PyTorch：

```bash
source examples/xujiawei/setup_wandb.sh
export WANDB_ENTITY=jiawei415
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_HTTPS_PROXY=http://127.0.0.1:18881
```

`setup_wandb.sh` 使用 uv 安装到 `/tmp/verl-wandb-runtime-0.23.1` 并设置 `PYTHONPATH`。未安装 uv 时会先在临时目录引导安装 uv。每个新 shell 都需重新 source；不会修改基础 Python 环境。

`common.sh` 尊重外部提供的 `WANDB_API_KEY`；若需要覆盖现有凭据，在关闭 xtrace 的情况下输入，不把 key 放进 Hydra 命令行：

```bash
set +x
read -rsp 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
```

保留原训练参数，示例：

```bash
export TIME_TAG=$(date +%Y%m%d%H%M%S)
export PROJECT_NAME=verl_research
export EXPERIMENT_NAME="${TIME_TAG}_math_grpo_bypass_noclip_8Bbase_rsseq_max_k2_1"
export MODEL_NAME=Qwen3-8B-Base
export CLIP_RATIO_LOW=100.0 CLIP_RATIO_HIGH=100.0 CLIP_RATIO_C=100.0
export MIN_P=0
export ROLLOUT_RS=seq_max_k2 ROLLOUT_RS_THRESHOLD=1.0
export ROLLOUT_CORR_LOSS_TYPE=reinforce
bash examples/xujiawei/run_grpo_math.sh
```

也适用于同目录中通过 `launch.sh` 启动的 math/search 多轮脚本。默认无需手写 Ray 的 W&B 环境覆盖；启动器解析最终 Hydra 配置，将 W&B 代理、NO_PROXY 和 SDK 路径传给 worker。凭据仍由新 trial 的启动环境继承，不写入 Hydra 参数或日志；不要复用带有其他凭据的旧 Ray 集群。

若选择 DPPO，请使用相应 `LOSS_MODE` 和散度阈值，并将 `ROLLOUT_RS`/`ROLLOUT_IS` 设为 `null`；上面的例子是原 reinforce 配方。

## 直连节点

无需启动隧道。先 source SDK 设置，再清除 W&B 专用代理并启动相同训练脚本：

```bash
source examples/xujiawei/setup_wandb.sh
unset WANDB_HTTP_PROXY WANDB_HTTPS_PROXY
export WANDB_ENTITY=jiawei415
bash examples/xujiawei/run_grpo_math.sh
```

`common.sh` 会合并并去重已有 `no_proxy`/`NO_PROXY`，加入 W&B 域名。使用专用代理时，清除会让 W&B 绕过隧道的 `*`、W&B 域名及其父域规则，保留内部网络的其他排除项，并加入回环地址。

## 状态检查和停止

- 新训练日志必须先出现 `W&B write/readback passed`，然后才会加载模型。
- 只有 `console` logger 时跳过 W&B 预检；`--cfg job` 等配置查询也不会写 W&B。
- `WANDB_PREFLIGHT=0` 可显式跳过在线预检，通常仅用于调试；这时不保证后续连接可用。
- 预检证明当时能写入，不能保证长期网络不会中断。训练过程中仍需查看正式 run 指标是否更新。
- 中转服务器或 tmux 进程退出后不能自动恢复；重启服务器后重新运行管理命令。管理进程存活时，SSH 断开会自动重连。
- trial 结束后，在中转端执行 `tmux send-keys -t "wandb-${TRIAL_ID}" C-c`，管理脚本会关闭自己的 SSH 子进程和代理。所有使用该私钥的转发结束后，可以删除专门导出的私钥。
- 不要把 `127.0.0.1:18881` 当成公共代理地址；每个新 trial 必须先建立自己的转发。
