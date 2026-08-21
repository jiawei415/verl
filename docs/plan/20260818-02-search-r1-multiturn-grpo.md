# Search-R1 Multiturn GRPO 训练计划

- 日期: 2026-08-18
- 作者: xujiawei.415
- 目标: 基于 Search-R1 数据 + 本地 dense retriever, 在 `verl/examples/xujiawei/` 建立多轮 GRPO 训练管线, 让 policy 学会 `<tool_call> query </tool_call>` 触发搜索并给出 `<answer>...</answer>`。
- Reference: https://github.com/PeterGriffinJin/Search-R1 (官方)

## 0. 已经决策 (grilling 输出)

| # | 决策 | 选择 |
|---|---|---|
| Q1 | Retriever 后端 | 官方 e5 + faiss-gpu |
| Q2 | 打包格式 | 不压缩, 直接 rsync |
| Q3 | 下载源 | HF `PeterJinGo/*` |
| Q4 | Index 变体 | `wiki-18-e5-index-HNSW64` (ANN, ~30-50GB) |
| Q5 | 运行时包内容 | index + corpus (`wiki-18.jsonl`) |
| Q6 | 模型 | env 入参 `MODEL_NAME`, 无默认 |
| Q7 | Val 子集 | 每 sub-dataset 前 500 行 (共 ~3500) |
| Q8 | max_turns | 4 |
| Q9 | Tool 协议 | 自定义 `search_r1` tool_parser (裸文本 query) |
| Q10 | 部署 | 两脚本分离 (`start_retriever.sh` + `run_grpo_search_multiturn.sh`) |
| Q11 | Reward | 官方 EM (`search_r1_like_qa_em`, 已在 verl) |
| Q12 | Retriever 起动 | 智能模式 (`/tmp/.ready` flag) |
| Q13 | 文件布局 | `verl/examples/xujiawei/` 私有目录 |
| Q14 | 下载策略 | 本地 `/tmp` 缓存 + rsync → HDFS |
| Q15 | GPU 分配 | env 入参 (例 `0,1,2,3`), retriever 与训练卡 ID 拆分 |

## 1. 数据集

| 用途 | HF repo | 内容 | 行数 |
|---|---|---|---|
| Train | `PeterJinGo/nq_hotpotqa_train` (`train.parquet`) | NQ + HotpotQA 混合 | ~170k |
| Val (7 domains) | `PeterJinGo/nq_hotpotqa_train` (`test.parquet`) 采样 | popqa/2wikimultihopqa/triviaqa/hotpotqa/nq/musique/bamboogle | 每 domain 500 行, 共 ~3500 |

字段 (verl 格式, `preprocess_search_r1_dataset.py` 已实现基础转换):
- `data_source` = `searchR1_<subset>` (nq/popqa/…) → 触发 `search_r1_like_qa_em` reward
- `prompt` = [{system}, {user 含 tool_call 教学 + question}]
- `reward_model.ground_truth.target` = list of golden answers
- `extra_info.tools_kwargs.search.create_kwargs` = {ground_truth, question, data_source}

## 2. Retrieval 环境

- **Encoder**: `intfloat/e5-base-v2` (存 `$MODEL_ROOT/e5-base-v2`, ~440MB)
- **Index**: `PeterJinGo/wiki-18-e5-index-HNSW64` (~30-50GB, ANN)
- **Corpus**: `PeterJinGo/wiki-18-corpus` → `wiki-18.jsonl` (~76GB 解压)
- **Server**: FastAPI + uvicorn + faiss-gpu, HTTP `POST /retrieve` (输入 queries + topk, 输出 [{document,score}])
- **GPU**: 单卡 5-7GB, `CUDA_VISIBLE_DEVICES` 由 env 指定

## 3. Tool 协议

Search-R1 官方 (非 hermes JSON):
```
<think> reasoning </think>
<tool_call> query </tool_call>
<tool_response> docs (来自 retriever) </tool_response>
<answer> final </answer>
```

需在 `verl/experimental/agent_loop/tool_parser.py` 追加 `@register("search_r1")` parser:
- 抓 `<tool_call>` 到 `</tool_call>` 裸文本作为 query
- 构造 `FunctionCall(name="search", arguments=json.dumps({"query": query}))`

## 4. HDFS 布局

```
/mnt/hdfs/byte_data_seed/hdd_hldy/user/xujiawei.415/
├── hf_models/
│   └── e5-base-v2/                       # 新: encoder
├── hf_datasets/
│   └── search_multiturn/                 # 新: preprocess 输出
│       ├── train.parquet
│       └── test.parquet                  # ~3500 行 (7 domains × 500)
└── searchR1/                             # 新
    ├── wiki-18.jsonl                     # 76GB, 解压后
    ├── wiki-18.jsonl.gz                  # 备份原始
    └── wiki-18-e5-index-HNSW64/          # ~30-50GB
```

## 5. 本地 `/tmp` 布局 (pod 运行时)

```
/tmp/searchR1/
├── wiki-18.jsonl
├── wiki-18-e5-index-HNSW64/
└── .ready                                # rsync 完成标记
```

## 6. 新增文件

### 6.1 `verl/examples/data_preprocess/xujiawei_search_r1.py`
- 从 HF 下载 `PeterJinGo/nq_hotpotqa_train` 的 `train.parquet` + `test.parquet` 到本地缓存
- Train: 全量, 加 `searchR1_` 前缀
- Test: 按 `data_source` group, 每组前 500 行
- 生成 verl 格式 (prompt/reward_model/extra_info/tools_kwargs)
- 输出 `/mnt/hdfs/.../hf_datasets/search_multiturn/{train,test}.parquet`
- 参考现有 `preprocess_search_r1_dataset.py` (无需下载脚本, 已有 pipeline)

### 6.2 `verl/examples/xujiawei/retriever/download.py`
- 从 HF 拉:
  - `PeterJinGo/wiki-18-e5-index-HNSW64` (index files)
  - `PeterJinGo/wiki-18-corpus` (`wiki-18.jsonl.gz`)
  - `intfloat/e5-base-v2` (encoder)
- 下载到 `/tmp/hf_stage/` (~150GB)
- `gunzip` corpus → `wiki-18.jsonl`
- `rsync` → `/mnt/hdfs/.../searchR1/` 和 `.../hf_models/e5-base-v2/`
- `rm -rf /tmp/hf_stage`

### 6.3 `verl/examples/xujiawei/retriever/retrieval_server.py`
- FastAPI + faiss-gpu, 从 `INDEX_PATH`, `CORPUS_PATH`, `E5_MODEL_PATH` 加载
- `POST /retrieve` (queries, topk=3, return_scores=True) → 每 query top-k docs (从 corpus 拿原文)
- Health `GET /health`
- Multi-worker uvicorn, 端口 8000 (env `RETRIEVAL_PORT` 可覆盖)

### 6.4 `verl/examples/xujiawei/search_tool.py`
- `@function_tool("search")` 装饰的 async 函数
- 接收 `query: str` (或 `query_list: list[str]`)
- HTTP POST 到 `RETRIEVAL_URL` (env, 默认 `http://localhost:8000/retrieve`)
- 组装 top-k docs 为字符串 (每 doc 加 title/content 摘要, 截断至 `MAX_TOOL_RESPONSE_CHARS`)
- 返回该字符串, 由 tool_agent_loop 拼成 `<tool_response>...</tool_response>`

### 6.5 `verl/examples/xujiawei/start_retriever.sh`
入参:
- `RETRIEVER_GPU_IDS` (默认 `"7"`) — 例 `"0,1"` 多卡
- `RETRIEVAL_PORT` (默认 `8000`)
- `HDFS_SEARCHR1_DIR` (默认 `$HDFS_PATH/searchR1`)

行为 (智能模式):
```bash
LOCAL=/tmp/searchR1
if [[ ! -f $LOCAL/.ready ]]; then
    mkdir -p $LOCAL
    rsync -aW $HDFS_SEARCHR1_DIR/wiki-18.jsonl $LOCAL/
    rsync -aW $HDFS_SEARCHR1_DIR/wiki-18-e5-index-HNSW64 $LOCAL/
    touch $LOCAL/.ready
fi
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_IDS nohup python retrieval_server.py \
    --index_path $LOCAL/wiki-18-e5-index-HNSW64 \
    --corpus_path $LOCAL/wiki-18.jsonl \
    --retriever_model $MODEL_ROOT/e5-base-v2 \
    --topk 3 --faiss_gpu --port $RETRIEVAL_PORT \
    > /tmp/retriever.log 2>&1 &
# 健康检查, 循环等 ready
```

### 6.6 `verl/examples/xujiawei/run_grpo_search_multiturn.sh`
入参 (在共享 common.sh 之外):
- `MODEL_NAME` (必需)
- `TRAIN_GPU_IDS` (例 `0,1,2,3,4,5,6`) — 训练卡
- `RETRIEVAL_URL` (默认 `http://localhost:8000/retrieve`)
- `MAX_TURNS` (默认 4)
- `VAL_N` / `ROLLOUT_N` 等 (继承 common.sh)

内容:
```bash
source common.sh
IFS=, read -ra _gpu_arr <<< "$TRAIN_GPU_IDS"
NGPUS_PER_NODE=${#_gpu_arr[@]}
export CUDA_VISIBLE_DEVICES=$TRAIN_GPU_IDS
: "${MAX_RESPONSE_LENGTH:=8192}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${TRAIN_FILES:=search_multiturn/train.parquet}"
: "${VAL_FILES:=search_multiturn/test.parquet}"

# multi-turn knobs
DATA=("${COMMON_DATA[@]}" algorithm.adv_estimator=grpo data.return_raw_chat=True)
ACTOR=(... offload=True)
ROLLOUT=(
    "${COMMON_ROLLOUT[@]}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
    actor_rollout_ref.rollout.multi_turn.function_tool_path=$SCRIPT_DIR/search_tool.py
    actor_rollout_ref.rollout.multi_turn.format=search_r1        # ← 新 parser
)
REF=("${COMMON_REF[@]}")
TRAINER=("${TRAINER_BASE[@]}" trainer.val_before_train=False trainer.log_val_generations=50)
source $SCRIPT_DIR/launch.sh "$@"
```

### 6.7 `verl/experimental/agent_loop/tool_parser.py` (追加)
```python
@register_tool_parser("search_r1")
class SearchR1ToolParser(ToolParser):
    @property
    def stop_token_ids(self):
        # optionally add stop on `</tool_call>` token IDs
        return []

    async def extract_tool_calls(self, response_ids, tools):
        text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        # regex 抓 <tool_call>...</tool_call>
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        calls = [FunctionCall(name="search", arguments=json.dumps({"query_list": [m.strip()]})) for m in matches]
        return text, calls
```

## 7. 训练时 GPU 分配

- 8 卡机, `TRAIN_GPU_IDS=0,1,2,3,4,5,6`, `RETRIEVER_GPU_IDS=7`
- 训练 tp=2 → rollout replicas = 3, DP = ceil(7/2) = 3 或 4
- Retriever HNSW QPS 单卡足以支撑 rollout 并发 (e5 batch encode + faiss HNSW query <10ms/query)

## 8. 里程碑

### M1: 数据 (估 2 小时)
- [ ] Preprocess 脚本 `xujiawei_search_r1.py`
- [ ] 下载 + 处理, 输出 HDFS train/test parquet
- [ ] Sanity: `data_source` 分布正确, prompt 格式对齐 Search-R1 官方

### M2: Retriever 部署 (估 1 天含下载)
- [ ] `download.py` (150GB, 单向 HDFS)
- [ ] `retrieval_server.py` + smoke test (`curl POST /retrieve`)
- [ ] `start_retriever.sh` (rsync + 起 server, `.ready` flag)

### M3: 集成 (估半天)
- [ ] `search_tool.py` (function_tool, HTTP client)
- [ ] `SearchR1ToolParser` 加入 `tool_parser.py`
- [ ] Unit test: agent_loop 模拟一次 `<tool_call>query</tool_call>` → 触发 search_tool → 拿 `<tool_response>` 拼回

### M4: 训练脚本 (估 2 小时)
- [ ] `run_grpo_search_multiturn.sh`
- [ ] 首次跑通 smoke (小 batch, TEST_FREQ=1), 观察 `val-core/searchR1_*/acc/mean@N`

### M5: 全量训练
- [ ] `MODEL_NAME=Qwen3-4B-Instruct-2507 TRAIN_GPU_IDS=0,1,2,3,4,5,6 bash run_grpo_search_multiturn.sh`
- [ ] Wandb 观察 7 个 domain 各自 acc 曲线

## 9. 风险 / 注意

- **Retriever server ↔ training 通信**: 训练 rollout 并发 (~batch × rollout_n × turns) 高, retriever HTTP 需承载. FastAPI + `workers=4` uvicorn 应可 handle. 若成瓶颈, 加 semaphore/rate_limit 或多 worker。
- **/tmp 空间**: 132GB index+corpus 若 /tmp 不够, 用 /home 或其他大盘。df 事先检查。
- **HNSW 召回损失**: 若发现学习曲线明显差于 Flat, 可切 Flat index (再下 132GB)。
- **verl 官方 `SearchTool` 已删除**: 本方案完全私有实现 (`xujiawei/search_tool.py`), 不侵入 upstream。
- **max_turns=4 不够**: musique/bamboogle 多跳可能需要更多轮, 训练中期若观察 `num_turns/mean` 接近 4, 考虑放宽到 6-8。
