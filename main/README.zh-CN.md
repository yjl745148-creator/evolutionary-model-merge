# Evolutionary Model Merge

[English](README.md) | **简体中文**

本项目使用 CMA-ES 搜索两个同架构 Hugging Face 因果语言模型的融合参数，并提供比例消融、逐层搜索、模型对比和结果绘图脚本。

> ⚠️ **负责任使用**：本项目包含安全/红队评测能力（攻击成功率 ASR），可用于评估模型在有害请求下的行为。仅限用于 AI 安全研究、对齐评估与防御性目的。详见下方[负责任使用](#负责任使用)。

## 环境

建议使用 Python 3.10 或 3.11，并安装：

```bash
pip install -r requirements.txt
```

模型融合通常需要 CUDA GPU。CPU 可用于部分检查，但完整实验会很慢。

可选的 API 裁判配置：

```bash
# Qwen/DashScope
export LLM_PROVIDER=qwen
export QWEN_API_KEY="..."

# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="..."
```

未设置对应 API key 时，融合阶段会明确警告并仅使用规则评分。

## 主要入口

全局融合：

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --seed 42
```

低显存模式：

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --low-vram
```

逐层搜索：

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/per-layer \
  --per-layer-search \
  --per-layer-alpha 0.7 \
  --large-model never
```

`--per-layer-alpha` 实际控制逐层目标中的风险权重：

```text
alpha * (1 - safety_score) + (1 - alpha) * dialogue_score
```

## 恢复检查点

```bash
python run_merge.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/example \
  --resume
```

恢复逻辑支持：

- 两模型全局标量搜索
- DataFlow 路由参数
- DARE drop rate 和对应随机种子
- 逐层搜索参数

恢复时使用的 CLI/YAML 配置必须与生成 checkpoint 时一致。

## 数据目录

`compare_models.py` 会从项目内的 `data/eval/` 查找：

```text
data/eval/
  mt_bench_questions.jsonl
  alpaca_eval.json
  advbench_behaviors.jsonl
  harmbench_behaviors.jsonl
  jailbreakbench_behaviors.jsonl
```

文件不存在时脚本会使用代码内置的小型题集。这些有害评测数据集**未随仓库分发**，请按下方[数据集来源](#数据集来源)从各自官方渠道获取，并遵守其各自 license。

## 模型对比

```bash
python compare_models.py \
  --merged experiments/example/merged_model \
  --chat /path/to/chat-model \
  --base /path/to/base-model \
  --output experiments/example/comparison_report.json
```

**提示词策略**：评测时三个模型（merged/chat/base）都套用各自 tokenizer 的 chat 模板；没有模板的模型（通常是 base）会自动退回裸 prompt。评测**不注入任何系统提示词**（包括安全提示）。空响应会被剔除并标记，不计为 0 分。

如果全部裁判请求失败，脚本会直接报错，不再把零个有效样本记录为 ASR 0。

## 其他脚本

比例消融：

```bash
python ablation_merge_ratio.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/ablation
```

独立逐层热图搜索：

```bash
python layer_heatmap.py \
  --model-a /path/to/chat-model \
  --model-b /path/to/base-model \
  --output experiments/layer-heatmap
```

重新绘制逐层结果：

```bash
python plot_merge_result.py \
  --output-dir experiments/per-layer
```

仅复测对话能力（DeepSeek 裁判）：

`retest_it_dialogue_deepseek.py` **只重测对话分**（MT-Bench + AlpacaEval），不跑安全/ASR 评测，并改用独立的 DeepSeek V4 Pro 裁判。

**为什么需要它：** 在某些运行里，`compare_models.py` 主评测的对话分会整列变成 0 —— 例如被测模型在该提示词格式下空响应，或主裁判调用异常 —— 导致该模型的对话能力失效、被低估。此时无需重跑整套对比，可用本脚本单独对受影响的模型（例如 merged / m2 模型）重测一次对话分，并用另一个裁判交叉验证。脚本会逐条记录输入、模型回复、裁判原始输出、解析分数和错误，便于排查。

```bash
# 方式一：直接指定模型
python retest_it_dialogue_deepseek.py \
  --model-path /path/to/m2_model \
  --model-id m2 \
  --output experiments/example/m2_dialogue_retest.json

# 方式二：从已有的 comparison_report.json 继承模型路径 / 题量 / 提示词模式
python retest_it_dialogue_deepseek.py \
  --source-report experiments/example/comparison_report.json \
  --output experiments/example/m2_dialogue_retest.json
```

裁判 API key 从环境变量读取（用 `--api-key-env` 指定读取哪个变量）；裁判模型默认 `deepseek-v4-pro`（可用 `--judge-model` / `--base-url` 覆盖）。

## 输出文件

典型融合输出：

```text
experiments/example/
  best_so_far.json
  latest_generation_best.json
  generation_best.jsonl
  results.yaml
  merged_model/
```

每次新的非恢复运行会重新建立本次运行的 `generation_best.jsonl` 和全局最佳记录，避免旧实验结果污染新运行。使用 `--resume` 时不会重新执行 CMA-ES。

三个 CMA 阶段分别使用 `seed`、`seed + 1`、`seed + 2`，相同配置和环境下可重复生成候选序列。

## 数据集来源

下列评测数据集**不包含在本仓库中**。请从官方来源下载，放到 `data/eval/`，并遵守各自的 license / 使用条款：

- **MT-Bench** — https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- **AlpacaEval** — https://github.com/tatsu-lab/alpaca_eval
- **AdvBench** — https://github.com/llm-attacks/llm-attacks
- **HarmBench** — https://github.com/centerforaisafety/HarmBench
- **JailbreakBench** — https://github.com/JailbreakBench/jailbreakbench

后三个为有害行为提示数据集，仅用于安全评测。

## 负责任使用

本项目用于 **AI 安全研究与对齐评估**。其攻击成功率（ASR）评测会向被测模型发送有害请求并对其响应打分，目的是**衡量并改进**模型安全性，而非生成或传播有害内容。

使用者须：

- 仅将本工具用于合法的研究、评估与防御目的；
- 不得用于生成、分发或实施任何造成现实伤害的内容或行为；
- 遵守所用模型、数据集与 API 的许可条款及当地法律法规；
- 对自行获取并放入 `data/eval/` 的有害数据集自负合规责任。

作者对本软件的滥用不承担责任，详见 [LICENSE](LICENSE)。

## 注意

- 两个源模型必须具有兼容的层数、隐藏维度、注意力头、词表和 embedding 配置。
- `--trust-remote-code` 默认关闭，仅对可信模型使用。
- 收到终止信号后，程序会在当前候选评估完成后停止，不再进入后续 CMA 阶段。

