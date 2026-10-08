# 安装与配置

## 环境

使用 Python 3.11。先安装支持自己显卡的 PyTorch 和 torchvision，再运行 README 中的安装命令。RTX 5090 需要支持其架构的 PyTorch 构建；本机使用的版本记录在 [evals/environment.json](../evals/environment.json)。

## 本地模型

可以先用 Lingshu-7B 验证流程：

```bash
python scripts/download_model_ranges.py 7b
python scripts/download_model_ranges.py cxr3b
python scripts/download_model_ranges.py review8
git clone https://github.com/bowang-lab/MedRAX.git ../MedRAX
python scripts/prepare_data.py
python scripts/expand_knowledge.py
CHEST_MODEL_PATH="$PWD/../models/Lingshu-7B" bash scripts/start.sh
```

多模型流程还需 NV-Reason-CXR-3B 与 Qwen3-VL-8B-Instruct；7B 命令仅将临床主模型替换成较小版本。下载脚本固定模型版本，并核对权重的 LFS SHA256。TorchXRayVision 的分类和分割权重按需下载。

`prepare_data.py` 使用人工病史和独立样例胸片构建演示病例，只用于检查流程。MedRAX 元数据及原始图片需单独下载，见 [上游仓库](https://github.com/bowang-lab/MedRAX)。医学资料保留来源 URL，当前检索库规模较小。

默认模型是 `../models/Lingshu-32B-NF4`。32B 量化需单独使用 GPU，先停止推理服务：

```bash
python scripts/download_model_ranges.py 32b
python scripts/quantize_local_model.py \
  --source ../models/Lingshu-32B --output ../models/Lingshu-32B-NF4
bash scripts/start.sh
```

Qwen 审查模型在加载时量化为 NF4，影像模型保持 BF16。辅助模型串行使用同一个显存槽，避免三份权重同时常驻。

转换使用 NF4 双重量化和 BF16 计算，视觉模块与 lm_head 保持原精度。首次从机械盘加载可能耗时数分钟。

## API 模型

```bash
cp .env.example .env
# 编辑 .env，设置 CHEST_BACKEND=api，并填写模型、地址和 Key
set -a
source .env
set +a
bash scripts/start.sh
```

API 模式需要 `CHEST_API_MODEL`、`OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。API 模式仅替换临床主模型；影像和审查 Agent 仍使用本地权重与 GPU。真实凭据只保存在本地，`.env` 已加入 `.gitignore`。

## 远程访问

默认监听 `127.0.0.1:7860`。通过 VSCode「端口」面板转发 7860，或在局域网监听：

```bash
CHEST_HOST=0.0.0.0 CHEST_PORT=7860 bash scripts/start.sh
```

页面和 API 使用同一地址。配置示例见 [.env.example](../.env.example)。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `CHEST_MODEL_PATH` | `../models/Lingshu-32B-NF4` | 模型目录 |
| `CHEST_HOST` / `CHEST_PORT` | `127.0.0.1` / `7860` | 监听地址和端口 |
| `CHEST_BACKEND` | `local` | `local` 或 `api` |
| `CHEST_RADIOLOGY_MODEL_PATH` | `../models/NV-Reason-CXR-3B` | 影像 Agent 模型 |
| `CHEST_REVIEW_MODEL_PATH` | `../models/Qwen3-VL-8B-Instruct` | 审查 Agent 模型 |
| `CHEST_RECHECK_IMAGE` | `1` | 诊断及复核阶段接收原图 |
| `CHEST_CONSTRAINED_JSON` | `1` | 本地 JSON Schema 解码约束 |
| `CHEST_MAX_INPUT_TOKENS` | `8192` | 输入 token 上限 |
