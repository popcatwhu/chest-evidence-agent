# Chest Evidence Agent

**本地医学多模态 Agent：从病史与胸片，到可追溯的候选诊断报告。**

基于 FastAPI、LangGraph 和本地视觉语言模型，提供中文工作台、图像工具、医学检索、证据检查、病例更新与可复查评测。项目面向 Agent 应用开发研究与工程演示，未经临床验证。

![诊断工作台空状态](docs/images/workbench.png)

> 截图展示界面空状态，不包含真实患者资料、胸片或预设诊断。

## 能做什么

- 上传病史与胸片，生成最可能候选、鉴别诊断、支持和反对依据及进一步评估方向。
- 使用本地 Lingshu-32B NF4，支持显存内运行；也可选择7B模型或OpenAI兼容API。
- 调用胸片分类、肺/心脏区域分割及专用胸片观察模型。分类分数不是患病概率，分割不定位病灶。
- 检索带来源的医学资料；将病例原文、模型观察和参考资料分别保存。
- 核对关键病例线索，比较候选病因与证据冲突；检查引用、重复诊断及确认检查建议，最多一次修订。
- 补充资料后重新分析，保存每次输入快照、执行事件和历史报告。
- 提供原图放大、证据跳转、任务刷新恢复、记录下载与移动端布局。

## 流程与工程结构

```mermaid
flowchart LR
  A[病史与原始胸片] --> B[事实抽取与原文核对]
  B --> C[影像观察与工具计划]
  C --> D[分类 / 分割 / 医学检索]
  D --> E[临床证据对照]
  E --> F[候选诊断报告]
  F --> G[引用与语义复核]
  G --> H[最多一次修订]
  H --> I[报告、证据与历史记录]
```

| 部分 | 实现 |
| --- | --- |
| API与任务执行 | FastAPI、单工作线程串行GPU任务 |
| Agent状态流转 | LangGraph，有界修订 |
| 模型 | Transformers、bitsandbytes NF4、BF16计算 |
| 结构化输出 | LM Format Enforcer + Pydantic |
| 检索 | 小规模TF-IDF字符检索，保留URL和来源类型 |
| 数据持久化 | SQLite、输入快照、事件及原始模型输出 |
| 前端 | 原生HTML/CSS/JavaScript，同源API |

代码位于 `chest_agent/`，前端位于 `static/`，准备与评测工具位于 `scripts/`，测试位于 `tests/`。

## 真实评测结果

所有模型对照均使用真实推理输出。参考答案、最终诊断和图注只交给评分端；失败任务保留在分母。以下各批患者互不重叠，任务口径不同，不合并准确率。

**50位公开患者：简洁选择题接口，原题 + 胸片。**

| 模型 | 直接答题 | 原始工具辅助 | 核验工具后 |
| --- | --- | --- | --- |
| Qwen3-VL-32B NF4 | 24/50，48% | 23/50，46% | 23/50，46% |
| Lingshu-7B BF16 | 28/50，56% | 23/50，46% | 27/50，54% |
| Lingshu-32B NF4 | 32/50，64% | 33/50，66% | 33/50，66% |

**另20位新患者：观察工具对照，仍使用简洁答题接口。**

| 策略 | 正确选项 | 中位耗时 |
| --- | --- | --- |
| Lingshu32直接答题 | 12/20，60% | 6.75秒 |
| 旧观察工具辅助 | 13/20，65% | 7.08秒 |
| NV-Reason工具辅助 | 12/20，60% | 15.71秒 |

**再20位新患者：相同原始病史、胸片和原题，生成完整报告。**

| 流程 | 正确选项 | 完成报告 | 中位耗时 |
| --- | --- | --- | --- |
| 直接推理 | 13/20，65% | 17/20 | 9.83秒 |
| 旧Agent | 15/20，75% | 20/20 | 41.12秒 |
| 加临床证据对照的Agent | 15/20，75% | 20/20 | 58.25秒 |

新旧Agent同对15位、同错5位，新增步骤未提高这批患者的得分，耗时及待复核标记增加。已知6位开发患者的具体诊断名称匹配从3/6变为4/6，但不能作为独立准确率。结核仍误判，张力性气胸仍只识别到大类。

本机69项工程测试通过；RTX5090 32GB运行主模型的完整报告测试中，PyTorch峰值保留显存约21.6–21.8GiB。公开样本存在未知预训练污染，未进行临床专家评分。75%衡量的是完整报告里的选择题结论，不是开放诊断或临床准确率。

详细方法及迭代历史见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。实际汇总包括 [模型对照](medical_model_validation.json)、[工具对照](diagnostic_quality_validation.json) 和 [完整报告对照](full_report_quality_validation.json)。这些是测量记录，首次启动不会自动产生相同成绩。

## 安装

建议Python3.11。先安装适合自己硬件的PyTorch/torchvision，再安装项目；无需依赖开发者的其他Python环境。

```bash
git clone https://github.com/popcatwhu/chest-evidence-agent.git
cd chest-evidence-agent
python3.11 -m venv .venv
source .venv/bin/activate
# 先按照PyTorch官方安装说明安装匹配GPU的版本
python -m pip install -e '.[dev,quantized]'
python -m pytest -q
```

本机测试环境版本记录在 [environment.json](environment.json)：PyTorch2.11、CUDA13运行时、Transformers4.57.6。RTX5090需要支持其架构的PyTorch构建。

### 本地模型

先用7B验证流程，或者准备32B的NF4版本。下载脚本固定官方版本，分块下载脚本对权重核对官方LFS SHA256，不执行远端模型代码。

```bash
# 7B，BF16
python scripts/download_model_ranges.py 7b
# 辅助观察模型
hf download IAMJB/chexpert-mimic-cxr-findings-baseline --local-dir ../models/CXR-Findings
# 可选：加载通用公开资料，参考图与元数据需要MedRAX仓库
git clone https://github.com/bowang-lab/MedRAX.git ../MedRAX
python scripts/prepare_data.py
python scripts/expand_knowledge.py
CHEST_MODEL_PATH="$PWD/../models/Lingshu-7B" bash scripts/start.sh
```

`prepare_data.py`中的教学演示由人工病史和独立样例胸片组成，仅验证流程，不用于准确率评估。扩展资料保留来源及摘要类型，不伪装成完整临床指南库。图像分类/分割检查点由TorchXRayVision按需下载。

32B转换需单独占用GPU，先停止推理服务：

```bash
python scripts/download_model_ranges.py 32b
python scripts/quantize_local_model.py \
  --source ../models/Lingshu-32B --output ../models/Lingshu-32B-NF4
bash scripts/start.sh
```

默认模型目录是 `../models/Lingshu-32B-NF4`。转换保留视觉模块与lm_head原精度，保存分词器与来源版本；模型权重不随仓库分发。大模型初次从机械盘加载可能耗时数分钟，暖运行与冷启动不能混用。

### API后端

也可以配置OpenAI兼容服务。真实Key只放本地环境变量，示例见 [.env.example](.env.example)。

```bash
cp .env.example .env
# 编辑.env，填写自己的提供商地址、模型名和Key
set -a
source .env
set +a
bash scripts/start.sh
```

API模式设置 `CHEST_BACKEND=api`、`CHEST_API_MODEL`、`OPENAI_BASE_URL` 和 `OPENAI_API_KEY`。本地没有专用观察权重时可设置 `CHEST_CXR_EXPERT=0`；工具失败不会被当成正常影像。

## 访问与开关

默认地址是 `http://127.0.0.1:7860`。服务运行在远端时，可用VSCode「端口」面板转发7860，使用面板给出的本地地址。局域网监听可配置：

```bash
CHEST_HOST=0.0.0.0 CHEST_PORT=7860 bash scripts/start.sh
```

页面与API使用相同地址，无需修改前端后端地址。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `CHEST_MODEL_PATH` | `../models/Lingshu-32B-NF4` | 本地模型路径 |
| `CHEST_HOST` / `CHEST_PORT` | `127.0.0.1` / `7860` | 监听地址和端口 |
| `CHEST_BACKEND` | `local` | local或api |
| `CHEST_CXR_EXPERT` | `1` | 启用专用观察工具 |
| `CHEST_CXR_READER` | `legacy` | 默认旧工具；nvreason仅作实验 |
| `CHEST_CLINICAL_CONTRAST` | `1` | 可用0关闭新增临床对照 |
| `CHEST_RECHECK_IMAGE` | `1` | 诊断、复核、修订接收原图 |
| `CHEST_CONSTRAINED_JSON` | `1` | 本地JSON Schema解码约束 |
| `CHEST_MAX_INPUT_TOKENS` | `8192` | 当前输入上限 |

## 评测与复现

需要另行获取上游数据及模型，仓库只提供代码与实际汇总，不包含原始胸片、患者数据库或模型权重。准备脚本保留输入指纹和病例划分，参考字段不会送入推理请求。

```bash
# 准备公开基准候选池和小型开发数据
python scripts/prepare_benchmark.py --limit 5
# 需先准备MedRAX元数据，再冻结按患者隔离的样本
python scripts/prepare_independent_benchmark.py
# 对常驻服务评测完整报告，默认比较direct和verified
python scripts/evaluate_http.py --output data/independent/results.json
# 简洁答题实验单独执行，避免与完整报告混用
python scripts/benchmark_mcq.py --output data/minimal_mcq.json
```

新的实验应另存结果，不覆盖冻结历史。准备新的测试样本时使用 `--exclude-manifest`、`--seed`、`--output-dir`；在查看测试成绩前锁定候选策略。

主要接口：`/api/health`、`/api/cases`、`/api/cases/{id}/runs`、`/api/runs/{id}`，以及图像、补充资料和完整记录下载接口。

## 边界与来源

这是受MedRAX思路启发的独立应用扩展，不是MedAgent-Pro或MedRAX论文指标复现。引用匹配和结构有效不能证明医学正确；缺失确认检查、模型观察错误和病因误判仍可能发生。

代码采用MIT许可。依赖、数据和模型遵循上游条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
