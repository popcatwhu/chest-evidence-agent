# Chest Evidence Agent

结合病史和胸片生成候选诊断报告的多模型 Agent 系统。LangGraph 协调独立影像分析、临床诊断、证据审查和报告修订，保留各 Agent 的意见及处理记录。

![工作台](docs/images/workbench.png)

## 功能

- 上传胸片和病史，生成候选诊断、鉴别诊断及检查建议。
- 调用胸片分类、解剖区域分割和专用观察模型，检索医学参考资料。
- 对照候选病因；Qwen 先独立判断，再检查临床报告，最多修订一次。
- 判断不一致时规划鉴别检索，协调模型依据患者证据决定保留、修订或保留分歧。
- 临床线索和审查引用通过原文片段编号选择，保存可追踪的来源位置。
- 补充病例资料后重新分析，查看历史报告和判断变化，下载完整记录。

| Agent | 模型 | 职责 |
| --- | --- | --- |
| 影像 | NV-Reason-CXR-3B，BF16 | 独立观察胸片，不接收临床结论 |
| 临床 | Lingshu-32B，NF4 | 根据原始病史与胸片独立形成初稿，不接收其他模型意见 |
| 审查 | Qwen3-VL-8B-Instruct，NF4 | 先根据病史与原图独立判断，再对照报告及原始证据 |
| 协调 | Lingshu-32B，NF4 | 检索分歧依据、裁决及修订报告，保存未解决问题 |

三个模型在 RTX 5090 32 GB 上分阶段执行。主模型常驻，影像和审查模型共用一个辅助显存槽，切换前释放前一个模型。

## 运行

需要 Python 3.11，先安装与显卡匹配的 PyTorch 和 torchvision。

```bash
git clone https://github.com/popcatwhu/chest-evidence-agent.git
cd chest-evidence-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,quantized]'
```

准备模型和参考资料后启动：

```bash
bash scripts/start.sh
```

浏览器打开 `http://127.0.0.1:7860`。模型下载、32B 量化、API 配置和远程访问见 [安装说明](docs/SETUP.md)。权重和病例数据需要单独准备。

## 实现

```text
病史 / 胸片 → 独立影像观察、临床初稿、工具与检索
          → Qwen 独立判断 → 分歧检索与裁决（按需）→ Qwen 对照审查 → 协调修订
```

LangGraph 管理流程，Pydantic 校验结构化输出。推理任务串行执行，SQLite 保存输入快照、报告和事件。前端使用 HTML、CSS 和 JavaScript；检索采用字符 TF-IDF。

独立判断不接收临床报告、计划、检索结果或其他模型的观察。意见不同会记录分歧；只有能定位原始证据的质疑才触发修订。

临床初稿同样只接收原始输入。协调模型改诊断必须引用患者证据和可定位的专业医学判据；仅修改引用或检查建议时，不允许顺带改变结论。

```text
chest_agent/   后端、模型调用和诊断流程
static/        工作台
scripts/       数据准备、模型下载和评测
tests/         单元测试
evals/         评测汇总和运行环境
docs/          安装、设计和评测说明
```

[设计说明](docs/DESIGN.md)介绍工具证据、检索和复核的处理方式。

## 评测

已完成60位公开患者的独立测试，排除此前使用过的128位患者。每位患者使用原始病史、一张匹配胸片和一道诊断选择题；失败任务计入分母。

| 指标 | 结果 |
| --- | --- |
| 报告中的选择题结论 | 44/60，73.33% |
| 完整报告生成 | 59/60 |
| 全部专业 Agent 完整参与 | 58/60 |
| 耗时中位数 | 71.74 秒 |
| GPU 峰值保留显存 | 28.92 GiB |
| 仍需复核的报告 | 51/60 |

95% Wilson 区间为60.99%–82.87%。这衡量公开题目在完整报告中的选项结论，尚未经临床专家评分；预训练污染情况未知。待复核标记包含引用问题、临床对照不足或辅助步骤未完成，不直接等于诊断错误数。

成绩对应 [冻结评测代码](https://github.com/popcatwhu/chest-evidence-agent/tree/81e67666ffd0da6ce11bed9cdc9d756dd4cb072a)，首批20例记录单独保留。项目用于研究和演示，未经临床验证。

[评测方法](docs/EVALUATION.md) · [60例结果](evals/summary.json) · [抽样记录](evals/sample.json) · [首批20例](evals/batches/test20-summary.json)

另用12道MedXpertQA胸片相关临床题做开发评测。固定版本的初版分歧流程未带来收益：临床初稿答对4/12，最终答对3/12，纠错0/8、误改1/4；同题单模型直接报告答对5/12。记录见[完整流程](evals/correction-development.json)和[单模型对照](evals/direct-development.json)。这些是另一来源的开发题，不与60例成绩合并。

据此改为原始输入独立起草，并收紧改诊断的判据要求。两例已知错误的[针对性回归](evals/correction-safeguard-regression.json)通过；修正后的总体准确率尚未测量。[24道保留测试题](evals/medxpert-splits.json)仍未运行。

## 开发

当前113项测试通过，覆盖结构化输出、证据引用、生成循环及协作流程。测试环境见 [运行环境](evals/environment.json)。

```bash
pytest -q
ruff format --check chest_agent scripts tests
```

本地审查模型的[合成回归测试](evals/review-regression.json)发现3/3条注入的事实错误，3条正确对照未误报。复现命令为 `python -m scripts.review_regression`，应在没有其他 GPU 推理任务时执行。这是工程检查，不计入诊断准确率。

两例已知病例的[多模型流程回归](evals/independent-consultation-regression.json)均完成独立判断与对照审查，峰值保留显存28.68 GiB。两例中未验证审查模型纠正诊断错误的能力；这些结果不替换冻结评测。

## 来源与许可

项目参考了 [MedRAX](https://github.com/bowang-lab/MedRAX) 的工具协作思路，使用 Lingshu 和 TorchXRayVision 等模型与工具。相关论文见 [参考资料](docs/REFERENCES.md)。

代码使用 [MIT](LICENSE) 许可；模型、数据和参考资料遵循各自条款，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
