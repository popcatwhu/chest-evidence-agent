# 多模型评测

## 协作流程

NV-Reason 影像 Agent 独立描述原图；Lingshu 临床 Agent 结合病史、工具和医学资料形成报告；Qwen 独立审查报告与原始证据。有效质疑触发一次 Lingshu 协调修订，然后再次交给 Qwen。辅助 Agent 未完成或质疑无法匹配原文时，保留备注并标记待复核。

## 测试集合

本轮使用60位尚未使用过的公开患者，每位使用一张胸片和一道诊断选择题。排除128位此前使用过的患者，模型和流程在读取测试成绩前固定。抽样种子、病例编号和元数据指纹见 [evals/sample.json](../evals/sample.json)。

模型输入只包含病史、匹配的胸片和公开问题。参考答案、图注、最终诊断和解释仅保存在评测器侧。失败任务仍计入分母，不重新提交已经失败的病例来替换结果。

这是小样本公开题目测试，预训练污染情况未知，尚未经临床专家评分。选择题结论得分、报告完成率和全部 Agent 完整参与率分别统计；引用检查通过不表示医学判断正确。

## 复现

先按照安装说明获取模型、MedRAX 元数据和 ChestAgentBench 数据：

```bash
python scripts/prepare_benchmark.py --limit 5
python scripts/prepare_independent_benchmark.py --limit 60 --seed 20261009 --exclude-manifest evals/sample.json --output-dir data/team60/test60
python scripts/run_expanded_evaluation.py --manifest data/team60/test60/manifest.json --results data/team60/test60_results.json
```

汇总脚本检查任务完整性、输入指纹、病史、问题、原图像素、选项评分及实际模型身份。报告中的 `collaboration` 保存意见、质疑和协调记录。汇总输出位于 [evals/summary.json](../evals/summary.json)。

## 延迟

主模型常驻 GPU，影像模型与审查模型分阶段加载。完整任务耗时包含辅助模型切换、工具、检索、复核和修订，不能与单次模型生成速度混用。GPU 峰值按任务记录。

`direct` 和 `tools` 仅供消融实验，工作台使用 `verified` 完整流程。不同任务、样本和输入协议的分数不合并。

## 首批20例记录

20例报告均完成，选择题结论答对15例。所有专业 Agent 完整参与的为19例；1例影像 Agent 的结构化输出未完成，保留为失败观察，没有重新提交病例替换结果。报告耗时中位数63.94秒，GPU峰值保留显存28.14 GiB。

16份报告仍需复核；这些标记包括临床对照不足、无法锚定的模型质疑和辅助 Agent 未完成等情况。它们不等同于16例临床诊断错误。具体医疗解释与检查建议尚未进行专家评分。

扩展评测支持断点续跑，同一结果目录只能运行一个进程。完成后自动审计并更新网页；中断时保留任务和进度。首批20例单独保留于 `evals/batches/`，两批病例没有重叠。
