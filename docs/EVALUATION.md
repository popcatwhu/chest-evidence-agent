# Agent 评测

诊断流程固定包含临床证据对照：整理病例、观察胸片、规划工具、检索医学资料、比较候选病因、生成报告、复核及必要时修订。临床证据对照没有关闭开关；步骤失败时记录未完成状态，并将报告标记为待复核。

## 当前记录

20位独立公开测试患者的完整报告均生成成功，报告中的选择题结论答对15例，耗时中位数58.25秒。患者与开发集合及此前使用的患者集合不重叠，测试结果未用于策略选择。详细指标与输入清单指纹见根目录的 `full_report_quality_validation.json`。

这些是公开数据上的选择题结论得分，尚未经临床专家评分，不能表述为临床诊断准确率。6位开发患者的名称匹配仅用于开发检查。保留待复核标记、失败任务与不确定性，不把引用有效等同于医学正确。

## 运行

先准备上游公开数据及模型，再冻结测试集合：

```bash
python scripts/prepare_benchmark.py --limit 5
python scripts/prepare_independent_benchmark.py
python scripts/evaluate_http.py --modes verified --output data/independent/results.json
python scripts/audit_full_report_results.py --manifest data/independent/manifest.json --results data/independent/results.json --output data/independent/audit.json
```

使用独立结果文件，不覆盖已经冻结的记录。审计核对输入指纹、病史、问题、原始图片像素、任务完整性和选项评分；参考答案不进入模型请求。失败任务保留在分母中。

`direct` 和 `tools` 是显式的消融实验模式，分别用于研究模型直接推理和跳过复核的工具推理，工作台只提供 `verified` 完整流程。它们不是可切换的 Agent 版本，其结果不得混入当前完整流程成绩。

发布到评测页面的汇总由 `scripts/export_frontend_evaluation.py` 从根目录的当前评测记录导出。页面不读取本地患者数据库或模型权重。
