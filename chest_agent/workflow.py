import json
import time
from threading import Lock
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from . import store, config
from .llm import backend
from .schemas import (
    Evidence,
    Plan,
    Report,
    Review,
    Change,
    CaseProfile,
    Observation,
    validate_evidence,
    report_schema,
)
from .quality import ground_facts, anchor_review, known_test_conflicts
from .recommendations import check_recommendations
from .clinical_consistency import clinical_inconsistencies
from .diagnosis_checks import diagnosis_issues
from .clinical_contrast import ClinicalContrast, ground_contrast, contrast_prompt
from .tools import image_tools, retrieve

RUN_LOCK = Lock()


class State(TypedDict, total=False):
    case: dict
    run_id: str
    question: str
    mode: str
    evidence: list[Evidence]
    plan: Plan
    report: Report
    issues: list[str]
    repairs: int
    previous: dict | None
    previous_context: str | None
    profile: CaseProfile
    review_notes: list[str]
    review_details: list[dict]
    clinical_contrast: ClinicalContrast | None
    contrast_notes: list[str]


def log(state, stage, message):
    store.event(state["run_id"], stage, message)


def prepare(state):
    log(state, "prepare", "整理病例，保存本次输入快照")
    evidence = [
        Evidence(id=f"C-{i + 1}", kind="case", title=f"病例资料 {i + 1}", content=line)
        for i, line in enumerate(state["case"]["context"].splitlines())
        if line.strip()
    ]
    if state["case"].get("image_path"):
        evidence.append(
            Evidence(
                id="I-original",
                kind="image",
                title="原始胸片",
                content="本次输入的原始胸片；引用此项时必须明确是视觉观察，不是工具测量。",
            )
        )
    return {
        "evidence": evidence,
        "repairs": 0,
        "previous": store.previous_report(state["case"]["id"], state["run_id"]),
        "previous_context": store.previous_context(
            state["case"]["id"], state["run_id"]
        ),
    }


def plan(state):
    if state["mode"] == "direct":
        log(state, "plan", "基线模式：直接多模态推理")
        return {"plan": Plan(tools=[], query=state["question"], reason="直接推理基线")}
    prompt = (
        "为胸部病例制定简短分析计划。工具 classify 可给出胸片病理模型分数；"
        "segment 可展示肺和心脏解剖区域，不定位病灶。只选择有必要的工具。"
        "有胸片时通常选择 classify；无胸片则 tools 必须为空。"
        "query 用英文医学关键词检索资料；reason 用简短中文解释。"
        "hypotheses列出至多四个值得鉴别的具体疾病；每项只写名称，不写理由，不局限于影像分类标签，尚未确诊。"
        "query必须包含关键风险因素、已知检查及这些疾病的英文名称。"
        f"\n有胸片：{bool(state['case'].get('image_path'))}\n病例：{state['case']['context']}\n问题：{state['question']}"
        f"\n核对过的事实：{state.get('profile', CaseProfile(facts=[])).model_dump_json()}"
        f"\n图像观察：{json.dumps([e.content if e.id == 'I-observe' else json.dumps({'visual_findings': json.loads(e.content)['visual_findings']}) for e in state['evidence'] if e.id in {'I-observe', 'I-radiology'} and e.status == 'completed'], ensure_ascii=False)}"
    )
    result = backend.json(prompt, Plan, max_tokens=350)
    if not state["case"].get("image_path"):
        result.tools = []
    result.tools = list(dict.fromkeys(result.tools))
    log(
        state,
        "plan",
        result.reason + "；工具：" + (", ".join(result.tools) or "无需图像工具"),
    )
    return {"plan": result}


def profile(state):
    if state["mode"] == "direct":
        return {"profile": CaseProfile(facts=[])}
    log(state, "facts", "提取症状、风险因素及已提供检查，并逐项核对原文")
    sources = [e.model_dump() for e in state["evidence"] if e.kind == "case"]
    raw = backend.json(
        "提取病例中明确提供的临床事实。name为简短中文项目名或检查英文缩写（如CD4、LDH）。"
        "value必须保留原始值及单位；已明确升高但没有数值时写升高，不编造数值。"
        "source_id使用C编号，source_quote逐字复制对应来源中的片段，英文原文不得翻译。"
        "包括年龄性别、全部症状、全部检查和补充结果；每个症状、风险因素、检查单独一项，不能把整段病史作为一个事实。不要提取编号或网址。"
        "不推断病因或诊断，不把未提到某症状当阴性；至多16项。\n"
        + json.dumps(sources, ensure_ascii=False),
        CaseProfile,
        max_tokens=1500,
    )
    grounded, rejected = ground_facts(raw, state["evidence"])
    log(
        state,
        "facts",
        f"核对通过 {len(grounded.facts)} 项事实；排除 {len(rejected)} 项无原文依据的提取",
    )
    return {"profile": grounded}


def observe(state):
    if state["mode"] == "direct" or not state["case"].get("image_path"):
        return {}
    log(state, "observe", "独立描述胸片可见的形态与位置，不先给疾病结论")
    result = backend.json(
        "Describe this chest radiograph carefully: opacities, large masses, nodules, mediastinal displacement and pleural changes. Use image-left/image-right unless orientation is clear. Do not diagnose diseases or assume normality. Provide 2 to 5 specific observations in Chinese or English; uncertainty belongs in limitations.",
        Observation,
        state["case"]["image_path"],
        max_tokens=550,
    )
    evidence = Evidence(
        id="I-observe",
        kind="image",
        title="视觉模型的独立胸片观察",
        content=result.model_dump_json(),
    )
    observations = [*state["evidence"], evidence]
    if config.CXR_EXPERT_ENABLED:
        log(state, "radiology", "调用胸片专用模型，获得第二种影像观察")
        try:
            observations.append(
                image_tools.findings(state["case"]["image_path"], state["run_id"])
            )
        except Exception as error:
            observations.append(
                Evidence(
                    id="I-radiology",
                    kind="image",
                    title="胸片专用模型执行失败",
                    content=str(error),
                    status="failed",
                )
            )
    return {"evidence": observations}


def analyze(state):
    evidence = list(state["evidence"])
    for tool_name in state["plan"].tools:
        log(state, "tools", f"正在执行 {tool_name}")
        try:
            result = getattr(image_tools, tool_name)(
                state["case"]["image_path"], state["run_id"]
            )
            log(state, "tools", f"{tool_name} 完成")
        except Exception as error:
            result = Evidence(
                id=f"I-{tool_name}",
                kind="image",
                title=f"{tool_name} 执行失败",
                content=str(error),
                status="failed",
            )
            log(state, "tools", f"{tool_name} 失败；不会将失败视为正常检查结果")
        evidence.append(result)
    return {"evidence": evidence}


def search(state):
    evidence = list(state["evidence"])
    if state["mode"] != "direct":
        log(state, "retrieve", "检索诊断资料，保留指南、专业资料和科普资料的来源区别")
        query = " ".join(
            e.content
            for e in state["evidence"]
            if e.kind == "case" and not e.content.startswith(("来源", "输入仅"))
        )
        # Keep the complete patient history; hypotheses are retrieval leads, not patient facts.
        query += "\nCandidate retrieval terms (unconfirmed): " + state["plan"].query
        query += " " + " ".join(state["plan"].hypotheses)
        for e in state["evidence"]:
            if e.id in {"I-observe", "I-radiology"} and e.status == "completed":
                data = json.loads(e.content)
                query += " " + " ".join(
                    data.get("visual_findings", data.get("observations", []))
                )
        found = retrieve(query)
        evidence.extend(found)
        log(state, "retrieve", f"检索到 {len(found)} 条资料片段")
    return {"evidence": evidence}


def reasoning_image(state):
    """Use the same image policy for drafting, reviewing and repairing a report."""
    if (
        state["mode"] == "direct"
        or not config.CXR_EXPERT_ENABLED
        or config.RECHECK_IMAGE
    ):
        return state["case"].get("image_path")
    return None


def reasoning_context(state):
    sections = []
    for evidence in state["evidence"]:
        if evidence.id == "I-original" and not reasoning_image(state):
            continue
        if evidence.status != "completed":
            sections.append(f"[{evidence.id}] 工具失败，不能用于支持或排除疾病。")
            continue
        content = evidence.content
        if evidence.id == "I-classify":
            data = json.loads(content)
            scores = sorted(data["scores"].items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            content = "模型辅助分数（不是患病概率）：" + ", ".join(
                f"{k}={v:.3f}" for k, v in scores
            )
        elif evidence.id == "I-radiology":
            content = "专用胸片模型观察（仍需原图核对）：" + "; ".join(
                json.loads(content)["visual_findings"]
            )
        elif evidence.id == "I-observe":
            content = "通用视觉模型观察（可能错误）：" + "; ".join(
                json.loads(content)["observations"]
            )
        sections.append(f"[{evidence.id}] {evidence.title}\n{content}")
    return "\n\n".join(sections)


def report_prompt(state):
    return (
        "你是研究用途的胸部诊断辅助系统。用中文给出疑似诊断，不要宣称未经确认的病因已确诊。"
        "根据病史和下列影像观察及资料，给出最可能疾病及最多两个鉴别诊断；每条支持或反对理由引用对应证据ID。"
        "I-observe和I-radiology都是模型观察，可能互相冲突；不要盲信它们。未知事实不能当已知，未提到症状不等于否认。"
        "静脉药物使用史不能代替HIV/免疫检查。分数不能代替影像诊断。没有石棉暴露不能排除胸膜肿瘤。"
        "解释病史与影像形态如何支持病因，不要把影像描述当疾病。无法区分时明确不确定并提出确认方法。"
        "优先解释有鉴别价值的时间关系、反复发作、风险因素和关键检查；不能被影像工具的病因猜测覆盖明确病史。"
        "模型观察相互矛盾时，不要无声拼接为一致事实；指出矛盾，不能把另一工具的否定语句当作排除依据。"
        "最可能诊断与鉴别项必须是不同疾病，不得把中文、英文、缩写或旧称当成多个诊断。"
        "recommended_checks必须含name、purpose、资料K编号evidence_ids；遵循资料中的确认方法，不提供处方。"
        "若没有相关K资料可支持检查建议，则recommended_checks和next_checks均为空；不得用病例C编号冒充确认检查的知识来源。"
        "assessment取有限/较充分/存在冲突，未做确认检查时用有限。next_checks可为空。"
        "只有选择题才填写answer_choice字母，其他问题为null。"
        + (
            "你同时接收原始胸片。独立核对胸膜线及其外侧肺纹理、肺容量、纵隔位置和局灶阴影；"
            "观察结果与工具冲突时指出具体形态和不确定性，不能把工具遗漏当作原图正常。直接观察引用I-original。"
            if reasoning_image(state)
            else "当前推理没有直接接收原始胸片，不能引用I-original作为直接观察依据。"
        )
        + "\n已与病例原文核对的事实（不是额外病史）："
        + (
            state.get("profile", CaseProfile(facts=[])).model_dump_json()
            if not state.get("clinical_contrast")
            else "见下方原文线索与病例来源；避免重复工作记忆。"
        )
        + "\n问题："
        + state["question"]
        + "\n\n"
        + reasoning_context(state)
        + contrast_context(state)
    )


def contrast(state):
    if state["mode"] != "verified":
        return {"clinical_contrast": None, "contrast_notes": []}
    log(state, "contrast", "对照关键病例线索、候选病因和互相矛盾的观察")
    prompt = (
        contrast_prompt(reasoning_context(state))
        + "\nTask instructions and options (not patient facts):\n"
        + state["question"]
    )
    try:
        raw = backend.json(
            prompt, ClinicalContrast, reasoning_image(state), max_tokens=1800
        )
        grounded, notes = ground_contrast(
            raw, state["evidence"], allow_original=bool(reasoning_image(state))
        )
    except ValueError:
        log(
            state,
            "contrast",
            "临床对照结构无效；保留原证据生成草稿，并标记此步骤未完成",
        )
        return {
            "clinical_contrast": None,
            "contrast_notes": ["临床证据对照未完成：模型没有返回有效结构"],
        }
    log(
        state,
        "contrast",
        f"保留{len(grounded.key_case_clues)}条原文线索、{len(grounded.hypotheses)}个候选、{len(grounded.observation_conflicts)}组观察分歧",
    )
    return {
        "clinical_contrast": grounded if len(grounded.hypotheses) >= 2 else None,
        "contrast_notes": notes,
    }


def contrast_context(state):
    memo = state.get("clinical_contrast")
    if not memo:
        return ""
    return (
        "\n临床证据对照（模型的中间候选，不是新增患者事实，也不是已经确认的诊断）：\n"
        + memo.model_dump_json()
        + "\n报告需解释最有鉴别价值的已知病史，比较具体病因与尚未知的确认条件。"
        "若观察存在分歧，直接查看原图并明确哪些分歧尚不能解决；不得把中间候选当作事实。"
        "若病史支持一个具体关联病因，应给出限定为疑似的具体候选，不必等待病理才提出鉴别；"
        "若支持不足，应保留不确定性，不能强行确诊罕见病。"
    )


def diagnose(state):
    log(state, "diagnose", "综合病例与图像证据，生成候选诊断")
    image = reasoning_image(state)
    result = backend.json(
        report_prompt(state), report_schema(state["question"]), image, max_tokens=1800
    )
    return {"report": result}


def update_report(state):
    result = state["report"]
    if state.get("previous"):
        log(state, "update", "比较本次报告与上次判断，解释补充资料的影响")
        old_lines = set((state.get("previous_context") or "").splitlines())
        new_info = [
            e.model_dump()
            for e in state["evidence"]
            if e.kind == "case" and e.content not in old_lines
        ]
        if not new_info:
            previous_name = state["previous"]["most_likely"]["name"]
            result.change_summary = f"本次没有新增病例资料；重新分析的候选判断为{result.most_likely.name}，上次为{previous_name}。"
            return {"report": result}
        change = backend.json(
            "用中文说明新资料如何影响诊断判断。候选诊断可以不变，但要明确证据变化；"
            "只引用下列新增资料的证据ID，不得写首次分析。\n新增资料："
            + json.dumps(new_info, ensure_ascii=False)
            + "\n上次报告："
            + json.dumps(state["previous"], ensure_ascii=False)
            + "\n本次报告："
            + result.model_dump_json(),
            Change,
            max_tokens=450,
        )
        valid_ids = {e["id"] for e in new_info}
        if any(ref not in valid_ids for ref in change.evidence_ids):
            raise ValueError("更新说明引用了不存在的新增证据")
        result.change_summary = change.summary
    else:
        result.change_summary = "首次分析"
    return {"report": result}


def verify(state):
    if state["mode"] != "verified":
        return {"issues": []}
    log(state, "verify", "检查证据引用、失败工具引用，并复核结论是否超出证据")
    issues = (
        validate_evidence(
            state["report"],
            state["evidence"],
            allow_original=bool(reasoning_image(state)),
        )
        + known_test_conflicts(state["report"], state["profile"])
        + check_recommendations(state["report"], state["evidence"])
        + clinical_inconsistencies(state["report"], state["evidence"])
        + diagnosis_issues(state["report"])
    )
    review_prompt = (
        "复核以下诊断报告。仅列实际发现的问题，例如编造病例事实、把工具失败当阴性、"
        "把模型分数当疾病概率、引用资料来证明患者症状、忽略明显证据冲突。"
        "重点检查：资料未提及某症状不能改写成无该症状；低分类分数不能改写成影像未见病变。"
        "不要因存在不确定性而要求确定诊断；合理的鉴别诊断不算错误。无实际问题时issues=[]。"
        "每个问题的report_quote必须逐字复制报告中有问题的完整句子或字段值，reason解释与证据的矛盾，evidence_ids引用对应证据。"
        "source_quote必须逐字引用能证明矛盾的证据原文。I-observe是视觉观察而不是分类分数。只复核报告，不要把证据中的免责声明当错误。已经提供的检查不应被再次列为缺失。"
        "没有引用模型分数的句子，不能指责报告把分数当成概率；不要输出泛泛的风险提醒。"
        "这是模型复核，不是临床验证。\n证据："
        + reasoning_context(state)
        + "\n待复核报告："
        + state["report"].model_dump_json()
    )
    try:
        review = backend.json(
            review_prompt, Review, reasoning_image(state), max_tokens=800
        )
    except ValueError as error:
        log(state, "verify", "模型复核输出无效；保留诊断草稿，标记需要复核")
        return {
            "issues": issues,
            "review_notes": [
                *state.get("contrast_notes", []),
                "语义复核未完成：模型未返回有效的复核结构",
            ],
            "review_details": [],
        }
    accepted, notes = anchor_review(review, state["report"], state["evidence"])
    issues = list(
        dict.fromkeys(issues + [f"“{i.report_quote}”：{i.reason}" for i in accepted])
    )
    log(state, "verify", "发现 " + str(len(issues)) + " 项待处理问题")
    return {
        "issues": issues,
        "review_notes": [*state.get("contrast_notes", []), *notes],
        "review_details": [i.model_dump() for i in accepted],
    }


def repair(state):
    log(state, "repair", "根据证据问题修订报告，随后重新校验")
    result = backend.json(
        report_prompt(state)
        + "\n修改以下报告以解决问题，只保留证据支持的结论："
        + state["report"].model_dump_json()
        + "\n问题："
        + json.dumps(state["issues"], ensure_ascii=False),
        report_schema(state["question"]),
        reasoning_image(state),
        max_tokens=1800,
    )
    return {"report": result, "repairs": state["repairs"] + 1}


def route(state):
    return "repair" if state["issues"] and state["repairs"] < 1 else "update"


def build_graph():
    graph = StateGraph(State)
    for name, fn in [
        ("prepare", prepare),
        ("profile", profile),
        ("observe", observe),
        ("plan", plan),
        ("analyze", analyze),
        ("search", search),
        ("contrast", contrast),
        ("diagnose", diagnose),
        ("verify", verify),
        ("repair", repair),
        ("update", update_report),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    for first, second in zip(
        [
            "prepare",
            "profile",
            "observe",
            "plan",
            "analyze",
            "search",
            "contrast",
            "diagnose",
        ],
        [
            "profile",
            "observe",
            "plan",
            "analyze",
            "search",
            "contrast",
            "diagnose",
            "verify",
        ],
    ):
        graph.add_edge(first, second)
    graph.add_conditional_edges(
        "verify", route, {"repair": "repair", "update": "update"}
    )
    graph.add_edge("repair", "verify")
    graph.add_edge("update", END)
    return graph.compile()


graph = build_graph()


def execute(run_id):
    # Serial GPU jobs avoid simultaneous model initialization and competing allocations.
    with RUN_LOCK:
        run = store.get_run(run_id)
        case = store.get_case(run["case_id"])
        case["context"] = run["context"]
        store.update_run(run_id, "running")
        start = time.monotonic()
        metric_start = len(backend.metrics)
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            state = graph.invoke(
                {
                    "case": case,
                    "run_id": run_id,
                    "question": run["question"],
                    "mode": run["mode"],
                }
            )
            report = state["report"]
            remaining = (
                validate_evidence(
                    report,
                    state["evidence"],
                    allow_original=bool(reasoning_image(state)),
                )
                + known_test_conflicts(report, state["profile"])
                + check_recommendations(
                    report, state["evidence"], require_sources=run["mode"] != "direct"
                )
                + clinical_inconsistencies(report, state["evidence"])
                + diagnosis_issues(report)
            )
            issues = list(dict.fromkeys(state["issues"] + remaining))
            result = {
                "report": report.model_dump(),
                "evidence": [e.model_dump() for e in state["evidence"]],
                "verification": {
                    "performed": run["mode"] == "verified",
                    "issues": issues,
                    "notes": state.get("review_notes", []),
                    "details": state.get("review_details", []),
                    "status": "needs_review"
                    if issues or state.get("review_notes")
                    else ("passed_checks" if run["mode"] == "verified" else "not_run"),
                    "note": "引用检查与模型复核不能证明临床正确性",
                },
                "metrics": {
                    "elapsed_seconds": round(time.monotonic() - start, 2),
                    "model_calls": backend.metrics[metric_start:],
                    "tool_calls": len(state["plan"].tools)
                    + int(any(e.id == "I-radiology" for e in state["evidence"])),
                    "repairs": state["repairs"],
                    "gpu_peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 2**30, 2
                    )
                    if torch.cuda.is_available()
                    else None,
                    "gpu_peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 2**30, 2
                    )
                    if torch.cuda.is_available()
                    else None,
                },
                "plan": state["plan"].model_dump(),
                "profile": state["profile"].model_dump(),
                "model": config.MODEL.name,
                "cxr_expert_enabled": config.CXR_EXPERT_ENABLED,
                "clinical_contrast": state["clinical_contrast"].model_dump()
                if state.get("clinical_contrast")
                else None,
                "clinical_contrast_enabled": run["mode"] == "verified",
                "cxr_reader": config.CXR_READER,
                "constrained_json_enabled": config.CONSTRAINED_JSON
                and config.BACKEND == "local",
                "reasoning_reads_original_image": bool(reasoning_image(state)),
            }
            store.update_run(run_id, "completed", result)
            log(
                state,
                "complete",
                "分析完成" + ("；仍有证据问题，需要复核" if issues else ""),
            )
        except Exception as error:
            store.event(run_id, "error", str(error))
            store.update_run(run_id, "failed", error=str(error))
