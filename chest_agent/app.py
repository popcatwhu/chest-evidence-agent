from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from . import store, config, agents
from .llm import backend
from .workflow import execute

executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app):
    store.init_db()
    store.recover_interrupted()
    yield


app = FastAPI(title="Chest Evidence Agent", lifespan=lifespan)
app.mount(
    "/artifacts", StaticFiles(directory=config.DATA / "artifacts"), name="artifacts"
)
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static/index.html")


@app.get("/api/health")
def health():
    return {
        "backend": config.BACKEND,
        "model_name": config.MODEL.name,
        "model_ready": backend.ready() and not config.INFERENCE_PAUSED,
        "cxr_reader": config.CXR_READER,
        "clinical_contrast": True,
        "agent_team": agents.roster(),
        "team_models_ready": all(
            a.model.ready() for a in (agents.radiologist, agents.reviewer)
        ),
        "inference_paused": config.INFERENCE_PAUSED,
        "load_nf4": config.LOAD_NF4,
        "recheck_original_image": config.RECHECK_IMAGE,
        "constrained_json": config.CONSTRAINED_JSON and config.BACKEND == "local",
        "max_input_tokens": config.MAX_INPUT_TOKENS,
        "model_loaded": backend.model is not None,
        "knowledge_documents": len(list((config.DATA / "knowledge").glob("*.json"))),
    }


@app.get("/api/cases")
def cases():
    return store.list_cases()


@app.post("/api/cases")
async def new_case(
    title: str = Form(...),
    context: str = Form(...),
    image: UploadFile | None = File(None),
):
    if not context.strip():
        raise HTTPException(422, "请填写病例资料")
    image_path = None
    if image and image.filename:
        image_path = config.DATA / "uploads" / f"{uuid4().hex}.png"
        try:
            with Image.open(image.file) as img:
                img.convert("RGB").save(image_path)
        except (OSError, ValueError):
            raise HTTPException(422, "请上传有效的 PNG 或 JPEG 图像")
    return store.create_case(title, context, str(image_path) if image_path else None)


@app.get("/api/cases/{case_id}/image")
def case_image(case_id: str):
    case = store.get_case(case_id)
    if not case or not case["image_path"]:
        raise HTTPException(404, "没有病例图像")
    return FileResponse(case["image_path"])


class Supplement(BaseModel):
    text: str = Field(min_length=1)


@app.post("/api/cases/{case_id}/supplement")
def supplement(case_id: str, body: Supplement):
    if not store.get_case(case_id):
        raise HTTPException(404, "病例不存在")
    store.append_context(case_id, body.text)
    return store.get_case(case_id)


class RunInput(BaseModel):
    question: str = (
        "最可能的诊断是什么？列出鉴别诊断、支持和反对证据，以及需要补充的检查。"
    )
    mode: str = "verified"


@app.post("/api/cases/{case_id}/runs")
def run_case(case_id: str, body: RunInput):
    if not store.get_case(case_id):
        raise HTTPException(404, "病例不存在")
    if body.mode not in {"direct", "tools", "verified"}:
        raise HTTPException(422, "未知分析模式")
    if config.INFERENCE_PAUSED:
        raise HTTPException(
            503, "正在进行模型对照评测，病例推理暂时暂停；历史报告仍可查看"
        )
    if not backend.ready():
        raise HTTPException(503, "模型尚未准备好，请等待权重下载完成或配置模型后端")
    run_id = store.create_run(case_id, body.question, body.mode)
    executor.submit(execute, run_id)
    return {"run_id": run_id}


@app.get("/api/cases/{case_id}/runs")
def case_runs(case_id: str):
    if not store.get_case(case_id):
        raise HTTPException(404, "病例不存在")
    with store.connect() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT id,status,created,mode FROM runs WHERE case_id=? ORDER BY created DESC",
                (case_id,),
            )
        ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "任务不存在")
    return run


@app.get("/api/runs/{run_id}/download")
def download(run_id: str):
    from fastapi.responses import JSONResponse

    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "任务不存在")
    return JSONResponse(
        run,
        headers={"Content-Disposition": f'attachment; filename="report-{run_id}.json"'},
    )
