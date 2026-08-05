import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.inference import AllusionEngine
from backend.knowledge_store import KnowledgeStore


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
MODEL_MODE = os.environ.get("DUOSHI_MODEL_MODE", "4bit").lower()
knowledge_path_setting = os.environ.get("DUOSHI_KNOWLEDGE_PATH")
KNOWLEDGE_PATH = (
    Path(knowledge_path_setting)
    if knowledge_path_setting
    else ROOT_DIR / "duoshi" / "data" / "user" / "user_allusions.xlsx"
)
knowledge_store = KnowledgeStore(KNOWLEDGE_PATH)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class KnowledgeEntryRequest(BaseModel):
    allusion_name: str = Field(min_length=1, max_length=200)
    source_text: str = Field(min_length=1, max_length=10000)
    allusion_mean: str = Field(min_length=1, max_length=10000)
    semantic_tags: str = Field(default="", max_length=2000)
    allusion_variants: str = Field(default="", max_length=2000)
    poem_example: str = Field(min_length=1, max_length=10000)

    @field_validator("allusion_name", "source_text", "allusion_mean", "poem_example")
    @classmethod
    def validate_required_text(cls, value):
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("必填字段不能为空。")
        return cleaned_value

    @field_validator("semantic_tags", "allusion_variants")
    @classmethod
    def clean_optional_text(cls, value):
        return value.strip()


@asynccontextmanager
async def lifespan(app):
    print(f"正在加载 {MODEL_MODE} 模型与典故索引，请稍候…")
    app.state.engine = AllusionEngine(MODEL_MODE)
    print("模型与典故索引加载完成。")
    yield


app = FastAPI(
    title="多识 · 典故解析",
    description="基于本地模型和 FAISS 诗例索引的典故解析接口。",
    lifespan=lifespan,
)


@app.get("/api/health")
def health(request: Request):
    engine = request.app.state.engine
    return {"status": "ready", "mode": engine.mode, "mode_label": engine.mode_label}


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    engine = request.app.state.engine

    def event_stream():
        sent_done = False
        try:
            for event in engine.stream_analysis(payload.message.strip()):
                sent_done = sent_done or event.get("type") == "done"
                data = json.dumps(event, ensure_ascii=False)
                yield f"data: {data}\n\n"
        except Exception as error:
            data = json.dumps(
                {"type": "error", "message": f"解析过程中出现错误：{error}"},
                ensure_ascii=False,
            )
            yield f"data: {data}\n\n"
        finally:
            if not sent_done:
                data = json.dumps({"type": "done"}, ensure_ascii=False)
                yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/knowledge")
def knowledge_summary():
    return {"count": knowledge_store.count()}


@app.post("/api/knowledge")
def add_knowledge(payload: KnowledgeEntryRequest):
    allusion_id = knowledge_store.add(payload.model_dump())
    return {
        "status": "saved",
        "allusion_id": allusion_id,
        "message": "典故资料已保存到本地知识库表格。",
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
