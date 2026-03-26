from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qa_engine_core import FINAL_TOP_K, MarkdownInstallQAHybrid

MARKDOWN_DIR = os.getenv("MARKDOWN_DIR", "./data/raw/out_md")
DEBUG = os.getenv("DEBUG", "true").lower() in {"1", "true", "yes"}
HAYSTACK_ENABLED = os.getenv("HAYSTACK_ENABLED", "true").lower() in {"1", "true", "yes"}
USE_OLLAMA_GENERATION = os.getenv("USE_OLLAMA_GENERATION", "false").lower() in {"1", "true", "yes"}
AUTO_BUILD_ON_STARTUP = os.getenv("AUTO_BUILD_ON_STARTUP", "false").lower() in {"1", "true", "yes"}
APP_TITLE = os.getenv("APP_TITLE", "Markdown Install QA API")
APP_VERSION = os.getenv("APP_VERSION", "1.1.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pregunta que llega desde la interfaz")
    top_k: int = Field(default=FINAL_TOP_K, ge=1, le=50)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    nodes_retrieved: int
    ollama_available: bool
    haystack_enabled: bool


class BuildResponse(BaseModel):
    message: str
    total_nodes: int
    ollama_available: bool
    haystack_enabled: bool


class EngineManager:
    def __init__(self) -> None:
        self._engine: MarkdownInstallQAHybrid | None = None
        self._lock = threading.Lock()
        self._is_building = False
        self._last_error: str | None = None

    def build(self, force: bool = False) -> MarkdownInstallQAHybrid:
        with self._lock:
            if self._engine is not None and not force:
                return self._engine
            self._is_building = True
            self._last_error = None

        try:
            engine = MarkdownInstallQAHybrid(
                markdown_dir=MARKDOWN_DIR,
                debug=DEBUG,
                enable_haystack=HAYSTACK_ENABLED,
                use_ollama_generation=USE_OLLAMA_GENERATION,
            )
            engine.build()
            with self._lock:
                self._engine = engine
                self._is_building = False
            return engine
        except Exception as exc:
            with self._lock:
                self._is_building = False
                self._last_error = str(exc)
            raise

    def get(self) -> MarkdownInstallQAHybrid:
        if self._engine is None:
            return self.build(force=False)
        return self._engine

    def status(self) -> dict[str, Any]:
        engine = self._engine
        return {
            "built": engine is not None,
            "is_building": self._is_building,
            "last_error": self._last_error,
            "total_nodes": len(engine.nodes) if engine else 0,
            "ollama_available": engine.ollama_available if engine else None,
            "haystack_enabled": engine.enable_haystack if engine else HAYSTACK_ENABLED,
            "embeddings_indexed": engine.embeddings_indexed if engine else False,
            "markdown_dir": MARKDOWN_DIR,
        }


manager = EngineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No bloquea el startup salvo que el usuario lo pida explícitamente.
    if AUTO_BUILD_ON_STARTUP:
        manager.build(force=False)
    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    lifespan=lifespan,
    description="API para consultar documentación Markdown usando BM25 + embeddings + reglas determinísticas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_TITLE,
        "version": APP_VERSION,
        **manager.status(),
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    status = manager.status()
    status["ready"] = status["built"] and not status["is_building"] and not status["last_error"]
    return status


@app.post("/build", response_model=BuildResponse)
def rebuild_index() -> BuildResponse:
    engine = manager.build(force=True)
    return BuildResponse(
        message="Índice reconstruido correctamente",
        total_nodes=len(engine.nodes),
        ollama_available=engine.ollama_available,
        haystack_enabled=engine.enable_haystack,
    )


@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest) -> QueryResponse:
    try:
        engine = manager.get()
        nodes = engine.search(payload.query, top_k=payload.top_k)
        answer = engine.answer(payload.query)
        sources = [
            {
                "doc_title": node.doc_title,
                "heading": node.heading,
                "page_url": node.page_url,
                "file_path": node.file_path,
                "heading_path": node.heading_path,
            }
            for node in nodes
        ]
        return QueryResponse(
            query=payload.query,
            answer=answer,
            sources=sources,
            nodes_retrieved=len(nodes),
            ollama_available=engine.ollama_available,
            haystack_enabled=engine.enable_haystack,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {exc}") from exc


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Markdown Install QA API activa",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "query": "/query",
        "build": "/build",
    }
