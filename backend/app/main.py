"""FastAPI application factory.

Run from the backend/ directory with: uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, load_settings
from app.container import Container, build_container
from app.errors import AppError

logger = logging.getLogger("doctutor")


def _error(status_code: int, code: str, message: str, detail: str | None) -> JSONResponse:
    body = {"code": code, "message": message, "detail": detail}
    return JSONResponse(status_code=status_code, content={"error": body})


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO)
    app = FastAPI(title="DocTutor", version="1.0.0")
    app.state.container = container or build_container(settings)

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.info("%s: %s | %s", exc.code, exc.message, exc.detail)
        return _error(exc.status_code, exc.code, exc.message, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        message = f"Invalid {field}: {first.get('msg', 'bad value')}" if field else "Invalid request."
        return _error(422, "invalid_input", message, str(exc.errors())[:1000])

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        detail = repr(exc) if settings.debug else "See server logs for details."
        return _error(500, "internal_error", "Something went wrong on the server.", detail)

    app.include_router(router)
    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    frontend = settings.frontend_dir
    if not (frontend / "index.html").is_file():
        logger.warning("Frontend not found at %s; serving API only.", frontend)
        return
    app.mount("/static", StaticFiles(directory=frontend), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(frontend / "index.html")

