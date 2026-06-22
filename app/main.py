"""FastAPI application setup for the Douyin Video Translator."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import confirmation_routes
from app.api.confirmation_routes import configure_confirmation_routes
from app.api.routes import configure_routes, router
from app.middleware.bot_protection import BotProtectionMiddleware
from app.services.checkpoint_manager import CheckpointManager
from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Sets up CORS, error handlers, and includes the API router.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Douyin Video Translator",
        description="Dịch video Douyin từ tiếng Trung sang tiếng Việt",
        version="0.1.0",
    )

    # Bot protection middleware (must be added before CORS so it runs first)
    app.add_middleware(BotProtectionMiddleware, max_404_per_minute=15, block_duration_seconds=300)

    # CORS middleware (allow all origins for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure route dependencies
    job_store = JobStore()
    checkpoint_manager = CheckpointManager(job_store)
    configure_routes(job_store=job_store, checkpoint_manager=checkpoint_manager)
    configure_confirmation_routes(checkpoint_manager=checkpoint_manager)

    # Include API routers
    app.include_router(router)
    app.include_router(confirmation_routes.router)

    # Global exception handlers
    _register_exception_handlers(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the app."""

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler for unhandled exceptions."""
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Đã xảy ra lỗi hệ thống, vui lòng thử lại sau",
                "step": None,
                "retryable": True,
                "retry_after": None,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError as 400 Bad Request."""
        return JSONResponse(
            status_code=400,
            content={
                "error": "VALIDATION_ERROR",
                "message": str(exc),
                "step": None,
                "retryable": False,
                "retry_after": None,
            },
        )


app = create_app()
