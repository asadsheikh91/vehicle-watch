import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.database import engine
from app.redis import init_redis, close_redis, get_redis_pool
from app.routers import auth, devices, telemetry, alerts, analytics
from app.workers.anomaly_worker import start_anomaly_worker

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting VehicleWatch API [env=%s]", settings.app_env)
    await init_redis()
    logger.info("Redis connection pool initialized")

    worker_task = asyncio.create_task(start_anomaly_worker())
    logger.info("Anomaly detection worker started")

    yield

    logger.info("Shutting down VehicleWatch API")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await close_redis()
    await engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(
    title="VehicleWatch",
    description="Real-Time Fleet Telemetry Ingestion and ML Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS fix: allow_credentials=True is incompatible with allow_origins=["*"].
# In development we allow localhost origins explicitly.
# In production, set ALLOWED_ORIGINS env var to your frontend domain.
_dev_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_dev_origins if not settings.is_production else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(devices.router, prefix=API_PREFIX)
app.include_router(telemetry.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Deep health check — verifies DB and Redis are reachable.
    Railway and load balancers use this to determine if the instance is healthy.
    """
    from sqlalchemy import text

    health: dict = {"status": "ok", "version": "1.0.0", "services": {}}

    # Check PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["services"]["postgres"] = "ok"
    except Exception as exc:
        health["services"]["postgres"] = f"error: {exc}"
        health["status"] = "degraded"

    # Check Redis
    try:
        redis = get_redis_pool()
        await redis.ping()
        health["services"]["redis"] = "ok"
    except Exception as exc:
        health["services"]["redis"] = f"error: {exc}"
        health["status"] = "degraded"

    return health
