import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Import database dependencies for table creation
# Note: You will need to create app/database.py with engine and Base
try:
    from database import init_db
except ImportError:
    init_db = None
    print("Warning: database not found. Database setup is mocked.")

# Import routers
# Note: You will need to create these routers in app/routers/
try:
    from routers import events, metrics, funnel, heatmap, anomalies, health
except ImportError:
    events = metrics = funnel = heatmap = anomalies = health = None
    print("Warning: routers not found. Routers are not included.")

# Ensure structlog outputs JSON
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run DB table creation on startup
    import models  # noqa: F401
    if init_db is not None:
        await init_db()
    else:
        logger.warning("startup", message="Database module missing, skipping table creation.")
    
    yield
    logger.info("shutdown", message="Application shutdown complete")

app = FastAPI(
    title="Store Intelligence API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware enabled for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for video feed
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(data_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=data_dir), name="static")

@app.middleware("http")
async def structlog_middleware(request: Request, call_next):
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    
    # Extract store_id from path if present (e.g. /stores/{store_id}/...)
    store_id = None
    path_parts = request.url.path.strip("/").split("/")
    if "stores" in path_parts:
        try:
            store_index = path_parts.index("stores")
            if store_index + 1 < len(path_parts):
                store_id = path_parts[store_index + 1]
        except ValueError:
            pass

    # Clear previous context and set new ones for this request
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id,
        endpoint=request.url.path,
        method=request.method,
    )
    if store_id:
        structlog.contextvars.bind_contextvars(store_id=store_id)
        
    try:
        response = await call_next(request)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        structlog.contextvars.bind_contextvars(
            latency_ms=latency_ms,
            status_code=response.status_code
        )
        logger.info("request_completed")
        return response
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        structlog.contextvars.bind_contextvars(
            latency_ms=latency_ms,
            status_code=503,
            error=str(e)
        )
        # Log without raw stack traces
        logger.error("request_failed", exc_info=False)
        return JSONResponse(status_code=503, content={"error": "service_unavailable", "detail": str(e)})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "detail": str(exc)},
    )

# Include routers if they were imported successfully
if events and metrics and funnel and heatmap and anomalies and health:
    app.include_router(events.router)
    app.include_router(metrics.router)
    app.include_router(funnel.router)
    app.include_router(heatmap.router)
    app.include_router(anomalies.router)
    app.include_router(health.router)

@app.get("/")
async def root():
    return {"status": "ok", "service": "store-intelligence-api"}

pipeline_process = None

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(ui_path, "r") as f:
        return f.read()

@app.post("/api/start-pipeline/{store_id}")
async def start_pipeline(store_id: str):
    global pipeline_process
    if pipeline_process is not None:
        try:
            pipeline_process.terminate()
            pipeline_process.wait(timeout=2)
        except Exception:
            pass
            
    pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    from datetime import datetime, timezone
    
    current_time_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    cmd = [
        "python", "detect.py",
        "--store1-dir", os.path.join(data_dir, "Store_1"),
        "--store2-dir", os.path.join(data_dir, "Store_2"),
        "--output", os.path.join(data_dir, "events.jsonl"),
        "--target-store", store_id,
        "--clip-start", current_time_iso
    ]
    pipeline_process = subprocess.Popen(cmd, cwd=pipeline_dir)
    return {"status": "started", "store_id": store_id}

