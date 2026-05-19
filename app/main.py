from contextlib import asynccontextmanager
import logging

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dashboard_routes import router as dashboard_router
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.model import load_model, ModelStore
from app.core.panel import PanelStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger = logging.getLogger("app.lifecycle")
    try:
        bundle = load_model(settings.model_path)
        ModelStore.set_bundle(bundle)
    except FileNotFoundError:
        if settings.model_required:
            raise
        logger.warning("Model file not found at %s", settings.model_path)

    try:
        panel = pd.read_parquet(settings.panel_path)
        PanelStore.set_panel(panel)
    except FileNotFoundError:
        if settings.panel_required:
            raise
        logger.warning("Panel file not found at %s", settings.panel_path)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(dashboard_router)
