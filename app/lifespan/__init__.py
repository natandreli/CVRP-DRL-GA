import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.config.logging import logger


def cleanup_old_sessions():
    """
    Remove session directories older than SESSION_MAX_AGE_HOURS.
    """
    if not settings.INSTANCES_DIR.exists():
        return

    cleaned = 0

    for session_dir in settings.INSTANCES_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        dir_age = time.time() - session_dir.stat().st_mtime

        if dir_age > settings.SESSION_MAX_AGE_HOURS * 3600:
            try:
                shutil.rmtree(session_dir)
                cleaned += 1
                logger.info(f"Cleaned old session directory: {session_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to clean {session_dir.name}: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} old session directories")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("Cleaning old session directories...")
    cleanup_old_sessions()

    settings.INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    settings.PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Instances directory ready: {settings.INSTANCES_DIR}")
    logger.info(f"Presets directory ready: {settings.PRESETS_DIR}")

    yield

    logger.info("Shutting down...")
