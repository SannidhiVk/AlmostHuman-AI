import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from models.whisper_processor import WhisperProcessor
from models.moondream_processor import MoondreamProcessor
from models.tts_processor import KokoroTTSProcessor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing models on startup...")
    try:
        # Initialize processors to load models
        whisper_processor = WhisperProcessor.get_instance()
        moondream_processor = MoondreamProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    from managers.connection_manager import manager

    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")
