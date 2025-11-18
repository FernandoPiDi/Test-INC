"""
Routes package - contiene los routers de datos e inferencia
"""

from routes.data import router as data_router
from routes.inference import router as inference_router

__all__ = ["data_router", "inference_router"]

