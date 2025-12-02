from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import router as root_router
from app.config import settings
from app.lifespan import lifespan

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint for the backend
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backend"}


app.include_router(root_router, prefix="/api")
