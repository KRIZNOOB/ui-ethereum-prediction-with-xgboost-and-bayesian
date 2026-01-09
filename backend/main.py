from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.core.config import settings
from app.routers import predictions

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Real-time Ethereum price prediction using XGBoost and Bayesian Optimization",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    predictions.router, 
    prefix=f"{settings.API_V1_STR}/predictions", 
    tags=["predictions"]
)

@app.get("/")
async def root():
    return {
        "message": f"{settings.PROJECT_NAME} API is running",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

@app.on_event("startup")
async def startup_event():
    from app.routers.predictions import predictor
    
    os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
    
    try:
        predictor.load_models()
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )