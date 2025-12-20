from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.core.config import settings
from app.routers import predictions

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Real-time Ethereum price prediction using XGBoost and Bayesian Optimization",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
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

# Create necessary directories on startup
@app.on_event("startup")
async def startup_event():
    from app.routers.predictions import predictor  # ✅ ADD THIS
    
    # Create models directory
    os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
    
    # ✅ ADD THIS - Load saved models
    try:
        predictor.load_models()
        print("Models loaded from saved_models/")
    except Exception as e:
        print(f"No saved models found: {e}")
    
    print(f"{settings.PROJECT_NAME} API started successfully!")
    print(f"Docs available at: http://localhost:8000/docs")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )