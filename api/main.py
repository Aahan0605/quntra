from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import portfolio, quantum, options, backtest
from api.routers import ml, analyze

app = FastAPI(
    title="Quantra API",
    description="Backend engine for the Quantra Hybrid Quantum-Classical Platform — with ML/DL Intelligence Layer",
    version="4.0.0"
)

# Enable CORS for the local terminal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolio.router)
app.include_router(quantum.router)
app.include_router(options.router)
app.include_router(backtest.router)
app.include_router(ml.router)
app.include_router(analyze.router)

@app.get("/health")
def health_check():
    """Simple API health check endpoint."""
    return {"status": "Quantra Engine Online", "version": "4.0.0", "ml_layer": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
