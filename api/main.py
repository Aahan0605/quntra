from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import portfolio, quantum, options, backtest
from api.routers import ml, analyze

app = FastAPI(
    title="Quantra API",
    description="Backend engine for the Quantra Hybrid Quantum-Classical Platform — with ML/DL Intelligence Layer",
    version="4.0.0"
)

# CORS for the local terminal only.
#
# This was allow_origins=["*"] together with allow_credentials=True — a
# combination browsers reject outright, and which states the intent to let
# any website on the internet make credentialed calls to a trading API.
# The terminal is served locally, so name the local origins explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8000", "http://127.0.0.1:8000",
        "null",  # file:// origin — quantra-terminal.html opened directly
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    import os

    import uvicorn

    # Loopback by default. This was 0.0.0.0, which publishes an
    # unauthenticated trading API to every device on the network — including
    # whatever else is on the cafe wifi. There is no auth on any router, so
    # binding publicly is equivalent to making it anonymous.
    # Override deliberately (behind a reverse proxy with auth) via QUNTRA_API_HOST.
    uvicorn.run("api.main:app", host=os.getenv("QUNTRA_API_HOST", "127.0.0.1"),
                port=int(os.getenv("QUNTRA_API_PORT", "8000")), reload=True)
