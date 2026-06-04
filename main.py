from fastapi import FastAPI
from src.api import router

# Δημιουργούμε την εφαρμογή FastAPI
app = FastAPI(
    title="Conversational AI Agent API",
    description="An agent that answers domain questions and makes predictions.",
    version="1.0.0"
)

# Ενώνουμε το router που φτιάξαμε στο api.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Ξεκινάμε τον server τοπικά στη θύρα 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)