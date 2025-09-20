from fastapi import FastAPI, Request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-logs")

app = FastAPI()

# Example route
@app.get("/")
async def root():
    logger.info("Root endpoint was called")
    return {"message": "Hello from FastAPI on Vercel!"}

# Route to show logs (keeps logs in memory)
log_memory = []

@app.get("/show-logs")
async def show_logs():
    return {"logs": log_memory}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log_entry = f"{request.method} {request.url.path}"
    log_memory.append(log_entry)
    logger.info(log_entry)
    response = await call_next(request)
    return response