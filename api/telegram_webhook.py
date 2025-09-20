from fastapi import FastAPI, Request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-logs")

app = FastAPI()

@app.get("/")
async def root(request: Request):
    log_entry = f"{request.method} {request.url.path}"
    logger.info(log_entry)   # Appears in Vercel function logs
    return {"message": "Hello from FastAPI!", "log": log_entry}