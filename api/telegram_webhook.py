from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
async def root(request: Request):
    log_entry = f"{request.method} {request.url.path}"
    print(log_entry)  # <- This WILL appear in Vercel function logs
    return {"message": "Hello from FastAPI!", "log": log_entry}