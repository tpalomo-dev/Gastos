import os
import aiohttp
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import asyncpg

app = FastAPI()

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

async def get_db_pool():
    if not hasattr(app.state, "db_pool"):
        app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
    return app.state.db_pool

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

@app.get("/favicon.ico")
async def faviconico():
    return Response(status_code=204)

@app.get("/favicon.png")
async def faviconpng():
    return Response(status_code=204)

@app.get("/")
def read_root():
    return {"message": "Hello World from FastAPI on Vercel!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message")

    if not message:
        return JSONResponse({"status": "no_message"})

    chat_id = message["chat"]["id"]

    async with aiohttp.ClientSession() as session:

        # Handle voice messages
        if message.get("voice"):
            file_id = message["voice"]["file_id"]

            # Get file info
            async with session.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
            ) as resp:
                file_info = await resp.json()

            file_path = file_info["result"]["file_path"]

            # Download audio
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            async with session.get(file_url) as audio_resp:
                audio_content = await audio_resp.read()

            tmp_path = f"/tmp/{file_id}.ogg"
            with open(tmp_path, "wb") as f:
                f.write(audio_content)

            print(f"Downloaded audio {file_id}")

            # Reply to user
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": "Audio received!"}
            )

            return JSONResponse({"status": "audio_received", "file_id": file_id})

        # Handle text messages
        elif message.get("text"):
            text = message["text"]
            msg_type = "text"
            amount = 0

            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                    text, msg_type, amount
                )

            # Reply to user
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": f"You said: {text}"}
            )

            return JSONResponse({"status": "text_received", "text": text})

        else:
            return JSONResponse({"status": "unknown_message_type"})
