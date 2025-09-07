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

@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

@app.get("/favicon.ico")
async def favicon():
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

    if message.get("voice"):
        file_id = message["voice"]["file_id"]

        async with aiohttp.ClientSession() as session:
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

        # Save to ephemeral storage
        tmp_path = f"/tmp/{file_id}.ogg"
        with open(tmp_path, "wb") as f:
            f.write(audio_content)

        # Optional: enqueue for transcription
        print(f"Downloaded audio {file_id}")

        return JSONResponse({"status": "audio_received", "file_id": file_id})

    # elif message.get("text"):
    #     text = message["text"]

    #     # Example: do something else with text
    #     # e.g., echo back, log it, or send to another API
    #     print(f"Received text message: {text}")

    #     # Example: send a reply to the user
    #     chat_id = message["chat"]["id"]
    #     async with aiohttp.ClientSession() as session:
    #         await session.post(
    #             f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
    #             json={"chat_id": chat_id, "text": f"You said: {text}"},
    #         )

    #     return JSONResponse({"status": "text_received", "text": text})

    elif message.get("text"):
        text = message["text"]
        chat_id = message["chat"]["id"]
        msg_type = "text"
        amount = 0  # or calculate if you want

        # Save to Neon
        async with app.state.db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                text, msg_type, amount
            )
    
        # Optional: send a reply
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": f"You said: {text}"},
            )
    
        return JSONResponse({"status": "text_received", "text": text})

    else:
        return JSONResponse({"status": "unknown_message_type"})
    
