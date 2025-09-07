import os
import io
import aiohttp
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import asyncpg

from huggingface_hub import InferenceClient

app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN)


@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message")

    if not message:
        return JSONResponse({"status": "no_message"})

    # Voice message
    if message.get("voice"):
        file_id = message["voice"]["file_id"]
        chat_id = message["chat"]["id"]

        async with aiohttp.ClientSession() as session:
            # Get file info
            async with session.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
            ) as resp:
                file_info = await resp.json()

            file_path = file_info["result"]["file_path"]

            # Download audio (OGG)
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            async with session.get(file_url) as audio_resp:
                ogg_bytes = await audio_resp.read()

        # Wrap bytes in a BytesIO for Hugging Face
        audio_file = io.BytesIO(ogg_bytes)
        output = client.automatic_speech_recognition(
            audio_file, model="openai/whisper-large-v3"
        )

        print(f"Processed audio {file_id}")

        # Send reply
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": f"Dijiste: {output['text']}"},
            )

        return JSONResponse({
            "status": "audio_received",
            "file_id": file_id,
            "transcription": output
        })

    # Text message
    elif message.get("text"):
        text = message["text"]
        chat_id = message["chat"]["id"]
        msg_type = "text"
        amount = 0

        # Use direct connection instead of pool
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute(
                "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                text, msg_type, amount
            )
        finally:
            await conn.close()

        # Send reply
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": f"You said: {text}"},
            )

        return JSONResponse({"status": "text_received", "text": text})

    else:
        return JSONResponse({"status": "unknown_message_type"})


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