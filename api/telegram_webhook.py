import os
import aiohttp
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import asyncpg
import asyncio

from huggingface_hub import InferenceClient

app = FastAPI()
logging.basicConfig(level=logging.INFO)


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN)

async def convert_ogg_to_mp3_bytes(ogg_bytes: bytes) -> bytes:
    """Convert OGG audio bytes to MP3 bytes asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "mp3",
        "pipe:1",
        "-y",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    mp3_bytes, err = await proc.communicate(input=ogg_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {err.decode()}")
    return mp3_bytes

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
            
            # Download audio
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            async with session.get(file_url) as audio_resp:
                ogg_bytes = await audio_resp.read()

        # Convert OGG -> MP3 in memory
        mp3_bytes = await convert_ogg_to_mp3_bytes(ogg_bytes)

        # Save to temp file for Hugging Face (required)
        tmp_mp3_path = f"/tmp/{file_id}.mp3"
        with open(tmp_mp3_path, "wb") as f:
            f.write(mp3_bytes)

        # Run ASR model
        output = client.automatic_speech_recognition(
            tmp_mp3_path, model="openai/whisper-large-v3"
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

# @app.post("/telegram_webhook")
# async def telegram_webhook(req: Request):
#     data = await req.json()
#     message = data.get("message")
    
#     if not message:
#         return JSONResponse({"status": "no_message"})
    
#     if message.get("voice"):
#         file_id = message["voice"]["file_id"]
#         async with aiohttp.ClientSession() as session:
#             # Get file info
#             async with session.get(
#                 f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
#             ) as resp:
#                 file_info = await resp.json()
            
#             file_path = file_info["result"]["file_path"]
            
#             # Download audio
#             file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
#             async with session.get(file_url) as audio_resp:
#                 audio_content = await audio_resp.read()
        
#         # Save to ephemeral storage
#         tmp_path = f"/tmp/{file_id}.ogg"
#         with open(tmp_path, "wb") as f:
#             f.write(audio_content)
        
#         print(f"Downloaded audio {file_id}")
#         return JSONResponse({"status": "audio_received", "file_id": file_id})
    
#     elif message.get("text"):
#         text = message["text"]
#         chat_id = message["chat"]["id"]
#         msg_type = "text"
#         amount = 0
        
#         # Use direct connection instead of pool
#         conn = await asyncpg.connect(DATABASE_URL)
        
#         try:
#             await conn.execute(
#                 "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
#                 text, msg_type, amount
#             )
#         finally:
#             await conn.close()
        
#         # Send reply
#         async with aiohttp.ClientSession() as session:
#             await session.post(
#                 f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#                 json={"chat_id": chat_id, "text": f"You said: {text}"},
#             )
    
#         return JSONResponse({"status": "text_received", "text": text})
    
#     else:
#         return JSONResponse({"status": "unknown_message_type"})
    
    