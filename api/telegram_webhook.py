# import os
# import aiohttp
# import logging
# from fastapi import FastAPI, Request, Response
# from fastapi.responses import JSONResponse
# import asyncpg

# app = FastAPI()

# logging.basicConfig(level=logging.INFO)

# TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# DATABASE_URL = os.getenv("DATABASE_URL")

# async def get_db_pool():
#     if not hasattr(app.state, "db_pool"):
#         app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
#     return app.state.db_pool

# @app.on_event("shutdown")
# async def shutdown():
#     await app.state.db_pool.close()

# @app.get("/favicon.ico")
# async def faviconico():
#     return Response(status_code=204)

# @app.get("/favicon.png")
# async def faviconpng():
#     return Response(status_code=204)

# @app.get("/")
# def read_root():
#     return {"message": "Hello World from FastAPI on Vercel!"}

# @app.get("/api/health")
# def health_check():
#     return {"status": "healthy"}

# @app.post("/telegram_webhook")
# async def telegram_webhook(req: Request):
#     data = await req.json()
#     message = data.get("message")

#     if not message:
#         return JSONResponse({"status": "no_message"})

#     chat_id = message["chat"]["id"]

#     async with aiohttp.ClientSession() as session:

#         # Handle voice messages
#         if message.get("voice"):
#             file_id = message["voice"]["file_id"]

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

#             tmp_path = f"/tmp/{file_id}.ogg"
#             with open(tmp_path, "wb") as f:
#                 f.write(audio_content)

#             print(f"Downloaded audio {file_id}")

#             # Reply to user
#             await session.post(
#                 f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#                 json={"chat_id": chat_id, "text": "Audio received!"}
#             )

#             return JSONResponse({"status": "audio_received", "file_id": file_id})

#         # Handle text messages
#         elif message.get("text"):
#             text = message["text"]
#             msg_type = "text"
#             amount = 0

#             pool = await get_db_pool()
#             async with pool.acquire() as conn:
#                 await conn.execute(
#                     "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
#                     text, msg_type, amount
#                 )

#             # Reply to user
#             await session.post(
#                 f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#                 json={"chat_id": chat_id, "text": f"You said: {text}"}
#             )

#             return JSONResponse({"status": "text_received", "text": text})

#         else:
#             return JSONResponse({"status": "unknown_message_type"})

import os
import aiohttp
import logging
import asyncio
import uuid
import json
import asyncpg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# Global connection pool (will be created on first request)
db_pool = None
http_session = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
    return db_pool

async def get_http_session():
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()
    return http_session

async def send_telegram_message(chat_id: int, text: str):
    session = await get_http_session()
    try:
        await session.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
        message = data.get("message")
        if not message:
            return JSONResponse({"status": "no_message"})

        chat_id = message["chat"]["id"]
        session = await get_http_session()

        # Handle voice messages
        if message.get("voice"):
            file_id = message["voice"]["file_id"]

            async with session.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
            ) as resp:
                file_info = await resp.json()

            file_path = file_info["result"]["file_path"]
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            tmp_path = f"/tmp/{uuid.uuid4()}.ogg"

            async with session.get(file_url) as audio_resp:
                audio_content = await audio_resp.read()
                with open(tmp_path, "wb") as f:
                    f.write(audio_content)

            logging.info(f"Downloaded audio {file_id} to {tmp_path}")

            # Send Telegram reply asynchronously (won't block)
            asyncio.create_task(send_telegram_message(chat_id, "Audio received!"))

            return JSONResponse({"status": "audio_received", "file_id": file_id})

        # Handle text messages
        elif message.get("text"):
            text = message["text"]
            msg_type = "text"
            amount = 0

            async def process_text():
                pool = await get_db_pool()
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await conn.execute(
                                "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                                text, msg_type, amount
                            )
                    await send_telegram_message(chat_id, f"You said: {text}")
                except Exception as e:
                    logging.error(f"DB insert or reply failed: {e}")

            # Run DB insert + reply concurrently without blocking webhook
            asyncio.create_task(process_text())

            return JSONResponse({"status": "text_received", "text": text})

        else:
            return JSONResponse({"status": "unknown_message_type"})

    except Exception as e:
        logging.exception("Webhook handler failed")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

# Add other endpoints
@app.get("/")
def read_root():
    return {"message": "Hello World from FastAPI on Vercel!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/favicon.ico")
async def faviconico():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/favicon.png")
async def faviconpng():
    from fastapi.responses import Response
    return Response(status_code=204)

# Vercel handler
handler = app
