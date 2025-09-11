import os
import io
import aiohttp
import logging
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import asyncpg
from huggingface_hub import InferenceClient
import sys

sys.path.append(os.path.dirname(__file__))
from functions_for_pred import predict_category

app = FastAPI()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not found in environment variables")
if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")

# Hugging Face client
client = InferenceClient(
    api_key=HF_TOKEN,
    headers={"Content-Type": "audio/ogg"}
)
# -------------------- Helper Functions -------------------- #

async def download_telegram_audio(file_id: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        file_info_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
        async with session.get(file_info_url) as resp:
            if resp.status != 200:
                raise Exception(f"Telegram API returned status {resp.status}")
            file_info = await resp.json()
        if not file_info.get("ok"):
            raise Exception(f"Telegram API error: {file_info.get('description')}")
        file_path = file_info["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
        async with session.get(file_url) as audio_resp:
            if audio_resp.status != 200:
                raise Exception(f"Failed to download audio: {audio_resp.status}")
            return await audio_resp.read()


async def send_telegram_message(chat_id: int, text: str):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        if response.status != 200:
            logger.error(f"Failed to send Telegram message: {response.status}")


async def save_text_to_db(text: str, category: str, amount: int = 0):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute(
                "INSERT INTO telegram_messages (text, type, amount, category) VALUES ($1, $2, $3)",
                text, category, amount
            )
            logger.info("Text message saved to database")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")

async def process_text_message(text: str, chat_id: int):
    # Predict category
    category = predict_category(text)
    
    # Save text + category to DB
    await save_text_to_db(text, category)
    
    # Send reply to Telegram
    await send_telegram_message(chat_id, f"Dijiste: {text} con categoría {category}")
    
    return JSONResponse({"status": "text_received", "text": text, "category": category})

async def process_voice_message(message: dict):
    file_id = message["voice"]["file_id"]
    chat_id = message["chat"]["id"]
    
    # Transcribe audio
    try:
        ogg_bytes = await download_telegram_audio(file_id)
        logger.info(f"Downloaded {len(ogg_bytes)} bytes of audio")
        output = client.automatic_speech_recognition(
            ogg_bytes,
            model="openai/whisper-large-v3-turbo"
        )
        transcription = output.get('text', '')
        logger.info(f"Transcription: {transcription}")
    except Exception as e:
        await send_telegram_message(chat_id, f"Error processing audio: {str(e)}")
        return JSONResponse({"status": "error", "error": "speech_recognition_failed"})

    # 2️⃣ Predict category and 3️⃣ Save transcription + category to DB
    return await process_text_message(transcription, chat_id)

@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
        logger.info(f"Received webhook data: {data}")
        message = data.get("message")
        if not message:
            return JSONResponse({"status": "no_message"})

        if message.get("voice"):
            return await process_voice_message(message)
        elif message.get("text"):
            return await process_text_message(message["text"], message["chat"]["id"])
        else:
            logger.warning(f"Unknown message type: {message.keys()}")
            return JSONResponse({"status": "unknown_message_type"})

    except Exception as e:
        logger.error(f"Unhandled error in webhook: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


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


@app.get("/api/test-hf")
async def test_hf():
    try:
        logger.info("Testing Hugging Face client")
        return {"status": "hf_client_initialized", "provider": "fal-ai"}
    except Exception as e:
        logger.error(f"HF client test failed: {str(e)}")
        return {"status": "error", "error": str(e)}
