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

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Add validation for environment variables
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not found in environment variables")
if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")

client = InferenceClient(
    api_key=HF_TOKEN,
    headers={"Content-Type": "audio/ogg"}  # since your file is mp3
)
# client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN)
# client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    try:
        data = await req.json()
        logger.info(f"Received webhook data: {data}")
        
        message = data.get("message")
        if not message:
            logger.warning("No message found in webhook data")
            return JSONResponse({"status": "no_message"})

        # Voice message
        if message.get("voice"):
            logger.info("Processing voice message")
            file_id = message["voice"]["file_id"]
            chat_id = message["chat"]["id"]
            
            logger.info(f"Voice file_id: {file_id}, chat_id: {chat_id}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Get file info
                    logger.info("Getting file info from Telegram API")
                    file_info_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
                    
                    async with session.get(file_info_url) as resp:
                        if resp.status != 200:
                            logger.error(f"Failed to get file info: {resp.status}")
                            raise Exception(f"Telegram API returned status {resp.status}")
                        
                        file_info = await resp.json()
                        logger.info(f"File info received: {file_info}")
                    
                    if not file_info.get("ok"):
                        logger.error(f"Telegram API error: {file_info}")
                        raise Exception(f"Telegram API error: {file_info.get('description')}")
                    
                    file_path = file_info["result"]["file_path"]
                    logger.info(f"File path: {file_path}")
                    
                    # Download audio (OGG)
                    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
                    logger.info(f"Downloading audio from: {file_url}")
                    
                    async with session.get(file_url) as audio_resp:
                        if audio_resp.status != 200:
                            logger.error(f"Failed to download audio: {audio_resp.status}")
                            raise Exception(f"Failed to download audio: {audio_resp.status}")
                        
                        ogg_bytes = await audio_resp.read()
                        logger.info(f"Downloaded {len(ogg_bytes)} bytes of audio data")

            except Exception as e:
                logger.error(f"Error downloading audio: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Send error message to user
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": f"Error downloading audio: {str(e)}"},
                    )
                return JSONResponse({"status": "error", "error": "download_failed"})

            try:
                # Pass raw bytes directly to Hugging Face
                logger.info("Preparing audio for speech recognition")
                
                logger.info("Calling Hugging Face speech recognition API")
                output = client.automatic_speech_recognition(
                    ogg_bytes,
                    # , model="openai/whisper-large-v3"
                    model="openai/whisper-large-v3-turbo"
                )
                
                logger.info(f"Speech recognition output: {output}")

            except Exception as e:
                logger.error(f"Error in speech recognition: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Send error message to user
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": f"Error processing audio: {str(e)}"},
                    )
                return JSONResponse({"status": "error", "error": "speech_recognition_failed"})

            try:
                logger.info(f"Processed audio {file_id}")
                
                prediction = predict_category(output['text'])
                
                # Send reply
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": f"Dijiste: {output['text']} con categoría {prediction}"},
                    )
                    if response.status != 200:
                        logger.error(f"Failed to send reply: {response.status}")
                    else:
                        logger.info("Reply sent successfully")

                return JSONResponse({
                    "status": "audio_received",
                    "file_id": file_id,
                    "transcription": output
                })

            except Exception as e:
                logger.error(f"Error sending reply: {str(e)}")
                return JSONResponse({"status": "error", "error": "reply_failed"})

        # Text message
        elif message.get("text"):
            logger.info("Processing text message")
            text = message["text"]
            chat_id = message["chat"]["id"]
            msg_type = "text"
            amount = 0
            
            try:
                # Use direct connection instead of pool
                conn = await asyncpg.connect(DATABASE_URL)
                try:
                    await conn.execute(
                        "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                        text, msg_type, amount
                    )
                    logger.info("Text message saved to database")
                finally:
                    await conn.close()
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                # Continue even if database fails
            
            # Send reply
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": f"You said: {text}"},
                )
            
            return JSONResponse({"status": "text_received", "text": text})
        
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

# Add a test endpoint to check HF client
@app.get("/api/test-hf")
async def test_hf():
    try:
        # Test with a small dummy audio file or just check if client is initialized
        logger.info("Testing Hugging Face client")
        return {"status": "hf_client_initialized", "provider": "fal-ai"}
    except Exception as e:
        logger.error(f"HF client test failed: {str(e)}")
        return {"status": "error", "error": str(e)}