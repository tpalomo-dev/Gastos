import os
import logging
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import sys
sys.path.append(os.path.dirname(__file__))
from functions_for_pred import process_voice_message, process_text_message, format_summaries_as_table

# Log a startup message immediately
# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("telegram_webhook")

# ---------- FastAPI App ----------
app = FastAPI()
# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    logger.info("where are my logs dude")
    
@app.post("/telegram_webhook")
async def telegram_webhook(req: Request):
    try:
        # Log immediately that the webhook endpoint was called
        logger.info("Telegram webhook endpoint called.")

        # Read JSON payload
        data = await req.json()
        logger.info(f"Received webhook data: {data}")

        message = data.get("message")
        if not message:
            logger.info("No message field in payload.")
            return JSONResponse({"status": "no_message"})

        # Handle voice messages
        if message.get("voice"):
            logger.info("Processing voice message.")
            return await process_voice_message(message)

        # Handle text messages
        elif message.get("text"):
            text = message["text"]
            chat_id = message["chat"]["id"]

            logger.info(f"Received text message: {text}")

            if text == "Reporte":
                logger.info("Generating report for chat_id: %s", chat_id)
                return await format_summaries_as_table(chat_id)
            else:
                logger.info("Processing regular text message for chat_id: %s", chat_id)
                return await process_text_message(text, chat_id)

        # Unknown message type
        else:
            logger.warning(f"Unknown message type: {message.keys()}")
            return JSONResponse({"status": "unknown_message_type"})

    except Exception as e:
        logger.error(f"Unhandled error in webhook: {str(e)}", exc_info=True)
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
