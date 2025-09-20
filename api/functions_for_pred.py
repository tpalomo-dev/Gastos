import numpy as np
import re
import os
import aiohttp
from fastapi.responses import JSONResponse
import asyncpg
from huggingface_hub import InferenceClient
import logging
logger = logging.getLogger(__name__)
import sys
sys.path.append(os.path.dirname(__file__))
from rescatar_valor_numerico import separar_texto_valor

# Build paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model/mlp_embeddings_model.npz")
EMB_DIR = os.path.join(BASE_DIR, "model/combined_embeddings.vec")

# ---- Utilities ----

def preprocess_spanish_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def relu(x): return np.maximum(0, x)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def load_local_spanish_embeddings(path=EMB_DIR):
    embeddings = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embeddings[word] = vector
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None


def create_embedding_matrix(embeddings_dict):
    vector_size = len(next(iter(embeddings_dict.values())))
    vocab_to_idx = {'<UNK>': 0}
    idx_to_vocab = {0: '<UNK>'}
    embedding_matrix = [np.random.normal(0, 0.1, vector_size)]
    for i, (word, vector) in enumerate(embeddings_dict.items(), 1):
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word
        embedding_matrix.append(vector)
    return np.array(embedding_matrix), vocab_to_idx, idx_to_vocab, vector_size

def get_text_embedding(text, embedding_matrix, vocab_to_idx, vector_size):
    words = text.split()
    vectors = [embedding_matrix[vocab_to_idx.get(word, 0)] for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# ---- Classifier ----

class SpanishTextClassifier:
    def __init__(self, model_path=MODEL_DIR, emb_path=EMB_DIR):
        # Load model weights
        data = np.load(model_path, allow_pickle=True)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.label_names = data["label_names"]

        # Load embeddings
        embeddings = load_local_spanish_embeddings(emb_path)
        self.embedding_matrix, self.vocab_to_idx, self.idx_to_vocab, self.vector_size = create_embedding_matrix(embeddings)

    def predict(self, text: str) -> str:
        text = preprocess_spanish_text(text)
        emb = get_text_embedding(text, self.embedding_matrix, self.vocab_to_idx, self.vector_size)
        x = emb.reshape(1, -1)
        h = relu(x @ self.W1 + self.b1)
        out = softmax(h @ self.W2 + self.b2)
        pred_idx = int(np.argmax(out, axis=1)[0])
        return self.label_names[pred_idx]

# ---- Singleton instance ----
_classifier_instance = None

def get_classifier() -> SpanishTextClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SpanishTextClassifier()
    return _classifier_instance

# ---- Function to keep your old interface ----
def predict_category(text: str) -> str:
    classifier = get_classifier()
    return classifier.predict(text)

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
    headers={"Content-Type": "audio/ogg"})

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


async def save_text_to_db(text: str, category: str, chat_id, amount: int = 0):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute(
                "INSERT INTO gastos_db (gasto, tipo_de_gasto, monto) VALUES ($1, $2, $3)",
                text, category, amount
            )
            await send_telegram_message(chat_id, "no problem with the writing")
        finally:
            await conn.close()
            await send_telegram_message(chat_id, "conn closed")
    except Exception as e:
        send_telegram_message(chat_id, f"Database error: {str(e)}")

async def process_text_message(text: str, chat_id: int):
    text_new, amount = separar_texto_valor(text)
    # Predict category
    category = predict_category(text_new)
    
    # Save text + category to DB
    await save_text_to_db(text = text_new, category = category, chat_id=chat_id, amount = amount)
    
    # Send reply to Telegram
    await send_telegram_message(chat_id, f"{text}, transformado a {text_new} con categoría {category} y precio {amount}")
    
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

    # Predict category and Save transcription + category to DB
    return await process_text_message(transcription, chat_id)