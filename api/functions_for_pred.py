import json
import numpy as np
import re
import unicodedata
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
MODEL_DIR = os.path.join(BASE_DIR, "lightweight_model")

# --- Load exported model ---
def load_model(output_dir):
    # TF-IDF
    with open(f"{output_dir}/tfidf_data.json", 'r', encoding='utf-8') as f:
        tfidf_data = json.load(f)
    
    tfidf_vocab = tfidf_data['vocabulary']
    tfidf_idf = np.array(tfidf_data['idf'])
    
    # Embeddings
    embedding_matrix = np.load(f"{output_dir}/embeddings.npy")
    with open(f"{output_dir}/embeddings_data.json", 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)
    vocab_to_idx = embeddings_data['vocab_to_idx']
    vector_size = embeddings_data['vector_size']
    
    # Classifier
    with open(f"{output_dir}/classifier_data.json", 'r', encoding='utf-8') as f:
        clf_data = json.load(f)
    coef = np.array(clf_data['coef'])
    intercept = np.array(clf_data['intercept'])
    classes = clf_data['classes']
    
    # Categories
    with open(f"{output_dir}/categories.json", 'r', encoding='utf-8') as f:
        categories = json.load(f)
    
    return {
        'tfidf_vocab': tfidf_vocab,
        'tfidf_idf': tfidf_idf,
        'embedding_matrix': embedding_matrix,
        'vocab_to_idx': vocab_to_idx,
        'vector_size': vector_size,
        'coef': coef,
        'intercept': intercept,
        'classes': classes,
        'categories': categories
    }

# --- Preprocessing ---

def preprocess_spanish_text(text):
    text = str(text).lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn' or c in 'ñü')
    text = re.sub(r'[^a-záéíóúüñ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- TF-IDF transform using saved vocab ---
def tfidf_transform(texts, tfidf_vocab, tfidf_idf):
    features = np.zeros((len(texts), len(tfidf_vocab)))
    word_to_index = tfidf_vocab
    idf = tfidf_idf
    
    for i, text in enumerate(texts):
        counts = {}
        for word in text.split():
            if word in word_to_index:
                idx = word_to_index[word]
                counts[idx] = counts.get(idx, 0) + 1
        max_count = max(counts.values(), default=1)
        for idx, count in counts.items():
            features[i, idx] = (count / max_count) * idf[idx]
    return features

# --- Embedding features ---
def get_text_embedding(text, embedding_matrix, vocab_to_idx, vector_size=100):
    vectors = []
    for word in text.split():
        idx = vocab_to_idx.get(word, 0)  # UNK
        vectors.append(embedding_matrix[idx])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# --- Logistic regression prediction ---
def predict(X, coef, intercept):
    logits = X @ coef.T + intercept
    probs = 1 / (1 + np.exp(-logits))
    
    # Multi-class (one-vs-rest)
    if probs.shape[1] > 1:
        preds = np.argmax(probs, axis=1)
    else:
        preds = (probs >= 0.5).astype(int).flatten()
    return preds

# --- Inference example ---

def predict_category(text: str):
    
    model = load_model(MODEL_DIR)
    
    # Preprocess
    text_proc = preprocess_spanish_text(text)
    
    # TF-IDF features
    tfidf_features = tfidf_transform([text_proc], model['tfidf_vocab'], model['tfidf_idf'])
    
    # Embedding features
    emb_features = get_text_embedding(text_proc, model['embedding_matrix'], model['vocab_to_idx'], model['vector_size'])
    emb_features = emb_features.reshape(1, -1)  # make it 2D to match hstack
    
    # Combine features
    X_combined = np.hstack([tfidf_features, emb_features])
    
    # Predict
    pred = predict(X_combined, model['coef'], model['intercept'])[0]
    pred_label = model['categories']['label_names'][pred]
    return pred_label

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
                "INSERT INTO telegram_messages (text, type, amount) VALUES ($1, $2, $3)",
                text, category, amount
            )
            logger.info("Text message saved to database")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")

async def process_text_message(text: str, chat_id: int):
    text_new, amount = separar_texto_valor(text)
    # Predict category
    category = predict_category(text_new)
    
    # Save text + category to DB
    await save_text_to_db(text = text_new, category = category, amount = amount)
    
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
