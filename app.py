from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import json
import traceback
import numpy as np
import faiss
import requests
import google.generativeai as genai

# ──────────────── FLASK SETUP ────────────────
app = Flask(__name__)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        h = resp.headers
        h["Access-Control-Allow-Origin"] = "https://cosmic-crostata-1c51df.netlify.app"
        h["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        h["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

CORS(app, origins=["https://cosmic-crostata-1c51df.netlify.app"])

# ──────────────── API KEYS & MODELS ────────────────
GROG_API_KEY = os.getenv("GROG_API_KEY")
if not GROG_API_KEY:
    raise RuntimeError("GROG_API_KEY environment variable is not set")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)

# Embedding přes Google models/embedding-001
def get_embedding(text: str) -> np.ndarray:
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([response["embedding"]], dtype="float32")

# Call Grog API
def call_llama(system_prompt: str, user_message: str) -> str:
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        app.logger.error("Grog API error:\n" + traceback.format_exc())
        raise

# Cached FAISS index and text chunks
index = None
chunks = None

# ──────────────── ENDPOINT ────────────────
@app.route("/ask", methods=["OPTIONS", "POST"])
@cross_origin(origin="https://cosmic-crostata-1c51df.netlify.app")
def ask():
    global index, chunks
    try:
        data = request.get_json(force=True)
        logs = data.get("logs", [])
        if not logs:
            return jsonify({"error": "Chyba: žádné logy nepřišly."}), 400

        # Sestavení dotazu z logů
        query = ""
        for log in logs:
            query += (
                f"{log['date']}: {log['activity']} {log['duration']}min {log['distance_km']}km "
                f"i1:{log['i1']} i2:{log['i2']} i3:{log['i3']} i4:{log['i4']} i5:{log['i5']} "
                f"poznámka: {log['note']}\n"
            )

        # Načtení FAISS indexu a chunks
        if index is None:
            base = os.path.dirname(__file__)
            index = faiss.read_index(os.path.join(base, "faiss.index"))
            with open(os.path.join(base, "chunks.json"), "r") as f:
                chunks = json.load(f)

        # Retrieval
        emb = get_embedding(query)
        D, I = index.search(emb, k=5)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(relevant_chunks)

        # System prompt pro Diggu
        system_prompt = f"""
Jsi osobní AI trenér běžeckého lyžování. Tvým klientem je mladý výkonnostní sportovec, který:
- má 18 let a je v první sezóně v juniorské kategorii
- věnuje se hlavně běžeckému lyžování, ski orienťáku a přes léto orientačnímu běhu
- aktuálně rozvíjí VO2max, rychlost a sprintové schopnosti
- technicky mu více sedí klasika než bruslení
- klidový tep sleduje pečlivě: zvýšený HR během závodů nebo soustředění je normální, ale ve školním týdnu je signál únavy
- školní dny ho vyčerpávají víc než trénink
- nejlépe závodí po stabilním objemu tréninků i za cenu lehké únavy
- moc závodů ho vyčerpává, ale rozzávodění pomáhá

Tvým úkolem je:
Na základě záznamů o tréninku za posledních 5 dní a poznámek navrhnout, co má sportovec trénovat **dnes**.

Zohledni:
- rozdělení intenzit (I1–I5)
- čas, vzdálenost, poznámky a únavu
- potřebu střídání těžkých a lehkých dnů
- VO2max, sprintové zaměření
- regeneraci podle poznámek a HR
- neboj se dát Rest den

Výstup:
Napiš pouze jednoduchý plán, například:
V - klus i2 45' + 3×100, na vyklepání nohou

Dodej lehké vysvětlení.

Kontext tréninků:
{context}
"""

        # Volání Llamy
        recommendation = call_llama(system_prompt, query)
        return jsonify({"recommendation": recommendation})

    except Exception:
        app.logger.error("Endpoint /ask error:\n" + traceback.format_exc())
        return jsonify({"error": "Vnitřní chyba serveru, mrkni do logů."}), 500

# ──────────────── RUN FOR RENDER ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
