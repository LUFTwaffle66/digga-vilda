from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import json
import traceback
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

# ──────────────── FLASK SETUP ────────────────
app = Flask(__name__)

# Globální CORS pro všechna URL na tvé Netlify doméně
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(
    app,
    resources={r"/*": {"origins": "https://cosmic-crostata-1c51df.netlify.app"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# ──────────────── API KEY & MODELS ────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=API_KEY)

def generate_from_gemini(prompt: str) -> str:
    try:
        resp = genai.generate_text(
            model="gemini-1.5-flash",
            prompt=prompt
        )
        return resp.text
    except Exception:
        app.logger.error("Gemini error:\n" + traceback.format_exc())
        raise

# Embedding model (Seznam/retromae-small-cs)
embedding_tokenizer = AutoTokenizer.from_pretrained("Seznam/retromae-small-cs")
embedding_model     = AutoModel.from_pretrained("Seznam/retromae-small-cs")

def get_embedding(text: str) -> np.ndarray:
    with torch.no_grad():
        inputs  = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = embedding_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        normed  = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
    return normed.cpu().numpy().astype("float32")

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

        # Načtení FAISS indexu a chunks jen jednou
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

        # System prompt
        system_prompt = f"""
Jsi osobní AI trenér běžeckého lyžování. Tvým klientem je mladý výkonnostní sportovec, který:
- má 18 let a je v první sezóně v juniorské kategorii
- věnuje se běžeckému lyžování, orientačnímu běhu a ski orienťáku
- aktuálně rozvíjí VO2max, rychlost a sprintové schopnosti
- technicky mu více sedí klasika než bruslení
- klidový tep sleduje jako indikátor regenerace, běžně se pohybuju 45 až 50, 55 a víc už není dobré
- školní dny ho vyčerpávají více než trénink
- nejlepší výkony podává po vyšším objemu tréninku

Tvým úkolem je:
Na základě záznamů o tréninku za posledních 5 dní a poznámek navrhnout, co má sportovec trénovat **dnes**.

Zohledni:
- rozdělení intenzit (I1–I5)
- čas, vzdálenost, poznámky a únavu
- potřebu střídání těžkých a lehkých dnů
- aktuální regeneraci podle poznámek a HR
- VO2max, sprintové zaměření
neboj se dát rest

Výstup:
Napiš pouze jednoduchý plán, například:
V - klus i2 45' + 3×100, na vyklepání nohou

Dodej lehké vysvětlení

Kontext tréninků:
{context}
"""
        recommendation = generate_from_gemini(system_prompt)
        return jsonify({"recommendation": recommendation})

    except Exception:
        app.logger.error("Endpoint /ask error:\n" + traceback.format_exc())
        return jsonify({"error": "Vnitřní chyba serveru, mrkni do logů."}), 500

# ──────────────── RUN FOR RENDER ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
