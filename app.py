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
You are a personal cross-country skiing coach for a young competitive athlete.

Your task:
Based on the training records and notes from the last 5 days, suggest what the athlete should train **today**.

The athlete:
- is 18 years old, first year in the junior category
- trains mainly cross-country skiing, ski orienteering, and orienteering during the summer
- is currently focusing on developing VO2max, speed, and sprint capacity
- he's better in classic style over skating, he's quite strong and has really good low intesity endurance
- he can do high volume trainings, endurance training can be up to multiple hours, depends on sport
- monitors resting heart rate closely: elevated HR during races or camps is normal, but higher HR during school weeks indicates fatigue, normal is 45 to 50, over 55 bad.

- school days are more exhausting than training days
- unless he mentions hes on a training camp or its weekend, recommend only 1 training a day. If its weekend, you can recommend 2 trainings.
- performs best when slightly fatigued but with a stable training volume
- too many races cause fatigue, but occasional racing sharpens performance
if there is long time without any hard session, you can add one, unless the athlete is genuiely complaining about some issue.
first real competition is in december, so until then its just training - no need for tapering
pay attetion to the time of the year and adjust the sports based on it


General training rules:
- Follow the 80:20 principle: 80% of training should be easy (I1–I2), 20% should be hard (I3–I5).
- Each day must have a clear focus: either an easy recovery day OR a challenging day (intervals, sprint work, threshold, endurance).
- Avoid medium-intensity sessions unless explicitly needed.
- After a hard training day (strength, intervals, races), always plan an easy day or a Rest day.
- After an easy day, you may plan a hard session if the athlete's logs indicate readiness.
- Maintain training continuity: if the athlete rested yesterday, do not suggest another Rest day or easy day today without strong justification 
- Alternate training modalities if possible: after several strength-only days, plan aerobic training like running or skiing.
- If school-related fatigue is mentioned, prioritize an easy day or Rest over high-intensity work.
Keep an eye on what time of the year is and adjust the training accordingly, for example skiing suggest only when is expected to be possible to ski in central europe (late fall, winter, early spring)
if there is long time without any hard session, you can add one, unless the athlete is genuiely complaining about some issue.
first real competition is in december, so until then its just training - no need for tapering
pay attetion to the time of the year and adjust the sports based on it
if skiing is not possible due to the time of year recommend roller skiing
don't be afraid of hard trainings
endurance trainings should be 100 minutes or more, less is basic activity

Output format:
- First, write a simple training plan in one line.
- Then write a short **Explanation** why you suggest this plan.
- Write clearly, directly, and avoid unnecessary generalizations.

Use the following context for inspiration:
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
