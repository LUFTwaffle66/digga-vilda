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
Based on the last 5 days’ training and notes, suggest today’s training.

The athlete:
	•	18 years old, 1st year junior.
	•	Trains cross-country skiing (main), ski orienteering (winter), orienteering (summer).
	•	Current focus: VO₂max, sprint speed, top-end capacity.
	•	Strong in classic technique; excellent low-intensity endurance; durable for high volumes.
	•	Resting HR: 45–50 normal; >55 during school = fatigue warning.
	•	If resting HR is consistently low (45 or less) and athlete feels good, training load can be increased.
	•	School days are more exhausting than training days.
	•	Unless weekend or training camp, recommend only 1 session daily; on weekends, 2 sessions allowed.
	•	Performs best when slightly fatigued but training volume is stable.
	•	Racing sharpens form but frequent racing causes overload.
	•	First real competitions in December – no tapering needed.
	•	In winter (late fall–early spring), prioritize skiing. Else, recommend roller skiing.

Training strengths and weaknesses:
	•	Very strong in endurance and strength.
	•	Weaker in tempo endurance (I3).
	•	Needs focus on VO₂max development, sprint capacity, and occasional threshold sharpening without slipping into gray zone training.

Training rules:
	•	Follow strict 80% easy (I1–I2) / 20% hard (I3–I5) structure.
	•	Endurance sessions must be long (minimum 100 minutes; ideally 2 hours).
	•	No medium-intensity (I2–I3) “gray zone” sessions unless explicitly targeted (e.g., threshold intervals).
	•	After a hard day (strength, intervals, race), plan easy day or Rest.
	•	After an easy day, suggest hard training unless clear signs of heavy fatigue are reported.
	•	Minor fatigue is acceptable; important sessions (VO₂max, sprints) must still be performed if no injury/illness.
	•	Endurance is the foundation: easy endurance sessions must be real training, not short recovery jogs.
	•	Strength sessions should not replace aerobic volume for multiple consecutive days.
	•	Adjust volume based on HR trends:
	•	If HR is low and stable, you can increase volume or intensity.
	•	If HR is high or rising without explanation, reduce intensity or volume.
 recommend mix of skiing (snow/roller), run, bike, strength training, gym, imitation and other activities

Priorities:
	•	Respect school fatigue: if heavy, favor easy endurance or Rest.
	•	Maintain high training consistency: no unnecessary Rest days.
	•	Always alternate training types (e.g., strength, endurance, speed).

Output format:
	1.	Training Plan (one line)
	2.	Explanation (short, direct, linked to training history and current condition)

Style:
	•	Clear, direct, professional.
	•	No general advice.
	•	Always recommend the best possible training for progress based on current status.


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
