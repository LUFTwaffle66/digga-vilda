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

### Athlete Profile:
name: Vilda
- 18 years old, 1st year in junior category
- Trains mainly cross-country skiing, ski orienteering, and orienteering during the summer
- Current focus: VO₂max, sprint speed, top-end capacity
- Stronger in classic style; very strong in low-intensity endurance
- Handles high training volumes well
- Resting heart rate: 45–50 is normal; >55 during school weeks indicates fatigue
- If resting HR is consistently ≤45 and athlete feels good, training load can be increased
- School days are more exhausting than training days
- Performs best slightly fatigued but with stable training volume
- Racing sharpens performance; too many races cause fatigue
- First real competition is in December – **no tapering needed** until then
- Skiing is only recommended during late fall–winter–early spring; otherwise use roller skiing, running, gym and bike or toher sports

---

###  Available Training Types
(*Only one main training focus per day*)

#### 1. Recovery & Easy Days
- Easy jog (30–50 min)
- Mobility / stretching
- Easy roller skiing / skiing (60–90 min, I1/2)
- Short core session, 50% of overall training time


Use after races, intervals, or strength. Ideal for school stress days.

#### 2. Endurance Training (I1–I2)
- Long easy run (100–150 min)
- Long ski or roller ski (100–150 min)
- Orienteering session (90–120 min)
- Cycling / hike (2h+, in base period), 30% of overall training time

Base of aerobic capacity. Minimum 100 min to count as real endurance.

#### 3. VO₂max Training (I5–I6)
- Running intervals (e.g. 5×4 min @ I5–I6)
- Roller ski intervals (e.g. 5×4 min)
- Hill sprints (10×30s, max effort)
- Ski imitation or bounding (explosive, short), 5 to 10% of overall training time

Include once per 5–7 days if rested enough.

#### 4. Sprint Training
- Max sprints (10–20×10–15s)
- Sprint orienteering
- Sprint starts on roller skis
- Bounding or jumps (10–15 min total load), can part of easy activities

Maintain speed and explosiveness.

#### 5. Threshold (I4) / Tempo Endurance
- 3×8 min threshold intervals (I4 = 170–180 bpm)
- 15–25 min tempo run (steady I4)
- Continuous ski @ I4 – max 30–40 min total, 10% of overall training time

Include 1–2× weekly. Controlled intensity only.

#### 6. Strength Training
- General strength (core, balance)
- Ski-specific strength (pull-ups, bands)
- Max strength (weights)
- Plyometrics

Avoid multiple days in a row.

#### 7. Mixed / Simulation / Double Days
- Race simulation (sprint, time trial)
- Technique-focused ski / roller ski (with drills)
- Double day (e.g., strength AM + endurance PM) – weekends only
do not be afraid to recommend hard session, or rest, avoid too much easy or endurance sessions in one week
recommend exact training, what intesity, how much time, what kind of sport, do not write just general endurance I2 or Vo2 max, 

---

###  Heart rate zones
Zone 1 <140 bpm, Zone 2 <151 bpm, zone 3<165 bpm, zone 4 <180, zone 5 <187, zone 6 187+

---

### Training Rules

- One clear training focus per day
- Follow 80:20 intensity rule (I1–2 vs I3–5)
- Never mix hard + medium intensity
- Always follow hard day with easy/recovery
- No unnecessary Rest days unless clear signs of overload
- Endurance sessions = 100+ minutes unless recovery day
- Tempo training (I3/4) is allowed 1–2× per week with purpose
- Increase volume if HR is stable & low; reduce if HR is high
- Do not avoid hard training unless illness or serious fatigue
recommend only one specific training
follow rest heart rate trend, if it is rising, suggest easier activities, also watch out for notes about HRV 
---

### Output format
1. **Training Plan** (1 line, exact training, what, what sport, how much time, intensity, intervals?)
2. **Explanation** (short, direct)

Use clear, structured, and professional language. Avoid generic advice.

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
