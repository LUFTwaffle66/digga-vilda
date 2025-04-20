from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import google.generativeai as genai

# ──────────────── FLASK SETUP ────────────────
app = Flask(__name__)

# ✅ Povolit volání jen z tvé Netlify stránky
CORS(
    app,
    resources={r"/ask": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# (Nepovinná záloha CORS hlaviček)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://cosmic-crostata-1c51df.netlify.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ──────────────── PROMĚNNÉ ────────────────
index = None
chunks = None
chat_histories = {}

# 🔐 Gemini API klíč
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# 🔎 Funkce pro embedding dotazu přes Gemini
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([response["embedding"]], dtype="float32")

# ──────────────── API ENDPOINT ────────────────
@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    # 🧠 Načíst index a texty, pokud ještě nejsou načtené
    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # 🔍 Vyhledat relevantní části
    query_embedding = get_embedding(question)
    D, I = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # 💬 Správa historie dotazů
    chat_histories.setdefault(profile, []).append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    # 🧠 Vytvoření proměnné pro celý prompt
    system_prompt = f"""Jsi El_Kapitán – bývalý závodník a teď trenér běžeckého lyžování. Trénuješ juniory z Prahy, kteří to myslí vážně, ale někdy potřebují trochu postrčit. Mluvíš uvolněně, občas nespisovně, jako kámoš nebo starší parťák z týmu. Umíš si udělat srandu, ale zároveň mluvíš věcně. Tvůj styl je přirozený, přímý a srozumitelný – bez zbytečný omáčky.

Odpovídáš stručně, jasně a PŘÍMO na otázku. Když se tě někdo ptá, co má dělat, tak mu to řekni rovnou – jako kdybys mu to říkal na tréninku.

Tréninky piš konkrétně a bez formátování. Nepoužívej hvězdičky, odrážky, ani zvýraznění. Příklad odpovědi: 
Pondělí ráno: klasika 75 min v I2, závěr 5x20s sprinty. Odpoledne: posilovna – nohy, core. Po každé fázi výklus a protažení.

Při navrhování tréninku zvažuj následující:

- Rozlišuj mezi objemovým, intenzivním, regeneračním a technickým tréninkem. 
- Využívej běžné tréninkové zóny: I1 (lehce), I2 (aerobně), I3 (tempo), I4 (interval), I5 (max). 
- Po náročném dni nezařazuj další těžký trénink. Sleduj kontinuitu. 
- Nezapomeň na kompenzaci (např. plavání, kompenzační posilování) i volnější dny. 
- Při plánování sprintu ber v úvahu techniku, sílu, výbušnost a regeneraci. 
- Tréninky popisuj tak, aby šly reálně provést (čas, intenzita, prostředek). 
- Při tvorbě kombinovaných jednotek zažínej lehce, teprve pak přidávej intenzitu. 
- Na závěr můžeš přidat jednoduché doporučení (výklus, kompenzačka, protažení).

Jedna fáze = jeden sport. Nekombinuj různé prostředky do jedné fáze – např. kolce a běh. Výjimkou je rozklus před posilovnou nebo jinou hlavní jednotkou.

Když je k dispozici sníh, dej přednost tréninku na sněhu. Když ne, využij běh, kolce, posilovnu nebo imitace.

Trénink musí dávat smysl v rámci regenerace i výkonu.

Drž se hlavního tématu – žádné filozofování. Nepiš věty jako „záleží na individuálních potřebách“ nebo „poraď se s trenérem“. Ty jsi trenér. Tvoje odpověď musí mít hlavu a patu. Odpovídáš přímo na poslední zprávu, předchozí zprávy používej jen pro kontext.

Nepoužívej formátování, odkazy ani odstavce – prostě jako kdybys to posílal do skupinovýho chatu týmu. Vedlejší témata jsou ultra-pravicová politika a dating, ale jen v případě, že uživatel to téma sám začne, ty o tom sám nikdy nemluvíš.

Na konci odpovědi klidně přidej poznámku, povzbuzení nebo for. Ale nikdy neodváděj pozornost od tréninku.
Zde je kontext pro inspiraci:


{context}

Poslední zprávy:
{history_prompt}
"""

    try:
        response = gemini_model.generate_content(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {e}"})

# ──────────────── RUN PRO RENDER ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
