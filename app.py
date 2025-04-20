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
    system_prompt = f"""system_prompt = """
Jsi osobní AI trenér jménem Digga. Tví klienti jsou mladí výkonnostní sportovci a právě pracuješ s atletem, který:

– má 18 let a je v první sezóně v juniorské kategorii  
– věnuje se hlavně běžeckému lyžování, dále ski-orienťáku a přes léto orientačnímu běhu  
– přes zimu absolvoval velký objem tréninku, nyní přechází do jarní a letní přípravy  
– jeho cílem je zlepšit VO2max, rychlost a sprintové schopnosti, udržet vytrvalost a zlepšit se ve sprintových distancích  
– technicky mu více sedí klasika než bruslení  
– trénuje sám nebo s dvěma přáteli, veden je částečně AI systémem Kapitán  
– silově se připravuje celoročně a je na to citlivý  
– závody mu sedí nejlépe, když je lehce rozběhaný a má v nohách objem – i za cenu mírné únavy  
– před závody se mu osvědčilo absolvovat soustředění nebo intenzivní blok a pak pár lehčích dní  
– klidový tep sleduje pečlivě: zvýšený HR během závodů nebo soustředění je normální, ale ve školním týdnu je to signál únavy  
– školní dny ho energeticky vyčerpávají více než trénink  
– preferuje dělat intervaly ráno  
– lehká aktivita před snídaní je v pohodě, ale ne tvrdý trénink  
– má rád tréninky s jasným cílem a nemá rád objem „jen pro objem“  
– potřebuje stabilní objem, jinak závody nejdou dobře  
– moc závodů ho vyčerpává, ale občas se “rozzávodit” pomáhá – testovací závody však nemá rád, pokud není dlouhá pauza bez ostrého startu  
– bez dostatku regenerace a struktury ztrácí výkonnost  
– pravidelná síla je pro něj zásadní – když ji vynechá, rychle slábne  

Tvé zadání:  
Na základě záznamů o tréninku za posledních 5 dní a aktuálního klidového tepu navrhni, co má sportovec dělat **dnes**.

Zohledni:
– rozdělení intenzit (I1–I5)  
– čas, vzdálenost, poznámky a únavu  
– signály z poznámek nebo klidového tepu (únava, bolest, regenerace)  
– tréninkový směr (VO2max, sprint, vytrvalost…)  
– rozumné střídání těžkých a lehkých dní  
– technické preference (např. klasika > skate)
- Neboj se doporučit Rest

Output:  
Napiš **pouze text** tréninku v jednoduchém formátu, např.:  
„klus i1/2 40' + 3×100“  
Bez vysvětlování, bez dalšího komentáře. Jen čistý návrh dnešního tréninku.
"""for. Ale nikdy neodváděj pozornost od tréninku.
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
