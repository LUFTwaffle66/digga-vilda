@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks

    data = request.get_json()
    logs = data.get("logs", [])

    if not logs:
        return jsonify({"recommendation": "Chyba: žádné logy nepřišly."}), 400

    query = ""
    for log in logs:
        query += f"{log['date']}: {log['activity']} {log['duration']}min {log['distance_km']}km i1:{log['i1']} i2:{log['i2']} i3:{log['i3']} i4:{log['i4']} i5:{log['i5']} poznámka: {log['note']}\n"

    if index is None:
        faiss_path = os.path.join(os.path.dirname(__file__), "faiss.index")
        index = faiss.read_index(faiss_path)
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    query_embedding = get_embedding(query)
    D, I = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

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

    try:
        response = gemini_model.generate_content(system_prompt)
        return jsonify({"recommendation": response.text})
    except Exception as e:
        return jsonify({"recommendation": f"Chyba: {e}"})
