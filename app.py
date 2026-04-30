import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, origins=['http://127.0.0.1:5000', 'http://localhost:5000'])

client = OpenAI(
    api_key="sk-9946d7c4b397427681466a5c9a98934b",   # <-- REEMPLAZA
    base_url="https://api.deepseek.com"
)

# Conectar a ChromaDB (misma carpeta que antes)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("cem_turo")

def buscar_contexto(pregunta, top_k=5):
    resultados = collection.query(query_texts=[pregunta], n_results=top_k)
    return resultados['documents'][0] if resultados['documents'] else []

@app.route('/')
def serve_index():
    return send_from_directory('', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.json
        user_message = data.get('message', '')

        fragmentos = buscar_contexto(user_message, top_k=4)
        contexto = "\n\n".join(fragmentos) if fragmentos else "No se encontró información específica en la web."

        system_prompt = f"""
Eres un asistente virtual del CEM El Turó, un complejo deportivo municipal.
Responde en el mismo idioma que el usuario (catalán o castellano).

Utiliza la siguiente información extraída de la página web del CEM El Turó para responder.
Si la información no está en el contexto, di que no la tienes disponible y sugiere contactar por teléfono o visitar la web.

CONTEXTO DE LA WEB:
{contexto}

Normas:
- Sé amable, breve y útil.
- No inventes datos; ciñete al contexto.
- Si el contexto no es suficiente, recomienda llamar al 93 545 15 50 o visitar www.cemelturo.cat.
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        bot_response = response.choices[0].message.content
        json_response = jsonify({"response": bot_response})
        json_response.headers.add('Access-Control-Allow-Origin', '*')
        return json_response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)