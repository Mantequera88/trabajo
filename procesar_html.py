import os
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Configuración
HTML_FOLDER = "."  # carpeta actual (y subcarpetas)
CHUNK_SIZE = 500  # caracteres por fragmento
OVERLAP = 50      # solapamiento entre fragmentos

# Inicializar el modelo de embeddings
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Conectar a ChromaDB (los datos se guardarán en ./chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Eliminar colección anterior si existe (para refrescar datos)
try:
    chroma_client.delete_collection("cem_turo")
except:
    pass
collection = chroma_client.create_collection(name="cem_turo", embedding_function=embedding_fn)

def extract_text_from_html(filepath):
    """Extrae texto limpio de un archivo HTML."""
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    # Eliminar etiquetas script, style, meta, etc.
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    texto = soup.get_text(separator=' ', strip=True)
    texto = re.sub(r'\s+', ' ', texto)  # espacios múltiples a uno
    return texto

def split_text(text, max_len=CHUNK_SIZE, overlap=OVERLAP):
    """Divide el texto en fragmentos con solapamiento."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_len
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

# Recorrer todos los archivos .html en el directorio actual y subdirectorios
html_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.html'):
            html_files.append(os.path.join(root, file))

print(f"📄 Se encontraron {len(html_files)} archivos HTML.")
for i, filepath in enumerate(html_files):
    print(f"Procesando {filepath}...")
    texto = extract_text_from_html(filepath)
    chunks = split_text(texto)
    # Añadir cada fragmento a la base de datos
    for j, chunk in enumerate(chunks):
        doc_id = f"{filepath}_{j}"
        collection.add(
            documents=[chunk],
            metadatas=[{"source": filepath, "chunk_index": j}],
            ids=[doc_id]
        )

print(f"✅ Base de datos creada con {collection.count()} fragmentos.")