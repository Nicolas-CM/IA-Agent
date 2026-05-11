import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# 1. Cargar variables de entorno
load_dotenv()


def ingest_docs():
    # Configuración de Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if not all([supabase_url, supabase_key, google_api_key]):
        print(
            "Error: Faltan credenciales en el archivo .env (SUPABASE o GOOGLE_API_KEY)"
        )
        return

    supabase = create_client(supabase_url, supabase_key)

    # 2. Cargar el documento
    file_path = "conocimiento.txt"
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        return

    loader = TextLoader(file_path)
    documents = loader.load()

    # 3. Fragmentación (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Documento dividido en {len(docs)} fragmentos.")

    # 4. Embeddings de Google (Dimensión 768)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", api_key=google_api_key
    )

    print("Subiendo vectores a Supabase con Gemini...")
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )
    print("¡Ingesta completada con éxito usando Google Gemini!")


if __name__ == "__main__":
    ingest_docs()
