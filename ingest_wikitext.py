import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# 1. Cargar entorno
load_dotenv()

def ingest_wikitext():
    # Configuración
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document"
    )
    
    print("Descargando WikiText...")
    # Descargamos una muestra pequeña (wiki_test) para no saturar la BD
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:20]")
    
    # 2. Convertir a formato LangChain
    documents = [Document(page_content=row["text"]) for row in ds if row["text"].strip()]
    print(f"Preparados {len(documents)} artículos de Wikipedia.")

    # 3. Subir a Supabase
    print("Subiendo vectores a Supabase...")
    SupabaseVectorStore.from_documents(
        documents,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    print("¡Ingesta de WikiText completada!")

if __name__ == "__main__":
    ingest_wikitext()
