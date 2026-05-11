import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from supabase.client import create_client

# 1. Cargar variables de entorno
load_dotenv()

def query_agent(question):
    # Configuración de clientes
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
    
    # 2. Generar embedding de la pregunta
    print(f"Pregunta: {question}")
    print("Generando embedding...")
    query_embedding = embeddings.embed_query(question)
    
    # 3. Buscar documentos similares usando RPC directamente
    print("Buscando documentos relevantes...")
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": 3
        }
    ).execute()
    
    # Convertir resultados a documentos de LangChain
    docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
            for doc in result.data]
    
    print(f"✅ Encontrados {len(docs)} documentos relevantes")
    
    # 4. Configurar el LLM y generar respuesta
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = ChatPromptTemplate.from_template("""
        Responde a la siguiente pregunta basándote únicamente en el contexto proporcionado:
        <context>
        {context}
        </context>
        Pregunta: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 5. Ejecutar consulta
    print("Generando respuesta...")
    response = document_chain.invoke({"context": docs, "input": question})
    
    print(f"\n{'='*80}")
    print(f"Respuesta: {response}")
    print(f"{'='*80}\n")
    
    return response

if __name__ == "__main__":
    query = "¿Qué es el RAG y cuáles son sus beneficios?"
    query_agent(query)
