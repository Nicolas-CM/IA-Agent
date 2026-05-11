import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Añadir el directorio raíz al path para importar lógica si fuera necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from supabase.client import create_client

# Cargar entorno
load_dotenv()

st.set_page_config(page_title="Agente RAG - Chat", page_icon="🤖")

def get_response(question, chat_history):
    """
    Lógica basada en query_agent.py adaptada para mantener contexto si se desea,
    aunque el prompt original es 'basado únicamente en el contexto'.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        return "Error: Configura SUPABASE_URL y SUPABASE_SERVICE_KEY en el .env"

    supabase = create_client(supabase_url, supabase_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
    
    # 1. Generar embedding de la pregunta
    query_embedding = embeddings.embed_query(question)
    
    # 2. Buscar documentos similares
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": 3
        }
    ).execute()
    
    docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
            for doc in result.data]
    
    if not docs:
        return "No encontré información relevante en mi base de datos para responder a eso."

    # 3. Configurar el LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Usando el mismo modelo que query_agent.py
    
    # Prompt que permite conversación (opcionalmente podrías pasar el historial al prompt)
    prompt = ChatPromptTemplate.from_template("""
        Eres un asistente experto. Responde a la siguiente pregunta basándote únicamente en el contexto proporcionado.
        Si la pregunta requiere seguimiento de la conversación anterior, usa el contexto para mantener la coherencia.
        
        <context>
        {context}
        </context>
        
        Pregunta: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Ejecutar consulta
    response = document_chain.invoke({"context": docs, "input": question})
    return response

# Interfaz de Streamlit
st.title("🤖 Chat con Agente RAG")
st.markdown("Interactúa con tu base de conocimientos en Supabase.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = get_response(prompt, st.session_state.messages)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar para utilidades adicionales (Evaluar)
with st.sidebar:
    st.header("Opciones")
    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("Evaluación Ragas")
    if st.button("Ejecutar Evaluación"):
        # Nota: Aquí se podría importar y llamar a evaluate_with_dataset de evaluate_with_json.py
        # Pero para mantenerlo simple y no modificar archivos base, damos la instrucción.
        st.warning("Para ver métricas detalladas, ejecuta: `python evaluate_with_json.py` en tu terminal.")
