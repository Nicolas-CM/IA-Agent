import streamlit as st
import os
import sys
import asyncio
from dotenv import load_dotenv

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from supabase.client import create_client

# Cargar entorno
load_dotenv()

st.set_page_config(page_title="Agente RAG + Email MCP", page_icon="🤖")

# --- HERRAMIENTAS ---


@tool
def consultar_base_conocimientos(query: str) -> str:
    """
    Consulta la base de conocimientos (Supabase) para responder preguntas sobre temas específicos
    que el asistente no conozca. Úsala para RAG.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", task_type="retrieval_query"
    )
    query_embedding = embeddings.embed_query(query)

    result = supabase.rpc(
        "match_documents",
        {"query_embedding": query_embedding, "match_threshold": 0.5, "match_count": 3},
    ).execute()

    if not result.data:
        return "No se encontró información relevante en la base de datos."

    context = "\n\n".join([doc["content"] for doc in result.data])
    return f"Información encontrada:\n{context}"


# --- INTEGRACIÓN MCP MEJORADA ---


def get_mcp_tools():
    """
    Carga las herramientas MCP de forma síncrona simplificada para evitar errores de __aenter__
    """
    import inspect
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_mcp_adapters.sessions import StdioConnection, StdioServerParameters

    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "email_server.py")
    )
    server_config = {
        "transport": "stdio",
        "command": sys.executable,
        "args": [server_script],
    }
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    # Usamos un enfoque de "fire and forget" para la conexión en el contexto de Streamlit
    # o intentamos cargar las herramientas directamente si la librería lo permite
    loop = None
    try:
        # Nota: load_mcp_tools es asíncrona, necesitamos un loop
        async def fetch():
            # API actual: load_mcp_tools acepta connection con transport
            sig = inspect.signature(load_mcp_tools)
            if "connection" in sig.parameters:
                return await load_mcp_tools(
                    session=None, connection=server_config, server_name="email"
                )

            # Preferimos el cliente multi-servidor cuando esta disponible
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
            except Exception:
                MultiServerMCPClient = None

            if MultiServerMCPClient is not None:
                client = MultiServerMCPClient({"email": server_config})
                return await client.get_tools()

            # API previa: acepta configuracion de servidores con transporte
            if "servers" in sig.parameters:
                return await load_mcp_tools(servers=[server_config])
            if "server" in sig.parameters:
                return await load_mcp_tools(server=server_config)

            conn_mgr = StdioConnection(server_params)

            # Compatibilidad entre versiones que exponen context manager async o sync
            if hasattr(conn_mgr, "__aenter__") and hasattr(conn_mgr, "__aexit__"):
                async with conn_mgr as conn:
                    return await load_mcp_tools(session=None, connection=conn)

            if hasattr(conn_mgr, "__enter__") and hasattr(conn_mgr, "__exit__"):
                with conn_mgr as conn:
                    return await load_mcp_tools(session=None, connection=conn)

            return await load_mcp_tools(session=None, connection=conn_mgr)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tools = loop.run_until_complete(fetch())
        return tools
    except Exception as e:
        st.sidebar.error(f"Error cargando herramientas MCP: `{e}`")
        return []
    finally:
        if loop is not None:
            try:
                loop.close()
            except Exception:
                pass


# Cacheamos las herramientas para no reiniciar el servidor MCP en cada recarga de Streamlit
if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = get_mcp_tools()

all_tools = [consultar_base_conocimientos] + st.session_state.mcp_tools

# --- CONFIGURACIÓN DEL AGENTE ---


def get_agent_executor():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente experto con acceso a una base de conocimientos y herramientas de correo electrónico.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, all_tools, prompt)
    return AgentExecutor(agent=agent, tools=all_tools, verbose=True)


def run_async(coro):
    """Run async code from a sync context (Streamlit)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- INTERFAZ STREAMLIT ---

st.title("🤖 Agente RAG Avanzado (MCP + Email)")
st.markdown(
    "Ahora puedo responder tus dudas y **enviar correos electrónicos** usando MCP."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Entrada de usuario
if user_input := st.chat_input("¿En qué puedo ayudarte hoy?"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            executor = get_agent_executor()

            # Convertir historial para LangChain
            history = st.session_state.messages[:-1]

            try:
                response = run_async(
                    executor.ainvoke({"input": user_input, "chat_history": history})
                )

                final_text = response["output"]
                st.markdown(final_text)
                st.session_state.messages.append(AIMessage(content=final_text))
            except Exception as e:
                st.error(f"Error en el agente: {e}")

with st.sidebar:
    st.header("Configuración")
    if st.button("Limpiar Conversación"):
        st.session_state.messages = []
        if "mcp_tools" in st.session_state:
            del st.session_state.mcp_tools
        st.rerun()

    st.divider()
    st.info(f"Herramientas activas: {len(all_tools)}")
    for t in all_tools:
        st.write(f"- {t.name}")
