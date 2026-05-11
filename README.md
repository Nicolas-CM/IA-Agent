# Agente RAG con LangChain, Supabase y Ragas

Sistema de Retrieval-Augmented Generation (RAG) con monitoreo y evaluación automática.

## 🏗️ Arquitectura

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Wikipedia  │─────▶│   Supabase   │◀─────│ Query Agent │
│   Dataset   │      │  Vector DB   │      │    (RAG)    │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            │                      ▼
                            │              ┌─────────────┐
                            └─────────────▶│  Evaluate   │
                                           │   (Ragas)   │
                                           └─────────────┘
```

## 📋 Requisitos

- Python 3.10+
- Cuenta de Supabase (con pgvector habilitado)
- API Key de Google Gemini
- API Key de LangSmith (opcional, para monitoreo)

## 🚀 Instalación

1. **Clonar y configurar entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configurar variables de entorno:**
Crea un archivo `.env` con:
```env
GOOGLE_API_KEY=tu_api_key_de_google
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agente-rag-supabase
LANGCHAIN_API_KEY=tu_api_key_de_langsmith
SUPABASE_URL=tu_url_de_supabase
SUPABASE_SERVICE_KEY=tu_service_key_de_supabase
```

3. **Configurar Supabase:**
Ejecuta este SQL en tu proyecto de Supabase:

```sql
-- Habilitar la extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Crear tabla para documentos
CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  content TEXT,
  metadata JSONB,
  embedding VECTOR(768)
);

-- Crear función de búsqueda por similitud
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
$$;

-- Crear índice para búsquedas rápidas
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## 📖 Uso

### 1. Ingestar Datos de Wikipedia

```bash
python ingest_wikitext.py
```

Este script:
- Descarga 20 artículos de WikiText-2
- Los convierte en embeddings usando Google Gemini
- Los almacena en Supabase

### 2. Consultar el Agente RAG

```bash
python query_agent.py
```

Realiza consultas sobre los datos almacenados. Puedes modificar la pregunta en el código.

### 3. Evaluar el Sistema con Ragas

Tienes **3 opciones** para evaluar:

#### Opción A: Dataset Manual (Recomendado para empezar)

```bash
python evaluate.py
```

Usa preguntas hardcodeadas en el código. Debes editar las preguntas y ground_truth según tu contenido.

#### Opción B: Dataset desde JSON (Más flexible) ⭐

```bash
python evaluate_with_json.py
```

1. Edita `evaluation_dataset.json` con tus preguntas y respuestas esperadas
2. Ejecuta el script
3. El sistema genera respuestas automáticamente y las compara con ground_truth

**Ejemplo de `evaluation_dataset.json`:**
```json
{
  "examples": [
    {
      "question": "What is machine learning?",
      "ground_truth": "Machine learning is a subset of AI that enables systems to learn from data."
    }
  ]
}
```

#### Opción C: Generación Sintética con Ragas

Descomenta la sección en `evaluate.py` para que Ragas genere preguntas automáticamente desde tus documentos.

**Métricas evaluadas:**
- **Faithfulness**: ¿La respuesta es fiel al contexto recuperado?
- **Answer Relevancy**: ¿La respuesta es relevante a la pregunta?
- **Context Precision**: ¿Los contextos recuperados son precisos?
- **Context Recall**: ¿Se recuperaron todos los contextos relevantes?

## 📁 Estructura del Proyecto

```
Agente/
├── ingest.py                  # Ingesta de documentos locales (conocimiento.txt)
├── ingest_wikitext.py         # Ingesta de Wikipedia
├── query_agent.py             # Agente de consultas RAG
├── evaluate.py                # Evaluación con dataset manual
├── evaluate_with_json.py      # Evaluación con dataset JSON (recomendado)
├── evaluation_dataset.json    # Dataset de preguntas y ground truth
├── check_models.py            # Utilidad para verificar modelos de Google
├── conocimiento.txt           # Datos de ejemplo locales
├── requirements.txt           # Dependencias
├── .env                       # Variables de entorno (NO SUBIR A GIT)
├── .env.example              # Plantilla de configuración
├── .gitignore                # Archivos a ignorar en Git
└── README.md                 # Este archivo
```

## 🔍 Flujo de Trabajo

### Opción A: Usar datos de Wikipedia (Recomendado para evaluación)

```bash
# 1. Ingestar datos de Wikipedia
python ingest_wikitext.py

# 2. Personalizar el dataset de evaluación
# Edita evaluation_dataset.json con preguntas relevantes a tu contenido

# 3. Evaluar con Ragas
python evaluate_with_json.py
```

### Opción B: Usar datos locales

```bash
# 1. Ingestar conocimiento.txt
python ingest.py

# 2. Consultar
python query_agent.py

# 3. Evaluar (ajusta las preguntas en evaluate.py)
python evaluate.py
```

## 📊 Métricas de Ragas

- **Faithfulness (0-1)**: Mide si la respuesta está fundamentada en el contexto recuperado
- **Answer Relevancy (0-1)**: Mide qué tan relevante es la respuesta a la pregunta
- **Context Precision (0-1)**: Mide la precisión de los contextos recuperados

Valores más cercanos a 1 indican mejor rendimiento.

## 🔧 Personalización

### Cambiar las preguntas de evaluación

Edita `evaluate.py` línea 44:
```python
test_questions = [
    "Tu pregunta 1",
    "Tu pregunta 2",
    "Tu pregunta 3"
]
```

### Ajustar el número de contextos recuperados

En `evaluate.py` línea 38:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Cambia el 3
```

### Cambiar el modelo LLM

En cualquier archivo:
```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # Cambia el modelo
```

## 📈 Monitoreo con LangSmith

Si configuraste LangSmith en el `.env`, todas las ejecuciones se registrarán automáticamente en:
https://smith.langchain.com/

Podrás ver:
- Trazas de ejecución
- Latencias
- Tokens consumidos
- Errores

## 🛠️ Utilidades

### Verificar modelos disponibles de Google

```bash
python check_models.py
```

## 🔒 Seguridad

⚠️ **IMPORTANTE**: Nunca subas el archivo `.env` a Git. Crea un `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
```

## 📚 Stack Tecnológico

- **LangChain**: Framework para aplicaciones LLM
- **Supabase**: Base de datos vectorial (PostgreSQL + pgvector)
- **Google Gemini**: LLM y embeddings
- **LangSmith**: Monitoreo y observabilidad
- **Ragas**: Evaluación de sistemas RAG
- **WikiText**: Dataset de Wikipedia

## 🤝 Contribuciones

Este es un proyecto académico para el curso de IA2 - Universidad Icesi.

## 📝 Licencia

MIT
