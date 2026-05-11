# Project Overview: Agente RAG con LangChain, Supabase y Ragas

A Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Supabase** (as a Vector Database), and **Google Gemini**. It includes automated evaluation using **Ragas** and monitoring with **LangSmith**.

## Core Stack
- **Language:** Python 3.10+
- **Orchestration:** LangChain
- **LLM & Embeddings:** Google Gemini (`gemini-1.5-flash` or `gemini-2.0-flash`, `models/gemini-embedding-001`)
- **Vector Store:** Supabase with `pgvector`
- **Evaluation:** Ragas (Faithfulness, Relevancy, Precision, Recall)
- **Monitoring:** LangSmith

---

## Building and Running

### 1. Environment Setup
Create a `.env` file based on `.env.example`:
```env
GOOGLE_API_KEY=your_google_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
# Optional: LangSmith for monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agente-rag-supabase
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 2. Database Initialization
Execute the following SQL in your Supabase SQL Editor to enable `pgvector` and create the required table and RPC function:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  content TEXT,
  metadata JSONB,
  embedding VECTOR(768)
);

CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (id BIGINT, content TEXT, metadata JSONB, similarity FLOAT)
LANGUAGE SQL STABLE
AS $$
  SELECT
    documents.id, documents.content, documents.metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
$$;
```

### 3. Key Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **Ingest Local Data** | `python ingest.py` | Processes `conocimiento.txt` and uploads to Supabase. |
| **Ingest Wikipedia** | `python ingest_wikitext.py` | Downloads 20 Wikipedia articles and uploads them. |
| **Query Agent** | `python query_agent.py` | Simple CLI to query the RAG system. |
| **Evaluate (JSON)** | `python evaluate_with_json.py` | **Recommended.** Evaluates using `evaluation_dataset.json`. |
| **Evaluate (Manual)** | `python evaluate.py` | Evaluates using questions hardcoded in the script. |
| **Evaluate (Synthetic)** | `python evaluate_synthetic.py` | Automatically generates evaluation cases using Ragas. See `SYNTHETIC_GENERATION_EXPLAINED.md` for details. |
| **Check Models** | `python check_models.py` | Lists available Google Gemini models. |

---

## Development Conventions

### Data Processing
- **Chunking:** Uses `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=100`.
- **Embeddings:** Uses `GoogleGenerativeAIEmbeddings` with `models/gemini-embedding-001` (768 dimensions).
- **Retrieval:** Uses the `match_documents` RPC with a default threshold of `0.5` and `k=3`.

### Evaluation Standards
- Always use `evaluation_dataset.json` for reproducible testing.
- Target Ragas metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.
- Scores > 0.8 are considered excellent; < 0.5 requires system adjustment (chunking, prompts, or retrieval params).

### Code Style
- Load environment variables using `python-dotenv`.
- Use `ChatGoogleGenerativeAI` with `gemini-1.5-flash` for fast and cost-effective generation.
- Ensure all Supabase interactions are authenticated via `SUPABASE_SERVICE_KEY`.

---

## Directory Structure
- `ingest*.py`: Scripts for data population.
- `query_agent.py`: Main entry point for user queries.
- `evaluate*.py`: Scripts for measuring RAG performance.
- `evaluation_dataset.json`: Source of truth for evaluation questions and ground truth answers.
- `conocimiento.txt`: Default local knowledge source.
