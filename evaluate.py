import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from supabase.client import create_client
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy, context_recall
from datasets import Dataset
from langchain_core.documents import Document

# 1. Cargar entorno
load_dotenv()

def evaluate_with_manual_dataset():
    """
    Evalúa el sistema RAG usando un dataset manual con ground truth
    """
    print("="*80)
    print("EVALUACIÓN CON DATASET MANUAL")
    print("="*80 + "\n")
    
    # Configuración de clientes
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # Configurar prompt
    prompt = ChatPromptTemplate.from_template("""
        Responde a la siguiente pregunta basándote únicamente en el contexto proporcionado:
        <context>
        {context}
        </context>
        Pregunta: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Dataset manual con preguntas basadas en el contenido real
    manual_dataset = {
        "question": [
            "¿Qué es el RAG?",
            "¿Cuáles son los beneficios del RAG?",
            "¿Qué es LangChain?"
        ],
        "ground_truth": [
            "El RAG (Retrieval-Augmented Generation) es una técnica que combina la potencia de los modelos de lenguaje con datos externos actualizados.",
            "Los beneficios del RAG son: 1) Reducción de alucinaciones porque el modelo se basa en hechos recuperados, 2) Conocimiento actualizado que no depende solo de la fecha de entrenamiento, 3) Privacidad al permitir usar datos locales sin re-entrenar el modelo.",
            "LangChain es el framework líder para construir aplicaciones con LLMs, permitiendo conectar modelos de lenguaje con bases de datos vectoriales como Supabase."
        ]
    }
    
    # Generar respuestas y recopilar contextos desde la BD
    answers = []
    contexts = []
    
    print("Generando respuestas desde la base de datos...\n")
    for i, question in enumerate(manual_dataset["question"]):
        print(f"[{i+1}/{len(manual_dataset['question'])}] Pregunta: {question}")
        
        try:
            # Generar embedding y buscar documentos
            query_embedding = embeddings.embed_query(question)
            result = supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.5,
                    "match_count": 3
                }
            ).execute()
            
            # Convertir a documentos
            docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
                    for doc in result.data]
            
            # Generar respuesta
            answer = document_chain.invoke({"context": docs, "input": question})
            retrieved_contexts = [doc.page_content for doc in docs]
            
            print(f"    Respuesta: {answer[:100]}...")
            print(f"    Contextos recuperados: {len(retrieved_contexts)}")
            print(f"    Ground Truth: {manual_dataset['ground_truth'][i][:100]}...")
            print("-" * 80)
            
            answers.append(answer)
            contexts.append(retrieved_contexts)
            
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            answers.append("Error al generar respuesta")
            contexts.append(["No se pudo recuperar contexto"])
    
    # Crear dataset para Ragas
    eval_dataset = Dataset.from_dict({
        "question": manual_dataset["question"],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": manual_dataset["ground_truth"]
    })
    
    return eval_dataset, llm, embeddings

def run_evaluation(dataset, llm, embeddings):
    """
    Ejecuta la evaluación con Ragas
    """
    print("\n" + "="*80)
    print("EJECUTANDO EVALUACIÓN CON RAGAS")
    print("="*80 + "\n")
    
    # Evaluar con Ragas
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,        # ¿La respuesta es fiel al contexto?
            answer_relevancy,    # ¿La respuesta es relevante a la pregunta?
            context_precision,   # ¿Los contextos son precisos?
            context_recall       # ¿Se recuperaron todos los contextos relevantes?
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n" + "="*80)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("="*80)
    print(result)
    
    # Mostrar métricas individuales
    print("\n📊 Métricas Detalladas:")
    print(f"  • Faithfulness (Fidelidad):        {result.get('faithfulness', 0):.4f}")
    print(f"  • Answer Relevancy (Relevancia):   {result.get('answer_relevancy', 0):.4f}")
    print(f"  • Context Precision (Precisión):   {result.get('context_precision', 0):.4f}")
    print(f"  • Context Recall (Recuperación):   {result.get('context_recall', 0):.4f}")
    
    print("\n💡 Interpretación:")
    print("  - Valores cercanos a 1.0 indican excelente rendimiento")
    print("  - Valores < 0.5 sugieren que el sistema necesita mejoras")
    
    return result

if __name__ == "__main__":
    print("\n🤖 SISTEMA DE EVALUACIÓN RAG CON RAGAS\n")
    print("Evaluando con dataset manual basado en tu contenido...\n")
    
    dataset, llm, embeddings = evaluate_with_manual_dataset()
    result = run_evaluation(dataset, llm, embeddings)
    
    print("\n✅ Evaluación completada exitosamente!")
