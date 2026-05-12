import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from supabase.client import create_client
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy, context_recall
from datasets import Dataset

# Cargar entorno
load_dotenv()

def load_evaluation_dataset(json_path="evaluation_dataset.json"):
    """
    Carga el dataset de evaluación desde un archivo JSON
    """
    # Asegurar que encuentre el archivo si se corre desde la raíz
    if not os.path.exists(json_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, json_path)
        
    print(f"📂 Cargando dataset desde: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [ex["question"] for ex in data["examples"]]
    ground_truths = [ex["ground_truth"] for ex in data["examples"]]
    
    print(f"✅ Dataset cargado: {len(questions)} preguntas\n")
    
    return questions, ground_truths

def evaluate_with_dataset(json_path="evaluation_dataset.json"):
    """
    Evalúa el sistema RAG usando el dataset del archivo JSON
    """
    print("="*80)
    print("EVALUACIÓN RAG CON DATASET PERSONALIZADO")
    print("="*80 + "\n")
    
    # Cargar dataset
    questions, ground_truths = load_evaluation_dataset(json_path)
    
    # Configuración de clientes
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # Configurar prompt
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Generar respuestas y recopilar contextos
    answers = []
    contexts = []
    
    print("🔄 Generando respuestas desde la base de datos...\n")
    for i, question in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] 📝 Pregunta: {question}")
        
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
            from langchain_core.documents import Document
            docs = [Document(page_content=doc["content"], metadata=doc.get("metadata", {})) 
                    for doc in result.data]
            
            # Generar respuesta
            answer = document_chain.invoke({"context": docs, "input": question})
            retrieved_contexts = [doc.page_content for doc in docs]
            
            print(f"    ✅ Respuesta generada: {answer[:100]}...")
            print(f"    📚 Contextos recuperados: {len(retrieved_contexts)}")
            print(f"    🎯 Ground Truth: {ground_truths[i][:100]}...")
            print("-" * 80)
            
            answers.append(answer)
            contexts.append(retrieved_contexts)
            
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            answers.append("Error al generar respuesta")
            contexts.append(["No se pudo recuperar contexto"])
            print("-" * 80)
    
    # Crear dataset para Ragas
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    # Evaluar con Ragas
    print("\n" + "="*80)
    print("🔬 EJECUTANDO EVALUACIÓN CON RAGAS")
    print("="*80 + "\n")
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,        # ¿La respuesta es fiel al contexto?
            answer_relevancy,    # ¿La respuesta es relevante a la pregunta?
            context_precision,   # ¿Los contextos son precisos?
            context_recall       # ¿Se recuperaron todos los contextos relevantes?
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("📊 RESULTADOS DE LA EVALUACIÓN")
    print("="*80)
    print(result)
    
    # Mostrar métricas individuales
    print("\n📊 Métricas Detalladas:")
    # Ragas EvaluationResult puede comportarse como un diccionario o tener los valores en .scores
    # Usamos la forma más segura de acceder a los resultados
    print(f"  • Faithfulness (Fidelidad):        {result['faithfulness']:.4f}")
    print(f"  • Answer Relevancy (Relevancia):   {result['answer_relevancy']:.4f}")
    print(f"  • Context Precision (Precisión):   {result['context_precision']:.4f}")
    print(f"  • Context Recall (Recuperación):   {result['context_recall']:.4f}")
    
    print("\n💡 Interpretación:")
    print("  - Valores cercanos a 1.0 = Excelente rendimiento")
    print("  - Valores entre 0.5-0.8 = Rendimiento aceptable")
    print("  - Valores < 0.5 = El sistema necesita mejoras")
    
    # Calcular promedio
    avg_score = sum([
        result['faithfulness'],
        result['answer_relevancy'],
        result['context_precision'],
        result['context_recall']
    ]) / 4
    
    print(f"\n🎯 Puntuación Promedio: {avg_score:.4f}")
    
    return result

if __name__ == "__main__":
    print("\n🤖 SISTEMA DE EVALUACIÓN RAG CON RAGAS\n")
    
    # Evaluar usando el dataset JSON
    result = evaluate_with_dataset("evaluation_dataset.json")
    
    print("\n✅ Evaluación completada exitosamente!")
    print("\n💡 Tip: Edita 'evaluation_dataset.json' para personalizar las preguntas según tu contenido de Wikipedia")
