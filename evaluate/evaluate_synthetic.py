"""
OPCIÓN 3: GENERACIÓN SINTÉTICA CON RAGAS
=========================================
Este script genera automáticamente un dataset de evaluación
a partir de tus documentos en Supabase, sin necesidad de escribir
preguntas manualmente.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    answer_relevancy,
    context_recall,
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

load_dotenv()


def generate_synthetic_evaluation():
    """
    Genera un dataset sintético de evaluación usando Ragas
    """
    print("=" * 80)
    print("🤖 GENERACIÓN SINTÉTICA DE DATASET CON RAGAS")
    print("=" * 80 + "\n")

    # PASO 1: Configurar conexiones
    print("📡 Conectando a Supabase y configurando LLM...")
    supabase = create_client(
        os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", task_type="retrieval_query"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # PASO 2: Recuperar documentos de la base de datos directamente
    print("📚 Recuperando documentos de la base de datos...")

    # Obtener todos los documentos
    result = supabase.table("documents").select("content, metadata").limit(20).execute()

    # Convertir a documentos de LangChain
    documents = [
        Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
        for doc in result.data
        if doc.get("content")
    ]

    print(f"✅ Recuperados {len(documents)} documentos\n")

    # Mostrar muestra de los documentos
    print("📄 Muestra de documentos:")
    for i, doc in enumerate(documents[:3]):
        print(f"  [{i+1}] {doc.page_content[:100]}...")
    print()

    # PASO 3: Crear el generador de testset
    print("🔧 Configurando generador de Ragas...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,  # LLM para generar preguntas
        critic_llm=llm,  # LLM para validar calidad de preguntas
        embeddings=embeddings,  # Embeddings para análisis semántico
    )

    # PASO 4: Generar el dataset sintético
    print("🎲 Generando preguntas sintéticas...")
    print("   Tipos de preguntas:")
    print("   • Simple (40%): Preguntas directas sobre hechos")
    print("   • Reasoning (40%): Preguntas que requieren razonamiento")
    print("   • Multi-context (20%): Preguntas que combinan múltiples documentos\n")

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=10,  # Número de preguntas a generar
        distributions={
            simple: 0.4,  # 40% preguntas simples
            reasoning: 0.4,  # 40% preguntas de razonamiento
            multi_context: 0.2,  # 20% preguntas multi-contexto
        },
    )

    print(f"✅ Dataset generado con {len(testset)} ejemplos\n")

    # PASO 5: Convertir a formato Dataset
    dataset = testset.to_dataset()

    # Mostrar ejemplos generados
    print("=" * 80)
    print("📋 EJEMPLOS DE PREGUNTAS GENERADAS")
    print("=" * 80 + "\n")

    for i in range(min(3, len(dataset))):
        print(f"Ejemplo {i+1}:")
        print(f"  Pregunta: {dataset['question'][i]}")
        print(f"  Ground Truth: {dataset['ground_truth'][i][:150]}...")
        print(f"  Contextos: {len(dataset['contexts'][i])} documentos")
        print("-" * 80)

    # PASO 6: Evaluar con Ragas
    print("\n" + "=" * 80)
    print("🔬 EVALUANDO DATASET GENERADO")
    print("=" * 80 + "\n")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,  # ¿Las respuestas son fieles al contexto?
            answer_relevancy,  # ¿Las respuestas son relevantes?
            context_precision,  # ¿Los contextos son precisos?
            context_recall,  # ¿Se recuperaron todos los contextos?
        ],
        llm=llm,
        embeddings=embeddings,
    )

    # PASO 7: Mostrar resultados
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DE LA EVALUACIÓN")
    print("=" * 80)
    print(result)

    print("\n📈 Métricas Detalladas:")
    print(f"  • Faithfulness (Fidelidad):        {result.get('faithfulness', 0):.4f}")
    print(
        f"  • Answer Relevancy (Relevancia):   {result.get('answer_relevancy', 0):.4f}"
    )
    print(
        f"  • Context Precision (Precisión):   {result.get('context_precision', 0):.4f}"
    )
    print(f"  • Context Recall (Recuperación):   {result.get('context_recall', 0):.4f}")

    avg_score = (
        sum(
            [
                result.get("faithfulness", 0),
                result.get("answer_relevancy", 0),
                result.get("context_precision", 0),
                result.get("context_recall", 0),
            ]
        )
        / 4
    )

    print(f"\n🎯 Puntuación Promedio: {avg_score:.4f}")

    # PASO 8: Guardar el dataset generado (opcional)
    print("\n💾 Guardando dataset generado...")
    testset.to_pandas().to_csv("synthetic_evaluation_dataset.csv", index=False)
    print("✅ Dataset guardado en: synthetic_evaluation_dataset.csv")

    return result, dataset


if __name__ == "__main__":
    print("\n🤖 EVALUACIÓN RAG CON GENERACIÓN SINTÉTICA\n")
    print("Este script:")
    print("  1. Lee tus documentos de Supabase")
    print("  2. Genera preguntas automáticamente con Ragas")
    print("  3. Crea respuestas esperadas (ground truth)")
    print("  4. Evalúa tu sistema RAG")
    print("  5. Guarda el dataset para uso futuro\n")

    input("Presiona Enter para continuar...")

    result, dataset = generate_synthetic_evaluation()

    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)
    print("\n💡 Ventajas de este método:")
    print("  • No necesitas escribir preguntas manualmente")
    print("  • Genera preguntas diversas (simples, razonamiento, multi-contexto)")
    print("  • Las respuestas están basadas en tus documentos reales")
    print(
        "  • Puedes reutilizar el dataset generado (synthetic_evaluation_dataset.csv)"
    )
    print("\n⚠️ Consideraciones:")
    print("  • Las preguntas pueden no ser perfectas")
    print("  • Revisa el CSV generado para validar calidad")
    print("  • Puedes ajustar test_size y distributions según necesites")
