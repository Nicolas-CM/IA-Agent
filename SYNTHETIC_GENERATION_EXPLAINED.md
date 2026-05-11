# 🤖 Generación Sintética con Ragas - Explicación Detallada

## 🎯 ¿Qué es la Generación Sintética?

Es un proceso **automático** donde Ragas:
1. Lee tus documentos
2. Genera preguntas inteligentes
3. Crea respuestas esperadas (ground truth)
4. Todo sin intervención manual

---

## 🔄 Flujo Completo

```
┌─────────────────────────────────────────────────────────────┐
│                    TUS DOCUMENTOS EN SUPABASE               │
│                                                             │
│  Doc 1: "Machine learning is a subset of AI that..."       │
│  Doc 2: "Neural networks consist of interconnected..."     │
│  Doc 3: "Deep learning uses multiple layers to..."         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              RAGAS TESTSET GENERATOR                        │
│                                                             │
│  1. Analiza el contenido semántico                         │
│  2. Identifica conceptos clave                             │
│  3. Encuentra relaciones entre documentos                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              GENERA 3 TIPOS DE PREGUNTAS                    │
│                                                             │
│  📝 SIMPLE (40%)                                            │
│     "What is machine learning?"                            │
│     → Pregunta directa sobre un hecho                      │
│                                                             │
│  🧠 REASONING (40%)                                         │
│     "Why are neural networks effective for pattern         │
│      recognition?"                                         │
│     → Requiere razonamiento y análisis                     │
│                                                             │
│  🔗 MULTI-CONTEXT (20%)                                     │
│     "How does deep learning differ from traditional        │
│      machine learning approaches?"                         │
│     → Combina información de múltiples documentos          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              GENERA GROUND TRUTH                            │
│                                                             │
│  Para cada pregunta, extrae la respuesta correcta          │
│  directamente de los documentos fuente                     │
│                                                             │
│  Pregunta: "What is machine learning?"                     │
│  Ground Truth: "Machine learning is a subset of AI that    │
│                 enables systems to learn from data..."     │
│  Contextos: [Doc 1, Doc 2]                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DATASET SINTÉTICO COMPLETO                     │
│                                                             │
│  {                                                          │
│    "question": [...],                                       │
│    "ground_truth": [...],                                   │
│    "contexts": [...],                                       │
│    "answer": [...]                                          │
│  }                                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              EVALUACIÓN CON RAGAS                           │
│                                                             │
│  Compara las respuestas generadas vs ground truth          │
│  Métricas: Faithfulness, Relevancy, Precision, Recall      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎲 Tipos de Preguntas Generadas

### 1️⃣ **Simple Questions (Preguntas Simples)**

**Características:**
- Respuesta directa en un solo documento
- Basadas en hechos concretos
- No requieren razonamiento complejo

**Ejemplo:**
```
Documento: "Python is a high-level programming language created by Guido van Rossum."

Pregunta generada: "Who created Python?"
Ground Truth: "Guido van Rossum"
```

### 2️⃣ **Reasoning Questions (Preguntas de Razonamiento)**

**Características:**
- Requieren análisis y comprensión
- Pueden involucrar causa-efecto
- Necesitan inferencia

**Ejemplo:**
```
Documento: "Python's simple syntax makes it ideal for beginners. 
           Its extensive libraries support rapid development."

Pregunta generada: "Why is Python popular among beginners?"
Ground Truth: "Python is popular among beginners because of its simple 
               syntax and extensive libraries that support rapid development."
```

### 3️⃣ **Multi-Context Questions (Preguntas Multi-Contexto)**

**Características:**
- Combinan información de múltiples documentos
- Requieren síntesis de información
- Más complejas y realistas

**Ejemplo:**
```
Doc 1: "Python is dynamically typed and interpreted."
Doc 2: "Java is statically typed and compiled."

Pregunta generada: "What are the main differences between Python and Java?"
Ground Truth: "Python is dynamically typed and interpreted, while Java is 
               statically typed and compiled."
```

---

## ⚙️ Parámetros Configurables

```python
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,              # Número total de preguntas
    distributions={
        simple: 0.4,           # 40% preguntas simples
        reasoning: 0.4,        # 40% preguntas de razonamiento
        multi_context: 0.2     # 20% preguntas multi-contexto
    }
)
```

### Ajustes Recomendados:

| Caso de Uso | test_size | simple | reasoning | multi_context |
|-------------|-----------|--------|-----------|---------------|
| **Evaluación rápida** | 5-10 | 0.5 | 0.3 | 0.2 |
| **Evaluación completa** | 20-50 | 0.4 | 0.4 | 0.2 |
| **Documentos simples** | 10-20 | 0.6 | 0.3 | 0.1 |
| **Documentos complejos** | 15-30 | 0.3 | 0.4 | 0.3 |

---

## 🔍 Ejemplo Real de Salida

```
📋 EJEMPLOS DE PREGUNTAS GENERADAS
================================================================================

Ejemplo 1:
  Pregunta: What is the primary function of neural networks?
  Ground Truth: Neural networks are designed to recognize patterns and 
                process information in a way similar to the human brain...
  Contextos: 2 documentos
--------------------------------------------------------------------------------

Ejemplo 2:
  Pregunta: Why is deep learning considered more powerful than traditional 
            machine learning?
  Ground Truth: Deep learning is more powerful because it can automatically 
                learn hierarchical representations from raw data...
  Contextos: 3 documentos
--------------------------------------------------------------------------------

Ejemplo 3:
  Pregunta: How do convolutional neural networks differ from recurrent 
            neural networks?
  Ground Truth: CNNs are specialized for processing grid-like data such as 
                images, while RNNs are designed for sequential data...
  Contextos: 4 documentos
```

---

## ✅ Ventajas de la Generación Sintética

| Ventaja | Descripción |
|---------|-------------|
| 🚀 **Velocidad** | Genera 10-50 preguntas en minutos |
| 🎯 **Precisión** | Ground truth extraído de tus documentos reales |
| 🔄 **Reproducible** | Puedes regenerar con diferentes parámetros |
| 📊 **Diversidad** | 3 tipos de preguntas (simple, reasoning, multi-context) |
| 💾 **Reutilizable** | Guarda el dataset en CSV para uso futuro |
| 🤖 **Automático** | Cero intervención manual |

---

## ⚠️ Limitaciones

| Limitación | Solución |
|------------|----------|
| Preguntas pueden ser genéricas | Revisa y filtra el CSV generado |
| Requiere documentos de calidad | Asegúrate de ingestar contenido relevante |
| Puede generar preguntas repetitivas | Aumenta la diversidad de documentos |
| Ground truth puede ser imperfecto | Valida manualmente las preguntas críticas |

---

## 🎓 Cuándo Usar Cada Opción

### Usa **Opción 1 (Manual)** si:
- ✅ Tienes pocas preguntas específicas (< 5)
- ✅ Necesitas control total sobre las preguntas
- ✅ Estás probando casos edge específicos

### Usa **Opción 2 (JSON)** si:
- ✅ Tienes un conjunto definido de preguntas (5-20)
- ✅ Quieres compartir el dataset con tu equipo
- ✅ Necesitas versionamiento del dataset

### Usa **Opción 3 (Sintética)** si:
- ✅ Tienes muchos documentos (> 10)
- ✅ Quieres evaluar rápidamente sin escribir preguntas
- ✅ Necesitas diversidad de tipos de preguntas
- ✅ Quieres automatizar completamente el proceso

---

## 🚀 Cómo Ejecutar

```bash
# Opción 3: Generación Sintética
python evaluate_synthetic.py
```

**Salida:**
1. Genera preguntas automáticamente
2. Evalúa tu sistema RAG
3. Guarda el dataset en `synthetic_evaluation_dataset.csv`
4. Muestra métricas de rendimiento

---

## 💡 Tips Avanzados

### 1. Ajustar la Calidad de las Preguntas

```python
# Más preguntas simples para documentos técnicos
distributions={simple: 0.6, reasoning: 0.3, multi_context: 0.1}

# Más preguntas complejas para contenido narrativo
distributions={simple: 0.2, reasoning: 0.5, multi_context: 0.3}
```

### 2. Filtrar Documentos Relevantes

```python
# En lugar de usar "information" genérico
retriever.get_relevant_documents("machine learning neural networks")
```

### 3. Validar el Dataset Generado

```python
# Después de generar, revisa el CSV
import pandas as pd
df = pd.read_csv("synthetic_evaluation_dataset.csv")
print(df[['question', 'ground_truth']].head())
```

---

## 📚 Referencias

- [Ragas Documentation](https://docs.ragas.io/)
- [Testset Generation Guide](https://docs.ragas.io/en/latest/concepts/testset_generation.html)
- [Evolution Types](https://docs.ragas.io/en/latest/concepts/testset_generation.html#evolution-types)
