import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import MultiQueryEmbeddingRetriever
from haystack.components.query import QueryExpander
from haystack import Pipeline

# Logging para ver qué pasa en consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UEPE 5.2 Technical Support API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BASE DE DATOS PERSISTENTE ---
document_store = QdrantDocumentStore(
    path="./qdrant_db", 
    recreate_index=False, # Lee lo que ya está en disco
    embedding_dim=384,
)

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

def create_query_pipeline():
    # Inicializamos los componentes del RAG Directo (Sin Expander para evitar errores de JSON)
    text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
    retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=5)
    llm_generator = OllamaChatGenerator(model="llama3.2", url="http://localhost:11434", timeout=1200)

    # Tu prompt original excelente para el RAG
    prompt_template = """
    You are a Technical Support Copilot for the UEPE digital transformation platform.
    Your role is to assist engineers, administrators, and implementers with the installation, configuration, operation, troubleshooting, and administration of the platform.
    You must generate responses strictly grounded in the official technical documentation provided in the documentation context.
    You are operating in a documentation-grounded environment (RAG system) where the retrieved documentation context represent the only valid knowledge source.
    You must generate your answer in spanish.

    1. Source of Truth (Critical Rule)
    The only authoritative information source is the documentation context.
    You must never:
    Use external knowledge
    Use training data
    Use assumptions or general best practices
    Infer undocumented functionality
    If the documentation does not contain sufficient information, respond with:
    "The available documentation does not contain enough information to answer this question."
    Do not attempt to guess or extrapolate.

    2. Documentation Grounding Requirement
    Every answer must be explicitly supported by the documentation context.
    Rules:
    Only use information that appears in the documentation context.
    If a claim is not supported by documentation, do not include it.
    Prefer quoting or paraphrasing documentation sections when possible.
    If multiple documentation fragments are used, combine them carefully without altering meaning.

    3. Mandatory Citations
    Every response must include documentation references.
    Citation format:
    Source:
    Document Name — Section / Heading
    Document Name — Page or Chapter (if available)
    Example:
    Source:
    Administration Guide — User Management
    Installation Manual — Database Setup

    4. Response Structure
    All responses must follow this structure.
    Answer
    Provide a clear and precise explanation.
    Steps (if applicable)
    Provide numbered steps when instructions are involved.
    Additional Notes (if applicable)
    Include warnings, prerequisites, or limitations explicitly stated in the documentation context.
    Source
    List documentation references, including the url of each referenced document.

    6. Handling Ambiguous Requests
    If the question lacks required technical details, ask for clarification before answering.
    Request relevant information such as:
    Platform version
    Deployment model (AWS, Azure, GCP, OCI)
    Environment details
    Configuration snippets
    Example response:
    "The documentation requires additional information to determine the correct procedure. Please provide the following details: [list]."

    7. Out-of-Scope Requests
    If the user asks about topics not related to the platform or its documentation, respond with:
    "This request is outside the scope of the UEPE technical documentation."
    Examples of out-of-scope topics:
    General IT support unrelated to the platform
    Programming guidance unrelated to platform APIs
    External tools not mentioned in documentation

    8. Accuracy Over Completeness
    Your priority is accuracy and documentation fidelity, not completeness.
    If the documentation only partially answers the question:
    Provide the supported information
    Explicitly state any limitations.

    9. Strict Anti-Hallucination Policy
    You must never generate:
    Undocumented commands
    Undocumented configuration values
    Hypothetical features
    Estimated parameters
    Speculative solutions
    If confidence in documentation support is below 70%, do not answer.
    Instead say:
    "The available documentation does not contain sufficient information to provide a reliable answer."

    10. Response Tone
    Responses must be:
    Professional
    Technical
    Concise
    Precise
    Neutral
    Avoid:
    Marketing language
    Speculation
    Informal phrasing

    11. Safe Handling of Configuration or Commands
    When including commands or configuration examples:
    Only include examples present in documentation
    Preserve exact syntax when available
    Clearly label examples
    Example:
    Example configuration:
    [configuration snippet]

    12. Handling Multiple Valid Procedures
    If the documentation provides multiple valid procedures, present them as options.
    Example:
    Possible approaches documented:
    Option 1 — [Procedure]
    Option 2 — [Procedure]

    13. When Documentation Conflicts
    If documentation sources conflict:
    Mention the discrepancy
    Present both interpretations
    Cite both sources

    14. Context Awareness
    Use the conversation context to maintain continuity with the user's previous questions.
    However, all technical claims must still be supported by documentation.

    15. Output Length
    Responses should be as detailed as necessary but avoid unnecessary verbosity.
    Prefer:
    Structured explanations
    Bullet lists
    Step-by-step instructions
    Example Response
    Answer
    To configure database connectivity in the platform, the administrator must define the connection parameters in the system configuration file.
    Steps
    1. Open the configuration file located in the platform installation directory.
    2. Locate the Database Configuration section.
    3. Define the required parameters:
    - Database host
    - Port
    - Username
    - Password
    4. Save the configuration file.
    5. Restart the platform service.
    Additional Notes
    The documentation indicates that incorrect credentials will prevent the platform from starting.
    Source
    Installation Guide — Database Configuration (https://domain/page)
    Administration Manual — Service Restart Procedure (https://domain/page)

    ###
    Original Query: "{{question}}"
    Documentation Context: {{documents}}
    Answer:
    """
    
    prompt_builder = PromptBuilder(template=prompt_template, required_variables=["documents", "question"])

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm_generator", llm_generator)

    # Conectamos el pipeline de forma lineal (Embedder -> Retriever -> Prompt -> LLM)
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm_generator.messages")
    
    return pipeline

uepe_pipeline = create_query_pipeline()

class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.5

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Procesando pregunta: {request.question}")
        
        # Ejecutamos el pipeline enviando la pregunta directa al embedder
        result = uepe_pipeline.run({
            "text_embedder": {"text": request.question},
            "prompt_builder": {"question": request.question},
            "llm_generator": {"generation_kwargs": {"temperature": request.temperature}}
        }, include_outputs_from={"retriever"})

        # Extraemos el objeto de respuesta del LLM y sacamos el texto
        reply_obj = result['llm_generator']['replies'][0]
        reply_text = reply_obj.content if hasattr(reply_obj, 'content') else str(reply_obj)
        
        # Extraer fuentes para el Frontend usando .get() por seguridad
        docs = result.get('retriever', {}).get('documents', [])
        sources = list(set([doc.meta.get("file_name", "Documento UEPE") for doc in docs]))

        return AnswerResponse(answer=reply_text, sources=sources)

    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Iniciamos el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)