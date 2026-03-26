from pathlib import Path

from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline
#from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.query import QueryExpander
from haystack.components.retrievers import MultiQueryEmbeddingRetriever


#document_store = InMemoryDocumentStore()
document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    return_embedding=True,
    embedding_dim = 384,
)
embedder_model = "sentence-transformers/all-MiniLM-L6-v2" 
#embedder_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 768

markdown_converter = MarkdownToDocument()
document_cleaner = DocumentCleaner()

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")

preprocessing_pipeline.connect("markdown_converter", "document_cleaner")

path = "./out_md"
files = list(Path(path).glob("*.md"))
print (files)

custom_metadata = {"Platform": "UEPE", "version": "5.2"}

print ("Preprocessing files...")
preprocessing_pipeline_result = preprocessing_pipeline.run({"markdown_converter": {"sources": files, "meta": custom_metadata}})

preprocessed_documents = preprocessing_pipeline_result["document_cleaner"]["documents"]

document_splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50, split_threshold=100)
document_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
document_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
indexing_pipeline.add_component(instance=document_writer, name="document_writer")

indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")

print ("Indexing documents...")
indexing_pipeline.run({"document_splitter": {"documents": preprocessed_documents}})


custom_template = """
You are a search query expansion assistant.
Generate {{ n_expansions }} alternative search queries for: "{{ query }}"

Return ONLY valid JSON in the following format:

{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}

Do not include explanations.
Do not include markdown.
Do not include text before or after the JSON.

Return the expanded queries translated to english.
Focus on technical terminology and domain-specific variations.
"""

query_expander = QueryExpander(
    chat_generator=OllamaChatGenerator(model="llama3.2", url="http://localhost:11434", timeout=600),
    prompt_template=custom_template, n_expansions=3,
)
#text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
llm_generator = OllamaChatGenerator(model="llama3.2", url="http://localhost:11434", timeout=1200)

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
prompt_builder = PromptBuilder(template=prompt_template,
    required_variables=["documents", "question"])

query_pipeline = Pipeline()
query_pipeline.add_component("expander", query_expander)
#query_pipeline.add_component(instance=text_embedder, name="text_embedder")
#query_pipeline.add_component(
#    "retriever",
#    QdrantEmbeddingRetriever(document_store=document_store, top_k=5),
#)
query_pipeline.add_component(
    "retriever",
    MultiQueryEmbeddingRetriever(
        retriever=QdrantEmbeddingRetriever(document_store=document_store, top_k=5),
        query_embedder=SentenceTransformersTextEmbedder(model=embedder_model),
    ),
)
query_pipeline.add_component("prompt_builder", prompt_builder)
query_pipeline.add_component("llm_generator", llm_generator)

query_pipeline.connect("expander.queries", "retriever.queries")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "llm_generator.messages")

query = "How can I create a EKS cluster for installing UEPE?"
#query = "¿Cuáles son los pasos que debo seguir para instalar UEPE en AWS?"
#query = "¿Cuál es el primer paso para instalar UEPE en AWS?"
#query = "¿Cómo genero el cluster de Kubernetes en AWS para instalar UEPE?"
result = query_pipeline.run(
    {
        "expander": {"query": query},
        "prompt_builder": {"question": query},
        "llm_generator": {"generation_kwargs": {"temperature": 0.1}}
    })

#print(result["retriever"]["documents"])
print(result['llm_generator']['replies'][0])