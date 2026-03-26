from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import requests

from haystack import Document, Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


# =========================================================
# CONFIG
# =========================================================

MARKDOWN_DIR = "./out_md"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

TOP_K = 5
USE_QUERY_EXPANSION = False   # Déjalo en False mientras depuras grounding
DEBUG_RETRIEVAL = False
MAX_CONTEXT_CHARS_PER_DOC = 1800


# =========================================================
# HELPERS
# =========================================================

def extract_trailing_json_metadata(text: str) -> tuple[str, dict[str, Any]]:
    """
    Intenta detectar un JSON al final del markdown y separarlo del contenido.
    Esto sirve para casos como:
      ... contenido markdown ...
      { "title": "...", "page_url": "...", ... }
    """
    raw = text.rstrip()
    candidate_positions = [m.start() for m in re.finditer(r"\{", raw)]

    for pos in reversed(candidate_positions):
        candidate = raw[pos:].strip()
        try:
            meta = json.loads(candidate)
            if isinstance(meta, dict) and any(
                k in meta for k in ("title", "page_url", "page_id", "files", "html_file")
            ):
                content = raw[:pos].rstrip()
                return content, meta
        except Exception:
            continue

    return text, {}


def merge_meta(*metas: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for meta in metas:
        if not meta:
            continue
        for k, v in meta.items():
            merged[k] = v
    return merged


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def simple_query_terms(text: str) -> list[str]:
    """
    Extrae términos simples para una heurística ligera de relevancia.
    """
    stop = {
        "how", "can", "i", "a", "an", "the", "for", "to", "of", "and", "in", "on",
        "what", "which", "is", "are", "de", "la", "el", "los", "las", "para", "que",
        "como", "cuál", "cual", "un", "una", "por", "con", "del", "uepe"
    }
    tokens = re.findall(r"[a-zA-Z0-9\-_\.]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in stop]


def lexical_overlap_score(query: str, content: str) -> int:
    q_terms = set(simple_query_terms(query))
    c_terms = set(simple_query_terms(content))
    return len(q_terms & c_terms)

def normalize_shell_command(cmd: str) -> str:
    cmd = (cmd or "").strip()

    # quitar backticks
    cmd = cmd.strip("`").strip()

    # quitar prompt de shell
    cmd = re.sub(r"^\$\s+", "", cmd)

    # unir continuaciones de línea
    cmd = cmd.replace("\\\n", " ")
    cmd = cmd.replace("\\", " ")

    # colapsar espacios
    cmd = re.sub(r"\s+", " ", cmd).strip()

    return cmd



def extract_allowed_commands_from_docs(documents: list[Document]) -> list[str]:
    commands = set()
    allowed_prefixes = ("eksctl ", "terraform ", "kubectl ", "helm ", "export ", "aws ")

    for doc in documents:
        content = doc.content or ""

        # 1) bloques de código markdown
        fenced_blocks = re.findall(r"```(?:bash|shell|sh)?\n(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced_blocks:
            lines = block.splitlines()
            current = []

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    if current:
                        cmd = normalize_shell_command(" ".join(current))
                        if any(cmd.startswith(p) for p in allowed_prefixes):
                            commands.add(cmd)
                        current = []
                    continue

                if any(line.startswith(p) for p in allowed_prefixes) or current:
                    current.append(line.rstrip("\\"))

                    if not raw_line.rstrip().endswith("\\"):
                        cmd = normalize_shell_command(" ".join(current))
                        if any(cmd.startswith(p) for p in allowed_prefixes):
                            commands.add(cmd)
                        current = []

            if current:
                cmd = normalize_shell_command(" ".join(current))
                if any(cmd.startswith(p) for p in allowed_prefixes):
                    commands.add(cmd)

        # 2) líneas normales
        lines = content.splitlines()
        current = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if current:
                    cmd = normalize_shell_command(" ".join(current))
                    if any(cmd.startswith(p) for p in allowed_prefixes):
                        commands.add(cmd)
                    current = []
                continue

            line = re.sub(r"^[-*]\s+", "", line).strip()
            line = re.sub(r"^\$\s+", "", line).strip()

            if any(line.startswith(p) for p in allowed_prefixes):
                current = [line.rstrip("\\")]
                if not raw_line.rstrip().endswith("\\"):
                    cmd = normalize_shell_command(" ".join(current))
                    if any(cmd.startswith(p) for p in allowed_prefixes):
                        commands.add(cmd)
                    current = []
            elif current:
                # continuación probable de comando multilinea
                if (
                    line.startswith("--")
                    or line.startswith("-")
                    or re.match(r"^[A-Za-z0-9_./:-]+$", line)
                ):
                    current.append(line.rstrip("\\"))
                    if not raw_line.rstrip().endswith("\\"):
                        cmd = normalize_shell_command(" ".join(current))
                        if any(cmd.startswith(p) for p in allowed_prefixes):
                            commands.add(cmd)
                        current = []
                else:
                    current = []

        if current:
            cmd = normalize_shell_command(" ".join(current))
            if any(cmd.startswith(p) for p in allowed_prefixes):
                commands.add(cmd)

    return sorted(commands)



def extract_generated_commands(text: str) -> list[str]:
    found = set()
    allowed_prefixes = ("eksctl ", "terraform ", "kubectl ", "helm ", "export ", "aws ")

    # comandos en backticks
    for match in re.findall(r"`([^`]+)`", text):
        cmd = normalize_shell_command(match)
        if any(cmd.startswith(p) for p in allowed_prefixes):
            found.add(cmd)

    # comandos en líneas
    for raw_line in text.splitlines():
        line = normalize_shell_command(raw_line)
        if any(line.startswith(p) for p in allowed_prefixes):
            found.add(line)

    return sorted(found)



def extract_urls(text: str) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s\)>\]]+", text or "")))


def render_documents_for_prompt(documents: list[Document], max_chars_per_doc: int = 1800) -> str:
    """
    Convierte los documentos recuperados en un contexto textual explícito.
    """
    blocks = []

    for i, doc in enumerate(documents, 1):
        meta = doc.meta or {}
        title = meta.get("title") or meta.get("source_file") or f"Document {i}"
        page_url = meta.get("page_url") or "N/A"
        page_id = meta.get("page_id") or "N/A"
        source_file = meta.get("source_file") or "N/A"
        content = (doc.content or "").strip()
        content = content[:max_chars_per_doc]

        block = (
            f"[DOCUMENT {i}]\n"
            f"Title: {title}\n"
            f"Page URL: {page_url}\n"
            f"Page ID: {page_id}\n"
            f"Source File: {source_file}\n"
            f"Content:\n{content}\n"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def print_retrieved_docs(documents: list[Document]) -> None:
    for i, doc in enumerate(documents, 1):
        meta = doc.meta or {}
        print("\n" + "=" * 80)
        print(f"DOC {i}")
        print("-" * 80)
        print("META:", json.dumps(meta, ensure_ascii=False, indent=2))
        print("-" * 80)
        print((doc.content or "")[:2000])
        print("=" * 80)


def shell_tokens(cmd: str) -> list[str]:
    return normalize_shell_command(cmd).split()


def is_supported_command(generated: str, allowed_commands: list[str]) -> bool:
    g = normalize_shell_command(generated)
    g_tokens = shell_tokens(g)

    if not g_tokens:
        return False

    for allowed in allowed_commands:
        a = normalize_shell_command(allowed)
        a_tokens = shell_tokens(a)

        # coincidencia exacta
        if g == a:
            return True

        # mismo prefijo completo
        if a.startswith(g) or g.startswith(a):
            return True

        # misma familia de comando: helm repo add / helm repo update / helm install ...
        if len(g_tokens) >= 2 and len(a_tokens) >= 2:
            if g_tokens[:2] == a_tokens[:2]:
                # subsecuencia ordenada
                idx = 0
                for tok in a_tokens:
                    if idx < len(g_tokens) and tok == g_tokens[idx]:
                        idx += 1
                if idx == len(g_tokens):
                    return True

        # comandos tipo helm install con más estructura
        if len(g_tokens) >= 3 and len(a_tokens) >= 3:
            if g_tokens[:3] == a_tokens[:3]:
                return True

    return False

def ollama_generate(prompt: str, temperature: float = 0.0) -> str:
    """
    Llama a Ollama directamente para tener control total del prompt.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=1200)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def expand_query_with_ollama(query: str) -> list[str]:
    """
    Expansión opcional de consulta. Se deja desactivada por defecto para evitar ruido.
    """
    prompt = f"""
You are a search query expansion assistant.
Generate 3 alternative search queries for this technical documentation question:

"{query}"

Rules:
- Return ONLY valid JSON.
- Use this format:
{{
  "queries": ["query 1", "query 2", "query 3"]
}}
- No explanations.
- No markdown.
- Prefer English technical terminology if useful.
"""
    try:
        raw = ollama_generate(prompt, temperature=0.0)
        data = json.loads(raw)
        queries = data.get("queries", [])
        if isinstance(queries, list):
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        pass
    return []


def build_safe_fallback_answer(documents: list[Document], query: str) -> str:
    if not documents:
        return (
            "Answer\n"
            "The available documentation does not contain enough information to answer this question.\n\n"
            "Source\n"
            "No relevant documentation fragments were retrieved."
        )

    top_meta = documents[0].meta or {}
    title = top_meta.get("title", "Unknown document")
    url = top_meta.get("page_url", "N/A")

    excerpts = build_grounded_excerpts(documents, query, max_docs=3)
    commands = extract_allowed_commands_from_docs(documents)

    parts = [
        "Answer",
        "La respuesta previa contenía elementos no verificables. A continuación se muestra únicamente información respaldada por la documentación recuperada.",
        "",
        "Supported Information",
    ]

    for i, excerpt in enumerate(excerpts, 1):
        parts.append(f"{i}. {excerpt}")

    if commands:
        parts.extend([
            "",
            "Commands explicitly found in the documentation",
        ])
        for cmd in commands:
            parts.append(f"- `{cmd}`")

    parts.extend([
        "",
        "Source",
        f"{title} ({url})"
    ])

    return "\n".join(parts)

def validate_answer(answer: str, documents: list[Document]) -> tuple[bool, list[str]]:
    problems = []

    allowed_urls = set()
    for doc in documents:
        meta = doc.meta or {}
        if meta.get("page_url"):
            allowed_urls.add(meta["page_url"])

    answer_urls = extract_urls(answer)
    external_urls = [u for u in answer_urls if u not in allowed_urls]
    if external_urls:
        problems.append(f"URLs externas detectadas: {external_urls}")

    allowed_commands = extract_allowed_commands_from_docs(documents)
    generated_commands = extract_generated_commands(answer)

    invalid_commands = []
    for cmd in generated_commands:
        if not is_supported_command(cmd, allowed_commands):
            invalid_commands.append(cmd)

    if invalid_commands:
        problems.append(f"Comandos no soportados por la documentación: {invalid_commands}")

    return (len(problems) == 0), problems





# =========================================================
# INDEXING
# =========================================================

def build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        return_embedding=True,
        embedding_dim=384,
    )


def load_and_prepare_markdowns(markdown_dir: str) -> list[Document]:
    path = Path(markdown_dir)
    files = sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".md"])

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .md en {markdown_dir}")

    print("Markdown files found:")
    for f in files:
        print(" -", f)

    custom_metadata = {"Platform": "UEPE", "version": "5.2"}

    markdown_converter = MarkdownToDocument()
    document_cleaner = DocumentCleaner()

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.connect("markdown_converter", "document_cleaner")

    print("\nPreprocessing files...")
    result = preprocessing_pipeline.run(
        {
            "markdown_converter": {
                "sources": files,
                "meta": custom_metadata,
            }
        }
    )

    raw_documents = result["document_cleaner"]["documents"]

    # Enriquecer documentos extrayendo metadata embebida al final del markdown
    enriched_docs: list[Document] = []
    for src_file, doc in zip(files, raw_documents):
        content = doc.content or ""
        cleaned_content, embedded_meta = extract_trailing_json_metadata(content)

        base_meta = dict(doc.meta or {})
        base_meta["source_file"] = str(src_file)

        final_meta = merge_meta(base_meta, embedded_meta)
        enriched_docs.append(
            Document(
                content=cleaned_content,
                meta=final_meta,
            )
        )

    return enriched_docs


def index_documents(document_store: QdrantDocumentStore, documents: list[Document]) -> None:
    document_splitter = DocumentSplitter(
        split_by="word",
        split_length=180,
        split_overlap=40,
        split_threshold=80,
        respect_sentence_boundary=True,
        language="en",
    )

    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDER_MODEL)
    document_writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    indexing_pipeline.add_component(instance=document_writer, name="document_writer")

    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    print("\nIndexing documents...")
    indexing_pipeline.run({"document_splitter": {"documents": documents}})


# =========================================================
# RETRIEVAL
# =========================================================

class GroundedUEPERetriever:
    def __init__(self, document_store: QdrantDocumentStore):
        self.document_store = document_store
        self.text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL)
        self.retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=TOP_K)

    def retrieve(self, query: str, expanded_queries: list[str] | None = None) -> list[Document]:
        queries = [query]
        if expanded_queries:
            queries.extend([q for q in expanded_queries if q.strip()])

        unique_docs: dict[str, Document] = {}

        for q in queries:
            embed_result = self.text_embedder.run(text=q)
            query_embedding = embed_result["embedding"]

            retrieved = self.retriever.run(query_embedding=query_embedding)
            docs = retrieved["documents"]

            for doc in docs:
                # Heurística ligera para filtrar resultados absurdamente alejados
                overlap = lexical_overlap_score(query, doc.content or "")
                meta = dict(doc.meta or {})
                meta["lexical_overlap"] = overlap
                doc.meta = meta

                key = f"{doc.id}"
                prev = unique_docs.get(key)

                if prev is None:
                    unique_docs[key] = doc
                else:
                    prev_score = getattr(prev, "score", 0.0) or 0.0
                    curr_score = getattr(doc, "score", 0.0) or 0.0
                    if curr_score > prev_score:
                        unique_docs[key] = doc

        docs = list(unique_docs.values())

        docs.sort(
            key=lambda d: (
                (d.meta or {}).get("lexical_overlap", 0),
                getattr(d, "score", 0.0) or 0.0,
            ),
            reverse=True,
        )

        # elimina documentos muy flojos si hay suficientes mejores
        strong_docs = [d for d in docs if (d.meta or {}).get("lexical_overlap", 0) >= 2]
        if len(strong_docs) >= 2:
            docs = strong_docs

        return docs[:TOP_K]


# =========================================================
# QA
# =========================================================

def _clean_excerpt(text: str, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""

    # corta por oraciones completas, no por caracteres arbitrarios
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out = []
    total = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if total + len(s) + 1 > max_chars:
            break
        out.append(s)
        total += len(s) + 1

    if not out:
        return text[:max_chars].rstrip()

    return " ".join(out).strip()


def build_grounded_excerpts(documents: list[Document], query: str, max_docs: int = 3) -> list[str]:
    excerpts = []

    q_terms = set(simple_query_terms(query))

    for doc in documents[:max_docs]:
        content = (doc.content or "").strip()
        if not content:
            continue

        # intenta localizar una oración que comparta términos con la consulta
        sentences = re.split(r'(?<=[.!?])\s+', re.sub(r"\s+", " ", content))
        ranked = []

        for s in sentences:
            score = len(q_terms & set(simple_query_terms(s)))
            if score > 0:
                ranked.append((score, s.strip()))

        if ranked:
            ranked.sort(reverse=True, key=lambda x: x[0])
            excerpt = ranked[0][1]
        else:
            excerpt = _clean_excerpt(content, max_chars=450)

        excerpts.append(excerpt)

    return excerpts

def build_prompt(question: str, documents: list[Document]) -> str:
    context = render_documents_for_prompt(documents, max_chars_per_doc=MAX_CONTEXT_CHARS_PER_DOC)
    allowed_commands = extract_allowed_commands_from_docs(documents)
    allowed_urls = [
        (doc.meta or {}).get("page_url")
        for doc in documents
        if (doc.meta or {}).get("page_url")
    ]
    allowed_urls = sorted(set(allowed_urls))

    allowed_commands_block = "\n".join(f"- {cmd}" for cmd in allowed_commands) if allowed_commands else "- None found"
    allowed_urls_block = "\n".join(f"- {url}" for url in allowed_urls) if allowed_urls else "- None found"

    prompt = f"""
You are a Technical Support Copilot for the UEPE platform.
You must answer in Spanish.

CRITICAL RULES:
1. Use ONLY the DOCUMENTATION CONTEXT below.
2. Do NOT use external knowledge.
3. Do NOT invent commands, URLs, section names, AWS procedures, or configuration values.
4. If the answer is not explicitly supported by the DOCUMENTATION CONTEXT, say exactly:
"The available documentation does not contain enough information to answer this question."
5. You may ONLY cite URLs that appear in the allowed URLs list.
6. You may ONLY mention commands that appear in the allowed commands list.
7. If the documentation contains an exact filename, command, YAML file, path, or step, preserve it exactly.
8. Forbidden examples:
   - Mentioning AWS docs not present in context
   - Mentioning `aws eks create-cluster` unless it appears in context
   - Mentioning placeholder URLs like https://domain/page
9. Prefer exact wording from the documentation when possible.

RESPONSE FORMAT:
Answer
<clear answer grounded in the docs>

Steps
<numbered steps only if explicitly supported by the docs>

Additional Notes
<only notes explicitly supported by the docs>

Source
<document title and URL taken only from context>

ALLOWED URLS:
{allowed_urls_block}

ALLOWED COMMANDS:
{allowed_commands_block}

QUESTION:
{question}

DOCUMENTATION CONTEXT:
{context}

Now produce the final answer in Spanish, strictly grounded in the context above.
"""
    return prompt.strip()


def answer_question(
    retriever: GroundedUEPERetriever,
    query: str,
    use_query_expansion: bool = False,
) -> str:
    expanded_queries = expand_query_with_ollama(query) if use_query_expansion else []
    docs = retriever.retrieve(query, expanded_queries=expanded_queries)

    if DEBUG_RETRIEVAL:
        print("\nRetrieved documents:")
        print_retrieved_docs(docs)

    if not docs:
        return "The available documentation does not contain enough information to answer this question."

    prompt = build_prompt(query, docs)
    answer = ollama_generate(prompt, temperature=0.0)

    is_valid, problems = validate_answer(answer, docs)
    if not is_valid:
        print("\n[WARNING] Invalid grounded answer detected:")
        for p in problems:
            print(" -", p)
        return build_safe_fallback_answer(docs, query)

    return answer


# =========================================================
# MAIN
# =========================================================

def main():
    document_store = build_document_store()

    preprocessed_documents = load_and_prepare_markdowns(MARKDOWN_DIR)
    index_documents(document_store, preprocessed_documents)

    retriever = GroundedUEPERetriever(document_store=document_store)

    # Cambia la pregunta aquí
    query = "How can I create a EKS cluster for installing UEPE?"
    # query = "¿Cuáles son los pasos para crear el clúster básico de Kubernetes en AWS para UEPE?"
    # query = "¿Cuál es el primer paso para instalar UEPE en AWS?"

    answer = answer_question(
        retriever=retriever,
        query=query,
        use_query_expansion=USE_QUERY_EXPANSION,
    )

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(answer)


if __name__ == "__main__":
    main()