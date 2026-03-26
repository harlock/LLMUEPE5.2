from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import requests

# =========================================================
# CONFIG
# =========================================================

MARKDOWN_DIR = "./out_md"
DEBUG = True

HAYSTACK_ENABLED = True
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
USE_OLLAMA_GENERATION = os.getenv("USE_OLLAMA_GENERATION", "false").lower() in {"1", "true", "yes"}
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "12"))
EMBED_TOP_K = int(os.getenv("EMBED_TOP_K", "12"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


# =========================================================
# OPTIONAL HAYSTACK IMPORTS
# =========================================================

try:
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import (
        InMemoryBM25Retriever,
        InMemoryEmbeddingRetriever,
    )
    HAYSTACK_AVAILABLE = True
except Exception:
    Document = None  # type: ignore
    InMemoryDocumentStore = None  # type: ignore
    InMemoryBM25Retriever = None  # type: ignore
    InMemoryEmbeddingRetriever = None  # type: ignore
    HAYSTACK_AVAILABLE = False


# =========================================================
# HELPERS
# =========================================================

STOPWORDS = {
    "how", "what", "which", "when", "where", "the", "for", "and", "are", "with",
    "using", "use", "into", "from", "that", "this", "must", "need", "should",
    "you", "your", "can", "que", "como", "cuál", "cual", "para", "con", "del",
    "las", "los", "uepe", "usage", "engine", "private", "edition",
    "installation", "installing", "aws", "eks", "kubernetes", "cluster",
    "commands", "command", "comando", "comandos", "show", "give", "after",
    "expected", "should", "do", "i", "me", "output", "outputs"
}

INSTALL_QUERY_TERMS = {
    "install", "installation", "setup", "set up", "create", "deploy", "configure",
    "bootstrap", "provision", "helm install", "helm upgrade"
}

VERIFY_HEADING_TERMS = {"verify", "validation", "validate", "check", "checks", "expected output", "output", "install helm chart", "post installation", "post-install"}


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def dedupe_preserve(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def simple_query_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_.\-/]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def lexical_overlap_score(query: str, text: str) -> int:
    q = set(simple_query_terms(query))
    t = set(simple_query_terms(text))
    return len(q & t)


def split_sentences_naive(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def load_sidecar_meta(md_path: Path) -> dict[str, Any]:
    meta_path = md_path.parent / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_title_from_markdown(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        m = re.match(r"^\s*#\s+(.*)$", line)
        if m:
            return m.group(1).strip()
    return fallback


def infer_page_url(meta: dict[str, Any]) -> str | None:
    return meta.get("page_url") or meta.get("url")


def is_install_command_query(query: str) -> bool:
    q = normalize_text(query)
    return any(term in q for term in INSTALL_QUERY_TERMS)


def extract_release_name_from_query(query: str) -> str | None:
    patterns = [
        r"helm\s+install\s+([a-zA-Z0-9._-]+)",
        r"helm\s+upgrade\s+([a-zA-Z0-9._-]+)",
        r"install\s+([a-zA-Z0-9._-]+)",
        r"deploy\s+([a-zA-Z0-9._-]+)",
    ]
    q = normalize_text(query)
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return m.group(1)
    return None


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, node_id in enumerate(ranking, start=1):
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (k + rank)
    return fused


# =========================================================
# COMMAND / YAML / OUTPUT EXTRACTION
# =========================================================


def extract_fenced_code_blocks(text: str) -> list[tuple[str, str]]:
    pattern = r"```([a-zA-Z0-9_-]*)\n(.*?)^```"
    out = []
    for lang, content in re.findall(pattern, text or "", flags=re.DOTALL | re.MULTILINE):
        out.append(((lang or "").strip().lower(), content.rstrip()))
    return out


def join_command_continuations(text: str) -> list[str]:
    lines = (text or "").splitlines()
    out = []
    i = 0

    while i < len(lines):
        current = lines[i].rstrip()
        if not current.strip():
            i += 1
            continue

        while current.rstrip().endswith("\\") and i + 1 < len(lines):
            current = current.rstrip("\\").rstrip() + " " + lines[i + 1].strip()
            i += 1

        while i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if re.match(r"^--[A-Za-z0-9_-]+", nxt) or re.match(r"^-[A-Za-z0-9]", nxt):
                current = current.rstrip() + " " + nxt
                i += 1
            else:
                break

        out.append(current)
        i += 1

    return out


def normalize_shell_command(cmd: str) -> str:
    cmd = (cmd or "").strip()
    cmd = cmd.strip("`").strip()
    cmd = re.sub(r"^\$\s+", "", cmd)
    cmd = cmd.replace("\\\n", " ")
    cmd = cmd.replace("\\", " ")
    cmd = re.sub(r"\s+", " ", cmd).strip()
    cmd = cmd.replace(" .alpha .", ".alpha.").replace(" .kubernetes .", ".kubernetes.")
    return cmd


def extract_inline_shell_commands(text: str) -> list[str]:
    patterns = [
        r"(helm\s+repo\s+add[^\n]+)",
        r"(helm\s+repo\s+update[^\n]*)",
        r"(helm\s+install[^\n]+)",
        r"(helm\s+upgrade[^\n]+)",
        r"(kubectl\s+apply\s+-f[^\n]+)",
        r"(kubectl\s+create\s+secret[^\n]+)",
        r"(kubectl\s+annotate[^\n]+)",
        r"(kubectl\s+get[^\n]+)",
        r"(kubectl\s+logs[^\n]+)",
        r"(terraform\s+init[^\n]*)",
        r"(terraform\s+plan[^\n]*)",
        r"(terraform\s+apply[^\n]*)",
        r"(eksctl\s+create\s+cluster[^\n]+)",
        r"(eksctl\s+get\s+nodegroup[^\n]+)",
        r"(export\s+KUBECONFIG[^\n]+)",
        r"(java\s+-jar[^\n]+)",
        r"(curl[^\n]+)",
        r"(\./[A-Za-z0-9._/\-]+\.sh[^\n]*)",
        r"(source\s+\./[^\n]+)",
        r"(SQL>@[^\n]+)",
        r"(SQL>[^\n]+)",
        r"(lsnrctl\s+[^\n]+)",
    ]

    found = []
    for line in join_command_continuations(text):
        for pattern in patterns:
            matches = re.findall(pattern, line, flags=re.IGNORECASE)
            for m in matches:
                cmd = normalize_shell_command(m)
                if cmd:
                    found.append(cmd)

        m = re.search(
            r":\s*(helm\s+repo\s+add[^\n]+|helm\s+repo\s+update[^\n]*|helm\s+install[^\n]+|helm\s+upgrade[^\n]+|kubectl\s+apply\s+-f[^\n]+|terraform\s+init[^\n]*|terraform\s+plan[^\n]*|terraform\s+apply[^\n]*|eksctl\s+create\s+cluster[^\n]+|export\s+KUBECONFIG[^\n]+)",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            cmd = normalize_shell_command(m.group(1))
            if cmd:
                found.append(cmd)

    return dedupe_preserve(found)


def is_yaml_start_line(line: str) -> bool:
    s = line.rstrip()
    stripped = s.strip()

    if not stripped:
        return False

    if stripped.lower().startswith(("note!", "info:", "example:", "warning:")):
        return False

    if re.match(r"^(kubectl |helm |terraform |eksctl |aws |curl |export |java -jar |\./)", stripped):
        return False

    return bool(re.match(r"^[A-Za-z0-9_.-]+:\s*.*$", stripped))


def is_yaml_continuation_line(line: str) -> bool:
    stripped = line.rstrip()
    s = stripped.strip()

    if not s:
        return True

    if s.lower().startswith(("note!", "info:", "example:", "warning:")):
        return False

    if re.match(r"^\d+\.\s+", s):
        return False

    if re.match(r"^(kubectl |helm |terraform |eksctl |aws |curl |export |java -jar |\./)", s):
        return False

    if re.match(r"^[A-Za-z0-9_.-]+:\s*.*$", s):
        return True

    if re.match(r"^\s+[A-Za-z0-9_.-]+:\s*.*$", stripped):
        return True

    if re.match(r"^\s*-\s+.*$", stripped):
        return True

    if re.match(r"^\s*\{.*\}\s*$", stripped):
        return True

    return False


def extract_inline_yaml_blocks(text: str) -> list[str]:
    lines = (text or "").splitlines()
    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not is_yaml_start_line(line):
            i += 1
            continue

        block = [line.rstrip()]
        j = i + 1
        while j < len(lines) and is_yaml_continuation_line(lines[j]):
            block.append(lines[j].rstrip())
            j += 1

        cleaned = "\n".join(block).strip()
        non_empty = [x for x in cleaned.splitlines() if x.strip()]
        colon_lines = sum(1 for x in non_empty if ":" in x)

        if len(non_empty) >= 3 and colon_lines >= 2:
            blocks.append(cleaned)

        i = j

    return dedupe_preserve(blocks)


def extract_inline_output_blocks(text: str) -> list[str]:
    lines = (text or "").splitlines()
    blocks = []
    i = 0

    header_patterns = [
        r"^\s*NAME\s+NAMESPACE\s+REVISION",
        r"^\s*NAME\s+READY\s+STATUS",
        r"^\s*CLUSTER\s+NODEGROUP\s+STATUS",
        r"^\s*NAMESPACE\s+NAME\s+CLASS",
        r"^\s*certificate_arn\s*=",
        r"^\s*db_endpoint\s*=",
        r"^\s*db_user\s*=",
        r"^\s*efs_id\s*=",
        r"^\s*name_servers\s*=",
    ]

    while i < len(lines):
        line = lines[i].rstrip()
        if any(re.match(p, line) for p in header_patterns):
            block = [line]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip()
                if not nxt.strip():
                    break
                if re.match(r"^(#|##|###|\d+\.)", nxt.strip()):
                    break
                block.append(nxt)
                j += 1
            if len(block) >= 2:
                blocks.append("\n".join(block).strip())
            i = j
            continue
        i += 1

    return dedupe_preserve(blocks)


def extract_numbered_steps(text: str) -> list[str]:
    steps = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m:
            step = m.group(2).strip()
            if step:
                steps.append(step)
    return dedupe_preserve(steps)


def is_install_like_command(cmd: str) -> bool:
    c = normalize_shell_command(cmd)
    prefixes = (
        "helm repo add ",
        "helm repo update",
        "helm install ",
        "helm upgrade ",
        "kubectl apply -f ",
        "kubectl create secret ",
        "terraform init",
        "terraform plan",
        "terraform apply",
        "eksctl create cluster ",
        "export KUBECONFIG",
        "java -jar ",
        "./",
        "source ./",
    )
    return any(c.startswith(p) for p in prefixes)


# =========================================================
# DATA MODEL
# =========================================================


@dataclass
class SectionNode:
    node_id: str
    file_path: str
    doc_title: str
    page_url: Optional[str]
    heading: str
    level: int
    heading_path: list[str]
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    raw_content: str = ""
    commands: list[str] = field(default_factory=list)
    yamls: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    @property
    def heading_path_str(self) -> str:
        return " > ".join(self.heading_path)


# =========================================================
# OLLAMA CLIENT
# =========================================================


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout = timeout

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=15)
            return r.ok
        except Exception:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.embed_model,
            "input": texts,
            "truncate": True,
        }
        r = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama devolvió una cantidad inesperada de embeddings")
        return embeddings

    def generate_answer(self, prompt: str) -> str:
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()


# =========================================================
# CORE ENGINE
# =========================================================


class MarkdownInstallQAHybrid:
    def __init__(
        self,
        markdown_dir: str,
        debug: bool = True,
        enable_haystack: bool = True,
        use_ollama_generation: bool = False,
    ):
        self.markdown_dir = Path(markdown_dir)
        self.debug = debug
        self.enable_haystack = enable_haystack and HAYSTACK_AVAILABLE
        self.use_ollama_generation = use_ollama_generation
        self.nodes: dict[str, SectionNode] = {}
        self.file_roots: dict[str, list[str]] = {}
        self.heading_index: dict[str, list[str]] = {}
        self.doc_title_index: dict[str, list[str]] = {}
        self.file_node_positions: dict[str, dict[str, int]] = {}
        self.haystack_documents: dict[str, Any] = {}
        self.document_store: Any = None
        self.bm25_retriever: Any = None
        self.embedding_retriever: Any = None
        self.ollama = OllamaClient(OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_CHAT_MODEL, REQUEST_TIMEOUT)
        self.ollama_available = self.ollama.healthcheck()
        self.embeddings_indexed = False

    def build(self) -> None:
        base_dir = Path.cwd()
        path = self.markdown_dir if self.markdown_dir.is_absolute() else (base_dir / self.markdown_dir).resolve()

        if not path.exists():
            raise FileNotFoundError(f"La carpeta no existe: {path}")

        files = sorted([p for p in path.rglob("*.md") if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos .md en {path}")

        self.nodes.clear()
        self.file_roots.clear()
        self.heading_index.clear()
        self.doc_title_index.clear()
        self.file_node_positions.clear()
        self.haystack_documents.clear()
        self.document_store = None
        self.bm25_retriever = None
        self.embedding_retriever = None
        self.embeddings_indexed = False

        for md_file in files:
            self._parse_markdown_file(md_file)

        self._build_indexes()
        if self.enable_haystack:
            self._build_haystack_index()

        if self.debug:
            print(f"Total section nodes: {len(self.nodes)}")

    def _parse_markdown_file(self, md_path: Path) -> None:
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        sidecar_meta = load_sidecar_meta(md_path)

        doc_title = sidecar_meta.get("title") or extract_title_from_markdown(raw, md_path.stem)
        page_url = infer_page_url(sidecar_meta)

        lines = raw.splitlines()
        current_node_id: Optional[str] = None
        stack: list[tuple[int, str]] = []
        file_nodes: list[str] = []

        def create_section(heading: str, level: int) -> str:
            nonlocal stack
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent_id = stack[-1][1] if stack else None
            heading_path = [self.nodes[x].heading for _, x in stack] + [heading]

            node_id = f"{md_path}::L{level}::{len(self.nodes)+1}"
            node = SectionNode(
                node_id=node_id,
                file_path=str(md_path),
                doc_title=doc_title,
                page_url=page_url,
                heading=heading,
                level=level,
                heading_path=heading_path,
                parent_id=parent_id,
            )
            self.nodes[node_id] = node
            file_nodes.append(node_id)

            if parent_id:
                self.nodes[parent_id].children_ids.append(node_id)

            stack.append((level, node_id))
            return node_id

        any_heading = any(re.match(r"^(#{1,6})\s+(.*)$", line.strip()) for line in lines)
        if not any_heading:
            current_node_id = create_section(doc_title, 1)

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if m:
                level = len(m.group(1))
                heading = m.group(2).strip()
                current_node_id = create_section(heading, level)
            else:
                if current_node_id is None:
                    current_node_id = create_section(doc_title, 1)
                self.nodes[current_node_id].raw_content += line + "\n"

        self.file_roots[str(md_path)] = file_nodes
        self.file_node_positions[str(md_path)] = {node_id: idx for idx, node_id in enumerate(file_nodes)}

        for node_id in file_nodes:
            self._finalize_node(self.nodes[node_id])

    def _finalize_node(self, node: SectionNode) -> None:
        text = node.raw_content.strip()

        commands: list[str] = []
        yamls: list[str] = []
        outputs: list[str] = []
        scripts: list[str] = []

        for lang, content in extract_fenced_code_blocks(text):
            lang = lang.lower()
            if lang in ("bash", "sh", "shell"):
                for line in join_command_continuations(content):
                    cmd = normalize_shell_command(line)
                    if cmd:
                        commands.append(cmd)
                        if cmd.startswith("./") or cmd.startswith("source ./") or cmd.startswith("java -jar") or cmd.startswith("SQL>@") or cmd.startswith("lsnrctl "):
                            scripts.append(cmd)
            elif lang in ("yaml", "yml"):
                if content.strip():
                    yamls.append(content.strip())
            elif lang in ("text", "plain"):
                if content.strip():
                    outputs.append(content.strip())
            else:
                for line in join_command_continuations(content):
                    cmd = normalize_shell_command(line)
                    if is_install_like_command(cmd):
                        commands.append(cmd)

        commands.extend(extract_inline_shell_commands(text))
        yamls.extend(extract_inline_yaml_blocks(text))
        outputs.extend(extract_inline_output_blocks(text))

        node.commands = dedupe_preserve(commands)
        node.yamls = dedupe_preserve(yamls)
        node.outputs = dedupe_preserve(outputs)
        node.scripts = dedupe_preserve(scripts)
        node.steps = extract_numbered_steps(text)

    def _build_indexes(self) -> None:
        for node_id, node in self.nodes.items():
            hkey = normalize_text(node.heading)
            self.heading_index.setdefault(hkey, []).append(node_id)

            tkey = normalize_text(node.doc_title)
            self.doc_title_index.setdefault(tkey, []).append(node_id)

    def _build_haystack_index(self) -> None:
        if not HAYSTACK_AVAILABLE:
            return

        self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        docs: list[Document] = []

        for node in self.nodes.values():
            content = self.build_retrieval_content(node)
            meta = {
                "node_id": node.node_id,
                "heading": node.heading,
                "doc_title": node.doc_title,
                "page_url": node.page_url,
                "file_path": node.file_path,
                "heading_path": node.heading_path_str,
                "has_commands": bool(node.commands),
                "has_yamls": bool(node.yamls),
                "has_outputs": bool(node.outputs),
                "has_scripts": bool(node.scripts),
                "has_steps": bool(node.steps),
                "level": node.level,
            }
            docs.append(Document(id=node.node_id, content=content, meta=meta))

        if self.ollama_available and docs:
            texts = [doc.content for doc in docs]
            try:
                embeddings = []
                for i in range(0, len(texts), EMBED_BATCH_SIZE):
                    batch = texts[i:i + EMBED_BATCH_SIZE]
                    embeddings.extend(self.ollama.embed(batch))
                docs = [replace(doc, embedding=emb) for doc, emb in zip(docs, embeddings)]
                self.embeddings_indexed = True
            except Exception as exc:
                self.embeddings_indexed = False
                if self.debug:
                    print(f"No se pudieron indexar embeddings con Ollama: {exc}")

        self.document_store.write_documents(docs)
        self.haystack_documents = {doc.id: doc for doc in docs}
        self.bm25_retriever = InMemoryBM25Retriever(self.document_store, top_k=BM25_TOP_K, scale_score=True)
        if self.embeddings_indexed:
            self.embedding_retriever = InMemoryEmbeddingRetriever(self.document_store, top_k=EMBED_TOP_K, scale_score=True)

    def build_retrieval_content(self, node: SectionNode) -> str:
        parts = [
            f"Document title: {node.doc_title}",
            f"Heading path: {node.heading_path_str}",
            f"Heading: {node.heading}",
        ]
        if node.commands:
            parts.append("Commands:\n" + "\n".join(node.commands))
        if node.outputs:
            parts.append("Outputs:\n" + "\n".join(node.outputs[:3]))
        if node.yamls:
            parts.append("YAML:\n" + "\n\n".join(node.yamls[:2]))
        if node.steps:
            parts.append("Steps:\n" + "\n".join(node.steps[:8]))
        if node.raw_content.strip():
            parts.append("Content:\n" + node.raw_content.strip())
        return "\n\n".join(parts)

    def get_descendants(self, node_id: str) -> list[SectionNode]:
        out: list[SectionNode] = []
        queue = list(self.nodes[node_id].children_ids)
        while queue:
            current_id = queue.pop(0)
            current = self.nodes[current_id]
            out.append(current)
            queue.extend(current.children_ids)
        return out

    def get_useful_descendants(self, node: SectionNode) -> list[SectionNode]:
        descendants = self.get_descendants(node.node_id)
        return [d for d in descendants if d.commands or d.yamls or d.outputs or d.scripts or d.steps]

    def get_siblings(self, node: SectionNode) -> list[SectionNode]:
        if not node.parent_id:
            return []
        parent = self.nodes[node.parent_id]
        return [self.nodes[nid] for nid in parent.children_ids if nid != node.node_id]

    def get_nearby_file_neighbors(self, node: SectionNode, radius: int = 2) -> list[SectionNode]:
        positions = self.file_node_positions.get(node.file_path, {})
        index = positions.get(node.node_id)
        if index is None:
            return []
        node_ids = self.file_roots.get(node.file_path, [])
        out: list[SectionNode] = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            j = index + offset
            if 0 <= j < len(node_ids):
                out.append(self.nodes[node_ids[j]])
        return out

    def get_structural_neighbors(self, node: SectionNode, radius: int = 2) -> list[SectionNode]:
        candidates = [node]
        if node.parent_id:
            candidates.append(self.nodes[node.parent_id])
        candidates.extend(self.get_descendants(node.node_id)[:6])
        candidates.extend(self.get_siblings(node))
        candidates.extend(self.get_nearby_file_neighbors(node, radius=radius))
        seen = set()
        out: list[SectionNode] = []
        for item in candidates:
            if item.node_id not in seen:
                seen.add(item.node_id)
                out.append(item)
        return out

    def detect_primary_subject(self, query: str) -> str | None:
        q = normalize_text(query)

        explicit_file = re.search(r"([a-zA-Z0-9._/-]+\.(yaml|yml|json|sh))", query, flags=re.IGNORECASE)
        if explicit_file:
            return explicit_file.group(1)

        heading_candidates: list[tuple[int, str]] = []
        for heading_key, node_ids in self.heading_index.items():
            if heading_key and heading_key in q:
                heading_candidates.append((len(heading_key), self.nodes[node_ids[0]].heading))

        if heading_candidates:
            heading_candidates.sort(reverse=True)
            return heading_candidates[0][1]

        doc_candidates: list[tuple[int, str]] = []
        for title_key, node_ids in self.doc_title_index.items():
            if title_key and title_key in q:
                doc_candidates.append((len(title_key), self.nodes[node_ids[0]].doc_title))

        if doc_candidates:
            doc_candidates.sort(reverse=True)
            return doc_candidates[0][1]

        release = extract_release_name_from_query(query)
        if release:
            return release

        terms = simple_query_terms(query)
        for term in sorted(terms, key=len, reverse=True):
            if term in self.heading_index:
                return self.nodes[self.heading_index[term][0]].heading

        return None

    def detect_query_intent(self, query: str) -> dict[str, Any]:
        q = normalize_text(query)
        subject = self.detect_primary_subject(query)

        wants_commands = any(x in q for x in ["command", "commands", "run", "install", "deploy", "setup", "how to install", "what commands"])
        wants_yaml = any(x in q for x in ["yaml", ".yaml", ".yml", "values file", "manifest", "configuration file"])
        wants_output = any(x in q for x in ["output", "outputs", "expected output", "what should i expect", "what should i see", "verify", "validation", "check"])
        wants_script = any(x in q for x in ["script", ".sh", "shell script"])
        wants_steps = any(x in q for x in ["steps", "procedure", "process", "how do i", "how to"])
        wants_prereq = any(x in q for x in ["prerequisite", "prerequisites", "before installing", "requirements"])

        return {
            "subject": subject,
            "wants_commands": wants_commands,
            "wants_yaml": wants_yaml,
            "wants_output": wants_output,
            "wants_script": wants_script,
            "wants_steps": wants_steps,
            "wants_prereq": wants_prereq,
        }

    def subject_score(self, node: SectionNode, subject: str | None) -> int:
        if not subject:
            return 0

        s = normalize_text(subject)
        heading = normalize_text(node.heading)
        path = normalize_text(node.heading_path_str)
        content = normalize_text(node.raw_content)
        title = normalize_text(node.doc_title)

        score = 0
        if heading == s:
            score += 100
        elif s in heading:
            score += 70
        if s in path:
            score += 60
        if s in content:
            score += 45
        if s in title:
            score += 20
        if subject.endswith((".yaml", ".yml")) and s in content:
            score += 30
        return score

    def score_node(self, node: SectionNode, query: str, intent: dict[str, Any]) -> float:
        subject = intent.get("subject")
        score = 0.0

        score += lexical_overlap_score(query, node.heading) * 12
        score += lexical_overlap_score(query, node.heading_path_str) * 8
        score += lexical_overlap_score(query, node.doc_title) * 6
        score += lexical_overlap_score(query, node.raw_content[:1600]) * 4
        score += self.subject_score(node, subject)

        heading_norm = normalize_text(node.heading)
        if any(term in heading_norm for term in VERIFY_HEADING_TERMS):
            score += 6

        if intent["wants_commands"] and node.commands:
            score += 18
        if intent["wants_yaml"] and node.yamls:
            score += 20
        if intent["wants_output"] and node.outputs:
            score += 38
        if intent["wants_script"] and node.scripts:
            score += 18
        if intent["wants_steps"] and node.steps:
            score += 10
        if intent["wants_prereq"] and ("prereq" in heading_norm or "preparation" in heading_norm or "requirements" in heading_norm):
            score += 18

        if intent["wants_output"] and is_install_command_query(query):
            release = extract_release_name_from_query(query)
            if release and any(release in normalize_text(cmd) for cmd in node.commands):
                score += 15
            if release and release in normalize_text(node.raw_content) and node.outputs:
                score += 18
            if any(term in heading_norm for term in {"install helm chart", "verify", "validation", "check"}):
                score += 12

        return score

    def lexical_search(self, query: str, intent: dict[str, Any], top_k: int = 12) -> dict[str, float]:
        scored = {
            node.node_id: self.score_node(node, query, intent)
            for node in self.nodes.values()
        }
        scored = {k: v for k, v in scored.items() if v > 0}
        return dict(sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k])

    def haystack_bm25_search(self, query: str, top_k: int = 12) -> dict[str, float]:
        if not self.bm25_retriever:
            return {}
        result = self.bm25_retriever.run(query=query, top_k=top_k)
        docs = result.get("documents", [])
        return {doc.id: float(getattr(doc, "score", 0.0) or 0.0) for doc in docs}

    def haystack_embedding_search(self, query: str, top_k: int = 12) -> dict[str, float]:
        if not self.embedding_retriever or not self.embeddings_indexed or not self.ollama_available:
            return {}
        try:
            query_embedding = self.ollama.embed([query])[0]
            result = self.embedding_retriever.run(query_embedding=query_embedding, top_k=top_k)
            docs = result.get("documents", [])
            return {doc.id: float(getattr(doc, "score", 0.0) or 0.0) for doc in docs}
        except Exception as exc:
            if self.debug:
                print(f"Embedding retrieval failed: {exc}")
            return {}

    def find_exact_subject_nodes(self, subject: str | None) -> list[SectionNode]:
        if not subject:
            return []
        s = normalize_text(subject)
        out = []
        for node in self.nodes.values():
            if normalize_text(node.heading) == s or normalize_text(node.doc_title) == s:
                out.append(node)
        return out

    def find_best_anchor_node(self, query: str, ranked_nodes: list[SectionNode], intent: dict[str, Any]) -> SectionNode | None:
        candidates: list[tuple[float, SectionNode]] = []
        subject = intent.get("subject")
        for node in ranked_nodes[:20]:
            score = self.score_node(node, query, intent)
            useful_desc = self.get_useful_descendants(node)
            if useful_desc:
                score += min(len(useful_desc), 10) * 4
            if subject and self.subject_score(node, subject) > 0:
                score += 10
            candidates.append((score, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else None

    def find_command_match_nodes(self, query: str) -> list[SectionNode]:
        release = extract_release_name_from_query(query)
        query_norm = normalize_text(query)
        found: list[SectionNode] = []
        for node in self.nodes.values():
            commands_blob = "\n".join(node.commands)
            blob_norm = normalize_text(commands_blob)
            if release and any(release == tok or release in normalize_text(cmd) for cmd in node.commands for tok in [release]):
                found.append(node)
                continue
            if "helm install" in query_norm and "helm install" in blob_norm:
                found.append(node)
                continue
            if "helm upgrade" in query_norm and "helm upgrade" in blob_norm:
                found.append(node)
                continue
        return found

    def output_context_nodes(self, query: str) -> list[SectionNode]:
        release = extract_release_name_from_query(query)
        command_nodes = self.find_command_match_nodes(query)
        candidates: list[SectionNode] = []
        for cmd_node in command_nodes:
            candidates.extend(self.get_structural_neighbors(cmd_node, radius=3))
            for node_id in self.file_roots.get(cmd_node.file_path, []):
                node = self.nodes[node_id]
                if node.outputs:
                    candidates.append(node)
                elif release and release in normalize_text(node.raw_content) and node.outputs:
                    candidates.append(node)
                elif release and release in normalize_text(node.heading_path_str) and node.outputs:
                    candidates.append(node)

        seen = set()
        filtered: list[SectionNode] = []
        for node in candidates:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            if node.outputs or any(term in normalize_text(node.heading) for term in VERIFY_HEADING_TERMS):
                filtered.append(node)
        return filtered

    def expand_anchor_context(self, anchor: SectionNode, top_k: int = 12) -> list[SectionNode]:
        useful_desc = self.get_useful_descendants(anchor)
        same_file_useful = [
            self.nodes[node_id] for node_id in self.file_roots.get(anchor.file_path, [])
            if node_id != anchor.node_id and (self.nodes[node_id].commands or self.nodes[node_id].yamls or self.nodes[node_id].outputs or self.nodes[node_id].scripts or self.nodes[node_id].steps)
        ]
        nodes = [anchor] + useful_desc + same_file_useful
        seen = set()
        out: list[SectionNode] = []
        for node in nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                out.append(node)
        return out[:top_k]

    def search(self, query: str, top_k: int = 10) -> list[SectionNode]:
        intent = self.detect_query_intent(query)
        lexical_scores = self.lexical_search(query, intent, top_k=max(top_k, 12))
        bm25_scores = self.haystack_bm25_search(query, top_k=max(top_k, 12)) if self.enable_haystack else {}
        embed_scores = self.haystack_embedding_search(query, top_k=max(top_k, 12)) if self.enable_haystack else {}

        normalized_lexical = normalize_score_map(lexical_scores)
        normalized_bm25 = normalize_score_map(bm25_scores)
        normalized_embed = normalize_score_map(embed_scores)
        rrf = reciprocal_rank_fusion([
            list(lexical_scores.keys()),
            list(bm25_scores.keys()),
            list(embed_scores.keys()),
        ])

        combined: dict[str, float] = {}
        all_ids = set(normalized_lexical) | set(normalized_bm25) | set(normalized_embed) | set(rrf)
        for node_id in all_ids:
            combined[node_id] = (
                normalized_lexical.get(node_id, 0.0) * 0.45 +
                normalized_bm25.get(node_id, 0.0) * 0.30 +
                normalized_embed.get(node_id, 0.0) * 0.25 +
                rrf.get(node_id, 0.0) * 10.0
            )

        if intent["wants_output"] and is_install_command_query(query):
            for node in self.output_context_nodes(query):
                combined[node.node_id] = combined.get(node.node_id, 0.0) + 2.75

        subject = intent["subject"]
        exact_subject_nodes = self.find_exact_subject_nodes(subject)
        if exact_subject_nodes:
            for node in exact_subject_nodes:
                combined[node.node_id] = combined.get(node.node_id, 0.0) + 2.0
                for desc in self.get_useful_descendants(node):
                    combined[desc.node_id] = combined.get(desc.node_id, 0.0) + 1.2

        ranked = [self.nodes[node_id] for node_id, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)]
        if not ranked:
            ranked = sorted(self.nodes.values(), key=lambda n: self.score_node(n, query, intent), reverse=True)

        anchor = self.find_best_anchor_node(query, ranked, intent)
        if intent["wants_output"] and is_install_command_query(query):
            output_neighbors = self.output_context_nodes(query)
            if output_neighbors:
                enriched = output_neighbors + ranked
                seen = set()
                out: list[SectionNode] = []
                for node in enriched:
                    if node.node_id not in seen:
                        seen.add(node.node_id)
                        out.append(node)
                return out[:top_k]

        if anchor and (intent["wants_commands"] or intent["wants_yaml"] or intent["wants_output"] or intent["wants_script"] or intent["wants_steps"]):
            expanded = self.expand_anchor_context(anchor, top_k=max(top_k, 12))
            if intent["wants_output"]:
                extra_output = [n for n in ranked if n.outputs][:6]
                expanded = expanded + extra_output
            seen = set()
            out: list[SectionNode] = []
            for node in expanded:
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    out.append(node)
            return out[:top_k]

        return ranked[:top_k]

    def collect_commands(self, nodes: list[SectionNode], install_only: bool = False) -> list[str]:
        commands: list[str] = []
        for node in nodes:
            cmds = node.commands
            if install_only:
                cmds = [c for c in cmds if is_install_like_command(c)]
            commands.extend(cmds)
        return dedupe_preserve(commands)

    def collect_yamls(self, nodes: list[SectionNode]) -> list[str]:
        yamls: list[str] = []
        for node in nodes:
            yamls.extend(node.yamls)
        return dedupe_preserve(yamls)

    def collect_scripts(self, nodes: list[SectionNode]) -> list[str]:
        scripts: list[str] = []
        for node in nodes:
            scripts.extend(node.scripts)
        return dedupe_preserve(scripts)

    def collect_outputs(self, nodes: list[SectionNode]) -> list[str]:
        outputs: list[str] = []
        for node in nodes:
            outputs.extend(node.outputs)
        return dedupe_preserve(outputs)

    def collect_steps(self, nodes: list[SectionNode]) -> list[str]:
        steps: list[str] = []
        for node in nodes:
            steps.extend(node.steps)
        return dedupe_preserve(steps)

    def unique_sources(self, nodes: list[SectionNode], max_items: int = 8) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        seen = set()
        for node in nodes:
            item = (node.doc_title, node.heading, node.page_url or "N/A")
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out[:max_items]

    def group_nodes_by_subtopic(self, nodes: list[SectionNode], anchor: SectionNode | None = None) -> dict[str, list[SectionNode]]:
        grouped: dict[str, list[SectionNode]] = {}
        anchor_id = anchor.node_id if anchor else None
        for node in nodes:
            if anchor_id and node.node_id == anchor_id:
                continue
            label = node.heading
            if anchor and node.heading_path:
                if len(node.heading_path) > len(anchor.heading_path):
                    label = node.heading_path[len(anchor.heading_path)]
                elif node.heading != anchor.heading:
                    label = node.heading
                else:
                    label = anchor.heading
            grouped.setdefault(label, []).append(node)
        if not grouped and anchor:
            grouped[anchor.heading] = [anchor]
        return grouped

    def build_grouped_command_answer(self, query: str, nodes: list[SectionNode], anchor: SectionNode | None = None) -> str:
        grouped_nodes = self.group_nodes_by_subtopic(nodes, anchor=anchor)
        grouped_commands: dict[str, list[str]] = {}
        for heading, group in grouped_nodes.items():
            cmds = self.collect_commands(group, install_only=is_install_command_query(query))
            if cmds:
                grouped_commands[heading] = cmds

        if not grouped_commands and anchor:
            anchor_cmds = self.collect_commands([anchor], install_only=is_install_command_query(query))
            if anchor_cmds:
                grouped_commands[anchor.heading] = anchor_cmds

        if not grouped_commands:
            return self.build_fallback_answer(nodes, query)

        title = anchor.heading if anchor else (self.detect_primary_subject(query) or "el tema solicitado")
        parts = ["Respuesta", f"La documentación muestra los siguientes comandos agrupados para {title}:"]
        for heading, commands in grouped_commands.items():
            parts.extend(["", heading, "```bash"])
            parts.extend(commands)
            parts.append("```")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_command_answer(self, query: str, nodes: list[SectionNode], anchor: SectionNode | None = None) -> str:
        if anchor and len(self.get_useful_descendants(anchor)):
            return self.build_grouped_command_answer(query, nodes, anchor=anchor)

        intent = self.detect_query_intent(query)
        commands = self.collect_commands(nodes, install_only=is_install_command_query(query))
        if not commands:
            return self.build_fallback_answer(nodes, query)

        subject = intent["subject"] or "el componente solicitado"
        parts = [
            "Respuesta",
            f"Los siguientes comandos están documentados para {subject}:",
            "",
            "Comandos",
            "```bash",
        ]
        parts.extend(commands)
        parts.append("```")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_yaml_answer(self, query: str, nodes: list[SectionNode]) -> str:
        subject = self.detect_primary_subject(query)
        yamls = self.collect_yamls(nodes)
        if not yamls:
            return self.build_fallback_answer(nodes, query)
        parts = ["Respuesta", f"Los siguientes bloques YAML están documentados para {subject or 'el tema solicitado'}:", "", "YAML"]
        for yml in yamls[:3]:
            parts.append("```yaml")
            parts.append(yml)
            parts.append("```")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_script_answer(self, query: str, nodes: list[SectionNode]) -> str:
        scripts = self.collect_scripts(nodes)
        if not scripts:
            return self.build_fallback_answer(nodes, query)
        parts = ["Respuesta", "La documentación muestra los siguientes scripts o entradas principales:", "", "Scripts"]
        for script in scripts[:4]:
            parts.append("```bash")
            parts.append(script)
            parts.append("```")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_output_answer(self, query: str, nodes: list[SectionNode]) -> str:
        intent = self.detect_query_intent(query)
        selected_nodes = nodes
        if intent["wants_output"] and is_install_command_query(query):
            bridge_nodes = self.output_context_nodes(query)
            if bridge_nodes:
                selected_nodes = bridge_nodes + nodes
                seen = set()
                deduped: list[SectionNode] = []
                for node in selected_nodes:
                    if node.node_id not in seen:
                        seen.add(node.node_id)
                        deduped.append(node)
                selected_nodes = deduped

        outputs = self.collect_outputs(selected_nodes)
        if not outputs:
            return self.build_fallback_answer(selected_nodes, query)

        parts = ["Respuesta", "La documentación muestra las siguientes salidas esperadas:", "", "Salida esperada"]
        for out in outputs[:3]:
            parts.append("```text")
            parts.append(out)
            parts.append("```")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(selected_nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_extractive_answer(self, query: str, nodes: list[SectionNode]) -> str:
        intent = self.detect_query_intent(query)
        excerpts: list[str] = []
        q_terms = set(simple_query_terms(query))
        for node in nodes[:5]:
            sentences = split_sentences_naive(node.raw_content)
            ranked: list[tuple[int, str]] = []
            for s in sentences:
                score = len(q_terms & set(simple_query_terms(s)))
                if score > 0:
                    ranked.append((score, s))
            if ranked:
                ranked.sort(reverse=True, key=lambda x: x[0])
                excerpts.append(ranked[0][1])
        excerpts = dedupe_preserve(excerpts)
        if not excerpts:
            return self.build_fallback_answer(nodes, query)
        parts = ["Respuesta", " ".join(excerpts[:3])]
        steps = self.collect_steps(nodes)
        if intent["wants_steps"] and steps:
            parts.extend(["", "Pasos"])
            for i, step in enumerate(steps[:8], 1):
                parts.append(f"{i}. {step}")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_fallback_answer(self, nodes: list[SectionNode], query: str) -> str:
        if not nodes:
            return "Respuesta\nNo encontré información suficiente en el corpus Markdown para responder esa pregunta."
        excerpts = []
        for node in nodes[:3]:
            excerpt = node.raw_content.strip()[:500]
            if excerpt:
                excerpts.append(excerpt)
        parts = [
            "Respuesta",
            "La información recuperada no permitió construir una respuesta más específica. A continuación se muestran fragmentos relevantes de la documentación:",
            "",
            "Información soportada",
        ]
        for i, ex in enumerate(excerpts, 1):
            parts.append(f"{i}. {ex}")
        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")
        return "\n".join(parts)

    def build_ollama_prompt(self, query: str, nodes: list[SectionNode], intent: dict[str, Any]) -> str:
        context_blocks = []
        for idx, node in enumerate(nodes[:6], 1):
            block = {
                "doc_title": node.doc_title,
                "heading": node.heading,
                "page_url": node.page_url,
                "commands": node.commands[:6],
                "outputs": node.outputs[:3],
                "yamls": node.yamls[:2],
                "steps": node.steps[:8],
                "content": node.raw_content[:1500],
            }
            context_blocks.append(f"Context {idx}:\n{json.dumps(block, ensure_ascii=False, indent=2)}")

        desired = "Responde en español, sin inventar datos, citando literalmente solo lo necesario."
        if intent["wants_output"]:
            desired += " Prioriza tablas o salidas esperadas."
        if intent["wants_commands"]:
            desired += " Prioriza comandos."

        return (
            "Eres un asistente de QA técnico para documentación Markdown.\n"
            f"Pregunta: {query}\n\n"
            f"{desired}\n\n"
            + "\n\n".join(context_blocks)
            + "\n\nSi el contexto no basta, dilo claramente."
        )

    def answer(self, query: str) -> str:
        intent = self.detect_query_intent(query)
        nodes = self.search(query, top_k=FINAL_TOP_K)

        ranked_all = sorted(self.nodes.values(), key=lambda n: self.score_node(n, query, intent), reverse=True)
        anchor = self.find_best_anchor_node(query, ranked_all, intent)

        if self.use_ollama_generation and self.ollama_available:
            try:
                prompt = self.build_ollama_prompt(query, nodes, intent)
                llm_answer = self.ollama.generate_answer(prompt)
                if llm_answer:
                    parts = ["Respuesta", llm_answer, "", "Fuente"]
                    for title, heading, url in self.unique_sources(nodes):
                        parts.append(f"- {title} — {heading} ({url})")
                    return "\n".join(parts)
            except Exception as exc:
                if self.debug:
                    print(f"Fallo la generación con Ollama: {exc}")

        if intent["wants_output"]:
            return self.build_output_answer(query, nodes)
        if intent["wants_yaml"] and not intent["wants_commands"]:
            return self.build_yaml_answer(query, nodes)
        if intent["wants_script"]:
            return self.build_script_answer(query, nodes)
        if intent["wants_commands"]:
            return self.build_command_answer(query, nodes, anchor=anchor)
        return self.build_extractive_answer(query, nodes)
