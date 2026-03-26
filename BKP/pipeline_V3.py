from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import requests

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


# =========================================================
# CONFIG
# =========================================================

MARKDOWN_DIR = "./out_md"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

TOP_K_RETRIEVE = 18
TOP_K_FINAL = 10
DEBUG_RETRIEVAL = True

MAX_CONTEXT_CHARS_PER_DOC = 2600
MAX_ALLOWED_BLOCKS_PER_TYPE = 8


# =========================================================
# GENERAL HELPERS
# =========================================================

def collapse_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").strip())


def normalize_text(text: str) -> str:
    return collapse_ws(text).lower()


def merge_meta(*metas: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for meta in metas:
        if meta:
            merged.update(meta)
    return merged


def dedupe_preserve(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def extract_urls(text: str) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s\)>\]]+", text or "")))


def simple_query_terms(text: str) -> list[str]:
    stop = {
        "how", "can", "what", "which", "when", "where", "the", "for", "and", "are", "with",
        "using", "use", "into", "from", "that", "this", "must", "need", "should", "would",
        "you", "your", "que", "como", "cuál", "cual", "para", "con", "del", "las", "los",
        "uepe", "usage", "engine", "private", "edition", "installing", "installation",
        "aws", "eks", "cluster", "kubernetes", "need", "commands", "command"
    }
    tokens = re.findall(r"[a-zA-Z0-9_.\-/]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in stop]


def lexical_overlap_score(query: str, content: str) -> int:
    q_terms = set(simple_query_terms(query))
    c_terms = set(simple_query_terms(content))
    return len(q_terms & c_terms)


def split_sentences_naive(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?\:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# =========================================================
# FILE / METADATA HELPERS
# =========================================================

def load_sidecar_meta(md_path: Path) -> dict[str, Any]:
    sidecar = md_path.parent / "meta.json"
    if not sidecar.exists():
        return {}
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)
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


# =========================================================
# SUBJECT / INTENT
# =========================================================

def detect_primary_subject(query: str) -> str | None:
    q = normalize_text(query)

    collection_subjects = [
        "aws add-ons",
        "aws addons",
        "kubernetes add-ons",
        "kubernetes addons",
        "add-ons for uepe",
        "addons for uepe",
        "uepe aws add-ons",
        "aws add-ons for uepe",
    ]
    for subject in collection_subjects:
        if subject in q:
            return "aws-addons-collection"

    known_subjects = [
        "external-dns",
        "ingress-nginx",
        "ingress-nginx-controller",
        "aws-load-balancer-controller",
        "aws load balancer controller",
        "efs-csi",
        "efs-csi-controller",
        "ebs-csi",
        "ebs-csi-controller",
        "cert-manager",
        "cluster-autoscaler",
        "fluent-bit",
        "kibana",
        "elasticsearch",
        "env-secrets",
        "ecr-cred",
        "uepe-eks.yaml",
        "kubeconfig.yaml",
        "uepe-values.yaml",
        "example-issuer.yaml",
        "platform",
        "oracle",
        "postgresql",
        "postgres",
        "saphana",
        "tls",
        "clusterissuer",
        "licensekey",
    ]

    for subject in known_subjects:
        if subject in q:
            return subject

    return None


def detect_query_intent(query: str) -> dict[str, Any]:
    q = normalize_text(query)

    wants_commands = any(x in q for x in [
        "what commands", "which commands", "commands do i need", "command do i need",
        "show commands", "installation commands", "install commands",
        "comandos", "comando", "ejecutar", "ejecuto", "run ", "execute "
    ])

    wants_yaml = any(x in q for x in [
        "yaml", ".yaml", ".yml", "manifest", "values file", "values.yaml",
        "show me the yaml", "show yaml", "muéstrame el yaml"
    ])

    wants_output = any(x in q for x in [
        "output", "salida", "expected output", "what should i see",
        "qué debo ver", "show output", "logs", "log output"
    ])

    wants_script = any(x in q for x in [
        "script", ".sh", "entry point", "main script", "sqlplus", "uepe-sys-db-tool", "jar"
    ])

    wants_steps = any(x in q for x in [
        "how", "cómo", "steps", "pasos", "install", "installation", "create", "set up", "setup"
    ])

    wants_prereq = any(x in q for x in [
        "prerequisite", "pre-requisite", "requirements", "tools", "iam", "license",
        "access key", "access keys", "prerequisito"
    ])

    wants_troubleshooting = any(x in q for x in [
        "error", "issue", "problem", "fails", "timeout", "troubleshoot", "debug"
    ])

    topic = None
    if any(x in q for x in ["external-dns", "ingress-nginx", "aws-load-balancer-controller", "cert-manager", "efs-csi", "ebs-csi", "cluster-autoscaler", "add-on", "addons", "add-ons"]):
        topic = "addons"
    elif any(x in q for x in ["cluster", "kubeconfig", "uepe-eks.yaml", "terraform.tfvars", "nodegroup"]):
        topic = "cluster"
    elif any(x in q for x in ["tls", "clusterissuer", "issuer", "keystore", "tls.crt", "tls.key"]):
        topic = "tls"
    elif any(x in q for x in ["secret", "env-secrets", "ecr-cred", "docker-registry"]):
        topic = "secrets"
    elif any(x in q for x in ["postgres", "oracle", "database", "jdbc", "saphana", "db"]):
        topic = "database"
    elif any(x in q for x in ["fluent-bit", "cloudwatch", "kibana", "elasticsearch", "logs", "logging"]):
        topic = "logging"
    elif any(x in q for x in ["helm install uepe", "uepe-values", "platform", "desktop-online", "licensekey"]):
        topic = "installation"

    return {
        "wants_commands": wants_commands,
        "wants_yaml": wants_yaml,
        "wants_output": wants_output,
        "wants_script": wants_script,
        "wants_steps": wants_steps,
        "wants_prereq": wants_prereq,
        "wants_troubleshooting": wants_troubleshooting,
        "topic": topic,
        "subject": detect_primary_subject(query),
    }


def is_install_command_query(query: str) -> bool:
    q = normalize_text(query)
    return (
        ("command" in q or "commands" in q or "comando" in q or "comandos" in q)
        and ("install" in q or "instalar" in q or "installation" in q)
    )


# =========================================================
# MARKDOWN PARSING
# =========================================================

def extract_fenced_code_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """
    Returns (lang, content, start_idx, end_idx)
    """
    blocks = []
    pattern = r"```([a-zA-Z0-9_-]*)\n(.*?)```"
    for m in re.finditer(pattern, text or "", flags=re.DOTALL):
        lang = (m.group(1) or "").strip().lower()
        content = (m.group(2) or "").rstrip()
        blocks.append((lang, content, m.start(), m.end()))
    return blocks


def remove_fenced_code_blocks(text: str) -> str:
    return re.sub(r"```([a-zA-Z0-9_-]*)\n(.*?)```", "", text or "", flags=re.DOTALL)


def parse_markdown_sections(text: str) -> list[dict[str, Any]]:
    lines = (text or "").splitlines()
    sections = []

    current_title = "Document"
    current_level = 1
    current_lines: list[str] = []
    seen_first_heading = False

    def flush():
        nonlocal current_title, current_level, current_lines
        body = "\n".join(current_lines).strip()
        sections.append(
            {
                "heading": current_title,
                "heading_level": current_level,
                "content": body,
            }
        )
        current_lines = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if seen_first_heading or current_lines:
                flush()
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            seen_first_heading = True
        else:
            current_lines.append(line)

    if current_lines or not sections:
        flush()

    return [s for s in sections if s["content"].strip() or s["heading"].strip()]


def normalize_shell_command(cmd: str) -> str:
    cmd = (cmd or "").strip()
    cmd = cmd.strip("`").strip()
    cmd = re.sub(r"^\$\s+", "", cmd)
    cmd = cmd.replace("\\\n", " ")
    cmd = cmd.replace("\\", " ")
    cmd = re.sub(r"\s+", " ", cmd).strip()
    cmd = cmd.replace(" .alpha .", ".alpha.").replace(" .kubernetes .", ".kubernetes.")
    return cmd


def shell_tokens(cmd: str) -> list[str]:
    return normalize_shell_command(cmd).split()


def join_command_continuations(text: str) -> list[str]:
    lines = (text or "").splitlines()
    out = []
    i = 0

    while i < len(lines):
        current = lines[i].rstrip()
        if not current.strip():
            out.append(current)
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


def extract_inline_shell_commands(text: str) -> list[str]:
    patterns = [
        r"(helm\s+repo\s+add[^\n]+)",
        r"(helm\s+repo\s+update[^\n]*)",
        r"(helm\s+install[^\n]+)",
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
    joined_lines = join_command_continuations(text)

    for line in joined_lines:
        for pattern in patterns:
            matches = re.findall(pattern, line, flags=re.IGNORECASE)
            for m in matches:
                cmd = normalize_shell_command(m)
                if cmd:
                    found.append(cmd)

        m = re.search(
            r":\s*(helm\s+repo\s+add[^\n]+|helm\s+repo\s+update[^\n]*|helm\s+install[^\n]+|kubectl\s+apply\s+-f[^\n]+|kubectl\s+get[^\n]+|terraform\s+init[^\n]*|terraform\s+plan[^\n]*|terraform\s+apply[^\n]*|eksctl\s+create\s+cluster[^\n]+|export\s+KUBECONFIG[^\n]+)",
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

    if re.match(r"^[A-Za-z0-9_.-]+:\s*.*$", stripped):
        return True

    return False


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


def classify_code_block(lang: str, content: str, heading: str, surrounding_text: str) -> str:
    lc = (lang or "").lower()
    text = (content or "").strip()
    head = normalize_text(heading)
    around = normalize_text(surrounding_text)

    if lc in ("yaml", "yml"):
        return "yaml"

    if lc in ("bash", "sh", "shell"):
        if text.startswith("#!/"):
            return "script"
        return "shell_command"

    if lc == "json":
        return "json"

    if lc == "sql":
        return "sql"

    if text.startswith("#!/"):
        return "script"

    if re.search(r"^(apiVersion:|kind:|metadata:|spec:|rules:|controller:|aws:|global:)", text, flags=re.MULTILINE):
        return "yaml"

    if re.search(r"^(kubectl |helm |terraform |eksctl |aws |docker |java -jar |curl |export |\./)", text, flags=re.MULTILINE):
        return "shell_command"

    if re.search(r"^(SQL>@|SQL>|lsnrctl |source \./)", text, flags=re.MULTILINE):
        return "script"

    if "output" in head or "output" in around:
        return "output"

    if re.search(r"^\s*NAME\s+NAMESPACE\s+REVISION", text, flags=re.MULTILINE):
        return "output"

    if re.search(r"^\s*NAME\s+READY\s+STATUS", text, flags=re.MULTILINE):
        return "output"

    return "code"


def infer_topic_tags(title: str, heading: str, content: str) -> list[str]:
    text = normalize_text(f"{title}\n{heading}\n{content}")
    tags = []

    mapping = {
        "cluster": ["uepe-eks.yaml", "eksctl", "kubeconfig", "terraform.tfvars", "create cluster", "nodegroup"],
        "addons": ["external-dns", "ingress-nginx", "aws-load-balancer-controller", "efs-csi", "ebs-csi", "cluster-autoscaler", "helm repo"],
        "tls": ["clusterissuer", "cert-manager", "tls", "keystore", "tls.crt", "tls.key"],
        "secrets": ["env-secrets", "kubectl create secret", "docker-registry", "ecr-cred"],
        "database": ["uepe-sys-db-tool.jar", "postgres", "oracle", "saphana", "jdbc", "dbadmin"],
        "installation": ["helm install uepe", "uepe-values.yaml", "licensekey", "desktop-online", "platform"],
        "logging": ["fluent-bit", "cloudwatch", "elasticsearch", "kibana", "log groups"],
        "output_examples": ["apply complete", "ready status", "kubectl get pods", "helm list", "create_complete"],
    }

    for tag, needles in mapping.items():
        if any(n in text for n in needles):
            tags.append(tag)

    return sorted(set(tags))


def build_documents_from_markdown(md_path: Path) -> list[Document]:
    raw = md_path.read_text(encoding="utf-8", errors="ignore")
    sidecar_meta = load_sidecar_meta(md_path)

    title = sidecar_meta.get("title") or extract_title_from_markdown(raw, md_path.stem)
    page_url = infer_page_url(sidecar_meta)

    sections = parse_markdown_sections(raw)
    docs: list[Document] = []

    base_meta = {
        "source_file": str(md_path),
        "doc_title": title,
        "page_url": page_url,
        "page_id": sidecar_meta.get("page_id"),
        "Platform": "UEPE",
        "version": "5.2",
    }

    for idx, section in enumerate(sections):
        heading = section["heading"]
        content = section["content"]
        topic_tags = infer_topic_tags(title, heading, content)

        section_meta = merge_meta(
            base_meta,
            {
                "doc_type": "section",
                "heading": heading,
                "heading_level": section["heading_level"],
                "section_index": idx,
                "topic_tags": topic_tags,
            },
        )

        section_text = f"# {title}\n## {heading}\n\n{remove_fenced_code_blocks(content).strip()}".strip()
        docs.append(Document(content=section_text, meta=section_meta))

        blocks = extract_fenced_code_blocks(content)
        for block_idx, (lang, block_content, start, end) in enumerate(blocks):
            before = content[max(0, start - 250):start]
            after = content[end:min(len(content), end + 250)]
            block_type = classify_code_block(lang, block_content, heading, before + "\n" + after)

            block_meta = merge_meta(
                base_meta,
                {
                    "doc_type": "block",
                    "block_type": block_type,
                    "block_lang": lang or "plain",
                    "heading": heading,
                    "heading_level": section["heading_level"],
                    "section_index": idx,
                    "block_index": block_idx,
                    "topic_tags": topic_tags,
                },
            )

            block_text = (
                f"# {title}\n"
                f"## {heading}\n"
                f"[BLOCK TYPE: {block_type}]\n"
                f"```{lang}\n{block_content}\n```"
            )
            docs.append(Document(content=block_text, meta=block_meta))

        for inline_idx, cmd in enumerate(extract_inline_shell_commands(content)):
            meta = merge_meta(
                base_meta,
                {
                    "doc_type": "block",
                    "block_type": "shell_command",
                    "block_lang": "shell",
                    "heading": heading,
                    "heading_level": section["heading_level"],
                    "section_index": idx,
                    "block_index": f"inline-shell-{inline_idx}",
                    "topic_tags": topic_tags,
                },
            )
            block_text = f"# {title}\n## {heading}\n[BLOCK TYPE: shell_command]\n```bash\n{cmd}\n```"
            docs.append(Document(content=block_text, meta=meta))

        for inline_idx, yml in enumerate(extract_inline_yaml_blocks(content)):
            meta = merge_meta(
                base_meta,
                {
                    "doc_type": "block",
                    "block_type": "yaml",
                    "block_lang": "yaml",
                    "heading": heading,
                    "heading_level": section["heading_level"],
                    "section_index": idx,
                    "block_index": f"inline-yaml-{inline_idx}",
                    "topic_tags": topic_tags,
                },
            )
            block_text = f"# {title}\n## {heading}\n[BLOCK TYPE: yaml]\n```yaml\n{yml}\n```"
            docs.append(Document(content=block_text, meta=meta))

        for inline_idx, out in enumerate(extract_inline_output_blocks(content)):
            meta = merge_meta(
                base_meta,
                {
                    "doc_type": "block",
                    "block_type": "output",
                    "block_lang": "text",
                    "heading": heading,
                    "heading_level": section["heading_level"],
                    "section_index": idx,
                    "block_index": f"inline-output-{inline_idx}",
                    "topic_tags": topic_tags,
                },
            )
            block_text = f"# {title}\n## {heading}\n[BLOCK TYPE: output]\n```text\n{out}\n```"
            docs.append(Document(content=block_text, meta=meta))

    return docs


def load_corpus(markdown_dir: str) -> list[Document]:
    base_dir = Path(__file__).resolve().parent
    path = (base_dir / markdown_dir).resolve()

    print(f"Base dir: {base_dir}")
    print(f"Searching markdown in: {path}")

    if not path.exists():
        raise FileNotFoundError(f"La carpeta no existe: {path}")

    files = sorted([p for p in path.rglob("*.md") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .md en {path}")

    print("Markdown files found:")
    for f in files:
        print(" -", f)

    all_docs: list[Document] = []
    for f in files:
        all_docs.extend(build_documents_from_markdown(f))

    print(f"Built {len(all_docs)} Haystack documents")
    return all_docs


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


def index_documents(document_store: QdrantDocumentStore, documents: list[Document]) -> None:
    embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDER_MODEL)
    writer = DocumentWriter(document_store)

    print("Embedding documents...")
    embed_result = embedder.run(documents=documents)
    embedded_docs = embed_result["documents"]

    print("Writing documents to Qdrant...")
    writer.run(documents=embedded_docs)


# =========================================================
# RETRIEVAL HELPERS
# =========================================================

def is_aws_addons_page(doc: Document) -> bool:
    meta = doc.meta or {}
    title = normalize_text(meta.get("doc_title", ""))
    source_file = normalize_text(meta.get("source_file", ""))

    return (
        "kubernetes cluster add-ons - aws" in title
        or "kubernetes_cluster_add_ons_aws" in source_file
    )


def filter_docs_for_subject(documents: list[Document], subject: str | None) -> list[Document]:
    if not subject:
        return documents

    if subject == "aws-addons-collection":
        aws_docs = [d for d in documents if is_aws_addons_page(d)]
        if not aws_docs:
            return documents

        useful = []
        for d in aws_docs:
            meta = d.meta or {}
            heading = normalize_text(meta.get("heading", ""))
            content = normalize_text(d.content or "")

            if heading in {"aws add-ons", "kubernetes add-ons"} and len(content.strip()) < 120:
                continue

            useful.append(d)

        return useful or aws_docs

    subject_norm = normalize_text(subject)

    exact_heading_docs = []
    heading_match_docs = []
    title_match_docs = []
    content_match_docs = []

    for doc in documents:
        meta = doc.meta or {}
        heading = normalize_text(meta.get("heading", ""))
        title = normalize_text(meta.get("doc_title", ""))
        content = normalize_text(doc.content or "")

        if heading == subject_norm:
            exact_heading_docs.append(doc)
        elif subject_norm in heading:
            heading_match_docs.append(doc)
        elif subject_norm in title:
            title_match_docs.append(doc)
        elif subject_norm in content:
            content_match_docs.append(doc)

    if exact_heading_docs:
        return exact_heading_docs
    if heading_match_docs:
        return heading_match_docs
    if title_match_docs:
        return title_match_docs
    if content_match_docs:
        return content_match_docs

    return documents


# =========================================================
# RETRIEVER
# =========================================================

class InstallationRetriever:
    def __init__(self, document_store: QdrantDocumentStore):
        self.document_store = document_store
        self.text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL)
        self.retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=TOP_K_RETRIEVE)

    def retrieve(self, query: str) -> list[Document]:
        intent = detect_query_intent(query)
        subject = intent["subject"]

        embed_result = self.text_embedder.run(text=query)
        query_embedding = embed_result["embedding"]
        retrieved = self.retriever.run(query_embedding=query_embedding)["documents"]

        scored_docs: list[Document] = []

        for doc in retrieved:
            meta = dict(doc.meta or {})
            content = doc.content or ""
            overlap = lexical_overlap_score(query, content)
            bonus = 0
            penalty = 0
            exact_subject_bonus = 0

            block_type = meta.get("block_type")
            topic_tags = set(meta.get("topic_tags") or [])
            heading_norm = normalize_text(meta.get("heading", ""))
            title_norm = normalize_text(meta.get("doc_title", ""))
            content_norm = normalize_text(content)

            if intent["wants_commands"] and block_type == "shell_command":
                bonus += 4
            if intent["wants_yaml"] and block_type == "yaml":
                bonus += 5
            if intent["wants_script"] and block_type in ("script", "sql"):
                bonus += 5
            if intent["wants_output"] and block_type == "output":
                bonus += 5
            if intent["wants_steps"] and meta.get("doc_type") == "section":
                bonus += 2

            if intent["topic"] and intent["topic"] in topic_tags:
                bonus += 4

            if subject:
                subject_norm = normalize_text(subject)
                if subject == "aws-addons-collection":
                    if is_aws_addons_page(doc):
                        exact_subject_bonus += 10
                else:
                    if heading_norm == subject_norm:
                        exact_subject_bonus += 10
                    elif subject_norm in heading_norm:
                        exact_subject_bonus += 8
                    elif subject_norm in title_norm:
                        exact_subject_bonus += 3
                    elif subject_norm in content_norm:
                        exact_subject_bonus += 4

            q_norm = normalize_text(query)

            if ("create" in q_norm or "set up" in q_norm or "cluster" in q_norm) and "ingress-nginx" in content_norm:
                penalty += 1

            if subject and normalize_text(subject) == "external-dns":
                if "external-dns.alpha.kubernetes.io" in content_norm and heading_norm != "external-dns":
                    penalty += 4

            if subject == "aws-addons-collection" and not is_aws_addons_page(doc):
                penalty += 4

            meta["lexical_overlap"] = overlap
            meta["intent_bonus"] = bonus
            meta["retrieval_penalty"] = penalty
            meta["exact_subject_bonus"] = exact_subject_bonus

            new_doc = replace(doc, meta=meta)
            scored_docs.append(new_doc)

        scored_docs.sort(
            key=lambda d: (
                (d.meta or {}).get("exact_subject_bonus", 0),
                (d.meta or {}).get("intent_bonus", 0),
                (d.meta or {}).get("lexical_overlap", 0),
                -((d.meta or {}).get("retrieval_penalty", 0)),
                getattr(d, "score", 0.0) or 0.0,
            ),
            reverse=True,
        )

        if subject:
            filtered = filter_docs_for_subject(scored_docs, subject)
            if filtered:
                scored_docs = filtered

        return scored_docs[:TOP_K_FINAL]


# =========================================================
# BLOCK COLLECTION
# =========================================================

def collect_allowed_blocks(documents: list[Document]) -> dict[str, list[str]]:
    allowed = {
        "shell_command": [],
        "yaml": [],
        "script": [],
        "output": [],
        "json": [],
        "sql": [],
    }

    for doc in documents:
        meta = doc.meta or {}
        block_type = meta.get("block_type")

        if meta.get("doc_type") == "block" and block_type in allowed:
            for lang, block, _, _ in extract_fenced_code_blocks(doc.content or ""):
                if block.strip():
                    allowed[block_type].append(block.strip())

        if meta.get("doc_type") == "section":
            content = doc.content or ""
            allowed["shell_command"].extend(extract_inline_shell_commands(content))
            allowed["yaml"].extend(extract_inline_yaml_blocks(content))
            allowed["output"].extend(extract_inline_output_blocks(content))

    for k in allowed:
        allowed[k] = dedupe_preserve(allowed[k])[:MAX_ALLOWED_BLOCKS_PER_TYPE]

    return allowed


def is_install_like_command(cmd: str, subject: str | None = None) -> bool:
    c = normalize_shell_command(cmd)
    s = normalize_text(subject or "")

    install_prefixes = (
        "helm repo add ",
        "helm repo update",
        "helm install ",
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

    if not any(c.startswith(p) for p in install_prefixes):
        return False

    if s:
        if s == "aws-addons-collection":
            return True
        if c.startswith("helm repo update"):
            return True
        if s in normalize_text(c):
            return True
        if s == "kubeconfig.yaml" and "kubeconfig" in normalize_text(c):
            return True
        return False

    return True


def is_supported_shell_command(generated: str, allowed_blocks: list[str]) -> bool:
    g = normalize_shell_command(generated)
    if not g:
        return False

    for block in allowed_blocks:
        if "\n" in block:
            lines = [normalize_shell_command(x) for x in block.splitlines() if normalize_shell_command(x)]
        else:
            lines = [normalize_shell_command(block)]

        for line in lines:
            if g == line:
                return True
            if line.startswith(g) or g.startswith(line):
                return True

            g_tokens = shell_tokens(g)
            a_tokens = shell_tokens(line)

            idx = 0
            for tok in a_tokens:
                if idx < len(g_tokens) and tok == g_tokens[idx]:
                    idx += 1
            if idx == len(g_tokens):
                return True

    return False


def normalize_yaml_block(text: str) -> list[str]:
    out = []
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("#"):
            continue
        out.append(line)
    return out


def is_supported_yaml_block(generated: str, allowed_blocks: list[str]) -> bool:
    g_lines = normalize_yaml_block(generated)
    if not g_lines:
        return False

    for block in allowed_blocks:
        a_lines = normalize_yaml_block(block)
        if g_lines == a_lines:
            return True

        idx = 0
        for line in a_lines:
            if idx < len(g_lines) and line == g_lines[idx]:
                idx += 1
        if idx == len(g_lines):
            return True

    return False


def is_supported_output_block(generated: str, allowed_blocks: list[str]) -> bool:
    g = normalize_text(generated)
    for block in allowed_blocks:
        a = normalize_text(block)
        if g == a or g in a:
            return True
    return False


def collect_commands_from_docs(documents: list[Document], subject: str | None = None, install_only: bool = False) -> list[str]:
    commands = []

    for doc in documents:
        meta = doc.meta or {}
        content = doc.content or ""

        if meta.get("doc_type") == "block" and meta.get("block_type") in ("shell_command", "script"):
            for lang, block, _, _ in extract_fenced_code_blocks(content):
                if lang in ("bash", "sh", "shell", "plain", ""):
                    for line in join_command_continuations(block):
                        cmd = normalize_shell_command(line)
                        if not cmd:
                            continue
                        if install_only and not is_install_like_command(cmd, subject):
                            continue
                        commands.append(cmd)

        for cmd in extract_inline_shell_commands(content):
            if install_only and not is_install_like_command(cmd, subject):
                continue
            commands.append(cmd)

    return dedupe_preserve(commands)


def collect_yaml_from_docs(documents: list[Document]) -> list[str]:
    yamls = []
    for doc in documents:
        meta = doc.meta or {}
        content = doc.content or ""

        if meta.get("doc_type") == "block" and meta.get("block_type") == "yaml":
            for lang, block, _, _ in extract_fenced_code_blocks(content):
                if lang in ("yaml", "yml", "plain", "") and block.strip():
                    yamls.append(block.strip())

        yamls.extend(extract_inline_yaml_blocks(content))

    return dedupe_preserve(yamls)


def collect_outputs_from_docs(documents: list[Document]) -> list[str]:
    outputs = []
    for doc in documents:
        meta = doc.meta or {}
        content = doc.content or ""

        if meta.get("doc_type") == "block" and meta.get("block_type") == "output":
            for _, block, _, _ in extract_fenced_code_blocks(content):
                if block.strip():
                    outputs.append(block.strip())

        outputs.extend(extract_inline_output_blocks(content))

    return dedupe_preserve(outputs)


def collect_scripts_from_docs(documents: list[Document]) -> list[str]:
    scripts = []
    for doc in documents:
        meta = doc.meta or {}
        content = doc.content or ""

        if meta.get("doc_type") == "block" and meta.get("block_type") in ("script", "sql"):
            for _, block, _, _ in extract_fenced_code_blocks(content):
                if block.strip():
                    scripts.append(block.strip())

        for cmd in extract_inline_shell_commands(content):
            if cmd.startswith("./") or cmd.startswith("source ./") or cmd.startswith("java -jar") or cmd.startswith("SQL>@") or cmd.startswith("lsnrctl "):
                scripts.append(cmd)

    return dedupe_preserve(scripts)


def unique_sources(documents: list[Document], max_items: int = 6) -> list[tuple[str, str, str]]:
    out = []
    seen = set()
    for doc in documents:
        meta = doc.meta or {}
        item = (
            meta.get("doc_title", "Unknown document"),
            meta.get("heading", ""),
            meta.get("page_url", "N/A"),
        )
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:max_items]


# =========================================================
# DETERMINISTIC ANSWERS
# =========================================================

def build_command_only_answer(query: str, documents: list[Document]) -> str:
    subject = detect_primary_subject(query)
    relevant_docs = filter_docs_for_subject(documents, subject)

    commands = collect_commands_from_docs(
        relevant_docs,
        subject=subject,
        install_only=is_install_command_query(query),
    )
    sources = unique_sources(relevant_docs)

    if not commands:
        return build_safe_fallback_answer(relevant_docs, query)

    parts = [
        "Respuesta",
        f"Los siguientes comandos están documentados para instalar {subject or 'el componente solicitado'}:",
        "",
        "Comandos",
        "```bash",
    ]
    parts.extend(commands)
    parts.append("```")

    parts.extend(["", "Fuente"])
    for title, heading, url in sources:
        parts.append(f"- {title} — {heading} ({url})")

    return "\n".join(parts)


def build_yaml_only_answer(query: str, documents: list[Document]) -> str:
    subject = detect_primary_subject(query)
    relevant_docs = filter_docs_for_subject(documents, subject)

    yamls = collect_yaml_from_docs(relevant_docs)
    sources = unique_sources(relevant_docs)

    if not yamls:
        return build_safe_fallback_answer(relevant_docs, query)

    parts = [
        "Respuesta",
        f"Los siguientes bloques YAML están documentados para {subject or 'el tema solicitado'}:",
        "",
        "YAML",
    ]
    for block in yamls[:3]:
        parts.append("```yaml")
        parts.append(block)
        parts.append("```")

    parts.extend(["", "Fuente"])
    for title, heading, url in sources:
        parts.append(f"- {title} — {heading} ({url})")

    return "\n".join(parts)


def build_output_only_answer(query: str, documents: list[Document]) -> str:
    subject = detect_primary_subject(query)
    relevant_docs = filter_docs_for_subject(documents, subject)

    outputs = collect_outputs_from_docs(relevant_docs)
    sources = unique_sources(relevant_docs)

    if not outputs:
        return build_safe_fallback_answer(relevant_docs, query)

    parts = [
        "Respuesta",
        "La documentación muestra las siguientes salidas esperadas:",
        "",
        "Salida esperada",
    ]
    for block in outputs[:3]:
        parts.append("```text")
        parts.append(block)
        parts.append("```")

    parts.extend(["", "Fuente"])
    for title, heading, url in sources:
        parts.append(f"- {title} — {heading} ({url})")

    return "\n".join(parts)


def build_script_only_answer(query: str, documents: list[Document]) -> str:
    subject = detect_primary_subject(query)
    relevant_docs = filter_docs_for_subject(documents, subject)

    scripts = collect_scripts_from_docs(relevant_docs)
    sources = unique_sources(relevant_docs)

    if not scripts:
        return build_safe_fallback_answer(relevant_docs, query)

    parts = [
        "Respuesta",
        "La documentación muestra los siguientes scripts o entradas principales:",
        "",
        "Scripts",
    ]
    for block in scripts[:3]:
        parts.append("```bash")
        parts.append(block)
        parts.append("```")

    parts.extend(["", "Fuente"])
    for title, heading, url in sources:
        parts.append(f"- {title} — {heading} ({url})")

    return "\n".join(parts)


def build_addons_collection_command_answer(query: str, documents: list[Document]) -> str:
    relevant_docs = filter_docs_for_subject(documents, "aws-addons-collection")

    grouped: dict[str, list[str]] = {}
    source_url = None
    source_title = None

    for doc in relevant_docs:
        meta = doc.meta or {}
        heading = meta.get("heading", "").strip()
        title = meta.get("doc_title", "").strip()
        page_url = meta.get("page_url", "").strip()

        if is_aws_addons_page(doc):
            source_title = title or source_title
            source_url = page_url or source_url

        if not heading or normalize_text(heading) in {"aws add-ons", "kubernetes add-ons"}:
            continue

        commands = collect_commands_from_docs([doc], subject="aws-addons-collection", install_only=True)

        commands = [
            c for c in commands
            if c.startswith("helm repo add ")
            or c.startswith("helm repo update")
            or c.startswith("helm install ")
            or c.startswith("kubectl apply -f ")
        ]

        if commands:
            grouped.setdefault(heading, [])
            for cmd in commands:
                if cmd not in grouped[heading]:
                    grouped[heading].append(cmd)

    if not grouped:
        return build_safe_fallback_answer(relevant_docs, query)

    parts = [
        "Respuesta",
        "La documentación de Kubernetes Cluster Add-ons - AWS (5.2) muestra los siguientes comandos para los AWS Add-ons de UEPE:",
    ]

    for heading, commands in grouped.items():
        parts.extend(["", heading, "```bash"])
        parts.extend(commands)
        parts.append("```")

    parts.extend(["", "Fuente"])
    if source_title and source_url:
        parts.append(f"- {source_title} ({source_url})")
    else:
        for title, heading, url in unique_sources(relevant_docs, max_items=3):
            parts.append(f"- {title} — {heading} ({url})")

    return "\n".join(parts)


def maybe_build_deterministic_answer(query: str, documents: list[Document]) -> str | None:
    intent = detect_query_intent(query)
    subject = intent["subject"]

    if subject == "aws-addons-collection" and intent["wants_commands"]:
        return build_addons_collection_command_answer(query, documents)

    if intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_output"] and not intent["wants_script"]:
        return build_command_only_answer(query, documents)

    if intent["wants_yaml"] and not intent["wants_commands"] and not intent["wants_output"] and not intent["wants_script"]:
        return build_yaml_only_answer(query, documents)

    if intent["wants_output"] and not intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_script"]:
        return build_output_only_answer(query, documents)

    if intent["wants_script"] and not intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_output"]:
        return build_script_only_answer(query, documents)

    if intent["wants_commands"] and intent["wants_yaml"]:
        relevant_docs = filter_docs_for_subject(documents, subject)
        commands = collect_commands_from_docs(
            relevant_docs,
            subject=subject,
            install_only=is_install_command_query(query),
        )
        yamls = collect_yaml_from_docs(relevant_docs)
        sources = unique_sources(relevant_docs)

        if commands or yamls:
            parts = ["Respuesta", "La documentación muestra los siguientes comandos y YAML relevantes:"]
            if commands:
                parts.extend(["", "Comandos", "```bash"])
                parts.extend(commands[:12])
                parts.append("```")
            if yamls:
                parts.extend(["", "YAML"])
                for y in yamls[:2]:
                    parts.append("```yaml")
                    parts.append(y)
                    parts.append("```")
            parts.extend(["", "Fuente"])
            for title, heading, url in sources:
                parts.append(f"- {title} — {heading} ({url})")
            return "\n".join(parts)

    return None


# =========================================================
# PROMPT FOR GENERAL QUESTIONS
# =========================================================

def render_documents_for_prompt(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, 1):
        meta = doc.meta or {}
        doc_title = meta.get("doc_title", "Unknown")
        heading = meta.get("heading", "")
        page_url = meta.get("page_url", "N/A")
        doc_type = meta.get("doc_type", "section")
        block_type = meta.get("block_type", "")
        topic_tags = ", ".join(meta.get("topic_tags") or [])

        content = (doc.content or "").strip()
        content = content[:MAX_CONTEXT_CHARS_PER_DOC]

        blocks.append(
            f"[DOCUMENT {i}]\n"
            f"Title: {doc_title}\n"
            f"Heading: {heading}\n"
            f"URL: {page_url}\n"
            f"Doc Type: {doc_type}\n"
            f"Block Type: {block_type}\n"
            f"Topic Tags: {topic_tags}\n"
            f"Content:\n{content}\n"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, documents: list[Document]) -> str:
    intent = detect_query_intent(question)
    context = render_documents_for_prompt(documents)

    allowed_urls = sorted(set((doc.meta or {}).get("page_url") for doc in documents if (doc.meta or {}).get("page_url")))
    allowed = collect_allowed_blocks(documents)

    allowed_urls_block = "\n".join(f"- {u}" for u in allowed_urls) if allowed_urls else "- None"

    shell_block = "\n\n".join(
        f"[SHELL {i+1}]\n```bash\n{b}\n```" for i, b in enumerate(allowed["shell_command"][:MAX_ALLOWED_BLOCKS_PER_TYPE])
    ) or "None"

    yaml_block = "\n\n".join(
        f"[YAML {i+1}]\n```yaml\n{b}\n```" for i, b in enumerate(allowed["yaml"][:MAX_ALLOWED_BLOCKS_PER_TYPE])
    ) or "None"

    script_block = "\n\n".join(
        f"[SCRIPT {i+1}]\n```bash\n{b}\n```" for i, b in enumerate(allowed["script"][:MAX_ALLOWED_BLOCKS_PER_TYPE])
    ) or "None"

    output_block = "\n\n".join(
        f"[OUTPUT {i+1}]\n```text\n{b}\n```" for i, b in enumerate(allowed["output"][:MAX_ALLOWED_BLOCKS_PER_TYPE])
    ) or "None"

    prompt = f"""
You are a Technical Support Copilot specialized in UEPE installation documentation.
You must answer in Spanish.

STRICT RULES:
1. Use ONLY the DOCUMENTATION CONTEXT.
2. Do NOT invent commands, steps, YAML, scripts, outputs, URLs, file names, placeholders, versions, or values.
3. If the answer is not explicitly supported by the documentation context, say exactly:
"The available documentation does not contain enough information to answer this question."
4. You may ONLY cite URLs from the ALLOWED URLS list.
5. You may ONLY include shell commands that appear in the ALLOWED SHELL COMMANDS or ALLOWED SCRIPTS blocks.
6. You may ONLY include YAML that appears in the ALLOWED YAML blocks.
7. You may ONLY include example outputs that appear in the ALLOWED OUTPUT blocks.
8. If the user asks "how", include the documented procedure in order, not only the final command.
9. If the user asks "what is" a file, explain its purpose only using wording found in the context.
10. Prefer exact filenames, commands, helm values, and YAML keys from the documentation.
11. Do not merge different YAML snippets unless the documentation itself presents them together.
12. For topic-specific questions, prefer the section whose heading matches the topic exactly.

QUESTION INTENT:
{json.dumps(intent, ensure_ascii=False, indent=2)}

RESPONSE FORMAT:
Respuesta
<respuesta grounded>

Pasos
<solo si aplica>

Comandos
<solo si aplica; usar bloque bash>

YAML
<solo si aplica; usar bloque yaml>

Scripts
<solo si aplica; usar bloque bash o text>

Salida esperada
<solo si aplica; usar bloque text>

Notas adicionales
<solo si están documentadas>

Fuente
- <Document Title> — <Heading> (<URL>)

ALLOWED URLS:
{allowed_urls_block}

ALLOWED SHELL COMMANDS:
{shell_block}

ALLOWED YAML:
{yaml_block}

ALLOWED SCRIPTS:
{script_block}

ALLOWED OUTPUTS:
{output_block}

DOCUMENTATION CONTEXT:
{context}

QUESTION:
{question}

Generate the final answer in Spanish, strictly grounded in the documentation context.
"""
    return prompt.strip()


# =========================================================
# OLLAMA
# =========================================================

def ollama_generate(prompt: str, temperature: float = 0.0) -> str:
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


# =========================================================
# VALIDATION / FALLBACK
# =========================================================

def extract_generated_code_blocks(text: str) -> list[tuple[str, str]]:
    return [(lang or "plain", content.strip()) for lang, content, _, _ in extract_fenced_code_blocks(text)]


def validate_answer(answer: str, documents: list[Document]) -> tuple[bool, list[str]]:
    problems = []

    allowed_urls = set((doc.meta or {}).get("page_url") for doc in documents if (doc.meta or {}).get("page_url"))
    answer_urls = extract_urls(answer)
    external_urls = [u for u in answer_urls if u not in allowed_urls]
    if external_urls:
        problems.append(f"URLs externas detectadas: {external_urls}")

    allowed = collect_allowed_blocks(documents)

    for lang, block in extract_generated_code_blocks(answer):
        lang_norm = (lang or "").lower()

        if lang_norm in ("bash", "sh", "shell", ""):
            cmd_lines = []
            for line in join_command_continuations(block):
                cmd = normalize_shell_command(line)
                if re.match(r"^(kubectl |helm |terraform |eksctl |aws |docker |export |java -jar |curl |\./|source \./|SQL>@|lsnrctl )", cmd):
                    cmd_lines.append(cmd)

            invalid = [c for c in cmd_lines if not is_supported_shell_command(c, allowed["shell_command"] + allowed["script"])]
            if invalid:
                problems.append(f"Comandos no soportados por la documentación: {invalid}")

        elif lang_norm in ("yaml", "yml"):
            if not is_supported_yaml_block(block, allowed["yaml"]):
                problems.append("Se detectó YAML que no coincide con los snippets permitidos.")

        elif lang_norm in ("text", "plain"):
            if re.search(r"(NAME\s+NAMESPACE\s+REVISION|NAME\s+READY\s+STATUS|CLUSTER\s+NODEGROUP\s+STATUS)", block):
                if not is_supported_output_block(block, allowed["output"]):
                    problems.append("Se detectó una salida esperada que no coincide con la documentación.")

    return (len(problems) == 0), problems


def build_grounded_excerpt(doc: Document, query: str) -> str:
    content = (doc.content or "").strip()
    sentences = split_sentences_naive(content)
    q_terms = set(simple_query_terms(query))

    ranked = []
    for s in sentences:
        score = len(q_terms & set(simple_query_terms(s)))
        if score > 0:
            ranked.append((score, s))
    if ranked:
        ranked.sort(reverse=True, key=lambda x: x[0])
        return ranked[0][1]
    return content[:500]


def build_safe_fallback_answer(documents: list[Document], query: str) -> str:
    if not documents:
        return (
            "Respuesta\n"
            "The available documentation does not contain enough information to answer this question.\n\n"
            "Fuente\n"
            "- No relevant documentation fragments were retrieved."
        )

    top = documents[0]
    meta = top.meta or {}
    title = meta.get("doc_title", "Unknown document")
    heading = meta.get("heading", "")
    url = meta.get("page_url", "N/A")

    excerpts = [build_grounded_excerpt(d, query) for d in documents[:3]]
    commands = collect_commands_from_docs(documents)[:5]
    yamls = collect_yaml_from_docs(documents)[:1]
    outputs = collect_outputs_from_docs(documents)[:1]

    parts = [
        "Respuesta",
        "La respuesta generada contenía elementos no verificables. A continuación se muestra únicamente información respaldada por la documentación recuperada.",
        "",
        "Información soportada",
    ]
    for i, ex in enumerate(excerpts, 1):
        parts.append(f"{i}. {ex}")

    if commands:
        parts.extend(["", "Comandos", "```bash"])
        parts.extend(commands)
        parts.append("```")

    if yamls:
        parts.extend(["", "YAML", "```yaml", yamls[0], "```"])

    if outputs:
        parts.extend(["", "Salida esperada", "```text", outputs[0], "```"])

    parts.extend([
        "",
        "Fuente",
        f"- {title} — {heading} ({url})"
    ])
    return "\n".join(parts)


# =========================================================
# MAIN QA
# =========================================================

def answer_question(retriever: InstallationRetriever, query: str) -> str:
    docs = retriever.retrieve(query)

    if DEBUG_RETRIEVAL:
        print("\nRetrieved documents:")
        for i, doc in enumerate(docs, 1):
            print("\n" + "=" * 80)
            print(f"DOC {i}")
            print("-" * 80)
            print("META:", json.dumps(doc.meta or {}, ensure_ascii=False, indent=2))
            print("-" * 80)
            print((doc.content or "")[:2200])

    if not docs:
        return "The available documentation does not contain enough information to answer this question."

    deterministic = maybe_build_deterministic_answer(query, docs)
    if deterministic:
        return deterministic

    prompt = build_prompt(query, docs)
    answer = ollama_generate(prompt, temperature=0.0)

    ok, problems = validate_answer(answer, docs)
    if not ok:
        print("\n[WARNING] Invalid grounded answer detected:")
        for p in problems:
            print(" -", p)
        return build_safe_fallback_answer(docs, query)

    return answer


def main():
    corpus = load_corpus(MARKDOWN_DIR)
    store = build_document_store()
    index_documents(store, corpus)

    retriever = InstallationRetriever(store)

    # Cambia la consulta aquí
    query = "What commands do I need to AWS Add-ons for UEPE"
    # query = "What commands do I need to install AWS Add-ons for UEPE?"
    # query = "What commands do I need to install external-dns?"
    # query = "How can I create a EKS cluster for installing UEPE and what is kubeconfig.yaml file?"
    # query = "Show me the YAML for ingress-nginx-values.yaml"
    # query = "What is the main entry point script for Oracle?"
    # query = "What output should I expect after helm install uepe?"
    # query = "How do I configure Fluent-bit to send logs to CloudWatch and Elasticsearch?"

    answer = answer_question(retriever, query)

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(answer)


if __name__ == "__main__":
    main()