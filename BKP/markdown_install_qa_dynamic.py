from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =========================================================
# CONFIG
# =========================================================

MARKDOWN_DIR = "./out_md"
DEBUG = True


# =========================================================
# HELPERS
# =========================================================


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def dedupe_preserve(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


STOPWORDS = {
    "how", "what", "which", "when", "where", "the", "for", "and", "are", "with",
    "using", "use", "into", "from", "that", "this", "must", "need", "should",
    "you", "your", "can", "que", "como", "cuál", "cual", "para", "con", "del",
    "las", "los", "uepe", "usage", "engine", "private", "edition",
    "installation", "installing", "aws", "eks", "kubernetes", "cluster",
    "commands", "command", "comando", "comandos", "show", "give", "after",
    "expected", "should", "do", "i", "me"
}


INSTALL_QUERY_TERMS = {
    "install", "installation", "setup", "set up", "create", "deploy", "configure",
    "bootstrap", "provision"
}


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


def load_sidecar_meta(md_path: Path) -> dict:
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


def infer_page_url(meta: dict) -> str | None:
    return meta.get("page_url") or meta.get("url")


def is_install_command_query(query: str) -> bool:
    q = normalize_text(query)
    return any(term in q for term in INSTALL_QUERY_TERMS)


@dataclass
class SubjectCandidate:
    kind: str
    value: str
    score: int


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
# CORE ENGINE
# =========================================================


class MarkdownInstallQA:
    def __init__(self, markdown_dir: str, debug: bool = True):
        self.markdown_dir = Path(markdown_dir)
        self.debug = debug
        self.nodes: dict[str, SectionNode] = {}
        self.file_roots: dict[str, list[str]] = {}
        self.heading_index: dict[str, list[str]] = {}
        self.doc_title_index: dict[str, list[str]] = {}

    # -----------------------------
    # Build corpus
    # -----------------------------
    def build(self) -> None:
        base_dir = Path(__file__).resolve().parent
        path = (base_dir / self.markdown_dir).resolve()

        if not path.exists():
            raise FileNotFoundError(f"La carpeta no existe: {path}")

        files = sorted([p for p in path.rglob("*.md") if p.is_file()])
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos .md en {path}")

        if self.debug:
            print(f"Base dir: {base_dir}")
            print(f"Searching markdown in: {path}")
            print("Markdown files found:")
            for f in files:
                print(" -", f)

        for md_file in files:
            self._parse_markdown_file(md_file)

        self._build_indexes()

        if self.debug:
            print(f"Total section nodes: {len(self.nodes)}")

    def _parse_markdown_file(self, md_path: Path) -> None:
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        sidecar_meta = load_sidecar_meta(md_path)

        doc_title = sidecar_meta.get("title") or extract_title_from_markdown(raw, md_path.stem)
        page_url = infer_page_url(sidecar_meta)

        lines = raw.splitlines()
        current_node_id = None
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

        for node_id in file_nodes:
            self._finalize_node(self.nodes[node_id])

    def _finalize_node(self, node: SectionNode) -> None:
        text = node.raw_content.strip()

        commands = []
        yamls = []
        outputs = []
        scripts = []

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

            elif lang == "sql":
                if content.strip():
                    scripts.append(content.strip())

        commands.extend(extract_inline_shell_commands(text))
        yamls.extend(extract_inline_yaml_blocks(text))
        outputs.extend(extract_inline_output_blocks(text))
        node.steps = extract_numbered_steps(text)

        for cmd in commands:
            if cmd.startswith("./") or cmd.startswith("source ./") or cmd.startswith("java -jar") or cmd.startswith("SQL>@") or cmd.startswith("lsnrctl "):
                scripts.append(cmd)

        node.commands = dedupe_preserve(commands)
        node.yamls = dedupe_preserve(yamls)
        node.outputs = dedupe_preserve(outputs)
        node.scripts = dedupe_preserve(scripts)

    def _build_indexes(self) -> None:
        for node_id, node in self.nodes.items():
            h = normalize_text(node.heading)
            self.heading_index.setdefault(h, []).append(node_id)

            t = normalize_text(node.doc_title)
            self.doc_title_index.setdefault(t, []).append(node_id)

    # -----------------------------
    # Subject / intent
    # -----------------------------
    def subject_candidates(self, query: str) -> list[SubjectCandidate]:
        q_norm = normalize_text(query)
        candidates: list[SubjectCandidate] = []

        file_match = re.search(r"\b[A-Za-z0-9_.-]+\.(?:yaml|yml|sh|jar)\b", query)
        if file_match:
            candidates.append(SubjectCandidate("file", file_match.group(0), 200))

        for heading_norm in self.heading_index.keys():
            if len(heading_norm) < 3:
                continue
            score = 0
            if heading_norm in q_norm:
                score += 80
            score += lexical_overlap_score(query, heading_norm) * 10
            if score >= 20:
                candidates.append(SubjectCandidate("heading", heading_norm, score))

        for title_norm in self.doc_title_index.keys():
            if len(title_norm) < 3:
                continue
            score = 0
            if title_norm in q_norm:
                score += 70
            score += lexical_overlap_score(query, title_norm) * 8
            if score >= 18:
                candidates.append(SubjectCandidate("doc_title", title_norm, score))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def detect_primary_subject(self, query: str) -> str | None:
        candidates = self.subject_candidates(query)
        return candidates[0].value if candidates else None

    def detect_query_intent(self, query: str) -> dict:
        q = normalize_text(query)
        subject = self.detect_primary_subject(query)

        wants_commands = any(x in q for x in [
            "what commands", "which commands", "commands do i need", "command do i need",
            "show commands", "installation commands", "install commands",
            "comandos", "comando", "ejecutar", "run ", "execute "
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
            "script", ".sh", "entry point", "main script", "sqlplus", "jar"
        ])

        wants_steps = any(x in q for x in [
            "how", "cómo", "steps", "pasos", "install", "installation", "create", "set up", "setup"
        ])

        wants_prereq = any(x in q for x in [
            "prerequisite", "requirements", "tools", "iam", "license", "access key", "access keys"
        ])

        return {
            "subject": subject,
            "wants_commands": wants_commands,
            "wants_yaml": wants_yaml,
            "wants_output": wants_output,
            "wants_script": wants_script,
            "wants_steps": wants_steps,
            "wants_prereq": wants_prereq,
        }

    # -----------------------------
    # Tree helpers
    # -----------------------------
    def get_descendants(self, node_id: str) -> list[SectionNode]:
        out = []
        stack = list(self.nodes[node_id].children_ids)
        while stack:
            cid = stack.pop(0)
            child = self.nodes[cid]
            out.append(child)
            stack.extend(child.children_ids)
        return out

    def get_file_nodes(self, file_path: str) -> list[SectionNode]:
        return [self.nodes[nid] for nid in self.file_roots.get(file_path, [])]

    def get_useful_descendants(self, node: SectionNode) -> list[SectionNode]:
        descendants = self.get_descendants(node.node_id)
        return [
            d for d in descendants
            if d.commands or d.yamls or d.outputs or d.scripts or d.steps
        ]

    def find_exact_subject_nodes(self, subject: str | None) -> list[SectionNode]:
        if not subject:
            return []

        s = normalize_text(subject)
        nodes: list[SectionNode] = []
        seen = set()

        for nid in self.heading_index.get(s, []):
            if nid not in seen:
                seen.add(nid)
                nodes.append(self.nodes[nid])

        for nid in self.doc_title_index.get(s, []):
            if nid not in seen:
                seen.add(nid)
                nodes.append(self.nodes[nid])

        if subject.endswith((".yaml", ".yml", ".sh", ".jar")):
            for node in self.nodes.values():
                if s in normalize_text(node.raw_content):
                    if node.node_id not in seen:
                        seen.add(node.node_id)
                        nodes.append(node)

        return nodes

    # -----------------------------
    # Search / ranking
    # -----------------------------
    def score_node(self, node: SectionNode, query: str, intent: dict) -> int:
        q_norm = normalize_text(query)
        subject = intent["subject"]
        subject_norm = normalize_text(subject) if subject else None

        title_norm = normalize_text(node.doc_title)
        heading_norm = normalize_text(node.heading)
        path_norm = normalize_text(node.heading_path_str)
        content_norm = normalize_text(node.raw_content)

        score = 0

        score += lexical_overlap_score(query, node.heading) * 10
        score += lexical_overlap_score(query, node.heading_path_str) * 8
        score += lexical_overlap_score(query, node.doc_title) * 5
        score += min(lexical_overlap_score(query, node.raw_content), 8) * 2

        if q_norm and q_norm in heading_norm:
            score += 30
        if q_norm and q_norm in path_norm:
            score += 24
        if q_norm and q_norm in title_norm:
            score += 18

        if subject_norm:
            if heading_norm == subject_norm:
                score += 45
            elif subject_norm in heading_norm:
                score += 30
            elif subject_norm in path_norm:
                score += 22
            elif subject_norm in title_norm:
                score += 18
            elif subject_norm in content_norm:
                score += 10

        if intent["wants_commands"] and node.commands:
            score += 12
        if intent["wants_yaml"] and node.yamls:
            score += 12
        if intent["wants_output"] and node.outputs:
            score += 12
        if intent["wants_script"] and node.scripts:
            score += 12
        if intent["wants_steps"] and node.steps:
            score += 8

        useful_desc_count = len(self.get_useful_descendants(node))
        if useful_desc_count:
            score += min(useful_desc_count, 8) * 3

        if subject_norm == "external-dns":
            if "external-dns.alpha.kubernetes.io" in content_norm and heading_norm != "external-dns":
                score -= 20

        return score

    def find_best_anchor_node(self, query: str, ranked: list[SectionNode], intent: dict) -> SectionNode | None:
        subject = intent["subject"]
        exact_subject_nodes = self.find_exact_subject_nodes(subject)

        anchor_candidates: list[tuple[int, SectionNode]] = []
        seen = set()

        for node in exact_subject_nodes + ranked[:20]:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)

            useful_desc = self.get_useful_descendants(node)
            if not useful_desc:
                continue

            anchor_score = 0
            anchor_score += lexical_overlap_score(query, node.heading) * 12
            anchor_score += lexical_overlap_score(query, node.heading_path_str) * 10
            anchor_score += lexical_overlap_score(query, node.doc_title) * 8
            anchor_score += min(len(useful_desc), 10) * 4

            if subject:
                s = normalize_text(subject)
                if normalize_text(node.heading) == s:
                    anchor_score += 50
                if s in normalize_text(node.doc_title):
                    anchor_score += 20
                if s in normalize_text(node.heading_path_str):
                    anchor_score += 18

            anchor_candidates.append((anchor_score, node))

        anchor_candidates.sort(key=lambda x: x[0], reverse=True)
        return anchor_candidates[0][1] if anchor_candidates else None

    def filter_nodes_for_exact_subject(
        self,
        nodes: list[SectionNode],
        subject: str | None,
        require: str | None = None,
    ) -> list[SectionNode]:
        if not subject:
            return nodes

        scored = []

        for node in nodes:
            if require == "yaml" and not node.yamls:
                continue
            if require == "commands" and not node.commands:
                continue
            if require == "scripts" and not node.scripts:
                continue
            if require == "outputs" and not node.outputs:
                continue

            score = self.node_subject_score(node, subject)
            if score > 0:
                scored.append((score, node))

        if not scored:
            return nodes

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored]

    def node_subject_score(self, node: SectionNode, subject: str | None) -> int:
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
            score += 50

        if s in title:
            score += 15

        if subject.endswith((".yaml", ".yml")) and s in content:
            score += 30

        return score

    def expand_anchor_context(self, anchor: SectionNode, top_k: int = 10) -> list[SectionNode]:
        useful_desc = self.get_useful_descendants(anchor)
        same_file_useful = [
            n for n in self.get_file_nodes(anchor.file_path)
            if n.node_id != anchor.node_id and (n.commands or n.yamls or n.outputs or n.scripts or n.steps)
        ]

        nodes = [anchor] + useful_desc + same_file_useful
        deduped = []
        seen = set()
        for node in nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                deduped.append(node)
        return deduped[:top_k]

    def search(self, query: str, top_k: int = 8) -> list[SectionNode]:
        intent = self.detect_query_intent(query)
        subject = intent["subject"]

        all_nodes = list(self.nodes.values())
        ranked = sorted(all_nodes, key=lambda n: self.score_node(n, query, intent), reverse=True)

        exact_subject_nodes = self.find_exact_subject_nodes(subject)
        if exact_subject_nodes:
            exact_subject_nodes = sorted(
                exact_subject_nodes,
                key=lambda n: self.score_node(n, query, intent),
                reverse=True,
            )
            best_exact = exact_subject_nodes[0]
            useful_desc = self.get_useful_descendants(best_exact)
            if useful_desc:
                return self.expand_anchor_context(best_exact, top_k=top_k)

        if intent["wants_yaml"] and subject and subject.endswith((".yaml", ".yml")):
            filtered = self.filter_nodes_for_exact_subject(ranked, subject, require="yaml")
            if filtered:
                return filtered[:top_k]

        anchor = self.find_best_anchor_node(query, ranked, intent)
        if anchor and (intent["wants_commands"] or intent["wants_yaml"] or intent["wants_output"] or intent["wants_script"] or intent["wants_steps"]):
            expanded = self.expand_anchor_context(anchor, top_k=top_k)
            if len(expanded) > 1:
                return expanded

        return ranked[:top_k]

    # -----------------------------
    # Deterministic answer builders
    # -----------------------------
    def collect_commands(self, nodes: list[SectionNode], install_only: bool = False) -> list[str]:
        commands = []
        for node in nodes:
            cmds = node.commands
            if install_only:
                cmds = [c for c in cmds if is_install_like_command(c)]
            commands.extend(cmds)
        return dedupe_preserve(commands)

    def collect_yamls(self, nodes: list[SectionNode]) -> list[str]:
        yamls = []
        for node in nodes:
            yamls.extend(node.yamls)
        return dedupe_preserve(yamls)

    def collect_scripts(self, nodes: list[SectionNode]) -> list[str]:
        scripts = []
        for node in nodes:
            scripts.extend(node.scripts)
        return dedupe_preserve(scripts)

    def collect_outputs(self, nodes: list[SectionNode]) -> list[str]:
        outputs = []
        for node in nodes:
            outputs.extend(node.outputs)
        return dedupe_preserve(outputs)

    def collect_steps(self, nodes: list[SectionNode]) -> list[str]:
        steps = []
        for node in nodes:
            steps.extend(node.steps)
        return dedupe_preserve(steps)

    def unique_sources(self, nodes: list[SectionNode], max_items: int = 6) -> list[tuple[str, str, str]]:
        out = []
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
        parts = [
            "Respuesta",
            f"La documentación muestra los siguientes comandos agrupados para {title}:",
        ]

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
        relevant_nodes = self.filter_nodes_for_exact_subject(nodes, subject, require="yaml")

        yamls = self.collect_yamls(relevant_nodes)
        sources = self.unique_sources(relevant_nodes)

        if not yamls:
            return self.build_fallback_answer(relevant_nodes, query)

        parts = [
            "Respuesta",
            f"Los siguientes bloques YAML están documentados para {subject or 'el tema solicitado'}:",
            "",
            "YAML",
        ]

        for yml in yamls[:3]:
            parts.append("```yaml")
            parts.append(yml)
            parts.append("```")

        parts.extend(["", "Fuente"])
        for title, heading, url in sources:
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
        outputs = self.collect_outputs(nodes)
        if not outputs:
            return self.build_fallback_answer(nodes, query)

        parts = ["Respuesta", "La documentación muestra las siguientes salidas esperadas:", "", "Salida esperada"]
        for out in outputs[:3]:
            parts.append("```text")
            parts.append(out)
            parts.append("```")

        parts.extend(["", "Fuente"])
        for title, heading, url in self.unique_sources(nodes):
            parts.append(f"- {title} — {heading} ({url})")

        return "\n".join(parts)

    def build_extractive_answer(self, query: str, nodes: list[SectionNode]) -> str:
        intent = self.detect_query_intent(query)

        excerpts = []
        q_terms = set(simple_query_terms(query))

        for node in nodes[:5]:
            sentences = split_sentences_naive(node.raw_content)
            ranked = []
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

        parts = ["Respuesta"]
        parts.append(" ".join(excerpts[:3]))

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
            return (
                "Respuesta\n"
                "No encontré información suficiente en el corpus Markdown para responder esa pregunta."
            )

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

    # -----------------------------
    # Main answer dispatcher
    # -----------------------------
    def answer(self, query: str) -> str:
        intent = self.detect_query_intent(query)
        nodes = self.search(query, top_k=10)

        all_nodes = list(self.nodes.values())
        ranked = sorted(all_nodes, key=lambda n: self.score_node(n, query, intent), reverse=True)
        anchor = self.find_best_anchor_node(query, ranked, intent)

        if self.debug:
            print("\nRetrieved nodes:")
            for i, node in enumerate(nodes, 1):
                print("\n" + "=" * 80)
                print(f"NODE {i}")
                print("-" * 80)
                print(json.dumps({
                    "file_path": node.file_path,
                    "doc_title": node.doc_title,
                    "page_url": node.page_url,
                    "heading": node.heading,
                    "heading_path": node.heading_path,
                    "level": node.level,
                    "commands": len(node.commands),
                    "yamls": len(node.yamls),
                    "scripts": len(node.scripts),
                    "outputs": len(node.outputs),
                    "steps": len(node.steps),
                }, ensure_ascii=False, indent=2))
                print("-" * 80)
                print(node.raw_content[:1800])

            if anchor:
                print("\n" + "=" * 80)
                print("ANCHOR NODE")
                print("-" * 80)
                print(json.dumps({
                    "doc_title": anchor.doc_title,
                    "heading": anchor.heading,
                    "heading_path": anchor.heading_path,
                    "useful_descendants": len(self.get_useful_descendants(anchor)),
                }, ensure_ascii=False, indent=2))

        if intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_output"] and not intent["wants_script"]:
            return self.build_command_answer(query, nodes, anchor=anchor)

        if intent["wants_yaml"] and not intent["wants_commands"] and not intent["wants_output"] and not intent["wants_script"]:
            return self.build_yaml_answer(query, nodes)

        if intent["wants_script"] and not intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_output"]:
            return self.build_script_answer(query, nodes)

        if intent["wants_output"] and not intent["wants_commands"] and not intent["wants_yaml"] and not intent["wants_script"]:
            return self.build_output_answer(query, nodes)

        if intent["wants_commands"] and intent["wants_yaml"]:
            commands = self.collect_commands(nodes, install_only=is_install_command_query(query))
            yamls = self.collect_yamls(nodes)
            if commands or yamls:
                parts = ["Respuesta", "La documentación muestra los siguientes comandos y YAML relevantes:"]
                if commands:
                    parts.extend(["", "Comandos", "```bash"])
                    parts.extend(commands[:12])
                    parts.append("```")
                if yamls:
                    parts.extend(["", "YAML"])
                    for yml in yamls[:2]:
                        parts.append("```yaml")
                        parts.append(yml)
                        parts.append("```")
                parts.extend(["", "Fuente"])
                for title, heading, url in self.unique_sources(nodes):
                    parts.append(f"- {title} — {heading} ({url})")
                return "\n".join(parts)

        return self.build_extractive_answer(query, nodes)


# =========================================================
# MAIN
# =========================================================


def main():
    qa = MarkdownInstallQA(MARKDOWN_DIR, debug=DEBUG)
    qa.build()

    # Ejemplos
    # query = "What commands do I need to install external-dns?"
    # query = "What commands do I need for AWS Add-ons for UEPE?"
    # query = "Show me the YAML for ingress-nginx-values.yaml"
    # query = "What is the main entry point script for Oracle?"
    query = "What output should I expect after helm install uepe?"

    answer = qa.answer(query)

    print("\n" + "=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)
    print(answer)


if __name__ == "__main__":
    main()
