# html_to_md_batch.py
#
# Etapa 2 (offline): Convertir HTML guardado -> Markdown.
# Lee carpetas generadas por el spider (meta.json + page.html/export_view.html),
# elige la mejor fuente (export_view.html > url_export_view.html > page.html),
# extrae SOLO el contenido principal (AkMainContent / content-body),
# y genera page.md con:
# - Título garantizado (# <title>)
# - Encabezados/subtítulos
# - Párrafos, listas, quotes, panels/notes
# - Tablas (HTML) + extracción de code dentro de celdas sin romper la tabla
# - Code blocks (con language-xxx o inferencia: json/yaml/shell/hcl/python/sql/xml)
#
# Run:
#   python html_to_md_batch.py --config config_batch.yaml
# Opcional:
#   python html_to_md_batch.py --input ./out_html --output ./out_md

import os
import re
import json
import yaml
import argparse
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag, NavigableString
from bs4.element import Comment


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def normalize_whitespace(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def prune_confluence_navigation(soup: BeautifulSoup):
    selectors = [
        "nav",
        "aside",
        "[role='navigation']",
        "[data-testid*='SideNavigation' i]",
        "[data-testid*='SpaceNavigation' i]",
        "[data-testid*='side-navigation' i]",
        "[data-testid*='sidebar' i]",
        "[aria-label*='sidebar' i]",
        "[aria-label*='navigation' i]",
    ]
    for sel in selectors:
        for t in soup.select(sel):
            t.decompose()


def remove_noise(scope: Tag):
    for bad in scope.find_all(["script", "style", "noscript", "template"]):
        bad.decompose()


def absolutize_links(scope: Tag, base_url: str):
    for a in scope.find_all("a"):
        href = a.get("href")
        if href:
            a["href"] = urljoin(base_url, href)
    for img in scope.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if src:
            img["src"] = urljoin(base_url, src)


# -------- language inference --------
def infer_lang_by_content(code_text: str) -> Optional[str]:
    t = (code_text or "").strip()
    if not t:
        return None

    if t.startswith("{") or t.startswith("["):
        try:
            json.loads(t)
            return "json"
        except Exception:
            pass

    if re.search(r"(?m)^\s*(apiVersion:|kind:|metadata:)\s*", t):
        return "yaml"

    first = (t.splitlines()[:1] or [""])[0].strip()
    if re.match(r"^(kubectl|aws|eksctl|terraform|helm|docker|git|curl|wget)\b", first):
        return "shell"

    if re.search(r'(?m)^\s*(resource|provider|variable|module)\s+"', t):
        return "hcl"

    if re.search(r"(?m)^\s*(def|class)\s+\w+", t) or re.search(r"(?m)^\s*import\s+\w+", t):
        return "python"

    if re.search(r"(?mi)^\s*(SELECT|INSERT|UPDATE|DELETE)\b", t):
        return "sql"

    if t.startswith("<") and re.search(r"</\w+>", t):
        return "xml"

    return None


# -------- code detection --------
def _class_str(tag: Tag) -> str:
    return " ".join(tag.get("class", [])).lower()


def is_confluence_code_container(tag: Tag) -> bool:
    if not isinstance(tag, Tag):
        return False

    dt = (tag.get("data-testid") or "").strip()
    if dt == "renderer-code-block":
        return True

    if tag.has_attr("data-ds--code--code-block"):
        return True

    cls = _class_str(tag)
    if "code-block" in cls or "prismjs" in cls:
        return True

    if (tag.name or "").lower() == "code":
        if "white-space: pre" in (tag.get("style") or "").lower():
            return True

    return False


def infer_code_language_dom(container: Tag) -> Optional[str]:
    code_tag = container if (container.name or "").lower() == "code" else container.find("code")
    if code_tag:
        for c in (code_tag.get("class", []) or []):
            if c.startswith("language-"):
                lang = c[len("language-") :].strip()
                return lang or None
    data_lang = (container.get("data-code-lang") or "").strip()
    return data_lang or None


def extract_code_text_dom(container: Tag) -> str:
    if (container.name or "").lower() == "pre":
        return container.get_text("\n", strip=False).replace("\r\n", "\n").replace("\r", "\n").strip("\n")

    code_tag = container if (container.name or "").lower() == "code" else container.find("code")
    if not code_tag:
        return container.get_text("\n", strip=False).replace("\r\n", "\n").replace("\r", "\n").strip("\n")

    line_spans = code_tag.find_all(attrs={"data-testid": re.compile(r"^renderer-code-block-line-")})
    if line_spans:
        lines = []
        for ls in line_spans:
            tmp = BeautifulSoup(str(ls), "html.parser")
            for ln in tmp.select(".linenumber, .ds-line-number"):
                ln.decompose()
            t = tmp.get_text("", strip=False).replace("\r\n", "\n").replace("\r", "\n")
            t = t[:-1] if t.endswith("\n") else t
            lines.append(t)
        return "\n".join(lines).rstrip("\n")

    return code_tag.get_text("\n", strip=False).replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def looks_like_big_code(code_text: str) -> bool:
    t = code_text or ""
    return ("\n" in t) or (len(t.strip()) >= 60)


def filter_top_level_containers(candidates: List[Tag]) -> List[Tag]:
    uniq = []
    seen = set()
    for t in candidates:
        if not isinstance(t, Tag):
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        uniq.append(t)

    cand_set = {id(t) for t in uniq}
    top = []
    for t in uniq:
        p = t.parent
        nested = False
        while isinstance(p, Tag):
            if id(p) in cand_set:
                nested = True
                break
            p = p.parent
        if not nested:
            top.append(t)

    return top


# -------- notes/callouts --------
def infer_note_kind(tag: Tag) -> Optional[str]:
    role = (tag.get("role") or "").lower()
    if role == "note":
        return "note"

    cls = _class_str(tag)
    if any(k in cls for k in ["warning", "confluence-warning", "ak-callout-warning"]):
        return "warning"
    if any(k in cls for k in ["note", "confluence-note", "ak-callout-note"]):
        return "note"
    if any(k in cls for k in ["tip", "confluence-tip", "ak-callout-tip"]):
        return "tip"
    if any(k in cls for k in ["info", "information", "confluence-information", "ak-callout-info"]):
        return "info"

    panel_type = (tag.get("data-panel-type") or "").lower().strip()
    if panel_type in {"info", "note", "tip", "warning"}:
        return panel_type

    return None


# -------- markdown rendering helpers --------
def md_escape_cell(text: str) -> str:
    t = (text or "").replace("|", "\\|")
    t = t.replace("\n", "<br>")
    return t.strip()


def paragraph_inline(p: Tag) -> str:
    """
    Preserve inline code and links.
    """
    parts: List[str] = []

    def walk(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            parts.append(str(node))
            return
        if not isinstance(node, Tag):
            return

        name = (node.name or "").lower()
        if name == "br":
            parts.append("\n")
            return
        if name == "code" and not (node.parent and isinstance(node.parent, Tag) and (node.parent.name or "").lower() == "pre"):
            txt = node.get_text("", strip=True)
            if txt:
                parts.append(f"`{txt}`")
            return
        if name == "a":
            href = node.get("href") or ""
            txt = normalize_whitespace(node.get_text(" ", strip=True)) or href
            if href:
                parts.append(f"[{txt}]({href})")
            else:
                parts.append(txt)
            return

        for ch in node.children:
            walk(ch)

    for ch in p.children:
        walk(ch)

    out = "".join(parts)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return normalize_whitespace(out)


def table_to_struct(table: Tag) -> dict:
    headers = []
    header_row = table.find("tr")
    if header_row:
        ths = header_row.find_all("th")
        if ths:
            headers = [normalize_whitespace(th.get_text(" ", strip=True)) for th in ths]

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [normalize_whitespace(c.get_text(" ", strip=True)) for c in cells]
        if tr.find_all("th") and headers and row == headers:
            continue
        rows.append(row)

    return {"headers": headers, "rows": rows}


def render_table_md(headers: List[str], rows: List[List[str]]) -> str:
    if not rows and not headers:
        return ""

    col_count = max([len(headers)] + [len(r) for r in rows] + [0])
    if col_count == 0:
        return ""

    def norm_row(r: List[str]) -> List[str]:
        r = (r + [""] * col_count)[:col_count]
        return [md_escape_cell(c) for c in r]

    if not headers:
        headers = [f"Col {i+1}" for i in range(col_count)]
    headers = norm_row(headers)
    body = [norm_row(r) for r in rows]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def extract_scope(soup: BeautifulSoup, scopes: List[str]) -> Tuple[Tag, str]:
    # Prefer #AkMainContent first to avoid sidebar
    for css in scopes:
        t = soup.select_one(css)
        if t:
            return t, css

    # fallbacks
    for css in ["#AkMainContent", "#content-body", "[data-test-id='content-body']", ".ak-renderer-document", "body"]:
        t = soup.select_one(css)
        if t:
            return t, css

    return soup, "document"


def html_to_markdown_full(html: str, base_url: str, title: str, scopes: List[str]) -> Tuple[str, Dict[str, int], str]:
    soup = make_soup(html)
    prune_confluence_navigation(soup)

    scope, scope_used = extract_scope(soup, scopes)
    absolutize_links(scope, base_url)
    remove_noise(scope)

    blocks: List[str] = []
    stats: Dict[str, int] = {}
    captured = set()

    def inc(k: str):
        stats[k] = stats.get(k, 0) + 1

    def inside_captured(tag: Tag) -> bool:
        p = tag.parent
        while isinstance(p, Tag):
            if id(p) in captured:
                return True
            p = p.parent
        return False

    def add(s: str = ""):
        s = s.rstrip()
        if s.strip():
            blocks.append(s)

    def add_blank():
        if not blocks or blocks[-1] != "":
            blocks.append("")

    def add_code(code_text: str, lang: Optional[str]):
        lang = (lang or "").strip() or (infer_lang_by_content(code_text) or "")
        fence = f"```{lang}".rstrip() if lang else "```"
        add(fence)
        blocks.append(code_text.rstrip("\n"))
        add("```")
        add_blank()
        inc("code")

    # Title guaranteed
    add(f"# {title}")
    add_blank()

    candidates = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "pre", "blockquote", "div", "span", "code", "img"]

    for tag in scope.find_all(candidates):
        if not isinstance(tag, Tag) or inside_captured(tag):
            continue
        name = (tag.name or "").lower()

        # headings
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            txt = normalize_whitespace(tag.get_text(" ", strip=True))
            if txt:
                lvl = int(name[1])
                lvl = max(1, min(lvl, 6))
                add(f"{'#' * lvl} {txt}")
                add_blank()
                inc("heading")
                captured.add(id(tag))
            continue

        # code blocks (confluence)
        if is_confluence_code_container(tag) and name in {"div", "span", "code"}:
            ct = extract_code_text_dom(tag)
            if ct.strip():
                lg = infer_code_language_dom(tag)
                add_code(ct, lg)
                captured.add(id(tag))
            continue

        # <pre>
        if name == "pre":
            ct = extract_code_text_dom(tag)
            if ct.strip():
                lg = infer_code_language_dom(tag)
                add_code(ct, lg)
                captured.add(id(tag))
            continue

        # paragraphs
        if name == "p":
            txt = paragraph_inline(tag)
            if txt:
                add(txt)
                add_blank()
                inc("p")
                captured.add(id(tag))
            continue

        # lists
        if name in {"ul", "ol"}:
            items = []
            for li in tag.find_all("li", recursive=False):
                t = normalize_whitespace(li.get_text(" ", strip=True))
                if t:
                    items.append(t)
            if items:
                if name == "ul":
                    for it in items:
                        add(f"- {it}")
                else:
                    for i, it in enumerate(items, 1):
                        add(f"{i}. {it}")
                add_blank()
                inc("list")
                captured.add(id(tag))
            continue

        # blockquote
        if name == "blockquote":
            txt = normalize_whitespace(tag.get_text("\n", strip=True))
            if txt:
                for ln in txt.splitlines():
                    add(f"> {ln}")
                add_blank()
                inc("quote")
                captured.add(id(tag))
            continue

        # notes/panels
        if name == "div":
            kind = infer_note_kind(tag)
            if kind:
                txt = normalize_whitespace(tag.get_text("\n", strip=True))
                if txt:
                    label = {"note": "Nota", "info": "Info", "tip": "Tip", "warning": "Warning"}.get(kind, kind.capitalize())
                    add(f"> **{label}:**")
                    for ln in txt.splitlines():
                        if ln.strip():
                            add(f"> {ln}")
                    add_blank()
                    inc("note")
                    captured.add(id(tag))
            continue

        # tables (and code in cells)
        if name == "table":
            t = table_to_struct(tag)
            if t["headers"] or t["rows"]:
                add(render_table_md(t["headers"], t["rows"]))
                add_blank()
                inc("table")

            # code inside cells
            snippets = []
            r_idx = -1
            for tr in tag.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                r_idx += 1
                for c_idx, cell in enumerate(cells):
                    cell_candidates: List[Tag] = []
                    for c in cell.find_all(True):
                        if is_confluence_code_container(c):
                            cell_candidates.append(c)
                    for ctag in cell.find_all("code"):
                        raw = ctag.get_text("\n", strip=False).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
                        if looks_like_big_code(raw):
                            cell_candidates.append(ctag)
                    cell_candidates = filter_top_level_containers(cell_candidates)

                    for c in cell_candidates:
                        ct = extract_code_text_dom(c).rstrip("\n")
                        if ct.strip():
                            lg = infer_code_language_dom(c) or infer_lang_by_content(ct)
                            snippets.append((r_idx, c_idx, ct, lg))

            if snippets:
                add("> Table embedded code blocks:")
                add_blank()
                for rr, cc, ct, lg in snippets:
                    add(f"**(table r{rr} c{cc})**")
                    add_blank()
                    add_code(ct, lg)

            captured.add(id(tag))
            continue

        # images (optional)
        if name == "img":
            src = tag.get("src") or tag.get("data-src") or ""
            alt = tag.get("alt") or ""
            if src and not src.startswith("data:"):
                add(f"![{alt}]({src})")
                add_blank()
                inc("img")
                captured.add(id(tag))
            continue

    md = "\n".join(blocks)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, stats, scope_used


def choose_input_html(folder: str, meta: dict) -> Tuple[str, str]:
    """
    Choose best HTML source:
    export_view.html > url_export_view.html > page.html
    """
    files = meta.get("files") or {}
    for key, fname in [("export_view", files.get("export_view_html")),
                       ("url_export_view", files.get("url_export_view_html")),
                       ("page", files.get("page_html"))]:
        if fname:
            path = os.path.join(folder, fname)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path, key

    # fallback raw
    for fname in ["export_view.html", "url_export_view.html", "page.html"]:
        path = os.path.join(folder, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path, fname

    return "", "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default=None, help="Override input dir (where html folders are)")
    ap.add_argument("--output", default=None, help="Override output dir (where md folders will be written)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = cfg.get("scrape", {}) or {}

    in_dir = args.input or sc.get("output_dir", "./out_html")  # same output of spider
    out_dir = args.output or sc.get("md_output_dir", "./out_md")

    scopes = sc.get("content_scope_css") or ["#AkMainContent [data-test-id='content-body']", "#AkMainContent", "#content-body", ".ak-renderer-document"]
    if not isinstance(scopes, list):
        scopes = [str(scopes)]

    os.makedirs(out_dir, exist_ok=True)

    folders = [os.path.join(in_dir, d) for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    folders.sort()

    for folder in folders:
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            continue

        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            continue

        page_url = meta.get("page_url") or meta.get("requested_url") or ""
        title = meta.get("title") or "Untitled"
        pid = meta.get("page_id")

        html_path, html_kind = choose_input_html(folder, meta)
        if not html_path:
            continue

        html = open(html_path, "r", encoding="utf-8", errors="ignore").read()

        md, stats, scope_used = html_to_markdown_full(html, page_url, title, scopes)

        # mirror folder name
        out_folder_name = os.path.basename(folder)
        out_folder = os.path.join(out_dir, out_folder_name)
        os.makedirs(out_folder, exist_ok=True)

        with open(os.path.join(out_folder, "page.md"), "w", encoding="utf-8") as f:
            f.write(md)

        # meta for md
        md_meta = {
            "title": title,
            "page_url": page_url,
            "page_id": pid,
            "html_source": html_kind,
            "html_file": os.path.basename(html_path),
            "scope_used": scope_used,
            "stats": stats,
            "files": {"markdown": "page.md"},
        }
        with open(os.path.join(out_folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(md_meta, f, ensure_ascii=False, indent=2)

    print(f"Done. MD written to: {out_dir}")


if __name__ == "__main__":
    main()