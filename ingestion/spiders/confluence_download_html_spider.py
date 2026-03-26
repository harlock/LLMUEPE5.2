# confluence_download_html_spider.py
#
# Etapa 1 (Scrapy): Descargar y guardar HTML completo por página.
# Guarda por cada URL en una carpeta:
#   - page.html               (HTML tal cual lo entrega la URL)
#   - export_view.html        (si se puede, desde REST API /wiki/rest/api/content/{id}?expand=body.export_view)
#   - url_export_view.html    (opcional, si se pide la URL con ?format=export_view)
#   - meta.json               (url, title, page_id, status, etc.)
#
# NO genera Markdown aquí.
#
# Run:
"""
scrapy runspider confluence_download_html_spider.py \
       -a config=config_batch.yaml \
       -a urls_file=urls_instalation.jsonl \
       -s LOG_LEVEL=INFO
"""
#
# Nota:
# - Si tu ambiente aún trae Scrapy 2.8, este spider ya no debería romperse por
#   "Trying to modify an immutable Settings object".
# - Si actualizas a Scrapy >= 2.11, sí podrá aplicar settings dinámicos desde YAML.

import os
import re
import json
import yaml
import hashlib
import scrapy
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_query_fragment(url: str) -> str:
    parts = list(urlparse(url))
    parts[4] = ""  # query
    parts[5] = ""  # fragment
    return urlunparse(parts)


def safe_folder_name(text: str, max_len: int = 90) -> str:
    text = (text or "").strip() or "untitled"
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return text[:max_len] if text else "untitled"


def extract_page_id(url: str) -> str | None:
    m = re.search(r"/pages/(\d+)(?:/|$)", url or "")
    return m.group(1) if m else None


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def ensure_url_export_view(url: str) -> str:
    """Agrega format=export_view al querystring sin perder otros params."""
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q["format"] = "export_view"
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))


class ConfluenceDownloadHtmlSpider(scrapy.Spider):
    name = "confluence_download_html"
    handle_httpstatus_list = [401, 403, 429, 500, 502, 503, 504]

    # Defaults seguros. Si Scrapy >= 2.11 y tu YAML trae overrides,
    # se intentan aplicar en from_crawler().
    custom_settings = {
        "FEED_EXPORT_ENCODING": "utf-8",
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 2.5,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "RETRY_TIMES": 8,
        "DOWNLOAD_TIMEOUT": 45,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 2.0,
        "AUTOTHROTTLE_MAX_DELAY": 120.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 0.5,
    }

    def __init__(self, config="config.yaml", urls_file="urls.jsonl", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = load_config(config)
        self.urls_file = urls_file

        site = self.cfg.get("site", {}) or {}
        sc = self.cfg.get("scrape", {}) or {}
        net = self.cfg.get("network", {}) or {}

        self.allowed_domains = site.get("allowed_domains") or []

        self.output_dir = sc.get("output_dir", "./out_html")
        os.makedirs(self.output_dir, exist_ok=True)

        self.use_rest_export_view = bool(sc.get("use_rest_export_view_fallback", True))
        self.export_view_expand = sc.get("export_view_expand", "body.export_view")
        self.also_download_url_export_view = bool(sc.get("also_download_url_export_view", False))

        self.user_agent = net.get("user_agent")
        self.extra_headers = dict(net.get("extra_headers", {}) or {})
        if self.user_agent and "User-Agent" not in self.extra_headers:
            self.extra_headers["User-Agent"] = self.user_agent

        # Fallback per-request, por si el crawler no permite set dinámico.
        self.download_timeout = int(net.get("timeout_seconds", 45))
        self.max_retry_times = int(net.get("retry_times", 8))

        self.seen = set()

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)

        cfg = spider.cfg
        site = cfg.get("site", {}) or {}
        net = cfg.get("network", {}) or {}

        desired_settings = {
            "ROBOTSTXT_OBEY": bool(site.get("obey_robots", True)),
            "DOWNLOAD_DELAY": float(net.get("download_delay", 2.5)),
            "RANDOMIZE_DOWNLOAD_DELAY": bool(net.get("randomize_download_delay", True)),
            "CONCURRENT_REQUESTS": int(net.get("concurrent_requests", 1)),
            "CONCURRENT_REQUESTS_PER_DOMAIN": int(net.get("concurrent_requests_per_domain", 1)),
            "RETRY_TIMES": int(net.get("retry_times", 8)),
            "DOWNLOAD_TIMEOUT": int(net.get("timeout_seconds", 45)),
            "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
            "AUTOTHROTTLE_ENABLED": bool(net.get("autothrottle_enabled", True)),
            "AUTOTHROTTLE_START_DELAY": float(net.get("autothrottle_start_delay", 2.0)),
            "AUTOTHROTTLE_MAX_DELAY": float(net.get("autothrottle_max_delay", 120.0)),
            "AUTOTHROTTLE_TARGET_CONCURRENCY": float(net.get("autothrottle_target_concurrency", 0.5)),
            "FEED_EXPORT_ENCODING": "utf-8",
        }

        extra_headers = dict(net.get("extra_headers", {}) or {})
        ua = net.get("user_agent")
        if ua and "User-Agent" not in extra_headers:
            extra_headers["User-Agent"] = ua
        if extra_headers:
            desired_settings["DEFAULT_REQUEST_HEADERS"] = extra_headers
        if ua:
            desired_settings["USER_AGENT"] = ua

        applied = []
        skipped = []

        for key, value in desired_settings.items():
            try:
                crawler.settings.set(key, value, priority="spider")
                applied.append(key)
            except TypeError:
                skipped.append(key)

        if skipped:
            spider.logger.warning(
                "No se pudieron aplicar settings dinámicos en runtime "
                "(probablemente Scrapy < 2.11). Se usarán custom_settings "
                "y fallbacks por request. Skipped: %s",
                ", ".join(skipped),
            )
        else:
            spider.logger.info("Settings dinámicos aplicados: %s", ", ".join(applied))

        return spider

    def _request_kwargs(self, meta: dict | None = None) -> dict:
        request_meta = dict(meta or {})
        request_meta.setdefault("download_timeout", self.download_timeout)
        request_meta.setdefault("max_retry_times", self.max_retry_times)

        return {
            "meta": request_meta,
            "dont_filter": True,
            "headers": self.extra_headers,
        }

    def _load_urls(self, path: str) -> list[tuple[str, str]]:
        """
        Soporta:
        - JSONL (cada línea: {"url":"...", "source":"..."})
        - TXT   (cada línea: URL)
        """
        rows: list[tuple[str, str]] = []

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return rows

        looks_like_jsonl = path.lower().endswith(".jsonl") or all(
            line.startswith("{") for line in lines[: min(5, len(lines))]
        )

        if looks_like_jsonl:
            for line in lines:
                try:
                    row = json.loads(line)
                    u = row.get("url")
                    src = row.get("source", "unknown")
                    if u:
                        rows.append((u, src))
                except Exception:
                    self.logger.warning("Línea JSONL inválida ignorada: %s", line[:200])
        else:
            for line in lines:
                if line and not line.startswith("#"):
                    rows.append((line, "txt"))

        return rows

    def _page_dir(self, title: str, pid: str | None, url: str) -> str:
        base_folder = safe_folder_name(title)
        folder = f"{base_folder}_{pid}" if pid else f"{base_folder}_{stable_hash(url)}"
        page_dir = os.path.join(self.output_dir, folder)
        os.makedirs(page_dir, exist_ok=True)
        return page_dir

    def _read_meta(self, page_dir: str) -> dict:
        meta_path = os.path.join(page_dir, "meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_meta(self, page_dir: str, meta: dict) -> None:
        meta_path = os.path.join(page_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def start_requests(self):
        for u, src in self._load_urls(self.urls_file):
            raw_url = (u or "").strip()
            if not raw_url:
                continue

            page_url = strip_query_fragment(raw_url)
            if not page_url:
                continue

            if page_url in self.seen:
                continue
            self.seen.add(page_url)

            if self.allowed_domains:
                if urlparse(page_url).netloc not in self.allowed_domains:
                    self.logger.info("URL fuera de allowed_domains, omitida: %s", page_url)
                    continue

            yield scrapy.Request(
                page_url,
                callback=self.parse_page,
                errback=self.on_error,
                **self._request_kwargs(
                    {
                        "list_source": src,
                        "page_url": page_url,
                        "request_kind": "page",
                    }
                ),
            )

            if self.also_download_url_export_view:
                ev = ensure_url_export_view(raw_url)
                yield scrapy.Request(
                    ev,
                    callback=self.parse_url_export_view,
                    errback=self.on_error,
                    **self._request_kwargs(
                        {
                            "list_source": src,
                            "page_url": page_url,
                            "export_view_url": ev,
                            "request_kind": "url_export_view",
                        }
                    ),
                )

    def on_error(self, failure):
        request = getattr(failure, "request", None)
        self.logger.warning("Request failed: %s", failure)

        if not request:
            return

        page_dir = request.meta.get("page_dir")
        if not page_dir:
            return

        meta = self._read_meta(page_dir)

        if request.meta.get("request_kind") == "rest_export_view":
            meta.setdefault("rest_export_view", {})
            meta["rest_export_view"]["attempted"] = True
            meta["rest_export_view"]["status"] = None
            meta["rest_export_view"]["error"] = str(failure.value)

        elif request.meta.get("request_kind") == "url_export_view":
            meta.setdefault("url_export_view", {})
            meta["url_export_view"]["attempted"] = True
            meta["url_export_view"]["status"] = None
            meta["url_export_view"]["error"] = str(failure.value)

        self._write_meta(page_dir, meta)

    def parse_page(self, response):
        page_url = response.meta.get("page_url") or response.url
        status = response.status
        html = response.text or ""
        title = response.css("title::text").get() or "Untitled"
        pid = extract_page_id(page_url) or extract_page_id(response.url)

        page_dir = self._page_dir(title, pid, page_url)

        with open(os.path.join(page_dir, "page.html"), "w", encoding="utf-8") as f:
            f.write(html)

        meta = {
            "page_url": page_url,
            "requested_url": response.url,
            "status": status,
            "title": title,
            "page_id": pid,
            "list_source": response.meta.get("list_source", "unknown"),
            "files": {
                "page_html": "page.html",
                "export_view_html": None,
                "url_export_view_html": None,
            },
            "rest_export_view": {
                "attempted": False,
                "status": None,
                "error": None,
            },
            "url_export_view": {
                "attempted": False,
                "status": None,
                "error": None,
            },
        }

        self._write_meta(page_dir, meta)

        if self.use_rest_export_view and pid:
            parsed = urlparse(page_url)
            api_url = (
                f"{parsed.scheme}://{parsed.netloc}"
                f"/wiki/rest/api/content/{pid}?expand={self.export_view_expand}"
            )

            yield scrapy.Request(
                api_url,
                callback=self.parse_rest_export_view,
                errback=self.on_error,
                **self._request_kwargs(
                    {
                        "page_dir": page_dir,
                        "page_url": page_url,
                        "page_id": pid,
                        "request_kind": "rest_export_view",
                    }
                ),
            )

        yield {
            "url": page_url,
            "status": status,
            "title": title,
            "output_folder": page_dir,
        }

    def parse_url_export_view(self, response):
        page_url = response.meta.get("page_url") or response.url
        html = response.text or ""
        title = response.css("title::text").get() or "Untitled"
        pid = extract_page_id(page_url) or extract_page_id(response.url)

        page_dir = self._page_dir(title, pid, page_url)

        with open(os.path.join(page_dir, "url_export_view.html"), "w", encoding="utf-8") as f:
            f.write(html)

        meta = self._read_meta(page_dir)
        meta.setdefault("files", {})
        meta["files"]["url_export_view_html"] = "url_export_view.html"
        meta.setdefault("url_export_view", {})
        meta["url_export_view"]["attempted"] = True
        meta["url_export_view"]["status"] = response.status
        meta["url_export_view"]["error"] = None

        self._write_meta(page_dir, meta)

    def parse_rest_export_view(self, response):
        page_dir = response.meta["page_dir"]

        meta = self._read_meta(page_dir)
        meta.setdefault("rest_export_view", {})
        meta["rest_export_view"]["attempted"] = True
        meta["rest_export_view"]["status"] = response.status
        meta["rest_export_view"]["error"] = None

        export_html = ""
        try:
            data = json.loads(response.text or "{}")
            export_html = (((data.get("body") or {}).get("export_view") or {}).get("value")) or ""
            if data.get("title"):
                meta["title"] = data["title"]
        except Exception as e:
            meta["rest_export_view"]["error"] = f"{type(e).__name__}: {e}"

        if export_html:
            with open(os.path.join(page_dir, "export_view.html"), "w", encoding="utf-8") as f:
                f.write(export_html)

            meta.setdefault("files", {})
            meta["files"]["export_view_html"] = "export_view.html"

        self._write_meta(page_dir, meta)