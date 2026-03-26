# discover_urls_spider_v2.py
# Run:
#   scrapy runspider discover_urls_spider.py -a config=config_discover.yaml -O urls_instalation.jsonl -s LOG_LEVEL=INFO
"""

scrapy runspider discover_urls_spider_v2.py \
  -a config=config_discover_v2.yaml \
  -O urls_instalation.jsonl \
  -s LOG_LEVEL=INFO

"""
import re
import json
import gzip
import yaml
import scrapy
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from xml.etree import ElementTree
from bs4 import BeautifulSoup


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_query_fragment(url: str) -> str:
    parts = list(urlparse(url))
    parts[4] = ""  # query
    parts[5] = ""  # fragment
    return urlunparse(parts)


def norm_url(url: str) -> str:
    return strip_query_fragment(url).rstrip("/")


def path_depth(url: str) -> int:
    p = urlparse(url).path.strip("/")
    if not p:
        return 0
    return len([seg for seg in p.split("/") if seg])


def compile_patterns(patterns):
    if not patterns:
        return []
    return [re.compile(p) for p in patterns]


def allowed_by_patterns(url: str, allow_pats, deny_pats) -> bool:
    if allow_pats and not any(p.search(url) for p in allow_pats):
        return False
    if deny_pats and any(p.search(url) for p in deny_pats):
        return False
    return True


def decode_json_string_literal(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except Exception:
        return s


def extract_adf_links_from_html(html: str) -> list[str]:
    out = []
    if not html:
        return out

    pats = [
        r'"href"\s*:\s*"([^"]+)"',
        r'"url"\s*:\s*"([^"]+)"',
    ]
    for pat in pats:
        for m in re.finditer(pat, html):
            raw = m.group(1)
            if not raw:
                continue
            out.append(decode_json_string_literal(raw))

    return out


def fix_common_typos(u: str) -> str:
    return u.replace("/paxges/", "/pages/")


class DiscoverUrlsSpider(scrapy.Spider):
    name = "discover_urls_v2"

    def __init__(self, config="config_discover.yaml", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = load_config(config)

        site = self.cfg.get("site", {})
        disc = self.cfg.get("discovery", {})
        net = self.cfg.get("network", {})
        ext = self.cfg.get("external", {})

        self.start_url = site.get("start_url")
        if not self.start_url:
            raise ValueError("Missing site.start_url in config")

        self.allowed_domains = site.get("allowed_domains") or [urlparse(self.start_url).netloc]
        self.follow_external = bool(site.get("follow_external", False))
        obey_robots = bool(site.get("obey_robots", True))

        # discovery config
        self.mode = (disc.get("mode", "crawl") or "crawl").lower()
        self.depth_limit = int(disc.get("depth_limit", 5))
        self.path_depth_limit = int(disc.get("path_depth_limit", 999))
        self.strip_query = bool(disc.get("strip_query", True))

        self.link_scope_css = (disc.get("link_scope_css") or "#AkMainContent").strip()
        self.fallback_scopes_css = disc.get("fallback_scopes_css", []) or []
        if isinstance(self.fallback_scopes_css, str):
            self.fallback_scopes_css = [self.fallback_scopes_css]

        self.min_links_in_scope = int(disc.get("min_links_in_scope", 3))
        self.extract_adf_links = bool(disc.get("extract_adf_links", True))
        self.fix_typos = bool(disc.get("fix_common_typos", True))
        self.use_export_view_param = bool(disc.get("use_export_view_param", False))

        self.allow_pats = compile_patterns(disc.get("allow_regex", []))
        self.deny_pats = compile_patterns(disc.get("deny_regex", []))

        self.sitemap_urls = disc.get("sitemap_urls", []) or []

        # external policy
        self.external_capture = bool(ext.get("capture", True))
        self.external_follow = bool(ext.get("follow", False))
        self.external_allow_domains = set(ext.get("allow_domains", []) or [])
        self.external_max_from_depth = int(ext.get("max_capture_from_internal_depth", 0))

        extra_headers = net.get("extra_headers", {}) or {}

        self.custom_settings = {
            "ROBOTSTXT_OBEY": obey_robots,
            "USER_AGENT": net.get("user_agent"),
            "DOWNLOAD_DELAY": float(net.get("download_delay", 2.5)),
            "RANDOMIZE_DOWNLOAD_DELAY": bool(net.get("randomize_download_delay", True)),
            "CONCURRENT_REQUESTS": int(net.get("concurrent_requests", 1)),
            "CONCURRENT_REQUESTS_PER_DOMAIN": int(net.get("concurrent_requests_per_domain", 1)),
            "RETRY_TIMES": int(net.get("retry_times", 8)),
            "DOWNLOAD_TIMEOUT": int(net.get("timeout_seconds", 45)),
            "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
            "DEFAULT_REQUEST_HEADERS": extra_headers,
            "FEED_EXPORT_ENCODING": "utf-8",
            "AUTOTHROTTLE_ENABLED": bool(net.get("autothrottle_enabled", True)),
            "AUTOTHROTTLE_START_DELAY": float(net.get("autothrottle_start_delay", 2.0)),
            "AUTOTHROTTLE_MAX_DELAY": float(net.get("autothrottle_max_delay", 120.0)),
            "AUTOTHROTTLE_TARGET_CONCURRENCY": float(net.get("autothrottle_target_concurrency", 0.5)),
        }

        self.seen = set()

    def _ensure_export_view(self, url: str) -> str:
        p = urlparse(url)
        q = dict(parse_qsl(p.query, keep_blank_values=True))
        q["format"] = "export_view"
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))

    def start_requests(self):
        # sitemap optional
        if self.mode in ("auto", "sitemap") and self.sitemap_urls:
            for sm in self.sitemap_urls:
                yield scrapy.Request(sm, callback=self.parse_sitemap, dont_filter=True)

        # crawl
        if self.mode in ("crawl", "auto"):
            url = self._ensure_export_view(self.start_url) if self.use_export_view_param else self.start_url
            yield scrapy.Request(url, callback=self.parse_crawl, dont_filter=True, meta={"depth": 0})

    def parse_sitemap(self, response):
        if response.status != 200:
            if self.mode == "auto":
                url = self._ensure_export_view(self.start_url) if self.use_export_view_param else self.start_url
                yield scrapy.Request(url, callback=self.parse_crawl, dont_filter=True, meta={"depth": 0})
            return

        body = response.body
        if response.url.endswith(".gz") or (b"<urlset" not in body and b"<sitemapindex" not in body):
            try:
                body = gzip.decompress(body)
            except Exception:
                pass

        try:
            root = ElementTree.fromstring(body)
        except Exception:
            return

        tag = root.tag.lower()
        if "sitemapindex" in tag:
            for loc in root.findall(".//{*}loc"):
                sm_url = (loc.text or "").strip()
                if sm_url:
                    yield scrapy.Request(sm_url, callback=self.parse_sitemap, dont_filter=True)
            return

        if "urlset" in tag:
            for loc in root.findall(".//{*}loc"):
                u = (loc.text or "").strip()
                if not u:
                    continue
                if self.strip_query:
                    u = strip_query_fragment(u)

                if urlparse(u).netloc and urlparse(u).netloc not in self.allowed_domains:
                    continue
                if path_depth(u) > self.path_depth_limit:
                    continue
                if not allowed_by_patterns(u, self.allow_pats, self.deny_pats):
                    continue

                if u not in self.seen:
                    self.seen.add(u)
                    yield {"url": u, "source": "sitemap", "depth": None, "path_depth": path_depth(u)}
            return

    def parse_crawl(self, response):
        depth = int(response.meta.get("depth", 0))

        current = response.url
        if self.strip_query:
            current = strip_query_fragment(current)

        if current not in self.seen:
            self.seen.add(current)
            yield {"url": current, "source": "crawl", "depth": depth, "path_depth": path_depth(current)}

        if depth >= self.depth_limit:
            return

        html = response.text or ""
        soup = BeautifulSoup(html, "lxml")

        def anchors_in_scope(css: str) -> list[str]:
            sc = soup.select_one(css) if css else None
            if not sc:
                return []
            for nav in sc.select("nav, aside, [role='navigation']"):
                nav.decompose()
            return [a.get("href") for a in sc.select("a[href]") if a.get("href")]

        primary = anchors_in_scope(self.link_scope_css)
        hrefs = list(primary)

        if len(primary) < self.min_links_in_scope:
            for css in self.fallback_scopes_css:
                hrefs.extend(anchors_in_scope(css))
            if self.extract_adf_links:
                hrefs.extend(extract_adf_links_from_html(html))

        # dedupe hrefs
        uniq = []
        seen_h = set()
        for h in hrefs:
            if not h:
                continue
            if h in seen_h:
                continue
            seen_h.add(h)
            uniq.append(h)

        for href in uniq:
            u = response.urljoin(href)
            if self.fix_typos:
                u = fix_common_typos(u)
            if self.strip_query:
                u = strip_query_fragment(u)

            if u.startswith(("mailto:", "tel:", "javascript:")):
                continue

            parsed = urlparse(u)
            netloc = parsed.netloc
            is_external = bool(netloc) and (netloc not in self.allowed_domains)

            if is_external:
                if self.external_capture and (depth < self.external_max_from_depth):
                    if self.external_allow_domains and netloc not in self.external_allow_domains:
                        continue
                    if u not in self.seen:
                        self.seen.add(u)
                        yield {"url": u, "source": "external_l1", "depth": depth + 1, "path_depth": path_depth(u), "from": current}
                continue

            if path_depth(u) > self.path_depth_limit:
                continue
            if not allowed_by_patterns(u, self.allow_pats, self.deny_pats):
                continue

            if u not in self.seen:
                self.seen.add(u)
                yield {"url": u, "source": "discovered", "depth": depth + 1, "path_depth": path_depth(u), "from": current}

            next_url = self._ensure_export_view(u) if self.use_export_view_param else u
            yield scrapy.Request(next_url, callback=self.parse_crawl, dont_filter=True, meta={"depth": depth + 1})