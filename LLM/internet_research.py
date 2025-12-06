"""
internet_research.py

Modular helper to add safe, auditable internet research to your RAG pipeline.

Features:
- Primary: uses OpenAI `web` tool (via OpenAI Python SDK) for live search & snippet extraction
- Fallback: SerpAPI search + simple fetch if OpenAI web tool is not available
- Sanitization: heuristic removal of step-by-step procedural sections (lab-safety protective)
- Allowlist: domain allowlist to restrict sources to trusted domains
- Caching: simple file-based TTL cache to reduce cost and latency
- Integration helper: `get_research_snippets(query, ...)` returns structured snippets ready to append to prompts

NOTE: this helper expects your environment to set OPENAI_API_KEY and optionally SERPAPI_API_KEY.
Also: for quick testing we include a path to the sample docs ZIP used earlier in this conversation:
    SAMPLE_ZIP_PATH = "/mnt/data/LabAssist_Sample_Docs.zip"

Use responsibly: do NOT forward raw procedural instructions from the web to end-users without human review.
"""

import os
import time
import json
import hashlib
from typing import List, Dict, Optional

# OpenAI Python SDK (v1+). The `OpenAI` client exposes chat completions with tools support.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# fallback search using SerpAPI
try:
    import requests
    SERPAPI_AVAILABLE = True
except Exception:
    SERPAPI_AVAILABLE = False

# Simple cache directory
CACHE_DIR = os.path.join(os.getcwd(), ".ir_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Sample zip path from the conversation (local file uploaded earlier)
SAMPLE_ZIP_PATH = "/mnt/data/LabAssist_Sample_Docs.zip"

# Domain allowlist (default). Modify as appropriate for your environment.
DEFAULT_TRUSTED_DOMAINS = {
    "cdc.gov",
    "who.int",
    "nih.gov",
    "nist.gov",
    "edu",
    "gov"
}

# TTL for cached search results (seconds)
CACHE_TTL = 60 * 60  # 1 hour


# ----------------------------- Utilities ---------------------------------

def _cache_key(prefix: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{h}.json"


def _cache_write(key: str, obj: dict):
    path = os.path.join(CACHE_DIR, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"ts": int(time.time()), "v": obj}, f)


def _cache_read(key: str, max_age: int) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, key)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if int(time.time()) - raw.get("ts", 0) > max_age:
        return None
    return raw.get("v")


# ----------------------------- Sanitization ------------------------------

def sanitize_text(text: str, max_snippet_chars: int = 1000) -> str:
    """Very small heuristic sanitizer that strips long step-lists and keeps short context.
    This is intentionally conservative: it removes lines that look like step-by-step instructions.
    """
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    safe_lines = []
    skip_block = False
    for line in lines:
        low = line.lower()
        # heuristics for procedural content
        if any(low.startswith(p) for p in ("step ", "step:", "procedure", "step") ):
            # skip following short block
            skip_block = True
            continue
        if any(kw in low for kw in ("do not", "warning", "caution", "danger", "avoid")):
            # keep safety warnings but strip detailed steps
            safe_lines.append(line)
            continue
        if skip_block:
            # heuristically skip until short paragraph end
            if len(line) < 40:
                # continue skipping
                continue
            else:
                skip_block = False
        # append only non-procedural lines
        safe_lines.append(line)
        if sum(len(s) for s in safe_lines) > max_snippet_chars:
            break
    snippet = " ".join(safe_lines)
    # final trim
    return snippet[:max_snippet_chars]


# ----------------------------- OpenAI web tool ----------------------------

def _openai_web_search(query: str, max_results: int = 3) -> List[Dict]:
    """Use OpenAI's web tool via the OpenAI Python client.
    Returns a list of dicts: {title, snippet, url, domain}
    NOTE: This requires an OpenAI account and a model that supports tools.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available in environment")

    client = OpenAI()

    # Use chat completion with `tools` param (OpenAI tooling). Implementation details of
    # returned structure may vary with SDK version — adapt if necessary.
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": query}],
            tools=[
                {"type": "web", "web": {"search": {"enable": True}, "browser": {"enable": False}}}
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        raise

    # The SDK wraps tool outputs in the message — SDK shapes vary; attempt several safe extraction patterns.
    tool_outputs = []
    try:
        # Newer SDKs may expose tool outputs under choices[0].message.tool_outputs
        choice = resp.choices[0]
        msg = choice.message
        # try several fields defensively
        if hasattr(msg, "tool_outputs") and msg.tool_outputs:
            tool_outputs = msg.tool_outputs
        elif isinstance(msg.content, dict) and msg.content.get("tool_outputs"):
            tool_outputs = msg.content.get("tool_outputs")
        else:
            # fallback: parse text for urls/snippets (best-effort)
            text = getattr(msg, "content", str(msg))
            tool_outputs = [{"title": "search", "snippet": text, "url": "", "domain": ""}]
    except Exception:
        # final fallback
        tool_outputs = []

    results = []
    for item in tool_outputs[:max_results]:
        title = item.get("title") if isinstance(item, dict) else item.get("name") if isinstance(item, dict) else ""
        snippet = item.get("snippet") or item.get("text") or (item.get("summary") if isinstance(item, dict) else "")
        url = item.get("url") or item.get("link") or ""
        domain = ""
        if url:
            try:
                import tldextract
                ext = tldextract.extract(url)
                domain = ".".join(p for p in (ext.domain, ext.suffix) if p)
            except Exception:
                domain = url
        results.append({"title": title or "", "snippet": sanitize_text(snippet), "url": url, "domain": domain})
    return results


# ----------------------------- SerpAPI fallback ---------------------------

def _serpapi_search(query: str, max_results: int = 3, serpapi_key: Optional[str] = None) -> List[Dict]:
    """Fallback to SerpAPI (or Bing) to retrieve top search results and short snippets.
    Requires SERPAPI_API_KEY env var or serpapi_key param.
    """
    key = serpapi_key or os.getenv("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("SerpAPI key not provided in environment")
    if not SERPAPI_AVAILABLE:
        raise RuntimeError("`requests` not available in environment")

    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": key,
    }
    resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("organic_results", [])[:max_results]:
        title = r.get("title")
        snippet = r.get("snippet") or r.get("rich_snippet", {}).get("top", "")
        url = r.get("link") or r.get("url")
        domain = ""
        if url:
            try:
                import tldextract
                ext = tldextract.extract(url)
                domain = ".".join(p for p in (ext.domain, ext.suffix) if p)
            except Exception:
                domain = url
        results.append({"title": title, "snippet": sanitize_text(snippet), "url": url, "domain": domain})
    return results


# ----------------------------- Public API --------------------------------

def get_research_snippets(query: str,
                          max_results: int = 3,
                          allowed_domains: Optional[set] = None,
                          use_cache: bool = True) -> List[Dict]:
    """Return a list of cleaned search snippets for use in RAG prompts.

    Each dict: {title, snippet, url, domain}
    """
    allowed = set(allowed_domains) if allowed_domains else set(DEFAULT_TRUSTED_DOMAINS)
    cache_key = _cache_key("web", query)
    if use_cache:
        out = _cache_read(cache_key, CACHE_TTL)
        if out:
            return out

    # Try OpenAI web tools first (if available)
    results = []
    if OPENAI_AVAILABLE:
        try:
            results = _openai_web_search(query, max_results=max_results)
        except Exception:
            results = []

    # If OpenAI returned nothing, fallback to SerpAPI
    if not results and SERPAPI_AVAILABLE:
        try:
            results = _serpapi_search(query, max_results=max_results)
        except Exception:
            results = []

    # Filter by allowlist and remove empty snippets
    filtered = []
    for r in results:
        domain = (r.get("domain") or "").lower()
        # accept if domain in allowlist OR allowlist contains a top-level token like 'edu' and domain endswith it
        accept = False
        if not allowed:
            accept = True
        else:
            for a in allowed:
                if a in domain or domain.endswith('.' + a) or domain.endswith(a):
                    accept = True
                    break
        if not accept:
            continue
        if not r.get("snippet"):
            continue
        filtered.append(r)

    # Cache and return
    if use_cache:
        _cache_write(cache_key, filtered)
    return filtered


def integrate_with_rag(sop_chunks: List[Dict], query: str, max_web: int = 2) -> List[Dict]:
    """Given local SOP chunks (list of {text, source, score}), decide whether to augment with web and return combined context.

    Returns a list of context blocks: first the local SOP chunks, then web snippets if any.
    """
    # Basic heuristic: if top local chunk has low semantic relevance (short text or low score), call web
    call_web = False
    if not sop_chunks:
        call_web = True
    else:
        # if the average length of texts is small => less confident
        avg_len = sum(len(c.get("text","")) for c in sop_chunks) / max(1, len(sop_chunks))
        if avg_len < 200:
            call_web = True

    combined = list(sop_chunks)
    if call_web:
        web = get_research_snippets(query, max_results=max_web)
        # append as pseudo-documents with source='web:domain'
        for w in web:
            combined.append({"text": w.get("snippet"), "source": f"web:{w.get('domain')}", "meta": {"url": w.get("url")}})
    return combined


# ----------------------------- Example usage -----------------------------
if __name__ == "__main__":
    print("internet_research helper - quick demo")
    q = "current CDC guidelines for laboratory fume hood certification"
    try:
        snippets = get_research_snippets(q, max_results=3)
        print("Returned snippets:")
        for s in snippets:
            print(json.dumps(s, indent=2))
    except Exception as e:
        print("Failed to run web search (missing SDK/API?). Error:", e)
        print("You can set OPENAI_API_KEY to enable OpenAI web tool or SERPAPI_API_KEY for SerpAPI fallback.")
