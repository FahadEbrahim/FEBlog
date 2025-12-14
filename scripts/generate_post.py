from __future__ import annotations

import os
import re
import time
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple


# -----------------------------
# Config
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "120"))
TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))

# Daily interval (UTC). Default: yesterday..today
DAYS_BACK = int(os.getenv("ARXIV_DAYS_BACK", "1"))
MAX_RESULTS_PER_TOPIC = int(os.getenv("ARXIV_MAX_RESULTS", "25"))

TOPICS = [
    "Code Plagiarism",
    "Software Plagiarism",
    "Code Models",
    "Code Pre-Trained Models",
    "Pre-Trained Code Models",
]

# Free-only model priority (no paid fallback)
FREE_MODEL_PRIORITY = [
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "mistralai/devstral-2512:free",
    "qwen/qwen3-coder:free",
    "kwaipilot/kat-coder-pro:free",
]

CODE_HOST_RE = re.compile(r"(https?://(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)/[^\s)]+)", re.I)


# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "post"

def extract_code_links(abstract: str) -> List[str]:
    return sorted({m.group(1).rstrip(".,;") for m in CODE_HOST_RE.finditer(abstract or "")})

def _is_free_model(model_obj: Dict[str, Any]) -> bool:
    pricing = (model_obj or {}).get("pricing") or {}
    # pricing fields like prompt/completion/image/request, "0" indicates free
    values = [str(v) for v in pricing.values() if v is not None]
    return bool(values) and all(v in ("0", "0.0") for v in values)

def choose_free_model() -> str:
    # If no key, we can still generate a post without summaries (placeholder).
    if not OPENROUTER_API_KEY:
        return ""

    r = requests.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    by_id = {m.get("id"): m for m in data}

    for mid in FREE_MODEL_PRIORITY:
        if mid in by_id and _is_free_model(by_id[mid]):
            return mid

    # Last resort: any free model
    for mid, mobj in by_id.items():
        if str(mid).endswith(":free") and _is_free_model(mobj):
            return mid

    return ""

def call_openrouter(model: str, messages: List[Dict[str, str]], temperature: float) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY.")
    if not model:
        raise RuntimeError("No free OpenRouter model available.")

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "top_p": 1.0}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# -----------------------------
# arXiv fetching
# -----------------------------
def arxiv_query(topic: str, start_dt: datetime, end_dt: datetime) -> List[Dict[str, str]]:
    # arXiv API: export.arxiv.org/api/query
    # We'll fetch recent results and filter by published date ourselves.
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f'all:"{topic}"',
        "start": 0,
        "max_results": MAX_RESULTS_PER_TOPIC,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    out: List[Dict[str, str]] = []
    for e in entries:
        published = e.findtext("atom:published", default="", namespaces=ns)
        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
        abstract = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        paper_id = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()

        if not published or not paper_id:
            continue

        pub_dt = datetime.strptime(published.split("T")[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if not (start_dt.date() <= pub_dt.date() <= end_dt.date()):
            continue

        # Convert abs -> pdf link
        pdf_link = paper_id.replace("/abs/", "/pdf/") + ".pdf"

        authors = []
        for a in e.findall("atom:author", ns):
            name = a.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name.strip())

        out.append(
            {
                "topic": topic,
                "title": title,
                "authors": ", ".join(authors),
                "published": published.split("T")[0],
                "abstract": abstract,
                "pdf": pdf_link,
                "abs": paper_id,
            }
        )

    # De-dup by abs link
    seen = set()
    uniq = []
    for item in out:
        if item["abs"] in seen:
            continue
        seen.add(item["abs"])
        uniq.append(item)
    return uniq


# -----------------------------
# Summarization prompt
# -----------------------------
def build_summary_prompt(item: Dict[str, str]) -> str:
    links = extract_code_links(item.get("abstract", ""))
    resources = ""
    if links:
        resources = "Code/resources mentioned by authors:\n" + "\n".join(f"- {u}" for u in links) + "\n\n"

    return f"""Write a concise research-digest summary.

Title: {item["title"]}
Authors: {item["authors"]}
Published: {item["published"]}
arXiv: {item["abs"]}
PDF: {item["pdf"]}

Abstract:
{item["abstract"]}

{resources}
Output format:
- 6 to 10 bullet points
- include: problem, method, data/benchmarks, key results, limitations, why it matters
- if resource links exist, mention them explicitly as “Resources: …”
No guessing details not supported by the abstract."""


def summarize_items(items: List[Dict[str, str]], model: str) -> List[Tuple[Dict[str, str], str, List[str]]]:
    results = []
    for it in items:
        links = extract_code_links(it.get("abstract", ""))
        if not OPENROUTER_API_KEY or not model:
            summary = "(No summary: missing OpenRouter key or free model.)"
        else:
            messages = [
                {"role": "system", "content": "You write crisp academic digests."},
                {"role": "user", "content": build_summary_prompt(it)},
            ]
            try:
                summary = call_openrouter(model, messages, temperature=TEMPERATURE).strip()
            except Exception as e:
                summary = f"(Summary failed: {e})"
            # tiny pacing so you don’t immediately slam into free-tier limits
            time.sleep(0.3)
        results.append((it, summary, links))
    return results


# -----------------------------
# Post generation
# -----------------------------
def write_post(date_str: str, topic_groups: Dict[str, List[Tuple[Dict[str, str], str, List[str]]]]) -> Path:
    POSTS_DIR.mkdir(parents=True, exist_ok=True)

    title = f"Daily arXiv digest ({date_str}, UTC)"
    filename = f"{date_str}-{slugify('arxiv-digest')}.md"
    out_path = POSTS_DIR / filename

    if out_path.exists():
        print(f"Post already exists: {out_path}")
        return out_path

    lines: List[str] = []
    lines.append("---")
    lines.append("layout: post")
    lines.append(f'title: "{title}"')
    lines.append("---\n")

    lines.append(f"Model: `{os.getenv('OPENROUTER_MODEL_USED','(free auto)')}`\n")

    for topic, items in topic_groups.items():
        lines.append(f"## {topic}\n")
        if not items:
            lines.append("_No matching papers in this interval._\n")
            continue

        for it, summary, links in items:
            lines.append(f"### {it['title']}\n")
            lines.append(f"**Authors:** {it['authors']}  ")
            lines.append(f"**Published:** {it['published']}  ")
            lines.append(f"**arXiv:** {it['abs']}  ")
            lines.append(f"**PDF:** {it['pdf']}\n")
            if links:
                lines.append("**Resources:**")
                for u in links:
                    lines.append(f"- {u}")
                lines.append("")
            lines.append("**Summary:**")
            lines.append(summary)
            lines.append("\n---\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return out_path


def main():
    # Use UTC to avoid timezone nonsense
    today = datetime.now(timezone.utc).date()
    start = datetime.combine(today - timedelta(days=DAYS_BACK), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)

    date_str = today.isoformat()

    model = choose_free_model()
    if model:
        os.environ["OPENROUTER_MODEL_USED"] = model

    all_by_topic: Dict[str, List[Tuple[Dict[str, str], str, List[str]]]] = {}

    for topic in TOPICS:
        items = arxiv_query(topic, start, today_as_dt(end))
        summarized = summarize_items(items, model)
        all_by_topic[topic] = summarized
    total_papers = sum(len(v) for v in all_by_topic.values())
    
    if total_papers == 0:
        print("No papers found in this interval. Skipping post creation.")
        return
    
    write_post(date_str, all_by_topic)


def today_as_dt(dt: datetime) -> datetime:
    # helper just to keep the call above readable
    return dt

if __name__ == "__main__":
    main()
