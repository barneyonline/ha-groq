"""Normalize provider source annotations without exposing arbitrary metadata."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def extract_citations(message: dict[str, Any]) -> list[dict[str, str]]:
    """Return unique HTTP source URLs from annotations, searches and browsing."""
    candidates: list[Any] = []
    for key in ("annotations", "citations"):
        if isinstance(message.get(key), list):
            candidates.extend(message[key])
    for tool in message.get("executed_tools") or []:
        if not isinstance(tool, dict):
            continue
        results = tool.get("search_results")
        if isinstance(results, dict):
            results = results.get("results", [])
        if isinstance(results, list):
            candidates.extend(results)
        if isinstance(tool.get("browser_results"), list):
            candidates.extend(tool["browser_results"])
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = {"url": candidate}
        if not isinstance(candidate, dict):
            continue
        candidate = candidate.get("url_citation", candidate)
        if not isinstance(candidate, dict):
            continue
        url = candidate.get("url")
        if not isinstance(url, str) or len(url) > 4096 or any(c.isspace() for c in url):
            continue
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"https", "http"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                continue
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        title = candidate.get("title")
        sources.append(
            {"url": url, "title": title[:512] if isinstance(title, str) else ""}
        )
        if len(sources) == 50:
            break
    return sources
