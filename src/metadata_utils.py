"""Backward-compatible provenance and citation helpers."""

from typing import Any, Dict, Optional


KNOWN_SOURCE_URLS = {
    "commands.md": (
        "https://raw.githubusercontent.com/gnea/grbl/master/"
        "doc/markdown/commands.md"
    ),
    "settings.md": (
        "https://raw.githubusercontent.com/gnea/grbl/master/"
        "doc/markdown/settings.md"
    ),
}


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Add provenance keys without changing or requiring older datasets."""

    normalized = dict(metadata)
    document = normalized.get("document")
    normalized.setdefault("source_url", KNOWN_SOURCE_URLS.get(document))
    normalized.setdefault("source_document", document)
    normalized.setdefault("source_section", normalized.get("section_path"))
    normalized.setdefault("source_type", normalized.get("authority") or "unknown")
    normalized.setdefault("retrieved_at", None)
    normalized.setdefault("upstream_revision", None)
    normalized.setdefault("checksum", None)
    return normalized


def citation_from_result(
    result: Dict[str, Any],
    excerpt_chars: int = 320,
) -> Dict[str, Optional[str]]:
    """Create a deterministic citation whose identity comes from retrieval."""

    nested = normalize_metadata(result.get("metadata") or {})
    merged = dict(nested)
    merged.update({key: value for key, value in result.items() if value is not None})
    merged = normalize_metadata(merged)
    text = str(merged.get("text") or "").strip()
    excerpt = " ".join(text.split())
    if len(excerpt) > excerpt_chars:
        excerpt = excerpt[: excerpt_chars - 1].rstrip() + "…"

    return {
        "citation_id": str(merged.get("chunk_id") or "unknown"),
        "chunk_id": merged.get("chunk_id"),
        "document": merged.get("source_document") or merged.get("document"),
        "section": merged.get("source_section") or merged.get("section"),
        "source": merged.get("source"),
        "source_url": merged.get("source_url"),
        "source_type": merged.get("source_type"),
        "excerpt": excerpt,
    }
