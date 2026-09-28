"""
GRBL CNC AI - ANSWER GENERATOR

Generates grounded technical answers using Groq first, with Ollama fallback.

Pipeline:
    User Question
        ↓
    Retrieved/Reranked Chunks
        ↓
    Context Builder
        ↓
    Groq (preferred) / Ollama (fallback)
        ↓
    Grounded Technical Answer
"""

import json
import os
import re
import sys
from time import perf_counter, sleep
from typing import List, Dict, Any

import requests

from config import CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    CONFIG.ollama_url
)

# Change this after checking:
# ollama list
#
# Example:
# OLLAMA_MODEL = "llama3.2:3b"
#
# We keep it configurable through an environment variable.
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    CONFIG.ollama_model
)

OLLAMA_NUM_PREDICT = CONFIG.ollama_num_predict

REQUEST_TIMEOUT = 180

# LLM provider selection:
#   auto   -> Groq when GROQ_API_KEY exists, otherwise Ollama
#   groq   -> Groq primary, Ollama fallback
#   ollama -> Ollama only
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
).rstrip("/")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
).strip()
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "30"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "400"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "1"))
GROQ_RETRY_DEFAULT_SECONDS = float(
    os.getenv("GROQ_RETRY_DEFAULT_SECONDS", "2.0")
)
GROQ_RETRY_MAX_SECONDS = float(
    os.getenv("GROQ_RETRY_MAX_SECONDS", "8.0")
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are GRBL CNC AI, a technical support assistant for CNC machines
using the GRBL controller.

Your job is to provide accurate, practical, safety-conscious technical
support based ONLY on the supplied knowledge-base context.

IMPORTANT RULES:

1. Use the supplied context as your primary source of truth.

2. Do NOT invent GRBL settings, alarm meanings, commands, electrical
   specifications, machine specifications, or procedures.

3. If the supplied context does not contain enough information to answer
   the question confidently, explicitly say:
   "The available GRBL documentation does not provide enough information
   to answer this confidently."

4. Distinguish between:
   - GRBL controller behavior
   - G-code behavior
   - CNC machine mechanics
   - spindle behavior
   - wiring/electrical problems

5. When mentioning a GRBL setting, show its exact number when available.

6. When mentioning a G-code command, show the exact command.

7. Never claim that a setting is responsible for a problem unless the
   supplied evidence supports that conclusion.

8. For troubleshooting questions:
   - Identify the most likely cause.
   - Give practical checks in order.
   - Explain what the technician should observe.
   - Explain what the result means.

9. For safety-related situations involving CNC motion, spindle operation,
   probing, limit switches, or machine movement, include an appropriate
   safety warning when necessary.

10. Do not blindly recommend changing settings.
    First explain what the setting controls and why it is relevant.

11. If the question is about a specific alarm, do not confuse the alarm
    with unrelated GRBL commands or settings.

12. Keep answers technically precise and reasonably concise.

13. Citation identities are attached by the application. Do not invent
    document names, section names, URLs, or chunk identifiers. In the SOURCES
    section write only: "See attached retrieved citations."

14. At the end, include:
    - Relevant GRBL setting/command if applicable
    - Confidence level

CONFIDENCE LEVELS:

HIGH:
The supplied documentation directly answers the question.

MEDIUM:
The documentation strongly supports the answer but some interpretation
is required.

LOW:
The documentation provides only partial information.

RESPONSE FORMAT:

ANSWER
<direct answer>

WHY
<technical explanation>

RECOMMENDED CHECKS
1. ...
2. ...
3. ...

GRBL COMMANDS / SETTINGS
<relevant commands or settings, if applicable>

SOURCES
See attached retrieved citations.

CONFIDENCE
HIGH / MEDIUM / LOW
"""


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """Simple client for a locally running Ollama server."""

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        num_predict: int = OLLAMA_NUM_PREDICT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_predict = num_predict

    def check_server(self) -> bool:
        """Check whether Ollama is reachable."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Return locally installed Ollama model names."""

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            return [
                model.get("name", "")
                for model in data.get("models", [])
            ]

        except requests.RequestException:
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> str:
        """Generate an answer using Ollama."""

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
            },
        }

        started_at = perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not communicate with Ollama: {exc}"
            ) from exc

        data = response.json()

        answer = data.get("response", "").strip()
        self.last_generation_seconds = perf_counter() - started_at

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer


# =============================================================================
# GROQ CLIENT + PROVIDER FALLBACK
# =============================================================================

class GroqClient:
    """Minimal Groq Chat Completions client using requests."""

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        base_url: str = GROQ_BASE_URL,
        model: str = GROQ_MODEL,
        max_tokens: int = GROQ_MAX_TOKENS,
        timeout: float = GROQ_TIMEOUT,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_generation_seconds = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        started_at = perf_counter()
        response = None
        last_error = None

        for attempt in range(GROQ_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429 and attempt < GROQ_MAX_RETRIES:
                    retry_after = response.headers.get("retry-after")
                    try:
                        wait_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        wait_seconds = GROQ_RETRY_DEFAULT_SECONDS

                    wait_seconds = max(
                        0.0,
                        min(wait_seconds, GROQ_RETRY_MAX_SECONDS),
                    )

                    print(
                        f"[GROQ RETRY] 429 rate limit. "
                        f"Waiting {wait_seconds:.2f}s before retry "
                        f"{attempt + 1}/{GROQ_MAX_RETRIES}."
                    )
                    sleep(wait_seconds)
                    continue

                response.raise_for_status()
                last_error = None
                break

            except requests.RequestException as exc:
                last_error = exc
                break

        if last_error is not None:
            raise RuntimeError(
                f"Could not communicate with Groq: {last_error}"
            ) from last_error

        if response is None:
            raise RuntimeError(
                "Groq request failed before receiving a response."
            )

        if response.status_code == 429:
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Could not communicate with Groq: {exc}"
                ) from exc

        self.last_generation_seconds = perf_counter() - started_at

        try:
            data = response.json()
            answer = (
                data["choices"][0]["message"]["content"]
                or ""
            ).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Groq returned an unexpected response."
            ) from exc

        if not answer:
            raise RuntimeError("Groq returned an empty response.")

        return answer


class ProviderFallbackClient:
    """Use Groq first and Ollama only if the primary provider fails."""

    def __init__(
        self,
        primary,
        fallback=None,
        primary_name: str = "primary",
        fallback_name: str = "fallback",
    ):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.last_provider_used = None
        self.last_generation_seconds = None
        self.last_primary_error = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> str:
        try:
            answer = self.primary.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            self.last_provider_used = self.primary_name
            self.last_generation_seconds = getattr(
                self.primary,
                "last_generation_seconds",
                None,
            )
            self.last_primary_error = None
            return answer
        except Exception as exc:
            self.last_primary_error = str(exc)

            print(
                f"[LLM FALLBACK] {self.primary_name} failed: "
                f"{self.last_primary_error}"
            )

            if self.fallback is None:
                raise

            print(
                f"[LLM FALLBACK] Switching to {self.fallback_name}."
            )

            answer = self.fallback.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            self.last_provider_used = self.fallback_name
            self.last_generation_seconds = getattr(
                self.fallback,
                "last_generation_seconds",
                None,
            )
            return answer


# =============================================================================
# CONTEXT BUILDER
# =============================================================================

def build_context(
    query: str,
    results: List[Dict[str, Any]],
    max_results: int = 5,
) -> str:
    """
    Convert reranked retrieval results into a structured LLM context.
    """

    selected = results[:max_results]

    context_parts = []

    for rank, item in enumerate(selected, start=1):

        chunk_id = item.get("chunk_id", "unknown")
        source = item.get("source", "unknown")
        document = item.get("document", "unknown")
        category = item.get("category", "unknown")
        section = item.get("section", "unknown")
        section_path = item.get("section_path", section)
        text = item.get("text", "")

        rerank_score = item.get("rerank_score")
        source_url = item.get("source_url") or (
            item.get("metadata") or {}
        ).get("source_url")

        part = f"""
--- EVIDENCE {rank} ---

Chunk ID:
{chunk_id}

Source:
{source}

Source URL:
{source_url}

Document:
{document}

Category:
{category}

Section:
{section}

Section Path:
{section_path}

Reranker Score:
{rerank_score}

Content:
{text}
"""

        context_parts.append(part.strip())

    return "\n\n".join(context_parts)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(
    query: str,
    results: List[Dict[str, Any]],
    query_info: Dict[str, Any] = None,
) -> str:
    """Build the final prompt sent to Ollama."""

    context = build_context(
        query=query,
        results=results,
        max_results=5,
    )

    query_type = (query_info or {}).get("query_type", "general")
    primary_section = results[0].get("section", "unknown") if results else "unknown"
    focus_entities = []
    info = query_info or {}
    if info.get("is_distance_query"):
        axis_settings = {"X": "$100", "Y": "$101", "Z": "$102"}
        focus_entities.extend(
            axis_settings[axis] for axis in info.get("axes", [])
            if axis in axis_settings
        )
    if info.get("is_homing_query"):
        focus_entities.extend(["$H", "$22"])
    if info.get("is_spindle_query"):
        focus_entities.extend(["M3", "M4", "M5"])
    if info.get("is_probe_query"):
        focus_entities.extend(["G38.2", "$6"])
    focus_entities = list(dict.fromkeys(focus_entities))
    evidence_text = "\n".join(
        f"{item.get('section', '')}\n{item.get('text', '')}" for item in results
    ).upper()
    focus_entities = [item for item in focus_entities if item in evidence_text]
    focus_text = ", ".join(focus_entities) if focus_entities else "None"

    prompt = f"""
USER QUESTION
=============
{query}

QUERY TYPE
==========
{query_type}

PRIMARY RETRIEVED SECTION
=========================
{primary_section}

EVIDENCE-DERIVED FOCUS IDENTIFIERS
==================================
{focus_text}

GRBL KNOWLEDGE BASE
===================

{context}

TASK
====
Answer the user's question using the GRBL knowledge-base evidence above.

Important:

- Prefer evidence that directly answers the question.
- Do not rely on general knowledge when the evidence is insufficient.
- Do not invent missing information.
- If the evidence contains conflicting information, explain the conflict.
- For troubleshooting, prioritize the evidence most directly related
  to the reported symptom.
- Quote exact GRBL setting numbers and G-code commands when relevant.
- For an exact identifier question, answer directly and concisely before any
  explanation.
- For troubleshooting, label documented facts separately from suggested
  diagnostic checks. Do not present a diagnostic suggestion as a documented
  GRBL fact.
- Do not create citations. Citation objects are attached by the application.
- Treat the primary retrieved section title as evidence, together with its
  content. Do not refuse merely because the defining identifier is in the
  section title rather than repeated in the content.
- Mention only identifiers that occur in the supplied evidence for this
  question. Do not introduce illustrative setting or command numbers.
- If evidence-derived focus identifiers are listed, explain their documented
  relevance before offering diagnostic suggestions. They are extracted from
  the retrieved evidence, not guessed.

Produce the final answer using the required response format.
"""

    return prompt.strip()


# =============================================================================
# ANSWER GENERATOR
# =============================================================================

class GRBLAnswerGenerator:
    """Generate grounded answers from reranked GRBL documents."""

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        provider: str = LLM_PROVIDER,
        groq_api_key: str = GROQ_API_KEY,
        groq_model: str = GROQ_MODEL,
    ):
        requested_provider = (provider or "auto").strip().lower()

        ollama_client = OllamaClient(
            base_url=ollama_url,
            model=model,
        )
        groq_client = GroqClient(
            api_key=groq_api_key,
            model=groq_model,
        )

        if requested_provider not in {"auto", "groq", "ollama"}:
            raise ValueError(
                "LLM_PROVIDER must be one of: auto, groq, ollama."
            )

        if requested_provider == "ollama":
            self.client = ollama_client
            self.provider_name = "ollama"

        elif requested_provider == "groq":
            if groq_client.configured:
                self.client = ProviderFallbackClient(
                    primary=groq_client,
                    fallback=ollama_client,
                    primary_name="groq",
                    fallback_name="ollama",
                )
                self.provider_name = "groq -> ollama fallback"
            else:
                self.client = ollama_client
                self.provider_name = "ollama (GROQ_API_KEY missing)"

        else:
            if groq_client.configured:
                self.client = ProviderFallbackClient(
                    primary=groq_client,
                    fallback=ollama_client,
                    primary_name="groq",
                    fallback_name="ollama",
                )
                self.provider_name = "groq -> ollama fallback"
            else:
                self.client = ollama_client
                self.provider_name = "ollama"

        self.last_grounding_check = {
            "passed": True,
            "unsupported_identifiers": [],
        }

    @staticmethod
    def extract_identifiers(text: str) -> set:
        patterns = re.findall(
            r"\$\$|\$[A-Z]+|\$\d{1,3}|\b[GM]\s?\d+(?:\.\d+)?\b|"
            r"\bALARM\s*:?\s*\d+\b",
            text.upper(),
        )
        return {
            re.sub(r"\s+|:(?=\d)", "", identifier)
            for identifier in patterns
        }

    def validate_grounding(
        self,
        answer: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence = "\n".join(
            f"{item.get('section', '')}\n{item.get('text', '')}"
            for item in results
        )
        answer_identifiers = self.extract_identifiers(answer)
        evidence_identifiers = self.extract_identifiers(evidence)
        unsupported = sorted(answer_identifiers - evidence_identifiers)
        unsupported = [
            identifier for identifier in unsupported
            if not (
                re.fullmatch(r"G\d+", identifier)
                and any(
                    evidence_identifier.startswith(identifier + ".")
                    for evidence_identifier in evidence_identifiers
                )
            )
        ]
        return {
            "passed": not unsupported,
            "answer_identifiers": sorted(answer_identifiers),
            "evidence_identifiers": sorted(evidence_identifiers),
            "unsupported_identifiers": unsupported,
        }

    def generate_answer(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_info: Dict[str, Any] = None,
    ) -> str:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not results:
            return (
                "I could not find relevant GRBL documentation for "
                "this question."
            )

        prompt = build_prompt(
            query=query,
            results=results,
            query_info=query_info,
        )

        answer = self.client.generate(prompt)
        self.last_grounding_check = self.validate_grounding(answer, results)

        if not self.last_grounding_check["passed"]:
            identifiers = ", ".join(
                self.last_grounding_check["unsupported_identifiers"]
            )
            return (
                "ANSWER\n"
                "The generated response was withheld because it contained "
                "GRBL identifiers not supported by the retrieved evidence.\n\n"
                f"Unsupported identifiers: {identifiers}\n\n"
                "SOURCES\nSee attached retrieved citations.\n\n"
                "CONFIDENCE\nLOW"
            )

        return answer


# =============================================================================
# TEST DATA
# =============================================================================

def load_test_results() -> List[Dict[str, Any]]:
    """
    Load retrieval results from an optional JSON file.

    This is mainly useful for testing the answer generator independently.
    """

    path = "data/processed/test_retrieval_results.json"

    if not os.path.exists(path):
        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception as exc:

        print(f"Could not load test results: {exc}")

        return []


# =============================================================================
# MAIN TEST
# =============================================================================

def main():

    print("=" * 80)
    print("GRBL CNC AI - ANSWER GENERATOR")
    print("=" * 80)
    print()

    answer_generator = GRBLAnswerGenerator()

    print(f"Configured provider: {answer_generator.provider_name}")
    print(f"Groq model: {GROQ_MODEL}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print()

    if isinstance(answer_generator.client, OllamaClient):
        print("Checking Ollama server...")
        if not answer_generator.client.check_server():
            print("Ollama server is not currently reachable.")
            return
        print("Ollama server: OK")
        print()

    test_results = load_test_results()

    if not test_results:
        print("No test retrieval result file found.")
        print()
        print("Answer generator is installed and ready.")
        return

    query = "What does $100 control?"
    print("=" * 80)
    print(f"TEST QUERY: {query}")
    print("=" * 80)

    answer = answer_generator.generate_answer(
        query=query,
        results=test_results,
    )

    print()
    print(answer)
    print()

    provider_used = getattr(
        answer_generator.client,
        "last_provider_used",
        None,
    )
    if provider_used:
        print(f"Provider used: {provider_used}")


if __name__ == "__main__":
    main()
