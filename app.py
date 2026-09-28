"""Professional Streamlit demo for the existing GRBL CNC AI orchestrator."""

import sys
import re
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grbl_ai import GRBLCNCAI


EXAMPLES = (
    "What does $100 control?",
    "What does G90 do?",
    "What does M3 do?",
    "My X axis moves the wrong distance. What should I check?",
)


def apply_styles():
    st.markdown(
        """
        <style>
        .stMainBlockContainer { max-width: 880px; padding-top: 3rem; padding-bottom: 4rem; }
        h1 { letter-spacing: -0.035em; margin-bottom: 0.1rem; }
        h2, h3 { letter-spacing: -0.015em; }

        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(120, 130, 140, 0.08);
            border: 1px solid rgba(120, 130, 140, 0.18);
            border-radius: 0.55rem;
            padding: 0.55rem 0.7rem;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 0.98rem;
        }

        .stButton > button[kind="primary"] {
            background: #176b73;
            border: 1px solid #176b73;
            border-radius: 0.45rem;
            color: white;
            font-weight: 650;
        }

        .stButton > button[kind="primary"]:hover {
            background: #12555b;
            border-color: #12555b;
            color: white;
        }

        .status-badge {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border: 1px solid rgba(23, 107, 115, 0.35);
            border-radius: 999px;
            background: rgba(23, 107, 115, 0.08);
            color: #176b73;
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.02em;
        }

        .brand-note {
            line-height: 1.35;
            opacity: 0.78;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading GRBL documentation and retrieval models...")
def get_assistant():
    return GRBLCNCAI()


def render_sidebar(assistant):
    degraded = getattr(assistant, "degraded_reasons", [])
    retrieval_mode = "BM25 degraded mode" if degraded else "Hybrid (BM25 + FAISS)"
    corpus_size = len(getattr(assistant.retriever.bm25, "chunks", []))

    with st.sidebar:
        st.subheader("System status")
        st.metric("AI engine", "Cloud")
        st.metric("Retrieval", retrieval_mode)
        st.metric("Knowledge base", f"{corpus_size} chunks")

        if degraded:
            st.warning("Some optional retrieval components are unavailable.")

        st.divider()
        st.markdown(
            '<div class="brand-note"><strong>TechSolAi</strong><br>'
            '<small>AI Solutions for Technical Knowledge</small></div>',
            unsafe_allow_html=True,
        )


def safe_markdown_text(value):
    """Render GRBL $ commands/settings literally without visible backslashes.

    Streamlit Markdown can interpret dollar signs as LaTeX delimiters. Wrap
    GRBL-style dollar tokens in inline-code spans instead, while leaving text
    already inside backticks unchanged.
    """
    value = str(value or "")

    parts = re.split(r"(`[^`]*`)", value)

    for index, part in enumerate(parts):
        if index % 2 == 1:
            continue

        # GRBL settings / status commands: $100, $101, $G, $H, $X, $$, etc.
        part = re.sub(
            r"(?<!\\)(\$\$|\$\d{1,3}\??|\$[A-Za-z](?:=[^\s,.;:]*)?\??)",
            r"`\1`",
            part,
        )
        parts[index] = part

    return "".join(parts)


def safe_expander_label(value):
    """Keep dollar signs literal in Streamlit expander labels."""
    return str(value or "").replace("$", r"\$")


def strip_duplicate_safety(answer, safety_message):
    """Avoid showing the same safety notice twice in the UI."""
    answer = str(answer or "").strip()
    safety_message = str(safety_message or "").strip()

    if not answer or not safety_message:
        return answer

    prefixes = (
        safety_message,
        f"SAFETY: {safety_message}",
    )

    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip(" \n:-")
            break

    return answer


def clean_source_label(value):
    """Clean display-only source labels without changing retrieval data."""
    value = str(value or "").strip()

    # Normalize common mojibake / punctuation artifacts seen in imported docs.
    replacements = {
        "â€”": "—",
        "â€“": "–",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
        "′": "'",
        "`": "",
    }
    for bad, good in replacements.items():
        value = value.replace(bad, good)

    # Collapse repeated whitespace / isolated punctuation fragments.
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"(?:\s*['‘’]+\s*){2,}", " ", value)
    value = re.sub(r"\s+>", " >", value)
    return value.strip(" -–—>'‘’")


def render_sources(citations):
    with st.container(border=True):
        st.subheader("Sources")

        if not citations:
            st.caption("No sources were returned.")
            return

        visible_citations = list(citations)[:3]

        for index, citation in enumerate(visible_citations, start=1):
            document = clean_source_label(
                citation.get("document") or "Unknown document"
            )
            section = clean_source_label(
                citation.get("section") or "Unknown section"
            )
            source = clean_source_label(
                citation.get("source") or "Unknown source"
            )

            label = safe_expander_label(
                f"{index}. {document} — {section}"
            )
            with st.expander(label):
                left, right = st.columns(2)

                left.caption("DOCUMENT")
                left.write(document)

                right.caption("SOURCE")
                right.write(source)

                st.caption("SECTION")
                st.write(section)

                if citation.get("excerpt"):
                    st.markdown("**Relevant excerpt**")
                    st.caption(
                        clean_source_label(citation["excerpt"])
                    )

                if citation.get("source_url"):
                    st.link_button(
                        "View original source",
                        citation["source_url"],
                    )

        if len(citations) > len(visible_citations):
            st.caption(
                f"Showing top {len(visible_citations)} of "
                f"{len(citations)} retrieved sources."
            )



def render_response(response):
    validation = response.get("validation", {})
    evidence = validation.get("level", "UNKNOWN")
    safety = response.get("safety", {})

    response_mode = (
        "Fast exact lookup"
        if response.get("answer_grounding", {}).get("mode") == "deterministic_exact"
        else None
    )

    status_left, status_right = st.columns(2)

    with status_left:
        st.caption("EVIDENCE STATUS")
        st.markdown(
            f'<span class="status-badge">{evidence}</span>',
            unsafe_allow_html=True,
        )

    with status_right:
        if response_mode:
            st.caption("RESPONSE MODE")
            st.markdown(
                f'<span class="status-badge">{response_mode}</span>',
                unsafe_allow_html=True,
            )

    st.write("")

    if not validation.get("supported", False):
        with st.container(border=True):
            st.warning(
                "The available documentation does not contain sufficient evidence."
            )

    if safety.get("required") and safety.get("message"):
        with st.container(border=True):
            st.markdown("#### Safety notice")
            st.warning(safety["message"])

    with st.container(border=True):
        st.subheader("Answer")
        raw_answer = response.get("answer") or "No answer was returned."
        clean_answer = strip_duplicate_safety(
            raw_answer,
            safety.get("message", ""),
        )
        st.markdown(safe_markdown_text(clean_answer))

    st.write("")
    render_sources(response.get("citations", []))


def main():
    st.set_page_config(
        page_title="GRBL CNC AI",
        page_icon="G",
        layout="centered",
    )

    apply_styles()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    heading, brand = st.columns([3, 1], vertical_alignment="bottom")

    with heading:
        st.title("GRBL CNC AI")
        st.markdown("#### Technical Support Assistant")
        st.caption("Grounded answers from GRBL documentation.")

    with brand:
        st.caption("TechSolAi")

    st.write("")

    try:
        assistant = get_assistant()
    except Exception:
        st.error(
            "The GRBL assistant could not be initialized. "
            "Please try again later."
        )
        return

    render_sidebar(assistant)

    selected = st.selectbox(
        "Example questions",
        ("Select an example...", *EXAMPLES),
    )

    if selected in EXAMPLES and st.button("Use example"):
        st.session_state["question_input"] = selected

    question = st.text_input(
        "Question",
        key="question_input",
        placeholder="Ask about a GRBL setting, command, or machine issue...",
    )

    ask_col, clear_col = st.columns([3, 1])

    with ask_col:
        ask_clicked = st.button(
            "Ask",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear history",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state["history"] = []
        st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a GRBL question first.")
        else:
            query_type = assistant.query_analyzer.analyze(question)["query_type"]

            loading_text = (
                "Generating a grounded troubleshooting answer..."
                if query_type == "troubleshooting"
                else "Searching the GRBL documentation..."
            )

            try:
                with st.spinner(loading_text):
                    response = assistant.ask(question)

                new_item = {
                    "question": question.strip(),
                    "response": response,
                }

                history = st.session_state["history"]
                if not history or history[0].get("question") != new_item["question"]:
                    history.insert(0, new_item)
                else:
                    # Refresh the latest answer without adding a duplicate row.
                    history[0] = new_item

                st.rerun()

            except RuntimeError:
                st.error(
                    "The AI service is temporarily unavailable. "
                    "Please try again in a moment."
                )

            except Exception:
                st.error(
                    "The assistant could not complete this request. "
                    "Please try again."
                )

    history = st.session_state["history"]

    if history:
        st.markdown("---")
        st.subheader("Latest Answer")

        latest = history[0]
        st.markdown(
            f"**Question:** {safe_markdown_text(latest['question'])}"
        )
        render_response(latest["response"])

        if len(history) > 1:
            st.markdown("---")
            st.subheader("History")

            for index, item in enumerate(history[1:], start=1):
                history_label = safe_expander_label(
                    f"{index}. {item['question']}"
                )
                with st.expander(
                    history_label,
                    expanded=False,
                ):
                    render_response(item["response"])

    else:
        st.caption(
            "No questions asked yet. The latest answer will appear here, "
            "with earlier questions kept in History."
        )


if __name__ == "__main__":
    main()
