from pathlib import Path
import json
import hashlib
import re
import unicodedata

from metadata_utils import KNOWN_SOURCE_URLS


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_DIR = BASE_DIR / "data" / "metadata"

# ============================================================
# DOCUMENT CONFIGURATION
# ============================================================

DOCUMENT_CONFIG = {
    "grbl": {
        "controller": "GRBL",
        "version": "1.1",
        "authority": "official",
        "source": "GRBL Official Documentation",
    },

    "gcode": {
        "controller": "GRBL",
        "version": "1.1",
        "authority": "official-derived",
        "source": "GRBL G-code Reference",
    },
}


# ============================================================
# TEXT CLEANING
# ============================================================

def repair_encoding(text: str) -> str:

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â·": "·",
        "Â°": "°",
        "Â±": "±",
        "Â": "",
        "\u00a0": " ",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def normalize_text(text: str) -> str:

    text = repair_encoding(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


# ============================================================
# MARKDOWN SECTIONS
# ============================================================

def detect_sections(text: str):

    lines = text.splitlines()

    sections = []

    current_title = "Introduction"
    current_level = 0
    current_lines = []

    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    for line in lines:

        match = heading_pattern.match(line)

        if match:

            if current_lines:

                content = "\n".join(
                    current_lines
                ).strip()

                if content:

                    sections.append(
                        {
                            "title": current_title,
                            "level": current_level,
                            "content": content,
                        }
                    )

            current_title = match.group(2).strip()

            current_level = len(
                match.group(1)
            )

            current_lines = []

        else:

            current_lines.append(line)

    if current_lines:

        content = "\n".join(
            current_lines
        ).strip()

        if content:

            sections.append(
                {
                    "title": current_title,
                    "level": current_level,
                    "content": content,
                }
            )

    return sections


# ============================================================
# SECTION HIERARCHY
# ============================================================

def build_section_context(sections):

    hierarchy = {}

    enriched = []

    for section in sections:

        level = section["level"]

        hierarchy[level] = section["title"]

        for old_level in list(
            hierarchy.keys()
        ):

            if old_level > level:

                del hierarchy[old_level]

        context_parts = []

        for current_level in sorted(
            hierarchy.keys()
        ):

            context_parts.append(
                hierarchy[current_level]
            )

        enriched.append(
            {
                **section,
                "section_path":
                    " > ".join(
                        context_parts
                    ),
            }
        )

    return enriched


# ============================================================
# CHUNKING
# ============================================================

def split_large_text(
    text,
    max_chars=1800
):

    if len(text) <= max_chars:

        return [text]

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:

            continue

        candidate = (
            paragraph
            if not current
            else current
            + "\n\n"
            + paragraph
        )

        if len(candidate) <= max_chars:

            current = candidate

        else:

            if current:

                chunks.append(
                    current
                )

            if len(paragraph) > max_chars:

                lines = paragraph.splitlines()

                current = ""

                for line in lines:

                    candidate = (
                        line
                        if not current
                        else current
                        + "\n"
                        + line
                    )

                    if len(candidate) <= max_chars:

                        current = candidate

                    else:

                        if current:

                            chunks.append(
                                current
                            )

                        current = line

                if current:

                    chunks.append(
                        current
                    )

                    current = ""

            else:

                current = paragraph

    if current:

        chunks.append(
            current
        )

    return chunks


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(
    title,
    section_path,
    content
):

    title_lower = title.lower()

    path_lower = section_path.lower()

    content_lower = content.lower()

    # Explicit G-code command
    if re.match(
        r"^(g\d+|m\d+|[fs])\b",
        title_lower
    ):

        return "gcode"

    # Explicit command sections
    if (
        "commands" in title_lower
        or "realtime commands" in title_lower
        or "'$' commands" in title_lower
        or "system commands" in title_lower
    ):

        return "commands"

    # Settings
    if (
        "settings" in title_lower
        or re.search(
            r"\$\d+",
            title_lower
        )
    ):

        return "settings"

    # Alarm
    if "alarm" in title_lower:

        return "alarms"

    # Errors
    if "error" in title_lower:

        return "errors"

    # Homing
    if "homing" in title_lower:

        return "homing"

    # Limits
    if "limit" in title_lower:

        return "limits"

    # Spindle
    if "spindle" in title_lower:

        return "spindle"

    # Coolant
    if "coolant" in title_lower:

        return "coolant"

    # G-code
    if (
        "g-code" in title_lower
        or "gcode" in title_lower
    ):

        return "gcode"

    # Coordinates
    if "coordinate" in title_lower:

        return "coordinates"

    # Probing
    if (
        "probe" in title_lower
        or "probing" in title_lower
    ):

        return "probing"

    # Motors
    if (
        "motor" in title_lower
        or "stepper" in title_lower
    ):

        return "motors"

    # Parent context
    if "alarm" in path_lower:

        return "alarms"

    if "homing" in path_lower:

        return "homing"

    if "limit" in path_lower:

        return "limits"

    if "spindle" in path_lower:

        return "spindle"

    if "g-code" in path_lower:

        return "gcode"

    # Strong content signals
    if (
        content_lower.count("alarm")
        >= 4
        and content_lower.count(
            "g-code"
        ) < 3
    ):

        return "alarms"

    if (
        content_lower.count("homing")
        >= 5
        and "command" not in title_lower
    ):

        return "homing"

    if (
        content_lower.count("spindle")
        >= 5
        and "command" not in title_lower
    ):

        return "spindle"

    return "general"


# ============================================================
# GRBL PARAMETER DETECTION
# ============================================================

def detect_grbl_parameters(text):

    parameters = sorted(
        set(
            re.findall(
                r"\$(\d{1,3})",
                text
            )
        ),
        key=lambda value: int(value)
    )

    return [
        f"${parameter}"
        for parameter in parameters
    ]


# ============================================================
# G-CODE COMMAND DETECTION
# ============================================================

def detect_gcode_commands(text):

    commands = re.findall(
        r"\b([GM]\d+(?:\.\d+)?)\b",
        text.upper()
    )

    return sorted(
        set(commands)
    )


# ============================================================
# ALARM DETECTION
# ============================================================

def detect_alarms(text):

    matches = re.findall(
        r"\b(?:alarm|ALARM)\s*:?\s*(\d+)",
        text
    )

    return sorted(
        set(matches),
        key=lambda value: int(value)
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(
    file_path,
    document_type
):

    print("=" * 80)

    print(
        f"Processing: "
        f"{file_path.relative_to(RAW_DIR)}"
    )

    print("=" * 80)

    raw_text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    cleaned_text = normalize_text(
        raw_text
    )

    source_checksum = hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()

    sections = detect_sections(
        cleaned_text
    )

    sections = build_section_context(
        sections
    )

    config = DOCUMENT_CONFIG[
        document_type
    ]

    chunks = []

    chunk_counter = 0

    for section in sections:

        section_chunks = split_large_text(
            section["content"]
        )

        for part_index, content in enumerate(
            section_chunks
        ):

            if not content.strip():

                continue

            chunk_counter += 1

            title = section["title"]

            section_path = section[
                "section_path"
            ]

            category = detect_category(
                title,
                section_path,
                content
            )

            parameters = (
                detect_grbl_parameters(
                    content
                )
            )

            gcode_commands = (
                detect_gcode_commands(
                    content
                )
            )

            alarms = detect_alarms(
                content
            )

            chunks.append(
                {
                    "chunk_id": (
                        f"{file_path.stem}_"
                        f"{chunk_counter:04d}"
                    ),

                    "source": config[
                        "source"
                    ],

                    "source_url": KNOWN_SOURCE_URLS.get(file_path.name),

                    "source_document": file_path.name,

                    "source_section": section_path,

                    "source_type": config["authority"],

                    "retrieved_at": None,

                    "upstream_revision": None,

                    "checksum": source_checksum,

                    "document":
                        file_path.name,

                    "document_type":
                        document_type,

                    "controller":
                        config[
                            "controller"
                        ],

                    "version":
                        config[
                            "version"
                        ],

                    "authority":
                        config[
                            "authority"
                        ],

                    "category":
                        category,

                    "section":
                        title,

                    "section_path":
                        section_path,

                    "chunk_index":
                        part_index,

                    "text":
                        content,

                    "grbl_parameters":
                        parameters,

                    "gcode_commands":
                        gcode_commands,

                    "alarms":
                        alarms,
                }
            )

    print(
        f"Raw characters:     "
        f"{len(raw_text):,}"
    )

    print(
        f"Clean characters:   "
        f"{len(cleaned_text):,}"
    )

    print(
        f"Sections detected:  "
        f"{len(sections):,}"
    )

    print(
        f"Chunks generated:   "
        f"{len(chunks):,}"
    )

    return chunks


# ============================================================
# SAVE
# ============================================================

def save_document_chunks(
    file_path,
    chunks
):

    relative_path = (
        file_path.relative_to(
            RAW_DIR
        )
    )

    output_name = (
        f"{relative_path.stem}_"
        f"{relative_path.parent.name}_"
        f"chunks.json"
    )

    output_path = (
        PROCESSED_DIR
        / output_name
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved chunks: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("GRBL CNC AI - UNIFIED DOCUMENT PROCESSOR")
    print("=" * 80)
    print()

    all_chunks = []

    document_counts = {}

    # --------------------------------------------------------
    # Process GRBL documents
    # --------------------------------------------------------

    grbl_dir = RAW_DIR / "grbl"

    if grbl_dir.exists():

        for file_path in sorted(
            grbl_dir.glob("*.md")
        ):

            chunks = process_document(
                file_path,
                "grbl"
            )

            save_document_chunks(
                file_path,
                chunks
            )

            all_chunks.extend(
                chunks
            )

            document_counts[
                file_path.name
            ] = len(chunks)

            print()

    # --------------------------------------------------------
    # Process G-code documents
    # --------------------------------------------------------

    gcode_dir = RAW_DIR / "gcode"

    if gcode_dir.exists():

        for file_path in sorted(
            gcode_dir.glob("*.md")
        ):

            chunks = process_document(
                file_path,
                "gcode"
            )

            save_document_chunks(
                file_path,
                chunks
            )

            all_chunks.extend(
                chunks
            )

            document_counts[
                file_path.name
            ] = len(chunks)

            print()

    # --------------------------------------------------------
    # Combined corpus
    # --------------------------------------------------------

    combined_path = (
        PROCESSED_DIR
        / "grbl_corpus.json"
    )

    with combined_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    categories = {}

    sources = {}

    for chunk in all_chunks:

        category = chunk[
            "category"
        ]

        source = chunk[
            "source"
        ]

        categories[
            category
        ] = (
            categories.get(
                category,
                0
            ) + 1
        )

        sources[
            source
        ] = (
            sources.get(
                source,
                0
            ) + 1
        )

    statistics = {

        "documents":
            len(document_counts),

        "chunks":
            len(all_chunks),

        "documents_detail":
            document_counts,

        "categories":
            categories,

        "sources":
            sources,
    }

    statistics_path = (
        METADATA_DIR
        / "dataset_statistics.json"
    )

    with statistics_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"Documents: "
        f"{len(document_counts)}"
    )

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    print()
    print("Documents:")

    for name, count in sorted(
        document_counts.items()
    ):

        print(
            f"  {name:<35} "
            f"{count}"
        )

    print()
    print("Categories:")

    for category, count in sorted(
        categories.items()
    ):

        print(
            f"  {category:<20} "
            f"{count}"
        )

    print()
    print("Sources:")

    for source, count in sorted(
        sources.items()
    ):

        print(
            f"  {source:<35} "
            f"{count}"
        )

    print()
    print(
        f"Combined corpus:"
    )

    print(
        combined_path
    )

    print()
    print(
        f"Statistics:"
    )

    print(
        statistics_path
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
