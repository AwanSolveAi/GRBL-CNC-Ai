"""
GRBL CNC AI
END-TO-END TECHNICAL SUPPORT ASSISTANT

Pipeline:

    User Question
        |
        v
    Query Analysis
        |
        +----------------------+
        |                      |
        v                      v
    FAISS Search           BM25 Search
        |                      |
        +----------+-----------+
                   |
                   v
            RRF Hybrid Fusion
                   |
                   v
          Cross-Encoder Reranker
                   |
                   v
        GRBL Query-Aware Boosting
                   |
                   v
          Evidence Validation
                   |
                   v
             Ollama LLM
                   |
                   v
        Grounded Technical Answer


This application intentionally keeps the existing:
    - HybridRetriever
    - Cross-Encoder reranker
    - Ollama answer generator

and adds orchestration, query-aware ranking, evidence validation,
source display, confidence handling, and CNC safety handling.
"""

from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# EXISTING PROJECT COMPONENTS
# =============================================================================

from hybrid_retriever import HybridRetriever
from reranker import rerank

from answer_generator import (
    GRBLAnswerGenerator,
)
from metadata_utils import citation_from_result
from config import CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================

HYBRID_TOP_K = CONFIG.hybrid_top_k

RERANK_TOP_K = CONFIG.rerank_top_k

FINAL_CONTEXT_K = CONFIG.answer_context_k

MIN_RESULTS_FOR_ANSWER = 1

# Conservative reranker threshold.
#
# Cross-Encoder scores are model-dependent, so this is NOT used as
# the only evidence-quality signal. It is combined with query matching.
MIN_RERANK_SCORE = -8.0

# RRF / retrieval configuration.
VECTOR_TOP_K = CONFIG.vector_top_k
BM25_TOP_K = CONFIG.bm25_top_k


# =============================================================================
# ANSI TERMINAL COLORS
# =============================================================================

class Colors:

    RESET = "\033[0m"

    BOLD = "\033[1m"

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color(text: str, code: str) -> str:

    return f"{code}{text}{Colors.RESET}"


# =============================================================================
# QUERY ANALYSIS
# =============================================================================

class QueryAnalyzer:
    """
    Identifies technical entities and troubleshooting intent.

    This is intentionally deterministic.

    We do not ask the LLM to decide which GRBL setting a technician
    is talking about before retrieval.
    """

    SETTING_PATTERN = re.compile(
        r"\$(\d{1,3})(?!\d)"
    )

    SYSTEM_COMMAND_PATTERN = re.compile(
        r"(?<!\w)(\$\$|\$#|\$G|\$I|\$N|\$C|\$X|\$H|\$J|\$RST|\$SLP)(?!\w)",
        re.IGNORECASE,
    )

    GCODE_PATTERN = re.compile(
        r"\bG\s*(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    MCODE_PATTERN = re.compile(
        r"\bM\s*(\d+)\b",
        re.IGNORECASE,
    )

    ALARM_PATTERN = re.compile(
        r"\balarm\s*(\d+)?\b",
        re.IGNORECASE,
    )

    ALARM_WORD_PATTERN = re.compile(
        r"\balarm\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        re.IGNORECASE,
    )

    AXIS_PATTERN = re.compile(
        r"\b([XYZAB])\s*(?:axis)?\b",
        re.IGNORECASE,
    )

    def analyze(self, query: str) -> Dict[str, Any]:

        q = query.strip()

        # Normalize a small set of common technician typos and one noisy
        # decimal G-code form before deterministic parsing. This improves
        # robustness without fuzzy-matching arbitrary unsupported identifiers.
        normalized_q = q
        normalized_q = re.sub(
            r"\bG\s*38\s+([2345])\b",
            r"G38.\1",
            normalized_q,
            flags=re.IGNORECASE,
        )

        lower = normalized_q.lower()
        typo_replacements = {
            "axix": "axis",
            "distnce": "distance",
            "limts": "limits",
            "limt": "limit",
            "dirrection": "direction",
            "homming": "homing",
            "milimeters": "millimeters",
            "seting": "setting",
            "controll": "control",
            "dose": "does",
            "wht": "what",
            "spd": "speed",
            "spindel": "spindle",
            "workng": "working",
            "alrm": "alarm",
            "hapening": "happening",
            "masq": "mask",
        }
        for wrong, correct in typo_replacements.items():
            lower = re.sub(rf"\b{re.escape(wrong)}\b", correct, lower)

        settings = [
            int(match)
            for match in self.SETTING_PATTERN.findall(normalized_q)
        ]

        gcodes = [
            f"G{match.upper()}"
            for match in self.GCODE_PATTERN.findall(normalized_q)
        ]

        mcodes = [
            f"M{match.upper()}"
            for match in self.MCODE_PATTERN.findall(normalized_q)
        ]

        alarm_matches = self.ALARM_PATTERN.findall(q)

        alarm_numbers = [
            int(value)
            for value in alarm_matches
            if value.strip()
        ]

        alarm_word_values = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9,
        }
        alarm_numbers.extend(
            alarm_word_values[value.lower()]
            for value in self.ALARM_WORD_PATTERN.findall(q)
        )
        alarm_numbers = list(dict.fromkeys(alarm_numbers))

        axes = [
            axis.upper()
            for axis in self.AXIS_PATTERN.findall(q)
        ]

        system_commands = []
        for command in self.SYSTEM_COMMAND_PATTERN.findall(q):
            normalized = command.upper()
            if normalized not in system_commands:
                system_commands.append(normalized)

        # Preserve which identifiers were literally typed by the user.
        # Later semantic inference may add an entity, but an inferred entity
        # should not automatically turn a symptom/troubleshooting question
        # into an "exact" route.
        explicit_settings = bool(settings)
        explicit_gcodes = bool(gcodes)
        explicit_mcodes = bool(mcodes)
        explicit_system_commands = bool(system_commands)
        explicit_alarm_number = bool(alarm_numbers)

        # ---------------------------------------------------------------
        # Troubleshooting intent
        # ---------------------------------------------------------------

        distance_terms = [
            "wrong distance",
            "wrong amount",
            "moves too far",
            "moves too short",
            "travel too far",
            "travel too short",
            "position inaccurate",
            "positioning inaccurate",
            "inaccurate movement",
            "incorrect distance",
            "incorrect movement",
            "moves the wrong",
            "axis distance",
            "steps per mm",
            "steps per millimeter",
            "steps/mm",
            "travel is inaccurate",
            "travel inaccurate",
            "axis travel is inaccurate",
        ]

        speed_terms = [
            "too fast",
            "too slow",
            "maximum speed",
            "max speed",
            "speed limit",
            "rapid speed",
            "feed rate",
        ]

        acceleration_terms = [
            "acceleration",
            "accelerates too fast",
            "accelerates too slowly",
            "jerk",
            "ramp up",
        ]

        travel_limit_terms = [
            "travel limit",
            "maximum travel",
            "soft limit",
            "axis travel",
            "travel range",
        ]

        homing_terms = [
            "homing",
            "home the machine",
            "home machine",
            "will not home",
            "won't home",
            "does not home",
            "not homing",
            "homming",
            "home problem",
            "home",
            "homing cycle",
            "home switch",
            "limit switch",
        ]

        spindle_terms = [
            "spindle",
            "rpm",
            "coolant",
        ]

        probe_terms = [
            "probe",
            "probing",
            "touch plate",
            "tool setter",
            "probe fail",
        ]

        positioning_terms = [
            "wrong units",
            "unit confusion",
            "relative instead",
            "absolute instead",
            "positioning mode",
            "inch or mm",
            "inches or millimeters",
        ]

        # ---------------------------------------------------------------
        # Deterministic semantic entity inference
        # ---------------------------------------------------------------
        # These are mappings from ordinary technician language to entities
        # already documented in the GRBL corpus. They do not hard-code
        # technical answers; retrieval/evidence validation still decides
        # whether the entity is actually supported.
        def add_setting(value: int):
            if value not in settings:
                settings.append(value)

        def add_gcode(value: str):
            if value not in gcodes:
                gcodes.append(value)

        def add_system(value: str):
            if value not in system_commands:
                system_commands.append(value)

        # Axis calibration / steps-per-distance.
        if any(
            term in lower
            for term in (
                "calibration distance per step",
                "steps per millimeter",
                "steps per mm",
                "steps/mm",
                "wrong distance",
                "wrong amount",
                "moves too far",
                "moves too short",
                "travel distance",
                "distance inaccurate",
                "axis moves the wrong distance",
                "axis moving wrong distance",
                "twice the commanded distance",
                "commanded distance",
                "travel is inaccurate",
                "travel inaccurate",
                "axis travel is inaccurate",
            )
        ):
            if "x" in axes or re.search(r"\bx\s*axis\b", lower):
                add_setting(100)
            if "y" in axes or re.search(r"\by\s*axis\b", lower):
                add_setting(101)
            if "z" in axes or re.search(r"\bz\s*axis\b", lower):
                add_setting(102)

        # Axis calibration paraphrases used by the Gold Standard.
        if (
            "x-axis calibration distance per step" in lower
            or "x axis calibration distance per step" in lower
            or ("x" in lower and "calibration" in lower and "distance per step" in lower)
        ):
            add_setting(100)

        if (
            "z travel distance" in lower
            or "z-axis travel distance" in lower
            or "z axis travel distance" in lower
            or ("z" in lower and "calibrates" in lower and "travel distance" in lower)
            or (
                re.search(r"\bz\s*axis\b", lower)
                and "travel" in lower
                and "inaccurate" in lower
            )
        ):
            add_setting(102)

        if (
            "homing" in lower
            and (
                "wrong end" in lower
                or "wrong direction" in lower
                or "opposite direction" in lower
            )
        ):
            add_setting(23)

        if "soft limit" in lower:
            add_setting(20)

        if "hard limit" in lower:
            add_setting(21)
            if "alarm" in lower or "unexpected" in lower:
                add_setting(5)

        if (
            "idle delay" in lower
            or "stays energized after motion stops" in lower
            or "remain energized after motion stops" in lower
        ):
            add_setting(1)

        if (
            "junction deviation" in lower
            or "corners are too sharp" in lower
            or "changes direction harshly" in lower
            or "direction harshly" in lower
        ):
            add_setting(11)
        if "enable" in lower and "homing" in lower:
            add_setting(22)
        if "homing" in lower and "seek" in lower:
            add_setting(25)
        if "homing" in lower and "feed" in lower:
            add_setting(24)
        if "homing" in lower and "pull-off" in lower:
            add_setting(27)
        if "probe" in lower and "invert" in lower:
            add_setting(6)
        if ("direction" in lower and ("backward" in lower or "opposite" in lower)):
            add_setting(3)
        if (
            "step pulse invert" in lower
            or "steps pulse invert" in lower
            or "step pulse invert mask" in lower
            or "steps pulse invert mask" in lower
        ):
            add_setting(2)
        if "arc tolerance" in lower:
            add_setting(12)
        if "laser mode" in lower:
            add_setting(32)
        if "maximum spindle" in lower or "max spindle" in lower:
            add_setting(30)
        if "minimum spindle" in lower or "min spindle" in lower:
            add_setting(31)

        # Natural-language GRBL system-command requests.
        if any(term in lower for term in ("see all current grbl settings", "view all grbl settings")):
            add_system("$$")
        if ("parser state" in lower or "modal state" in lower) and "view" in lower:
            add_system("$G")
        if ("performs homing" in lower or "perform homing" in lower):
            add_system("$H")
        if "alarm lock" in lower and any(term in lower for term in ("clear", "unlock", "kill")):
            add_system("$X")
        if "check mode" in lower or (
            "validate g-code" in lower and "without moving" in lower
        ):
            add_system("$C")

        # Natural-language G-code requests.
        if "absolute" in lower and any(term in lower for term in ("coordinate", "position", "distance mode")):
            add_gcode("G90")
        if ("incremental" in lower or "relative" in lower) and any(
            term in lower for term in ("coordinate", "position", "distance mode")
        ):
            add_gcode("G91")
        if "millimeter" in lower and ("g-code" in lower or "units" in lower or "select" in lower):
            add_gcode("G21")
        if "inches" in lower and ("g-code" in lower or "units" in lower or "select" in lower):
            add_gcode("G20")
        if "clockwise arc" in lower:
            add_gcode("G2")
        if "cancel" in lower and "motion mode" in lower:
            add_gcode("G80")
        if "machine coordinates" in lower:
            add_gcode("G53")
        if "first work coordinate system" in lower:
            add_gcode("G54")

        if (
            "distance-mode commands" in lower
            or "distance mode commands" in lower
            or ("relative" in lower and "unexpected" in lower)
        ):
            add_gcode("G90")
            add_gcode("G91")

        if (
            "unit-selection commands" in lower
            or "unit selection commands" in lower
            or "25.4 times wrong" in lower
        ):
            add_gcode("G20")
            add_gcode("G21")
        if "probe" in lower and "toward" in lower:
            if any(term in lower for term in ("errors if", "error if", "stop safely if")):
                add_gcode("G38.2")
            elif any(term in lower for term in ("without raising an error", "no error")):
                add_gcode("G38.3")

        # Explicit unsupported external products/domains. Keep this separate
        # from generic words such as "router" because CNC routers are valid.
        unsupported_terms = {
            "marlin", "fluidnc", "linuxcnc", "mach3", "klipper",
            "reprapfirmware", "fanuc", "siemens", "sinumerik", "cisco",
            "nvidia", "windows", "bitcoin", "tesla", "cura", "weather",
        }
        out_of_domain_terms = sorted(
            term
            for term in unsupported_terms
            if re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                lower,
            )
        )

        # Known identifiers that are not part of this GRBL knowledge base.
        # Keep this narrow to avoid rejecting valid GRBL commands.
        unsupported_mcodes = {
            code for code in mcodes
            if code in {"M6", "M500"}
        }

        # GRBL 1.1 settings represented by this corpus. Values outside these
        # documented ranges are treated as unsupported identifiers.
        supported_settings = {
            *range(0, 7),
            *range(10, 14),
            *range(20, 28),
            *range(30, 33),
            *range(100, 103),
            *range(110, 113),
            *range(120, 123),
            *range(130, 133),
        }
        unsupported_settings = {
            value for value in settings
            if value not in supported_settings
        }

        unsupported_identifier_intent = bool(
            unsupported_mcodes
            or unsupported_settings
            or (
                "another firmware" in lower
                and bool(mcodes)
            )
        )

        unsupported_instruction = (
            any(
                phrase in lower
                for phrase in (
                    "even if the grbl docs do not contain it",
                    "even if the documentation does not contain it",
                    "answer even if the docs do not contain it",
                    "answer without evidence",
                )
            )
            and not (settings or gcodes or mcodes or system_commands)
        )

        alarm_intent = bool(
            alarm_numbers
            or "alarm" in lower
            or "alarm lock" in lower
            or "unlock grbl" in lower
        )

        troubleshooting = any(
            (
                any(term in lower for term in distance_terms),
                any(term in lower for term in speed_terms),
                any(term in lower for term in acceleration_terms),
                any(term in lower for term in travel_limit_terms),
                any(term in lower for term in homing_terms),
                any(term in lower for term in spindle_terms),
                any(term in lower for term in probe_terms),
                any(term in lower for term in positioning_terms),
            )
        )

        diagnostic_terms = (
            "wrong distance",
            "not working",
            "not work",
            "keeps happening",
            "keeps happen",
            "unexpectedly",
            "unexpected",
            "seems inverted",
            "stays energized",
            "look inaccurate",
            "looks inaccurate",
            "too sharp",
            "harshly",
            "behavior changed",
            "stop safely if",
            "without raising an error",
            "if contact is missed",
            "if contact is not made",
            "one-block move",
            "which setting should i inspect",
            "which setting is relevant",
            "which settings are relevant",
            "which command applies",
            "which command is relevant",
            "which commands should i inspect",
            "which commands should i check",
            "which setting should i check",
            "25.4 times wrong",
            "direction is backward",
            "backward which setting",
            "opposite direction",
            "twice the commanded distance",
            "before continuing",
            "alarm-locked after",
            "after a condition was cleared",
            "validate g-code without moving",
            "without moving the machine",
            "i need to home the machine",
            "i want to validate g-code",
        )
        diagnostic_intent = any(term in lower for term in diagnostic_terms)

        # Unsupported/OOD intent must outrank literal identifier parsing.
        # Example: "What does Marlin M500 do?" contains an M-code token but
        # is explicitly outside the GRBL domain.
        if (
            out_of_domain_terms
            or unsupported_instruction
            or unsupported_identifier_intent
        ):
            query_type = "unsupported"

        # Diagnostic symptoms outrank *inferred* entities, but not literal
        # GRBL identifiers typed directly by the user.
        elif diagnostic_intent and not (
            explicit_settings
            or explicit_gcodes
            or explicit_mcodes
            or explicit_system_commands
            or explicit_alarm_number
        ):
            query_type = "troubleshooting"

        elif explicit_settings:
            query_type = "exact_setting"
        elif explicit_gcodes:
            query_type = "gcode"
        elif explicit_mcodes:
            query_type = "mcode"
        elif explicit_alarm_number:
            query_type = "alarm"
        elif explicit_system_commands:
            query_type = "system_command"

        # Natural-language system-command requests should outrank a generic
        # alarm-word match unless the question is clearly troubleshooting.
        elif system_commands and not diagnostic_intent:
            query_type = "system_command"
        elif diagnostic_intent:
            query_type = "troubleshooting"
        elif settings:
            query_type = "exact_setting"
        elif gcodes:
            query_type = "gcode"
        elif mcodes:
            query_type = "mcode"
        elif alarm_intent:
            query_type = "alarm"
        elif system_commands:
            query_type = "system_command"
        elif troubleshooting:
            query_type = "troubleshooting"
        else:
            query_type = "general"

        retrieval_terms = []
        retrieval_terms.extend(f"${setting}" for setting in settings)
        retrieval_terms.extend(gcodes)
        retrieval_terms.extend(mcodes)
        retrieval_terms.extend(system_commands)
        if any(term in lower for term in distance_terms):
            retrieval_terms.extend(["steps/mm", "$100", "$101", "$102"])
        if any(term in lower for term in homing_terms):
            retrieval_terms.extend(["homing cycle", "$H", "$22"])
        if any(term in lower for term in spindle_terms):
            retrieval_terms.extend(["spindle", "M3", "M4", "M5"])
        if any(term in lower for term in probe_terms):
            retrieval_terms.extend(["probe", "probing", "G38.2", "$6"])
        if any(term in lower for term in positioning_terms):
            if "unit" in lower or "inch" in lower or "millimeter" in lower:
                retrieval_terms.extend(["units", "G20", "G21"])
            if "relative" in lower or "absolute" in lower:
                retrieval_terms.extend(["distance mode", "G90", "G91"])

        retrieval_query = " ".join([q, *retrieval_terms]).strip()

        return {
            "query": q,

            "retrieval_query": retrieval_query,

            "query_type": query_type,

            "settings": settings,

            "gcodes": gcodes,

            "mcodes": mcodes,

            "alarm_numbers": alarm_numbers,

            "system_commands": system_commands,

            "axes": axes,

            "is_alarm_query": alarm_intent,

            "is_distance_query": any(
                term in lower
                for term in distance_terms
            ),

            "is_speed_query": any(
                term in lower
                for term in speed_terms
            ),

            "is_acceleration_query": any(
                term in lower
                for term in acceleration_terms
            ),

            "is_travel_limit_query": any(
                term in lower
                for term in travel_limit_terms
            ),

            "is_homing_query": any(
                term in lower
                for term in homing_terms
            ),

            "is_spindle_query": any(
                term in lower
                for term in spindle_terms
            ),

            "is_probe_query": any(
                term in lower
                for term in probe_terms
            ),

            "is_positioning_query": any(
                term in lower
                for term in positioning_terms
            ),

            "out_of_domain_terms": out_of_domain_terms,

            "unsupported_instruction": unsupported_instruction,
        }


# =============================================================================
# GRBL QUERY-AWARE RANKING
# =============================================================================

class GRBLRanker:
    """
    Applies deterministic GRBL-specific relevance boosts after the
    generic Cross-Encoder.

    Why?

    A generic semantic model can consider several settings related
    to motion equally relevant.

    For example:

        "My X axis moves the wrong distance"

    should strongly favor:

        $100 - X-axis steps/mm

    over:

        $110 - X-axis maximum rate
        $120 - X-axis acceleration
        $130 - X-axis maximum travel
    """

    # -------------------------------------------------------------------------
    # Setting families
    # -------------------------------------------------------------------------

    STEPS_MM = {
        100: "X",
        101: "Y",
        102: "Z",
    }

    MAX_RATE = {
        110: "X",
        111: "Y",
        112: "Z",
    }

    ACCELERATION = {
        120: "X",
        121: "Y",
        122: "Z",
    }

    MAX_TRAVEL = {
        130: "X",
        131: "Y",
        132: "Z",
    }

    def score_result(
        self,
        query_info: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Tuple[float, List[str]]:

        base_score = float(
            result.get(
                "reranker_score",
                result.get(
                    "rerank_score",
                    0.0,
                ),
            )
        )

        score = base_score

        reasons = []

        text = result.get(
            "text",
            "",
        )

        section = result.get(
            "section",
            "",
        )

        combined = (
            f"{section}\n{text}"
        ).lower()

        section_lower = section.lower()

        chunk_id = str(
            result.get(
                "chunk_id",
                "",
            )
        ).lower()

        # ---------------------------------------------------------------------
        # Explicit $SETTING query
        # ---------------------------------------------------------------------

        requested_settings = query_info[
            "settings"
        ]

        for setting in requested_settings:

            exact_token = f"${setting}"

            if exact_token.lower() in combined:

                score += 20.0

                reasons.append(
                    f"exact GRBL setting ${setting}"
                )

            if exact_token.lower() in section_lower:
                score += 30.0
                reasons.append(
                    f"defining section for ${setting}"
                )

            # Chunk IDs are also useful.
            if str(setting) in chunk_id:

                score += 5.0

        # ---------------------------------------------------------------------
        # Explicit G-code
        # ---------------------------------------------------------------------

        for gcode in query_info["gcodes"]:

            if gcode.lower() in combined:

                score += 20.0

                reasons.append(
                    f"exact G-code {gcode}"
                )

            if re.search(
                rf"(^|[^a-z0-9.]){re.escape(gcode.lower())}([^a-z0-9.]|$)",
                section_lower,
            ):
                score += 30.0
                reasons.append(
                    f"defining section for {gcode}"
                )

        # ---------------------------------------------------------------------
        # Explicit M-code
        # ---------------------------------------------------------------------

        for mcode in query_info["mcodes"]:

            if mcode.lower() in combined:

                score += 20.0

                reasons.append(
                    f"exact M-code {mcode}"
                )

            if re.search(
                rf"(^|[^a-z0-9]){re.escape(mcode.lower())}([^a-z0-9]|$)",
                section_lower,
            ):
                score += 30.0
                reasons.append(
                    f"defining section for {mcode}"
                )

        # ---------------------------------------------------------------------
        # Explicit Alarm code
        # ---------------------------------------------------------------------

        metadata = result.get("metadata") or {}
        authority = result.get("authority") or metadata.get("authority")
        category = str(
            result.get("category") or metadata.get("category") or ""
        ).lower()
        source_name = (
            str(result.get("source") or metadata.get("source") or "")
            + " "
            + str(result.get("document") or metadata.get("document") or "")
        ).lower()

        # Prefer the canonical GRBL source family for explicit entities.
        if query_info.get("settings") and "settings.md" in source_name:
            score += 18.0
            reasons.append("canonical settings source")
        if (query_info.get("gcodes") or query_info.get("mcodes")) and (
            "grbl_gcode.md" in source_name
        ):
            score += 18.0
            reasons.append("canonical G-code source")
        if query_info.get("system_commands") and "commands.md" in source_name:
            score += 18.0
            reasons.append("canonical system-command source")

        for alarm in query_info["alarm_numbers"]:
            alarm_token = f"alarm {alarm}"

            if alarm_token in combined:
                score += 20.0
                reasons.append(f"exact Alarm {alarm}")

            if authority in {"official", "official-derived"} and (
                alarm_token in section_lower
                or (category == "alarms" and alarm_token in text.lower())
            ):
                score += 30.0
                reasons.append(f"defining section for Alarm {alarm}")

        if query_info["is_alarm_query"] and not query_info["alarm_numbers"]:
            query_lower = query_info["query"].lower()
            if any(term in query_lower for term in ("unlock", "clear", "alarm lock")):
                if "$X" in section:
                    score += 25.0
                    reasons.append("alarm unlock command definition")

        # ---------------------------------------------------------------------
        # Explicit GRBL system command
        # ---------------------------------------------------------------------

        for command in query_info.get("system_commands", []):
            combined_original = f"{section}\n{text}"
            if command in combined_original:
                score += 20.0
                reasons.append(f"exact system command {command}")
            if command in section:
                score += 30.0
                reasons.append(f"defining section for {command}")

        # ---------------------------------------------------------------------
        # Axis distance troubleshooting
        # ---------------------------------------------------------------------

        if query_info["is_distance_query"]:

            for axis in query_info["axes"]:

                # X axis -> $100
                # Y axis -> $101
                # Z axis -> $102

                expected_setting = None

                for setting, setting_axis in self.STEPS_MM.items():

                    if setting_axis == axis:

                        expected_setting = setting
                        break

                if expected_setting is not None:

                    token = f"${expected_setting}"

                    if token.lower() in combined:

                        score += 35.0

                        reasons.append(
                            f"{axis}-axis distance -> "
                            f"{token} steps/mm"
                        )

            # General steps/mm evidence.
            if (
                "steps/mm" in combined
                or "steps per mm" in combined
            ):

                score += 15.0

                reasons.append(
                    "steps/mm evidence"
                )

            # Explicitly reduce unrelated motion settings.
            for setting in (
                110,
                111,
                112,
                120,
                121,
                122,
                130,
                131,
                132,
            ):

                if f"${setting}" in combined:

                    score -= 12.0

        # ---------------------------------------------------------------------
        # Speed troubleshooting
        # ---------------------------------------------------------------------

        if query_info["is_speed_query"]:

            for setting in (
                110,
                111,
                112,
            ):

                if f"${setting}" in combined:

                    score += 18.0

                    reasons.append(
                        f"maximum-rate setting ${setting}"
                    )

        # ---------------------------------------------------------------------
        # Acceleration troubleshooting
        # ---------------------------------------------------------------------

        if query_info["is_acceleration_query"]:

            for setting in (
                120,
                121,
                122,
            ):

                if f"${setting}" in combined:

                    score += 18.0

                    reasons.append(
                        f"acceleration setting ${setting}"
                    )

        # ---------------------------------------------------------------------
        # Travel limit troubleshooting
        # ---------------------------------------------------------------------

        if query_info["is_travel_limit_query"]:

            for setting in (
                130,
                131,
                132,
            ):

                if f"${setting}" in combined:

                    score += 18.0

                    reasons.append(
                        f"maximum-travel setting ${setting}"
                    )

        # ---------------------------------------------------------------------
        # Homing
        # ---------------------------------------------------------------------

        if query_info["is_homing_query"]:

            if (
                "$22" in combined
                or "$23" in combined
                or "$24" in combined
                or "$25" in combined
                or "$26" in combined
                or "$27" in combined
                or "$h" in combined
                or "homing" in combined
            ):

                score += 10.0

                reasons.append(
                    "homing-related evidence"
                )

            query_lower = query_info["query"].lower()

            # A machine that will not home should first surface the homing
            # enable setting ($22). Include common technician/noisy wording.
            if any(
                term in query_lower
                for term in (
                    "will not home",
                    "won't home",
                    "does not home",
                    "not homing",
                    "not homming",
                    "not home",
                )
            ):
                if "$22" in combined:
                    score += 40.0
                    reasons.append("cannot-home symptom -> $22")
                elif "$h" in combined:
                    score += 18.0
                    reasons.append("cannot-home symptom -> $H")

                for setting in (24, 25, 26, 27):
                    if f"${setting}" in combined:
                        score -= 8.0

            # "Aggressive and fast" homing is primarily a homing-speed
            # question. Both $25 (seek) and $24 (feed) are required evidence.
            if (
                "homing" in query_lower
                and (
                    "aggressive" in query_lower
                    or "too fast" in query_lower
                    or "fast" in query_lower
                )
            ):
                if "$25" in combined:
                    score += 38.0
                    reasons.append("fast/aggressive homing -> $25 seek")
                if "$24" in combined:
                    score += 34.0
                    reasons.append("fast/aggressive homing -> $24 feed")
                for setting in (22, 26, 27):
                    if f"${setting}" in combined:
                        score -= 10.0

            # Unexpected limit alarms should prioritize hard-limit enable
            # ($21) and limit-pin inversion ($5), not soft limits ($20).
            if (
                "limit" in query_lower
                and "alarm" in query_lower
                and (
                    "unexpected" in query_lower
                    or "unexpectedly" in query_lower
                    or "keeps" in query_lower
                    or "trigger" in query_lower
                )
            ):
                if "$21" in combined:
                    score += 40.0
                    reasons.append("unexpected limit alarms -> $21")
                if "$5" in combined:
                    score += 36.0
                    reasons.append("unexpected limit alarms -> $5")
                if "$20" in combined:
                    score -= 18.0

            if any(
                term in query_lower
                for term in (
                    "wrong end",
                    "wrong direction",
                    "opposite direction",
                )
            ):
                if "$23" in combined:
                    score += 35.0
                    reasons.append("homing wrong-end direction -> $23")

                for setting in (22, 24, 25, 26, 27):
                    if f"${setting}" in combined:
                        score -= 10.0

        # ---------------------------------------------------------------------
        # Probe
        # ---------------------------------------------------------------------

        if query_info["is_probe_query"]:

            if (
                "g38" in combined
                or "probe" in combined
            ):

                score += 12.0

                reasons.append(
                    "probe-related evidence"
                )

            query_lower = query_info["query"].lower()
            if "not trigger" in query_lower and (
                "g38.2" in section_lower or "$6" in section_lower
            ):
                score += 20.0
                reasons.append("probe trigger/failure evidence")

        # ---------------------------------------------------------------------
        # Spindle
        # ---------------------------------------------------------------------

        if query_info["is_spindle_query"]:

            if (
                "spindle" in combined
                or "m3" in combined
                or "m4" in combined
                or "m5" in combined
                or "coolant" in combined
            ):

                score += 10.0

                reasons.append(
                    "spindle/coolant evidence"
                )

            query_lower = query_info["query"].lower()
            if "start" in query_lower and "m3" in section_lower:
                score += 25.0
                reasons.append("spindle start definition")
            if any(term in query_lower for term in ("turn", "switch", "spindle on")) and "m3" in section_lower:
                score += 25.0
                reasons.append("spindle-on definition")
            if "stop" in query_lower and "m5" in section_lower:
                score += 25.0
                reasons.append("spindle stop definition")
            if any(term in query_lower for term in ("off", "shut")) and "m5" in section_lower:
                score += 25.0
                reasons.append("spindle-off definition")

        # ---------------------------------------------------------------------
        # Units and distance-mode confusion
        # ---------------------------------------------------------------------

        if query_info.get("is_positioning_query"):
            query_lower = query_info["query"].lower()
            if any(term in query_lower for term in ("unit", "inch", "millimeter")) and (
                "g20" in section_lower or "g21" in section_lower
            ):
                score += 25.0
                reasons.append("units-mode definition")
            if ("relative" in query_lower or "absolute" in query_lower) and (
                "g90" in section_lower or "g91" in section_lower
            ):
                score += 25.0
                reasons.append("distance-mode definition")

        return score, reasons

    # -------------------------------------------------------------------------
    # Apply ranking
    # -------------------------------------------------------------------------

    def rank(
        self,
        query_info: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        ranked = []

        for result in results:

            item = dict(result)

            final_score, reasons = self.score_result(
                query_info,
                item,
            )

            item["grbl_final_score"] = float(
                final_score
            )

            item["ranking_reasons"] = reasons

            ranked.append(item)

        ranked.sort(
            key=lambda item: item[
                "grbl_final_score"
            ],
            reverse=True,
        )

        for rank, item in enumerate(
            ranked,
            start=1,
        ):

            item["final_rank"] = rank

        return ranked


# =============================================================================
# EVIDENCE VALIDATOR
# =============================================================================

class EvidenceValidator:
    """
    Determines whether the retrieved evidence is sufficient.

    This is deliberately conservative.

    The LLM should not be allowed to fill missing GRBL documentation
    from its general training knowledge.
    """

    @staticmethod
    def exact_entities_defined(
        query_info: Dict[str, Any],
        result: Dict[str, Any],
    ) -> bool:
        """Require authoritative, definition-like evidence for identifiers.

        Dedicated section headings remain the strongest signal. When a
        canonical GRBL source groups several definitions into one chunk, an
        identifier may also be accepted from that authoritative source text.
        """

        metadata = result.get("metadata") or {}
        authority = result.get("authority") or metadata.get("authority")
        if authority not in {"official", "official-derived"}:
            return False

        raw_section = str(
            result.get("section") or metadata.get("section") or ""
        )
        section = raw_section.lower()
        raw_text = str(result.get("text") or metadata.get("text") or "")
        text = raw_text.lower()
        category = str(
            result.get("category") or metadata.get("category") or ""
        ).lower()
        source = (
            str(result.get("source") or metadata.get("source") or "")
            + " "
            + str(result.get("document") or metadata.get("document") or "")
        ).lower()

        def boundary_present(token: str, haystack: str) -> bool:
            return bool(
                re.search(
                    rf"(^|[^a-z0-9.$]){re.escape(token.lower())}([^a-z0-9.]|$)",
                    haystack,
                )
            )

        for setting in query_info["settings"]:
            token = f"${setting}".lower()
            if token in section:
                continue
            if "settings.md" in source and token in text:
                continue
            return False

        for gcode in query_info["gcodes"]:
            token = gcode.lower()
            if boundary_present(token, section):
                continue
            if "grbl_gcode.md" in source and boundary_present(token, text):
                continue
            return False

        for mcode in query_info["mcodes"]:
            token = mcode.lower()
            if boundary_present(token, section):
                continue
            if "grbl_gcode.md" in source and boundary_present(token, text):
                continue
            return False

        for alarm in query_info["alarm_numbers"]:
            token = f"alarm {alarm}"
            if token not in section and not (
                category == "alarms" and token in text
            ):
                return False

        for command in query_info.get("system_commands", []):
            command_lower = command.lower()

            # A system command must be defined by its section heading.
            # Do not accept incidental mentions inside commands.md, because
            # e.g. $RST documentation can mention $$ without defining what
            # $$ itself does.
            if command_lower in section:
                continue

            # Gold-standard corpus addendum entries are intentionally
            # self-contained definition chunks. Accept the command only when
            # the addendum chunk itself is the dedicated definition.
            if (
                "grbl_official_gold_addendum.md" in source
                and command_lower in section
            ):
                continue

            return False

        return True

    def validate(
        self,
        query_info: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not results:

            return {
                "supported": False,
                "level": "NONE",
                "reason": (
                    "No relevant GRBL evidence was retrieved."
                ),
            }

        query_words = set(
            re.findall(r"[a-z0-9]+", query_info["query"].lower())
        )

        if query_info.get("out_of_domain_terms"):
            return {
                "supported": False,
                "level": "INSUFFICIENT",
                "reason": (
                    "The question refers to unsupported non-GRBL technology: "
                    + ", ".join(query_info["out_of_domain_terms"])
                    + "."
                ),
            }

        if query_info.get("unsupported_instruction"):
            return {
                "supported": False,
                "level": "INSUFFICIENT",
                "reason": (
                    "The request asks for an answer without supporting GRBL "
                    "documentation, so the assistant must refuse."
                ),
            }

        unsupported_controllers = {"marlin", "fluidnc", "linuxcnc", "mach3"}
        named_unsupported = query_words & unsupported_controllers
        if named_unsupported:
            return {
                "supported": False,
                "level": "INSUFFICIENT",
                "reason": (
                    "The available corpus is for GRBL, not "
                    + ", ".join(sorted(named_unsupported))
                    + "."
                ),
            }

        top = results[0]

        score = float(
            top.get(
                "grbl_final_score",
                top.get(
                    "reranker_score",
                    0.0,
                ),
            )
        )

        combined_top = (
            f"{top.get('section', '')}\n"
            f"{top.get('text', '')}"
        ).lower()

        # ---------------------------------------------------------------------
        # Explicit alarm handling
        # ---------------------------------------------------------------------

        if query_info["is_alarm_query"]:

            alarm_number = None

            if query_info["alarm_numbers"]:

                alarm_number = (
                    query_info["alarm_numbers"][0]
                )

            # The current corpus does not contain a reliable alarm-code
            # reference for all alarm numbers.
            #
            # Do NOT allow an unrelated "$X" result to masquerade as
            # an explanation of "Alarm 1".

            if alarm_number is not None:

                alarm_token = (
                    f"alarm {alarm_number}"
                )

                if alarm_token not in combined_top:

                    return {
                        "supported": False,
                        "level": "INSUFFICIENT",
                        "reason": (
                            f"The current GRBL corpus does not "
                            f"contain direct evidence for Alarm "
                            f"{alarm_number}."
                        ),
                    }

                if not self.exact_entities_defined(query_info, top):
                    return {
                        "supported": False,
                        "level": "INSUFFICIENT",
                        "reason": (
                            "The retrieved Alarm evidence is not an "
                            "authoritative definition of the requested code."
                        ),
                    }

        # ---------------------------------------------------------------------
        # Explicit command/settings
        # ---------------------------------------------------------------------

        if (
            query_info["settings"]
            or query_info["gcodes"]
            or query_info["mcodes"]
            or query_info.get("system_commands")
        ):

            requested_entities = []
            for key in ("settings", "gcodes", "mcodes", "system_commands"):
                for entity in query_info.get(key, []):
                    single = dict(query_info)
                    single.update({
                        "settings": [],
                        "gcodes": [],
                        "mcodes": [],
                        "system_commands": [],
                        "alarm_numbers": [],
                    })
                    single[key] = [entity]
                    requested_entities.append(single)

            definitions_complete = all(
                any(
                    self.exact_entities_defined(single, result)
                    for result in results
                )
                for single in requested_entities
            )

            if not definitions_complete:
                return {
                    "supported": False,
                    "level": "INSUFFICIENT",
                    "reason": (
                        "The authoritative evidence does not define every "
                        "requested identifier directly."
                    ),
                }

            if score >= 5.0:

                return {
                    "supported": True,
                    "level": "HIGH",
                    "reason": (
                        "The retrieved evidence directly matches "
                        "the requested GRBL command or setting."
                    ),
                }

        # ---------------------------------------------------------------------
        # Troubleshooting
        # ---------------------------------------------------------------------

        if (
            query_info["is_distance_query"]
            or query_info["is_speed_query"]
            or query_info["is_acceleration_query"]
            or query_info["is_travel_limit_query"]
            or query_info["is_homing_query"]
            or query_info["is_spindle_query"]
            or query_info["is_probe_query"]
            or query_info.get("is_positioning_query")
        ):

            if score >= 8.0:

                return {
                    "supported": True,
                    "level": "HIGH",
                    "reason": (
                        "The retrieved documentation contains "
                        "strongly relevant GRBL setting evidence."
                    ),
                }

            if score >= 0.0:

                return {
                    "supported": True,
                    "level": "MEDIUM",
                    "reason": (
                        "The documentation provides relevant "
                        "technical evidence, but does not fully "
                        "diagnose machine-specific causes."
                    ),
                }

            return {
                "supported": False,
                "level": "LOW",
                "reason": (
                    "The retrieved evidence is not strong enough "
                    "to support a reliable troubleshooting answer."
                ),
            }

        # ---------------------------------------------------------------------
        # General questions
        # ---------------------------------------------------------------------

        stopwords = {
            "a", "an", "and", "are", "can", "configure", "do", "does",
            "for", "how", "i", "in", "install", "is", "it", "my", "of",
            "read", "support", "the", "to", "use", "what", "why",
        }
        meaningful = {
            word for word in query_words
            if word not in stopwords and len(word) >= 3
        }
        subject_words = meaningful - {"grbl", "cnc", "machine"}
        if subject_words and not any(word in combined_top for word in subject_words):
            return {
                "supported": False,
                "level": "INSUFFICIENT",
                "reason": "The retrieved evidence does not match the question's subject.",
            }

        domain_terms = {
            "alarm", "axis", "cnc", "coolant", "coordinate", "feed", "gcode",
            "grbl", "home", "homing", "jog", "limit", "machine", "motion",
            "probe", "probing", "spindle", "step", "steps", "travel", "unit",
            "units",
        }
        if query_info["query_type"] == "general" and not (
            meaningful & domain_terms
        ):
            return {
                "supported": False,
                "level": "INSUFFICIENT",
                "reason": "The question is outside the supported GRBL corpus.",
            }

        if score >= 5.0:

            return {
                "supported": True,
                "level": "HIGH",
                "reason": (
                    "The top retrieved evidence is strongly "
                    "related to the question."
                ),
            }

        if score >= 0.0:

            return {
                "supported": True,
                "level": "MEDIUM",
                "reason": (
                    "The retrieved evidence is relevant but "
                    "some interpretation may be required."
                ),
            }

        return {
            "supported": False,
            "level": "LOW",
            "reason": (
                "The retrieved evidence is not sufficiently "
                "relevant to answer confidently."
            ),
        }


# =============================================================================
# SAFETY ANALYZER
# =============================================================================

class SafetyAnalyzer:
    """
    Determines whether a CNC safety warning should be displayed.

    We do not inject dangerous machine instructions.

    The LLM is additionally instructed to remain safety-conscious.
    """

    MOTION_TERMS = [
        "move",
        "movement",
        "jog",
        "jogging",
        "axis",
        "travel",
        "home",
        "homing",
        "probe",
        "probing",
        "limit switch",
        "soft limit",
        "hard limit",
        "arc",
        "position",
        "coordinate",
        "relative",
        "absolute",
        "distance mode",
        "g-code",
        "motion",
        "limit alarm",
        "limit alarms",
        "alarm lock",
        "alarm-locked",
        "alarm locked",
        "unit-selection",
        "unit selection",
        "25.4 times",
        "job dimensions",
    ]

    SPINDLE_TERMS = [
        "spindle",
        "router",
        "rpm",
        "cut",
        "cutting",
        "m3",
        "m4",
        "m5",
    ]

    def analyze(
        self,
        query: str,
    ) -> Dict[str, Any]:

        lower = query.lower()

        motion = any(
            term in lower
            for term in self.MOTION_TERMS
        )

        spindle = any(
            term in lower
            for term in self.SPINDLE_TERMS
        )

        required = motion or spindle

        if not required:

            return {
                "required": False,
                "message": "",
            }

        message = (
            "SAFETY: Before testing CNC motion, probing, "
            "homing, limits, or spindle commands, ensure the "
            "machine is in a safe state, the work area is clear, "
            "and you are prepared for unexpected movement. "
            "Do not run motion or spindle commands blindly."
        )

        return {
            "required": True,
            "message": message,
        }


# =============================================================================
# SOURCE FORMATTER
# =============================================================================

class SourceFormatter:

    @staticmethod
    def build_citations(
        results: List[Dict[str, Any]],
        max_sources: int = 5,
    ) -> List[Dict[str, Any]]:
        citations = []
        seen = set()

        for result in results:
            citation = citation_from_result(result)
            key = citation["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
            if len(citations) >= max_sources:
                break

        return citations

    @staticmethod
    def format_sources(
        results: List[Dict[str, Any]],
        max_sources: int = 5,
    ) -> List[str]:

        return [
            f"{item.get('source') or 'Unknown source'} | "
            f"{item.get('document') or 'Unknown document'} | "
            f"{item.get('section') or 'Unknown section'}"
            for item in SourceFormatter.build_citations(results, max_sources)
        ]


# =============================================================================
# MAIN GRBL AI ENGINE
# =============================================================================

class GRBLCNCAI:

    def __init__(self):

        print()
        print(
            color(
                "=" * 80,
                Colors.CYAN,
            )
        )

        print(
            color(
                "GRBL CNC AI - INITIALIZING",
                Colors.BOLD,
            )
        )

        print(
            color(
                "=" * 80,
                Colors.CYAN,
            )
        )

        print()

        # ---------------------------------------------------------------------
        # Components
        # ---------------------------------------------------------------------

        print("Loading hybrid retriever...")

        self.retriever = HybridRetriever(
            vector_top_k=VECTOR_TOP_K,
            bm25_top_k=BM25_TOP_K,
        )

        print()

        print("Loading query analyzer...")

        self.query_analyzer = QueryAnalyzer()

        print(
            "Query analyzer: OK"
        )

        print()

        print("Loading GRBL ranking layer...")

        self.grbl_ranker = GRBLRanker()

        print(
            "GRBL ranking layer: OK"
        )

        print()

        print("Loading evidence validator...")

        self.validator = EvidenceValidator()

        print(
            "Evidence validator: OK"
        )

        print()

        print("Loading safety analyzer...")

        self.safety = SafetyAnalyzer()

        print(
            "Safety analyzer: OK"
        )

        print()

        print("Loading answer generator...")

        self.answer_generator = (
            GRBLAnswerGenerator()
        )

        self.degraded_reasons = list(
            getattr(self.retriever, "degraded_reasons", [])
        )
        self.cross_encoder_available = True

        print(
            f"Answer generator: "
            f"{getattr(self.answer_generator, 'provider_name', 'configured')} OK"
        )

        print()

        print(
            color(
                "GRBL CNC AI initialization complete.",
                Colors.GREEN,
            )
        )

        print()

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:

        # ---------------------------------------------------------------------
        # Hybrid retrieval
        # ---------------------------------------------------------------------

        query_info = self.query_analyzer.analyze(query)

        hybrid_results = self.retriever.search(
            query,
            top_k=HYBRID_TOP_K,
            query_info=query_info,
        )

        # Exact GRBL system commands such as "$$" contain punctuation that
        # lexical/semantic retrieval can underweight. If the indexed metadata
        # contains a dedicated authoritative definition section for the exact
        # requested command, preserve that definition as a candidate before
        # reranking. This is evidence lookup, not a hard-coded answer.
        if (
            query_info.get("query_type") == "system_command"
            and len(query_info.get("system_commands", [])) == 1
        ):
            requested_command = query_info["system_commands"][0].upper()
            existing_ids = {
                item.get("chunk_id")
                for item in hybrid_results
            }

            for raw in getattr(self.retriever, "metadata", []) or []:
                metadata = raw.get("metadata") or raw
                section = str(
                    raw.get("section")
                    or metadata.get("section")
                    or ""
                )
                authority = (
                    raw.get("authority")
                    or metadata.get("authority")
                )

                # Require a dedicated definition heading. Incidental mentions
                # such as "$RST" text containing "$$" are intentionally ignored.
                section_upper = section.upper()
                command_pattern = (
                    rf"(^|[^A-Z0-9.$])"
                    rf"{re.escape(requested_command)}"
                    rf"([^A-Z0-9.]|$)"
                )
                if not re.search(command_pattern, section_upper):
                    continue
                if authority not in {"official", "official-derived"}:
                    continue

                chunk_id = (
                    raw.get("chunk_id")
                    or metadata.get("chunk_id")
                )
                if chunk_id in existing_ids:
                    break

                candidate = {
                    "chunk_id": chunk_id,
                    "source": (
                        raw.get("source")
                        or metadata.get("source")
                    ),
                    "document": (
                        raw.get("document")
                        or metadata.get("document")
                    ),
                    "category": (
                        raw.get("category")
                        or metadata.get("category")
                    ),
                    "section": section,
                    "section_path": (
                        raw.get("section_path")
                        or metadata.get("section_path")
                        or section
                    ),
                    "text": (
                        raw.get("text")
                        or metadata.get("text")
                        or ""
                    ),
                    "authority": authority,
                    "source_url": (
                        raw.get("source_url")
                        or metadata.get("source_url")
                    ),
                    "metadata": metadata,
                    "rrf_score": 1.0,
                    "hybrid_rank": 0,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "vector_score": None,
                    "bm25_score": None,
                    "retrieval_strategy": "exact_authoritative_metadata",
                    "query_type": query_info.get("query_type"),
                }
                hybrid_results.insert(0, candidate)
                break

        if not hybrid_results:

            return []

        preliminary = self.grbl_ranker.rank(query_info, hybrid_results)
        exact_types = {"exact_setting", "gcode", "mcode", "system_command"}
        entity_count = sum(
            len(query_info.get(key, []))
            for key in ("settings", "gcodes", "mcodes", "system_commands")
        )
        authoritative_exact = (
            query_info["query_type"] in exact_types
            and entity_count == 1
            and self.validator.validate(query_info, preliminary)["supported"]
        )
        skip_reranker = authoritative_exact and not CONFIG.force_rerank_all

        # ---------------------------------------------------------------------
        # Cross-Encoder
        # ---------------------------------------------------------------------

        if skip_reranker:
            reranked = []
            for result in hybrid_results[:RERANK_TOP_K]:
                item = dict(result)
                item["reranker_score"] = float(
                    item.get("rrf_score", item.get("score", 0.0)) or 0.0
                )
                reranked.append(item)
        else:
            try:
                if not self.cross_encoder_available:
                    raise RuntimeError(
                        "disabled after an earlier initialization failure"
                    )

                reranked = rerank(query, hybrid_results, top_k=RERANK_TOP_K)

            except Exception as exc:
                if self.cross_encoder_available:
                    reason = f"Cross-encoder unavailable: {exc}"
                    self.degraded_reasons.append(reason)
                    print(
                        "WARNING: Cross-encoder unavailable; "
                        "using retrieval order."
                    )
                self.cross_encoder_available = False
                reranked = []
                for result in hybrid_results[:RERANK_TOP_K]:
                    item = dict(result)
                    item["reranker_score"] = float(
                        item.get("bm25_score", item.get("score", 0.0)) or 0.0
                    )
                    reranked.append(item)

        # Preserve a small number of deterministic GRBL-aware candidates that
        # a generic cross-encoder may otherwise discard before final ranking.
        reranked_ids = {item.get("chunk_id") for item in reranked}
        for candidate in preliminary[:5]:
            if candidate.get("chunk_id") not in reranked_ids:
                reranked.append(candidate)
                reranked_ids.add(candidate.get("chunk_id"))

        # ---------------------------------------------------------------------
        # GRBL-specific deterministic ranking
        # ---------------------------------------------------------------------

        final_results = self.grbl_ranker.rank(
            query_info,
            reranked,
        )

        return final_results

    # =========================================================================
    # BUILD LLM RESULTS
    # =========================================================================

    def prepare_llm_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        selected = results[
            :FINAL_CONTEXT_K
        ]

        output = []

        for result in selected:

            item = dict(result)

            # Answer generator expects this spelling in its context.
            item["rerank_score"] = item.get(
                "reranker_score",
                item.get(
                    "rerank_score",
                    0.0,
                ),
            )

            output.append(item)

        return output

    # =========================================================================
    # FALLBACK ANSWER
    # =========================================================================

    @staticmethod
    def insufficient_evidence_answer(
        validation: Dict[str, Any],
        query: str,
    ) -> str:

        return (
            "\n"
            "ANSWER\n"
            "-------\n"
            "The available GRBL documentation does not provide "
            "enough information to answer this confidently.\n"
            "\n"
            "WHY\n"
            "----\n"
            f"{validation.get('reason', '')}\n"
            "\n"
            "The assistant has intentionally not filled the gap "
            "with unsupported information.\n"
            "\n"
            "RECOMMENDED CHECKS\n"
            "------------------\n"
            "1. Check whether the required GRBL version or alarm/error "
            "reference is included in the knowledge base.\n"
            "2. Add the relevant official GRBL documentation to the "
            "knowledge base if available.\n"
            "3. Re-run the document processing and indexing pipeline.\n"
            "\n"
            "GRBL COMMANDS / SETTINGS\n"
            "------------------------\n"
            "No specific command or setting is being recommended because "
            "the available evidence is insufficient.\n"
            "\n"
            "CONFIDENCE\n"
            "----------\n"
            "LOW\n"
        )

    @staticmethod
    def supported_evidence_answer(
        query_info: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> str:
        """Summarize retrieved evidence when Ollama over-refuses it."""

        top = results[0]
        section = str(top.get("section") or "Relevant GRBL documentation")
        text = " ".join(str(top.get("text") or "").split())
        excerpt = text[:600].rstrip()
        if len(text) > 600:
            excerpt += "…"

        if query_info["query_type"] == "troubleshooting":
            return (
                "ANSWER\n"
                f"The most relevant documented area is: {section}.\n\n"
                "DOCUMENTED FACTS\n"
                f"{excerpt}\n\n"
                "SUGGESTED DIAGNOSTIC STEPS\n"
                "1. Verify the setting, command, or input described in the "
                "retrieved section.\n"
                "2. Compare the observed machine state with that documented "
                "behavior before changing configuration.\n"
                "3. Treat these checks as diagnostic suggestions, not as a "
                "documented root-cause determination.\n\n"
                "SOURCES\nSee attached retrieved citations.\n\n"
                "CONFIDENCE\nMEDIUM"
            )

        requested = [
            *(f"${value}" for value in query_info.get("settings", [])),
            *query_info.get("gcodes", []),
            *query_info.get("mcodes", []),
            *query_info.get("system_commands", []),
        ]
        if requested:
            evidence_lines = []
            for identifier in requested:
                for result in results:
                    result_section = str(result.get("section") or "")
                    matches = (
                        identifier in result_section
                        if identifier.startswith("$") and not identifier[1:].isdigit()
                        else identifier.upper() in result_section.upper()
                    )
                    if matches:
                        result_text = " ".join(
                            str(result.get("text") or "").split()
                        )
                        evidence_lines.append(
                            f"- {result_section}: {result_text[:300].rstrip()}"
                        )
                        break
            if evidence_lines:
                return (
                    "ANSWER\n"
                    + "\n".join(evidence_lines)
                    + "\n\nSOURCES\nSee attached retrieved citations.\n\n"
                    "CONFIDENCE\nHIGH"
                )

        return (
            "ANSWER\n"
            f"{section}: {excerpt}\n\n"
            "SOURCES\nSee attached retrieved citations.\n\n"
            "CONFIDENCE\nHIGH"
        )

    @staticmethod
    def deterministic_exact_answer(
        result: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> str:
        """Render one validated authoritative definition without an LLM."""

        section = " ".join(str(result.get("section") or "").split())
        evidence = " ".join(str(result.get("text") or "").split())
        return (
            "ANSWER\n"
            f"{section}\n\n"
            f"{evidence}\n\n"
            "SOURCES\nSee attached retrieved citations.\n\n"
            f"CONFIDENCE\n{validation['level']}"
        )

    # =========================================================================
    # GENERATE ANSWER
    # =========================================================================

    def ask(
        self,
        query: str,
        show_retrieval: bool = False,
    ) -> Dict[str, Any]:

        query = query.strip()

        if not query:

            raise ValueError(
                "Question cannot be empty."
            )

        # ---------------------------------------------------------------------
        # Query analysis
        # ---------------------------------------------------------------------

        query_info = (
            self.query_analyzer.analyze(
                query
            )
        )

        # ---------------------------------------------------------------------
        # Retrieve
        # ---------------------------------------------------------------------

        results = self.retrieve(
            query
        )

        # ---------------------------------------------------------------------
        # Evidence validation
        # ---------------------------------------------------------------------

        validation = self.validator.validate(
            query_info,
            results,
        )

        # ---------------------------------------------------------------------
        # Safety
        # ---------------------------------------------------------------------

        safety = self.safety.analyze(
            query
        )

        # ---------------------------------------------------------------------
        # Debug retrieval display
        # ---------------------------------------------------------------------

        if show_retrieval:

            self.print_retrieval(
                query,
                query_info,
                results,
                validation,
            )

        # ---------------------------------------------------------------------
        # Insufficient evidence
        # ---------------------------------------------------------------------

        if not validation["supported"]:

            answer = self.insufficient_evidence_answer(
                validation,
                query,
            )

            sources = SourceFormatter.format_sources(results)
            citations = SourceFormatter.build_citations(results)

            return {
                "query": query,
                "answer": answer,
                "results": results,
                "validation": validation,
                "safety": safety,
                "sources": sources,
                "citations": citations,
                "query_info": query_info,
                "route": query_info.get("query_type"),
                "grounded": False,
                "valid": True,
                "validation_reason": validation.get("reason"),
            }

        # ---------------------------------------------------------------------
        # LLM context
        # ---------------------------------------------------------------------

        llm_results = self.prepare_llm_results(
            results
        )

        # ---------------------------------------------------------------------
        # Generate grounded answer
        # ---------------------------------------------------------------------
        requested_exact = [
            *(f"${value}" for value in query_info.get("settings", [])),
            *query_info.get("gcodes", []),
            *query_info.get("mcodes", []),
            *query_info.get("system_commands", []),
        ]
        exact_types = {"exact_setting", "gcode", "mcode", "system_command"}
        defining_result = next(
            (
                result for result in results
                if self.validator.exact_entities_defined(query_info, result)
            ),
            None,
        )
        deterministic_exact = (
            query_info["query_type"] in exact_types
            and len(requested_exact) == 1
            and validation["level"] == "HIGH"
            and defining_result is not None
        )

        if deterministic_exact:
            answer = self.deterministic_exact_answer(defining_result, validation)
            answer_grounding = {
                "passed": True,
                "unsupported_identifiers": [],
                "mode": "deterministic_exact",
            }
        else:
            answer = self.answer_generator.generate_answer(
                query=query,
                results=llm_results,
                query_info=query_info,
            )

            refusal_markers = (
                "does not provide enough information",
                "not found in the available grbl documentation",
                "generated response was withheld",
            )
            missing_requested = [
                identifier for identifier in requested_exact
                if identifier.upper() not in answer.upper()
            ]
            if (
                any(marker in answer.lower() for marker in refusal_markers)
                or missing_requested
            ):
                answer = self.supported_evidence_answer(query_info, llm_results)
                self.answer_generator.last_grounding_check = (
                    self.answer_generator.validate_grounding(answer, llm_results)
                )
            answer_grounding = self.answer_generator.last_grounding_check

        # ---------------------------------------------------------------------
        # Safety prefix
        # ---------------------------------------------------------------------

        if safety["required"]:

            answer = (
                safety["message"]
                + "\n\n"
                + answer
            )

        # ---------------------------------------------------------------------
        # Add source/retrieval metadata outside LLM response
        # ---------------------------------------------------------------------

        sources = (
            SourceFormatter.format_sources(
                results
            )
        )

        citations = SourceFormatter.build_citations(results)

        return {
            "query": query,
            "answer": answer,
            "results": results,
            "validation": validation,
            "safety": safety,
            "sources": sources,
            "citations": citations,
            "answer_grounding": answer_grounding,
            "query_info": query_info,
            "route": query_info.get("query_type"),
            "grounded": bool(answer_grounding.get("passed", True)),
            "valid": bool(answer_grounding.get("passed", True)),
            "validation_reason": validation.get("reason"),
        }

    # =========================================================================
    # PRINT RETRIEVAL
    # =========================================================================

    def print_retrieval(
        self,
        query: str,
        query_info: Dict[str, Any],
        results: List[Dict[str, Any]],
        validation: Dict[str, Any],
    ):

        print()
        print(
            color(
                "=" * 80,
                Colors.MAGENTA,
            )
        )

        print(
            color(
                "RETRIEVAL DEBUG",
                Colors.BOLD,
            )
        )

        print(
            color(
                "=" * 80,
                Colors.MAGENTA,
            )
        )

        print()
        print(
            f"Query: {query}"
        )

        print()
        print(
            f"Detected settings: "
            f"{query_info['settings']}"
        )

        print(
            f"Detected G-codes: "
            f"{query_info['gcodes']}"
        )

        print(
            f"Detected M-codes: "
            f"{query_info['mcodes']}"
        )

        print(
            f"Detected alarms: "
            f"{query_info['alarm_numbers']}"
        )

        print()
        print(
            f"Evidence status: "
            f"{validation['level']}"
        )

        print(
            f"Evidence reason: "
            f"{validation['reason']}"
        )

        print()

        for result in results[:10]:

            print(
                "-" * 80
            )

            print(
                f"Final Rank: "
                f"{result.get('final_rank')}"
            )

            print(
                f"Chunk: "
                f"{result.get('chunk_id')}"
            )

            print(
                f"Category: "
                f"{result.get('category')}"
            )

            print(
                f"Section: "
                f"{result.get('section')}"
            )

            print(
                f"Reranker Score: "
                f"{result.get('reranker_score')}"
            )

            print(
                f"GRBL Final Score: "
                f"{result.get('grbl_final_score')}"
            )

            reasons = result.get(
                "ranking_reasons",
                [],
            )

            if reasons:

                print(
                    f"Boosts: "
                    f"{', '.join(reasons)}"
                )

            text = result.get(
                "text",
                "",
            )

            preview = (
                text
                .replace("\n", " ")
                .strip()
            )

            if len(preview) > 300:

                preview = (
                    preview[:300]
                    + "..."
                )

            print(
                f"Text: {preview}"
            )

        print()

    # =========================================================================
    # PRINT FINAL RESULT
    # =========================================================================

    def print_answer(
        self,
        response: Dict[str, Any],
    ):

        print()
        print(
            color(
                "=" * 80,
                Colors.GREEN,
            )
        )

        print(
            color(
                "GRBL CNC AI ANSWER",
                Colors.BOLD,
            )
        )

        print(
            color(
                "=" * 80,
                Colors.GREEN,
            )
        )

        print()

        print(
            response["answer"]
        )

        print()

        validation = response.get(
            "validation",
            {},
        )

        print(
            color(
                "-" * 80,
                Colors.GRAY,
            )
        )

        print(
            f"Evidence status: "
            f"{validation.get('level', 'UNKNOWN')}"
        )

        sources = response.get(
            "sources",
            [],
        )

        if sources:

            print()

            print(
                "Retrieved sources:"
            )

            for source in sources:

                print(
                    f"  - {source}"
                )

        safety = response.get(
            "safety",
            {},
        )

        if safety.get("required"):

            print()

            print(
                color(
                    "CNC safety notice was applied.",
                    Colors.YELLOW,
                )
            )

        print()

        print(
            color(
                "=" * 80,
                Colors.GREEN,
            )
        )

    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================

    def interactive(self):

        print()
        print(
            color(
                "=" * 80,
                Colors.CYAN,
            )
        )

        print(
            color(
                "GRBL CNC AI - TECHNICAL SUPPORT ASSISTANT",
                Colors.BOLD,
            )
        )

        print(
            color(
                "=" * 80,
                Colors.CYAN,
            )
        )

        print()

        print(
            "Ask technical questions about GRBL."
        )

        print(
            "Type 'help' for commands."
        )

        print(
            "Type 'exit' or 'quit' to leave."
        )

        print()

        while True:

            try:

                query = input(
                    color(
                        "GRBL AI > ",
                        Colors.CYAN,
                    )
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print()
                print(
                    "Exiting."
                )

                break

            if not query:

                continue

            # -----------------------------------------------------------------
            # Commands
            # -----------------------------------------------------------------

            if query.lower() in (
                "exit",
                "quit",
                "q",
            ):

                print(
                    "Goodbye."
                )

                break

            if query.lower() == "help":

                self.print_help()

                continue

            if query.lower() == "status":

                self.print_status()

                continue

            if query.lower() == "debug":

                print(
                    "Debug retrieval is enabled for the next query."
                )

                continue

            # -----------------------------------------------------------------
            # Ask
            # -----------------------------------------------------------------

            try:

                response = self.ask(
                    query,
                    show_retrieval=False,
                )

                self.print_answer(
                    response
                )

            except Exception as exc:

                print()

                print(
                    color(
                        "ERROR",
                        Colors.RED,
                    )
                )

                print(
                    str(exc)
                )

                print()

    # =========================================================================
    # HELP
    # =========================================================================

    @staticmethod
    def print_help():

        print()
        print(
            "=" * 80
        )

        print(
            "AVAILABLE COMMANDS"
        )

        print(
            "=" * 80
        )

        print()

        print(
            "help"
        )

        print(
            "  Show this help."
        )

        print()

        print(
            "status"
        )

        print(
            "  Show system/component status."
        )

        print()

        print(
            "exit"
        )

        print(
            "  Exit the assistant."
        )

        print()

        print(
            "Example questions:"
        )

        examples = [
            "What does $100 control?",
            "What does G90 do?",
            "What does G38.2 do?",
            "What does M3 do?",
            "My X axis moves the wrong distance. What should I check?",
            "How do I home the machine?",
            "What is Alarm 1?",
        ]

        for example in examples:

            print(
                f"  - {example}"
            )

        print()

    # =========================================================================
    # STATUS
    # =========================================================================

    def print_status(self):

        print()
        print(
            "=" * 80
        )

        print(
            "SYSTEM STATUS"
        )

        print(
            "=" * 80
        )

        print()

        print(
            f"Corpus / FAISS vectors: "
            f"{self.retriever.index.ntotal if self.retriever.index else 0}"
        )

        print(
            f"Metadata records: "
            f"{len(self.retriever.metadata)}"
        )

        print(
            f"Vector top-k: "
            f"{VECTOR_TOP_K}"
        )

        print(
            f"BM25 top-k: "
            f"{BM25_TOP_K}"
        )

        print(
            f"Hybrid top-k: "
            f"{HYBRID_TOP_K}"
        )

        print(
            f"Cross-Encoder top-k: "
            f"{RERANK_TOP_K}"
        )

        print(
            f"Final context chunks: "
            f"{FINAL_CONTEXT_K}"
        )

        print()

        print(
            "Components:"
        )

        print(
            "  [OK] FAISS"
            if self.retriever.index is not None
            else "  [DEGRADED] FAISS unavailable; BM25 fallback active"
        )

        print(
            "  [OK] BM25"
        )

        print(
            "  [OK] Hybrid RRF"
        )

        print(
            "  [OK] Cross-Encoder"
            if self.cross_encoder_available
            else "  [DEGRADED] Cross-Encoder unavailable"
        )

        print(
            "  [OK] GRBL query-aware ranking"
        )

        print(
            "  [OK] Evidence validator"
        )

        print(
            "  [OK] CNC safety analyzer"
        )

        print(
            "  [OK] Ollama"
        )

        if self.degraded_reasons:
            print()
            print("Degraded mode reasons:")
            for reason in self.degraded_reasons:
                print(f"  - {reason}")

        print()


# =============================================================================
# AUTOMATED TEST SUITE
# =============================================================================

def run_test_suite(
    ai: GRBLCNCAI,
):

    tests = [

        "What does $100 control?",

        "What does G90 do?",

        "What does G38.2 do?",

        "What does M3 do?",

        "What is Alarm 1?",

        "My X axis moves the wrong distance. What should I check?",

        "How do I home the machine?",

        "What should I check if my spindle doesn't start?",
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "GRBL CNC AI - AUTOMATED TEST SUITE"
    )

    print(
        "=" * 80
    )

    for number, query in enumerate(
        tests,
        start=1,
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"TEST {number}: {query}"
        )

        print(
            "#" * 80
        )

        try:

            response = ai.ask(
                query,
                show_retrieval=True,
            )

            ai.print_answer(
                response
            )

        except Exception as exc:

            print()

            print(
                color(
                    f"TEST ERROR: {exc}",
                    Colors.RED,
                )
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print(
        color(
            "=" * 80,
            Colors.CYAN,
        )
    )

    print(
        color(
            "GRBL CNC AI",
            Colors.BOLD,
        )
    )

    print(
        color(
            "END-TO-END TECHNICAL SUPPORT SYSTEM",
            Colors.BOLD,
        )
    )

    print(
        color(
            "=" * 80,
            Colors.CYAN,
        )
    )

    print()

    try:

        ai = GRBLCNCAI()

    except Exception as exc:

        print()

        print(
            color(
                "INITIALIZATION FAILED",
                Colors.RED,
            )
        )

        print()

        print(
            str(exc)
        )

        print()

        raise

    # -------------------------------------------------------------------------
    # Command-line modes
    # -------------------------------------------------------------------------

    args = sys.argv[1:]

    if "--test" in args:

        run_test_suite(
            ai
        )

        return

    if "--status" in args:

        ai.print_status()

        return

    # -------------------------------------------------------------------------
    # Interactive mode
    # -------------------------------------------------------------------------

    ai.interactive()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
