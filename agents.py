"""Specialized AI agents for the Autonomous Multi-Agent Software Development Swarm.

Each agent is powered by a specific Groq-hosted LLM and operates on strongly-typed
Pydantic v2 schemas imported from contracts.py.

Agents:
    - ArchitectAgent: Decomposes a user request into a SwarmProject.
    - CoderAgent: Refines file implementations.
    - DebuggerAgent: Root-cause analysis and patch generation.
"""

import json
import logging
import os
import re
from typing import Any, List, Optional

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

from contracts import DebuggingPatch, FileArtifact, SwarmProject

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoHealSwarm.Agents")

# ---------------------------------------------------------------------------
# Environment & Client Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
    )

_client = Groq(api_key=_GROQ_API_KEY, timeout=30.0)

# ---------------------------------------------------------------------------
# Model Fallback Cascade
# ---------------------------------------------------------------------------

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_CASCADE = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]

DEFAULT_MAX_TOKENS = 3500

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _clean_json_string(s: str) -> str:
    """Clean unescaped control characters and common JSON artifacts."""
    return s.replace("\u2011", "-").replace("\u00a0", " ")


def _repair_truncated_json(s: str) -> Optional[dict]:
    """Attempt to repair and parse a truncated JSON string by closing unclosed structures."""
    content = s.strip()
    if not content.startswith("{"):
        first_brace = content.find("{")
        if first_brace != -1:
            content = content[first_brace:]
        else:
            return None

    # Stack tracking for open quotes and brackets
    in_string = False
    escape = False
    stack = []

    for char in content:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack:
                    stack.pop()

    # Build repair suffix
    suffix = ""
    if in_string:
        suffix += '"'

    while stack:
        opener = stack.pop()
        if opener == "{":
            suffix += "}"
        elif opener == "[":
            suffix += "]"

    repaired = content + suffix
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _parse_json_robust(raw_content: str) -> dict:
    """Parse JSON from raw LLM output, extracting from markdown code blocks or repairing partial responses."""
    content = _clean_json_string(raw_content.strip())
    if not content:
        raise ValueError("LLM returned empty response.")

    # 1. Direct parse attempt
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json ... ``` code fence
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(fence_pattern, content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Extract outermost balanced { ... }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        extracted = content[first_brace:last_brace + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # 4. Attempt repair if truncated
    repaired = _repair_truncated_json(content)
    if repaired is not None:
        logger.info("Successfully repaired and parsed truncated JSON response.")
        return repaired

    raise ValueError(f"Could not parse valid JSON from response: {content[:300]}")


def _try_extract_failed_generation(err: Exception) -> Optional[dict]:
    """Attempt to extract and parse the JSON payload from Groq's failed_generation error field."""
    try:
        body: dict = getattr(err, "body", {}) or {}
        err_dict = body.get("error", {}) if isinstance(body, dict) else {}
        failed_gen = err_dict.get("failed_generation")
        if failed_gen and isinstance(failed_gen, str):
            logger.info("Rescuing valid JSON payload from Groq failed_generation field...")
            return _parse_json_robust(failed_gen)
    except Exception:
        pass
    return None


def _chat_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fallback_models: Optional[List[str]] = None,
) -> dict:
    """Call Groq Chat Completion with multi-model fallback cascade and fault-tolerant parsing."""
    messages = [
        {"role": "system", "content": system_prompt + "\n\nCRITICAL: Output strictly pure, valid JSON with NO markdown fences."},
        {"role": "user",   "content": user_prompt},
    ]

    candidates = [model]
    for fb in (fallback_models or FALLBACK_CASCADE):
        if fb not in candidates:
            candidates.append(fb)

    last_error: Optional[Exception] = None

    for m in candidates:
        logger.info("Calling model '%s' (max_tokens=%d)...", m, max_tokens)
        
        # 1. Try standard JSON mode call
        try:
            response = _client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            return _parse_json_robust(raw_content)
        except BadRequestError as bad_req:
            rescued = _try_extract_failed_generation(bad_req)
            if rescued is not None:
                return rescued
            logger.warning("Model '%s' json_mode failed. Retrying without json_mode...", m)
            try:
                response = _client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_content = response.choices[0].message.content or ""
                return _parse_json_robust(raw_content)
            except Exception as retry_err:
                rescued_retry = _try_extract_failed_generation(retry_err)
                if rescued_retry is not None:
                    return rescued_retry
                last_error = retry_err
        except (RateLimitError, Exception) as err:
            logger.warning("Model '%s' failed (%s). Trying next available fallback...", m, err)
            last_error = err
            continue

    raise RuntimeError(
        f"All models in the fallback cascade {candidates} failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# ArchitectAgent
# ---------------------------------------------------------------------------

_ARCHITECT_SYSTEM = """You are a Principal Software Architect specializing in Python.
Your role is to decompose a user's high-level software request into a complete, modular project plan.

You MUST respond with a single, valid JSON object matching this schema exactly:
{
  "project_name":         "<snake_case_name>",
  "architecture_summary": "<2-4 sentence design rationale>",
  "dependencies":         ["<pypi-package-name>", ...],
  "test_command":         "python -m unittest discover -s . -p 'test*.py'",
  "files": [
    {
      "filename":  "<relative_path.py>",
      "content":   "<complete raw Python source, NO markdown fences>",
      "file_type": "source" | "test" | "config"
    }
  ]
}

Design principles:
- Strictly separate concerns: core logic, utilities, unit tests.
- Keep file paths simple (e.g. core.py, utils.py, test_core.py).
- Every source file must include clean docstrings and type hints.
- Every test file must use the built-in `unittest` framework.
- The `content` field must contain COMPLETE, RUNNABLE Python code.
- If no third-party dependencies are needed, return an empty array for 'dependencies'."""


def run_architect(user_request: str, model: Optional[str] = None) -> SwarmProject:
    """Decompose a high-level user request into a fully-specified SwarmProject."""
    selected_model = model or PRIMARY_MODEL
    logger.info("ArchitectAgent starting with model '%s' for request: %r", selected_model, user_request[:80])

    user_prompt = (
        f"Design and generate a complete Python project for the following request:\n\n"
        f"{user_request}\n\n"
        f"Return only the JSON object."
    )

    raw = _chat_json(
        model=selected_model,
        system_prompt=_ARCHITECT_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    project = SwarmProject.model_validate(raw)
    logger.info(
        "ArchitectAgent complete -- project='%s', files=%d",
        project.project_name,
        len(project.files),
    )
    return project


# ---------------------------------------------------------------------------
# CoderAgent
# ---------------------------------------------------------------------------

_CODER_SYSTEM = """You are a Senior Python Engineer specializing in clean, production-grade code.
You will receive a list of Python files from a SwarmProject and your task is to refine and complete them.

You MUST respond with a valid JSON object matching this schema exactly:
{
  "files": [
    {
      "filename":  "<same relative path as input>",
      "content":   "<complete, refined raw Python source, NO markdown fences>",
      "file_type": "source" | "test" | "config"
    }
  ]
}

Refinement requirements:
- Ensure all public functions and classes have docstrings and type hints.
- Handle standard edge cases.
- Ensure unit tests use `unittest`.
- Never use markdown code fences inside JSON string values.
- Return ALL input files."""


def run_coder(project: SwarmProject, model: Optional[str] = None) -> SwarmProject:
    """Refine all file implementations in the project with production-quality code."""
    selected_model = model or PRIMARY_MODEL
    logger.info(
        "CoderAgent starting -- project='%s', files=%d",
        project.project_name,
        len(project.files),
    )

    files_context = json.dumps(
        [f.model_dump() for f in project.files],
        indent=2,
    )

    user_prompt = (
        f"Project: {project.project_name}\n"
        f"Architecture: {project.architecture_summary}\n\n"
        f"Refine the following Python files to production quality:\n\n"
        f"{files_context}\n\n"
        f"Return only the JSON object containing the refined 'files' array."
    )

    raw = _chat_json(
        model=selected_model,
        system_prompt=_CODER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    refined_files = [
        FileArtifact.model_validate(f)
        for f in raw.get("files", [])
    ]
    project.update_files(refined_files)

    logger.info(
        "CoderAgent complete -- refined %d file(s)",
        len(refined_files),
    )
    return project


# ---------------------------------------------------------------------------
# DebuggerAgent
# ---------------------------------------------------------------------------

_DEBUGGER_SYSTEM = """You are an expert Python Debugging Engineer.
You will receive failing source code files along with the test output/traceback.

You MUST respond with a valid JSON object matching this schema exactly:
{
  "root_cause_analysis": "<detailed explanation of what went wrong and why>",
  "files_to_update": [
    {
      "filename":  "<relative path of the file that needs to be fixed>",
      "content":   "<complete corrected Python source, NO markdown fences>",
      "file_type": "source" | "test" | "config"
    }
  ]
}

Debugging requirements:
- Provide a clear root_cause_analysis.
- Only include files in 'files_to_update' that need modifications.
- Each updated file must be COMPLETE and RUNNABLE."""


def run_debugger(
    project: SwarmProject,
    traceback: str,
    stdout: str = "",
    stderr: str = "",
    model: Optional[str] = None,
) -> DebuggingPatch:
    """Analyse a test failure and produce a DebuggingPatch to fix the project."""
    selected_model = model or PRIMARY_MODEL
    logger.info(
        "DebuggerAgent starting -- project='%s'",
        project.project_name,
    )

    files_context = json.dumps(
        [f.model_dump() for f in project.files],
        indent=2,
    )

    terminal_output = "\n".join(filter(None, [
        f"=== TRACEBACK ===\n{traceback.strip()}" if traceback else "",
        f"=== STDOUT ===\n{stdout.strip()}" if stdout else "",
        f"=== STDERR ===\n{stderr.strip()}" if stderr else "",
    ]))

    user_prompt = (
        f"Project: {project.project_name}\n"
        f"Architecture: {project.architecture_summary}\n\n"
        f"The following test command failed:\n  {project.test_command}\n\n"
        f"Terminal output:\n{terminal_output}\n\n"
        f"Source files:\n{files_context}\n\n"
        f"Diagnose the root cause and return the corrected files as a JSON patch object."
    )

    raw = _chat_json(
        model=selected_model,
        system_prompt=_DEBUGGER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    patch = DebuggingPatch.model_validate(raw)
    logger.info(
        "DebuggerAgent complete -- root_cause=%r, fix_count=%d",
        patch.root_cause_analysis[:80],
        len(patch.files_to_update),
    )
    return patch
