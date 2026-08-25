"""Secure sandbox for writing, statically analysing, and executing generated code.

Provides two public interfaces:
    - lint_code_security(code_str)         — AST-based static security analysis.
    - execute_project_tests(project, ...)  — Ephemeral workspace + subprocess test runner.

Security model:
    All generated code is scanned for forbidden AST nodes (dangerous builtins,
    OS/subprocess calls, network sockets, dynamic code execution) before any file
    is written to disk or executed. Execution is further isolated via a time-bounded
    subprocess with a clean, ephemeral working directory that is removed on exit.
"""

import ast
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

from contracts import SwarmProject

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("AutoHealSwarm.Sandbox")

# ---------------------------------------------------------------------------
# Security Policy
# ---------------------------------------------------------------------------

# Forbidden bare names (builtins used directly as identifiers)
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "memoryview",
})

# Forbidden attribute access patterns: module.attribute
# Stored as frozenset of (object_name, attr_name) tuples.
_FORBIDDEN_ATTRS: frozenset[tuple[str, str]] = frozenset({
    # os module
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "execvpe"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
    ("os", "spawnlpe"),
    ("os", "spawnv"),
    ("os", "spawnve"),
    ("os", "spawnvp"),
    ("os", "spawnvpe"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    # os.path destructive helpers
    ("os.path", "abspath"),   # not destructive, but disallowed as escape vector
    # shutil
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("shutil", "copy"),
    ("shutil", "copytree"),
    ("shutil", "disk_usage"),
    # subprocess
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "Popen"),
    ("subprocess", "getoutput"),
    ("subprocess", "getstatusoutput"),
    # socket
    ("socket", "socket"),
    ("socket", "create_connection"),
    ("socket", "create_server"),
    # ctypes
    ("ctypes", "CDLL"),
    ("ctypes", "cdll"),
    ("ctypes", "windll"),
    # builtins via module path
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "compile"),
    ("builtins", "__import__"),
})

# Forbidden module imports (top-level names only)
_FORBIDDEN_IMPORTS: frozenset[str] = frozenset({
    "subprocess",
    "socket",
    "ctypes",
    "pty",
    "pexpect",
    "paramiko",
    "ftplib",
    "telnetlib",
    "smtplib",
    "http.server",
    "xmlrpc",
    "multiprocessing",
    "cffi",
    "winreg",
    "msvcrt",
    "posix",
})


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class _ViolationCollector(ast.NodeVisitor):
    """Walks an AST and collects all security policy violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    # --- Imports ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden import '{alias.name}'."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root in _FORBIDDEN_IMPORTS:
            self.violations.append(
                f"Line {node.lineno}: Forbidden from-import from '{module}'."
            )
        # Also block: from builtins import eval / exec
        if module in ("builtins", "__builtin__"):
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    self.violations.append(
                        f"Line {node.lineno}: Forbidden import of builtin '{alias.name}' "
                        f"from '{module}'."
                    )
        self.generic_visit(node)

    # --- Dangerous builtins used as Name nodes ---

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self.violations.append(
                f"Line {node.lineno}: Forbidden builtin '{node.id}()' detected."
            )
        self.generic_visit(node)

    # --- Attribute access: obj.attr ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        obj_name = self._extract_name(node.value)
        if obj_name is not None:
            key = (obj_name, node.attr)
            if key in _FORBIDDEN_ATTRS:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden call '{obj_name}.{node.attr}' detected."
                )
        self.generic_visit(node)

    # --- __dunder__ attribute access used as escape hatch ---

    def visit_Call(self, node: ast.Call) -> None:
        # Detect getattr(obj, 'system') style evasion
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr = node.args[1].value
                self.violations.append(
                    f"Line {node.lineno}: Suspicious getattr() with literal attribute "
                    f"'{attr}' — potential policy bypass."
                )
        self.generic_visit(node)

    # --- Helpers ---

    @staticmethod
    def _extract_name(node: ast.expr) -> str | None:
        """Recursively extract a dotted name string from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = _ViolationCollector._extract_name(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return None


# ---------------------------------------------------------------------------
# Public: lint_code_security
# ---------------------------------------------------------------------------

def lint_code_security(code_str: str) -> tuple[bool, str]:
    """Statically analyse a Python code string for security policy violations.

    Uses Python's built-in ``ast`` module to walk the syntax tree without
    executing any code. Checks for forbidden imports, dangerous builtins,
    OS/subprocess/socket manipulation, and dynamic code execution patterns.

    Args:
        code_str: Raw Python source code to inspect.

    Returns:
        A ``(is_safe, message)`` tuple where:
        - ``(True, "Safe")`` if no violations were found.
        - ``(False, "<details>")`` if one or more violations were detected.
    """
    if not code_str or not code_str.strip():
        return False, "Code string is empty."

    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return False, f"Syntax error — cannot parse code: {exc}"

    collector = _ViolationCollector()
    collector.visit(tree)

    if collector.violations:
        detail = "\n".join(f"  • {v}" for v in collector.violations)
        return False, f"Security violations detected:\n{detail}"

    return True, "Safe"


# ---------------------------------------------------------------------------
# Internal Workspace Helpers
# ---------------------------------------------------------------------------

class _WorkspaceResult(NamedTuple):
    success: bool
    output: str


def _prepare_workspace(project: SwarmProject, base_dir: Path) -> list[str]:
    """Write all project files to the ephemeral workspace directory.

    Args:
        project:  The SwarmProject whose files will be written.
        base_dir: Absolute path to the workspace root directory.

    Returns:
        List of written file paths (relative to base_dir).

    Raises:
        PermissionError: If a file cannot be written.
        OSError: On unexpected filesystem errors.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for artifact in project.files:
        dest = base_dir / artifact.filename
        # Ensure nested directories exist (e.g. "src/utils.py")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(artifact.content, encoding="utf-8")
        written.append(str(dest.relative_to(base_dir)))
        logger.debug("Written: %s", dest)

    return written


def _cleanup_workspace(base_dir: Path) -> None:
    """Recursively remove the ephemeral workspace directory.

    Args:
        base_dir: Path to the workspace root to delete.
    """
    if base_dir.exists():
        try:
            shutil.rmtree(base_dir)
            logger.debug("Workspace cleaned: %s", base_dir)
        except OSError as exc:
            logger.warning("Failed to clean workspace '%s': %s", base_dir, exc)


def _run_security_checks(project: SwarmProject) -> _WorkspaceResult | None:
    """Run lint_code_security on every file in the project.

    Args:
        project: SwarmProject containing files to lint.

    Returns:
        A failed ``_WorkspaceResult`` if any file fails linting, or ``None``
        if all files pass.
    """
    for artifact in project.files:
        is_safe, reason = lint_code_security(artifact.content)
        if not is_safe:
            msg = (
                f"Security linting FAILED for '{artifact.filename}':\n"
                f"{textwrap.indent(reason, '  ')}"
            )
            logger.error(msg)
            return _WorkspaceResult(success=False, output=msg)
        logger.info("Security lint passed: %s", artifact.filename)
    return None


# ---------------------------------------------------------------------------
# Public: execute_project_tests
# ---------------------------------------------------------------------------

def execute_project_tests(
    project: SwarmProject,
    base_dir: str = ".swarm_workspace",
    timeout: int = 15,
) -> tuple[bool, str]:
    """Write project files to an ephemeral sandbox and execute the test suite.

    Workflow:
        1. Run ``lint_code_security`` on **every** file — abort on first failure.
        2. Write all files to ``base_dir`` on disk.
        3. Execute the test command via a time-bounded subprocess.
        4. Clean up the workspace directory unconditionally on exit.

    Args:
        project:  The SwarmProject to test.
        base_dir: Relative or absolute path for the ephemeral workspace.
                  Defaults to ``.swarm_workspace`` in the current directory.
        timeout:  Maximum seconds the test subprocess may run. Default is 15.

    Returns:
        A ``(success, output)`` tuple where:
        - ``(True, stdout)``            — all tests passed.
        - ``(False, stderr + stdout)``  — tests failed, timed out, or were blocked.
    """
    workspace = Path(base_dir).resolve()
    logger.info(
        "Sandbox starting — project='%s', workspace='%s'",
        project.project_name,
        workspace,
    )

    # --- Phase 1: Security linting (pre-write) ---
    lint_failure = _run_security_checks(project)
    if lint_failure is not None:
        return lint_failure.success, lint_failure.output

    # --- Phase 2: Write files to workspace ---
    try:
        written = _prepare_workspace(project, workspace)
        logger.info("Wrote %d file(s) to workspace.", len(written))
    except OSError as exc:
        msg = f"Failed to write workspace files: {exc}"
        logger.error(msg)
        return False, msg

    # --- Phase 3: Execute tests in subprocess ---
    # Ensure current workspace is on PYTHONPATH so imports work seamlessly
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(workspace)

    # If test files are in subdirectories, make sure __init__.py exists
    for path in workspace.rglob("*"):
        if path.is_dir() and not (path / "__init__.py").exists():
            try:
                (path / "__init__.py").touch()
            except OSError:
                pass

    cmd = [
        sys.executable,
        "-m", "unittest",
        "discover",
        "-s", ".",
        "-t", ".",
        "-p", "test*.py",
        "-v",
    ]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), workspace)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=sub_env,
        )
    except subprocess.TimeoutExpired:
        msg = (
            f"Test execution timed out after {timeout}s. "
            f"Possible infinite loop or blocking call in generated code."
        )
        logger.error(msg)
        _cleanup_workspace(workspace)
        return False, msg
    except OSError as exc:
        msg = f"Subprocess launch failed: {exc}"
        logger.error(msg)
        _cleanup_workspace(workspace)
        return False, msg

    # --- Phase 4: Cleanup ---
    _cleanup_workspace(workspace)

    # --- Phase 5: Evaluate results ---
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    combined_output = "\n".join(filter(None, [stderr, stdout]))

    if result.returncode == 0:
        logger.info("Tests PASSED (exit code 0).")
        return True, stdout

    logger.warning("Tests FAILED (exit code %d).", result.returncode)
    return False, combined_output


# ---------------------------------------------------------------------------
# Cleanup Utility (public, for external orchestrators)
# ---------------------------------------------------------------------------

def cleanup_workspace(base_dir: str = ".swarm_workspace") -> None:
    """Manually remove the sandbox workspace directory.

    Safe to call even if the directory does not exist.

    Args:
        base_dir: Path to the workspace directory to remove.
    """
    _cleanup_workspace(Path(base_dir).resolve())
