"""AutoHeal Swarm — Streamlit Command Dashboard.

Unifies contracts.py, agents.py, and sandbox.py into a live autonomous
multi-agent code generation and self-healing web application.
"""

import io
import time
import traceback
import zipfile

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AutoHeal Swarm",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1627 50%, #0a1520 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0a1628 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

.swarm-header {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem;
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(154,230,180,0.05) 100%);
    border-radius: 16px;
    border: 1px solid rgba(99,179,237,0.12);
    margin-bottom: 1.5rem;
}
.swarm-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9ae6b4, #63b3ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.swarm-header p { color: #718096; margin: 0.5rem 0 0; font-size: 1.05rem; }

.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.4px;
}
.badge-architect { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-coder     { background: rgba(154,230,180,0.12); color: #9ae6b4; border: 1px solid rgba(154,230,180,0.3); }
.badge-debugger  { background: rgba(252,129,74,0.12);  color: #fc814a; border: 1px solid rgba(252,129,74,0.3); }
.badge-sandbox   { background: rgba(214,188,250,0.12); color: #d6bcfa; border: 1px solid rgba(214,188,250,0.3); }

.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c7a7b) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 2.5rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important; transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(43,108,176,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(43,108,176,0.55) !important;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
}
.metric-card .label { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; margin-top: 2px; }

code, pre { font-family: 'JetBrains Mono', monospace !important; }

.banner-success {
    background: linear-gradient(135deg, rgba(72,187,120,0.15), rgba(56,161,105,0.1));
    border: 1px solid rgba(72,187,120,0.4); border-left: 4px solid #48bb78;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.banner-failure {
    background: linear-gradient(135deg, rgba(245,101,101,0.15), rgba(197,48,48,0.1));
    border: 1px solid rgba(245,101,101,0.4); border-left: 4px solid #f56565;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.banner-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem; }
.banner-body  { font-size: 0.9rem; color: #a0aec0; }

[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}

hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from agents import run_architect, run_coder, run_debugger
from sandbox import execute_project_tests
from contracts import SwarmProject

# ── Constants ─────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
    "groq/compound-mini",
]
FILE_TYPE_ICON = {"source": "🐍", "test": "🧪", "config": "⚙️"}
FILE_TYPE_LANG = {"source": "python", "test": "python", "config": "toml"}

SAMPLE_PROMPTS = {
    "🔢 Calculator Library": "Build a modular Python calculator library with functions for add, subtract, multiply, and divide with unit tests.",
    "📚 Stack Data Structure": "Build a Python Stack data structure with push, pop, peek, and is_empty methods, plus unit tests.",
    "🌳 Binary Search Tree": "Build a Python Binary Search Tree with insert, search, and in-order traversal methods, plus comprehensive unit tests.",
}

# ── Session state ─────────────────────────────────────────────────────────────
def _init_session() -> None:
    defaults: dict = {
        "project": None,
        "run_complete": False,
        "run_success": False,
        "attempt_logs": [],
        "zip_bytes": None,
        "prompt_text": "Build a modular Python calculator library with functions for add, subtract, multiply, and divide with unit tests.",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_zip(project: SwarmProject) -> bytes:
    """Pack all project files into an in-memory ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in project.files:
            zf.writestr(artifact.filename, artifact.content)
        if project.dependencies:
            zf.writestr("requirements.txt", "\n".join(project.dependencies))
    return buf.getvalue()


def _render_file_tabs_in_container(container, proj: SwarmProject | None) -> None:
    """Render a tabbed file browser into a specific Streamlit container."""
    with container:
        container.empty()
        st.markdown("### 📂 Generated Project Files")
        if not proj or not proj.files:
            st.info("No files generated yet. Launch the swarm to generate code.")
            return

        tab_labels = [
            f"{FILE_TYPE_ICON.get(f.file_type, '📄')} {f.filename}"
            for f in proj.files
        ]
        tabs = st.tabs(tab_labels)
        for tab, artifact in zip(tabs, proj.files):
            with tab:
                badge_color = {
                    "source": "#63b3ed",
                    "test": "#9ae6b4",
                    "config": "#d6bcfa",
                }.get(artifact.file_type, "#a0aec0")
                st.markdown(
                    f'<span style="background:rgba(0,0,0,0.3);border:1px solid {badge_color}44;'
                    f'color:{badge_color};padding:2px 10px;border-radius:999px;font-size:0.75rem;'
                    f'font-weight:600;">{artifact.file_type.upper()}</span><br><br>',
                    unsafe_allow_html=True,
                )
                lang = FILE_TYPE_LANG.get(artifact.file_type, "python")
                st.code(artifact.content, language=lang, line_numbers=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Swarm Configuration")
    st.markdown("---")

    selected_model = st.selectbox(
        "Architect Model",
        AVAILABLE_MODELS,
        index=0,
        help="Primary LLM for code generation and architecture planning.",
    )

    max_retries = st.slider(
        "Max Self-Healing Retries",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="How many times the Debugger Agent may attempt to fix failing tests.",
    )

    st.markdown("---")
    st.markdown("### 💡 Quick Load Samples")
    for title, prompt_sample in SAMPLE_PROMPTS.items():
        if st.button(title, use_container_width=True):
            st.session_state.prompt_text = prompt_sample
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Active Agent Roster")
    st.markdown("""
<div class="badge-row">
  <span class="badge badge-architect">🧠 Architect</span>
  <span class="badge badge-coder">💻 Coder</span>
  <span class="badge badge-debugger">🔍 Debugger</span>
  <span class="badge badge-sandbox">🔒 Sandbox</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Models")
    st.caption(f"**Architect** — `{selected_model}`")
    st.caption("**Coder** — `openai/gpt-oss-120b`")
    st.caption("**Debugger** — `openai/gpt-oss-120b`")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="swarm-header">
    <h1>🤖 AutoHeal Swarm</h1>
    <p>Autonomous Multi-Agent Software Development · Powered by Groq LPUs</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics Row ───────────────────────────────────────────────────────────────
_project: SwarmProject | None = st.session_state.project
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Files Generated</div>
        <div class="value">{len(_project.files) if _project else "0"}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Healing Attempts</div>
        <div class="value">{len(st.session_state.attempt_logs)}</div>
    </div>""", unsafe_allow_html=True)
with m3:
    _deps = len(_project.dependencies) if _project else 0
    st.markdown(f"""<div class="metric-card">
        <div class="label">Dependencies</div>
        <div class="value">{_deps}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    if not st.session_state.run_complete:
        _status_text = "Ready"
    elif st.session_state.run_success:
        _status_text = "✅ Passed"
    else:
        _status_text = "❌ Failed"
    st.markdown(f"""<div class="metric-card">
        <div class="label">Test Status</div>
        <div class="value" style="font-size:1.1rem">{_status_text}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prompt input ──────────────────────────────────────────────────────────────
user_prompt = st.text_area(
    "🎯 Describe the software or algorithm to build",
    value=st.session_state.prompt_text,
    height=110,
    key="user_prompt_input",
)

launch_col, _ = st.columns([1, 3])
with launch_col:
    launch = st.button("🚀 Launch Autonomous Swarm", use_container_width=True)

st.markdown("---")

# ── Dual-Column Layout ────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")
right_container = right_col.container()

# ── Swarm Execution ───────────────────────────────────────────────────────────
if launch:
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a software description before launching.")
        st.stop()

    st.session_state.prompt_text = user_prompt.strip()
    st.session_state.project = None
    st.session_state.run_complete = False
    st.session_state.run_success = False
    st.session_state.attempt_logs = []
    st.session_state.zip_bytes = None

    with left_col:
        st.markdown("### 📡 Live Execution Stream")

        # Phase 1 — Architect
        with st.status("🧠 **Architect Agent** — Designing project…", expanded=True) as arch_status:
            try:
                st.write("Decomposing your request into modular files…")
                t0 = time.time()
                project = run_architect(user_prompt.strip(), model=selected_model)
                elapsed = round(time.time() - t0, 1)
                st.write(f"✅ `{project.project_name}` designed in **{elapsed}s**")
                st.write(f"📁 **{len(project.files)}** file(s)  ·  📦 **{len(project.dependencies)}** dep(s)")
                arch_status.update(
                    label=f"🧠 Architect — Done ({elapsed}s)", state="complete"
                )
                st.session_state.project = project
                # Render initial files immediately into the right container!
                _render_file_tabs_in_container(right_container, project)
            except Exception:
                arch_status.update(label="🧠 Architect — Failed", state="error")
                st.error(f"**Architect failed:**\n```\n{traceback.format_exc()}\n```")
                st.stop()

        # Phase 2 — Coder
        with st.status("💻 **Coder Agent** — Refining implementations…", expanded=True) as coder_status:
            try:
                st.write("Enhancing docstrings, type hints, and edge-case handling…")
                t0 = time.time()
                project = run_coder(project, model=selected_model)
                elapsed = round(time.time() - t0, 1)
                st.write(f"✅ Refined **{len(project.files)}** file(s) in **{elapsed}s**")
                coder_status.update(
                    label=f"💻 Coder — Done ({elapsed}s)", state="complete"
                )
                st.session_state.project = project
                # Update files in real-time
                _render_file_tabs_in_container(right_container, project)
            except Exception:
                coder_status.update(label="💻 Coder — Failed", state="error")
                st.error(f"**Coder failed:**\n```\n{traceback.format_exc()}\n```")
                st.stop()

        # Phase 3 — Self-Healing Loop
        success = False
        test_output = ""

        for attempt in range(1, max_retries + 2):
            is_initial = attempt == 1
            if is_initial:
                status_label = "🔒 **Sandbox** — Running test suite…"
            else:
                status_label = f"🔍 **Debugger** — Healing attempt {attempt - 1}/{max_retries}…"

            with st.status(status_label, expanded=True) as loop_status:
                st.write(f"Executing `{project.test_command}`…")
                t0 = time.time()
                success, test_output = execute_project_tests(
                    project, base_dir=f".swarm_ws_{attempt}"
                )
                elapsed = round(time.time() - t0, 1)

                st.session_state.attempt_logs.append({
                    "attempt": attempt,
                    "success": success,
                    "output": test_output,
                    "elapsed": elapsed,
                })

                if success:
                    st.write(f"✅ All tests passed in **{elapsed}s**!")
                    loop_status.update(
                        label=f"✅ Tests Passed (attempt {attempt}, {elapsed}s)",
                        state="complete",
                    )
                    break

                remaining = max_retries - (attempt - 1)
                if remaining <= 0:
                    st.write("❌ Tests failed. No retries remaining.")
                    loop_status.update(
                        label="❌ Tests Failed — Max retries exceeded", state="error"
                    )
                    break

                st.write(f"⚠️ Tests failed in **{elapsed}s**. Engaging Debugger Agent…")
                st.code(test_output[:1500], language="text")

                try:
                    patch = run_debugger(project, traceback=test_output, model=selected_model)
                    project.apply_patch(patch)
                    st.session_state.project = project
                    _render_file_tabs_in_container(right_container, project)
                    st.write(f"🩹 Patch applied — **{len(patch.files_to_update)}** file(s) updated")
                    st.caption(f"Root cause: {patch.root_cause_analysis[:240]}")
                    loop_status.update(
                        label=f"🔍 Debugger — Patch applied (attempt {attempt})",
                        state="complete",
                    )
                except Exception:
                    loop_status.update(label="🔍 Debugger — Failed", state="error")
                    st.error(f"**Debugger failed:**\n```\n{traceback.format_exc()}\n```")
                    break

        # Outcome banners
        st.session_state.run_complete = True
        st.session_state.run_success = success

        if success:
            st.session_state.zip_bytes = _build_zip(project)
            st.markdown("""
<div class="banner-success">
    <div class="banner-title">🎉 Swarm Succeeded!</div>
    <div class="banner-body">All tests passed. Your verified project is ready to download below.</div>
</div>""", unsafe_allow_html=True)
            proj_name = project.project_name
            st.download_button(
                label="⬇️ Download Project ZIP",
                data=st.session_state.zip_bytes,
                file_name=f"{proj_name}_verified.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.markdown(f"""
<div class="banner-failure">
    <div class="banner-title">💥 Swarm Exhausted Retries</div>
    <div class="banner-body">Tests could not be fixed within {max_retries} attempt(s).
    Inspect the output below.</div>
</div>""", unsafe_allow_html=True)
            with st.expander("📋 Final test output", expanded=True):
                st.code(test_output or "(no output captured)", language="text")

else:
    # ── When not actively launching a new run ─────────────────────────────────
    with left_col:
        st.markdown("### 📡 Execution Stream")
        if st.session_state.attempt_logs:
            for log in st.session_state.attempt_logs:
                icon = "✅" if log["success"] else "❌"
                label = "Tests passed" if log["success"] else "Tests failed"
                with st.expander(f"{icon} Attempt {log['attempt']} — {label} ({log['elapsed']}s)"):
                    st.code(log["output"][:2000] or "(no output)", language="text")

            if st.session_state.run_success and st.session_state.zip_bytes:
                st.markdown("---")
                st.markdown("### 📦 Export Verified Project")
                proj_name = st.session_state.project.project_name if st.session_state.project else "project"
                st.download_button(
                    label="⬇️ Download Project ZIP",
                    data=st.session_state.zip_bytes,
                    file_name=f"{proj_name}_verified.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.markdown("""
<div style="text-align:center; padding: 3rem 1.5rem; color: #4a5568; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;">
    <div style="font-size:3rem; margin-bottom:0.8rem;">🤖</div>
    <div style="font-size:1.1rem; font-weight:600; color:#718096;">Ready to Launch</div>
    <div style="font-size:0.9rem; margin-top:0.3rem; color:#4a5568;">
        Click <strong>Launch Autonomous Swarm</strong> above or select a sample from the sidebar.
    </div>
</div>
""", unsafe_allow_html=True)

    _render_file_tabs_in_container(right_container, st.session_state.project)
