"""
app.py — ChronoMind Streamlit Application.

A modern, multi-tab Streamlit UI for:
  1. Chatbot — Ask questions via RAG + Persona
  2. Persona Viewer — Browse extracted user personas
  3. Topic Explorer — Navigate detected topic segments
  4. Checkpoints Explorer — Review 100-message summaries

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.persona_extractor import load_personas
from src.vector_store import VectorStore
from chatbot import ChronoBot
from app_v2_tabs import (
    render_persona_drift_tab,
    render_intent_tab,
    render_conflict_tab,
    render_sync_architecture_tab,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoMind — Conversation Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #c4b5fd;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    .metric-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #c4b5fd;
        margin-top: 0.25rem;
    }

    /* Chat message */
    .chat-msg {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        line-height: 1.6;
    }
    .chat-msg.user {
        background: linear-gradient(135deg, #312e81, #4338ca);
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: 2rem;
    }
    .chat-msg.bot {
        background: linear-gradient(135deg, #1e1b4b, #1e1b4b);
        color: #e0e7ff;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-bottom-left-radius: 4px;
        margin-right: 2rem;
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #1e1b4b, #1a1a3e);
        color: #e0e7ff;
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-bottom-left-radius: 4px;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin: 0.75rem 0;
        line-height: 1.8;
        font-size: 0.95rem;
        margin-right: 2rem;
    }
    .answer-card h4 {
        color: #a78bfa;
        margin: 0 0 0.75rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .answer-card hr {
        border-color: rgba(99,102,241,0.2);
        margin: 0.8rem 0;
    }

    /* Evidence item */
    .evidence-item {
        background: rgba(15, 12, 41, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.82rem;
        color: #c4b5fd;
    }
    .evidence-item .ev-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .evidence-item .ev-text {
        color: #9ca3af;
        font-size: 0.79rem;
        line-height: 1.45;
        max-height: 4.5em;
        overflow: hidden;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-topic { background: #7c3aed; color: white; }
    .badge-chunk { background: #2563eb; color: white; }
    .badge-checkpoint { background: #059669; color: white; }
    .score-pill {
        font-size: 0.72rem;
        color: #6b7280;
        background: rgba(255,255,255,0.05);
        padding: 0.1rem 0.4rem;
        border-radius: 10px;
    }

    /* Persona card */
    .persona-section {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    .persona-section h3 {
        color: #a78bfa;
        margin: 0 0 1rem;
        font-size: 1.1rem;
    }

    /* Trait item */
    .trait-item {
        background: rgba(67, 56, 202, 0.15);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #7c3aed;
    }
    .trait-item .trait-name {
        font-weight: 600;
        color: #c4b5fd;
    }
    .trait-item .confidence {
        color: #818cf8;
        font-size: 0.8rem;
    }
    .trait-item .evidence {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Topic card */
    .topic-card {
        background: linear-gradient(135deg, #1e1b4b, #1e293b);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border: 1px solid rgba(99, 102, 241, 0.25);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .topic-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
    }
    .topic-card .topic-id {
        color: #818cf8;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .topic-card .topic-range {
        color: #6b7280;
        font-size: 0.8rem;
    }
    .topic-card .topic-summary {
        color: #d1d5db;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1e1b4b);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c4b5fd;
    }

    /* Suggested question buttons */
    .suggested-btn {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s;
        display: inline-block;
        margin: 0.25rem;
    }
    .suggested-btn:hover {
        background: rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# State initialization
# ──────────────────────────────────────────────────────────────
def init_state():
    """Initialize Streamlit session state."""
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "config" not in st.session_state:
        st.session_state.config = None
    if "personas" not in st.session_state:
        st.session_state.personas = None
    if "topics" not in st.session_state:
        st.session_state.topics = None
    if "checkpoints" not in st.session_state:
        st.session_state.checkpoints = None


def load_system():
    """Load the ChronoBot and all data."""
    config = load_config()
    st.session_state.config = config

    # Initialize chatbot
    bot = ChronoBot(config)
    if bot.initialize():
        st.session_state.bot = bot
    else:
        st.session_state.bot = None

    # Load personas
    st.session_state.personas = load_personas(config.paths.persona_dir)

    # Load topics
    topics_path = Path(config.paths.checkpoints_dir) / "topics.json"
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            st.session_state.topics = json.load(f)

    # Load checkpoints
    cp_path = Path(config.paths.checkpoints_dir) / "checkpoints.json"
    if cp_path.exists():
        with open(cp_path, "r", encoding="utf-8") as f:
            st.session_state.checkpoints = json.load(f)


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 ChronoMind <span style='font-size:0.55em;color:#818cf8;vertical-align:middle;'>v2</span></h1>
        <p>Explainable Conversation Intelligence — Persona Drift · Intent Classification · Conflict-Aware RAG</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ System Status")

        bot = st.session_state.bot
        personas = st.session_state.personas
        topics = st.session_state.topics
        checkpoints = st.session_state.checkpoints

        if bot and bot.is_ready:
            st.success("✅ RAG System Ready")
        else:
            st.error("❌ RAG System Not Built")
            st.info("Run `python build_rag.py` to build.")

        if personas:
            st.success(f"✅ {len(personas)} Personas Loaded")
        else:
            st.warning("⚠️ No Personas Found")
            st.info("Run `python build_persona.py`")

        if topics:
            st.info(f"📋 {len(topics)} Topics Detected")
        if checkpoints:
            st.info(f"🔖 {len(checkpoints)} Checkpoints")

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

        if topics:
            total_msgs = sum(
                t.get("end_message", 0) - t.get("start_message", 0) + 1
                for t in topics
            )
            st.metric("Total Messages", f"{total_msgs:,}")
            st.metric("Topics", len(topics))

        if checkpoints:
            st.metric("Checkpoints", len(checkpoints))

        if personas:
            st.metric("Users Analyzed", len(personas))

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#6b7280;font-size:0.75rem;'>"
            "ChronoMind v1.0 · Local AI · No Paid APIs</p>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# Tab 1: Chatbot
# ──────────────────────────────────────────────────────────────
def render_chatbot_tab():
    bot = st.session_state.bot

    if not bot or not bot.is_ready:
        st.warning(
            "The RAG system is not built yet. Please run the build scripts first:\n\n"
            "```bash\npython build_rag.py\npython build_persona.py\n```"
        )
        return

    # ── Chat history ────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(
                f'<div class="chat-msg user">💬 {entry["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_bot_response(entry)

    # ── Suggested questions (only when no history) ──────────────
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        suggestions = bot.get_suggested_questions()
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            if col.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                _process_question(suggestion)
                st.rerun()

    # ── Chat input ──────────────────────────────────────────────
    question = st.chat_input("Ask a question about the conversations...")
    if question:
        _process_question(question)
        st.rerun()

    # ── Clear button ────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def _render_bot_response(entry: dict):
    """Render a v2 bot response with intent badge, conflict badge, and evidence."""
    from src.intent_classifier import INTENT_LABELS
    answer    = entry.get("content", "")
    evidence  = entry.get("evidence", [])
    intent    = entry.get("intent", {})
    conflicts = entry.get("conflicts", {})

    # ── v2 metadata badges ─────────────────────────────────────
    badge_row = []
    if intent and intent.get("intent") and intent["intent"] != "unknown":
        int_label = INTENT_LABELS.get(intent["intent"], intent["intent"])
        int_conf  = int(intent.get("confidence", 0) * 100)
        badge_row.append(
            f'<span style="background:rgba(124,58,237,0.25);border:1px solid #7c3aed;'
            f'color:#c4b5fd;padding:0.15rem 0.6rem;border-radius:20px;font-size:0.72rem;'
            f'font-weight:600;margin-right:0.4rem;">🎯 {int_label} ({int_conf}%)</span>'
        )
    if conflicts.get("has_conflicts"):
        n = conflicts["n_conflicts"]
        badge_row.append(
            f'<span style="background:rgba(239,68,68,0.2);border:1px solid #ef4444;'
            f'color:#fca5a5;padding:0.15rem 0.6rem;border-radius:20px;font-size:0.72rem;'
            f'font-weight:600;">⚠️ {n} Conflict(s) Found</span>'
        )
    elif evidence:
        badge_row.append(
            f'<span style="background:rgba(52,211,153,0.15);border:1px solid #34d399;'
            f'color:#6ee7b7;padding:0.15rem 0.6rem;border-radius:20px;font-size:0.72rem;'
            f'font-weight:600;">✅ No Conflicts</span>'
        )
    if badge_row:
        st.markdown(
            '<div style="margin-bottom:0.35rem;">' + "".join(badge_row) + "</div>",
            unsafe_allow_html=True,
        )

    # ── Primary answer card ──────────────────────────────────────
    st.markdown(
        f'<div class="answer-card">'
        f'<h4>🧠 ChronoMind v2</h4>'
        f'{answer}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if evidence:
        n_topics = sum(1 for e in evidence if e["type"] == "topic")
        n_chunks = sum(1 for e in evidence if e["type"] == "chunk")
        n_cps    = sum(1 for e in evidence if e["type"] == "checkpoint")
        has_persona = bool(st.session_state.personas)

        # ── How This Answer Was Generated ─────────────────────
        with st.expander("⚙️ How This Answer Was Generated", expanded=False):
            int_display = INTENT_LABELS.get(intent.get("intent", ""), "—") if intent else "—"
            int_conf_pct = f"{intent.get('confidence', 0):.0%}" if intent else "—"
            conflict_note = (
                f"Applied — {conflicts.get('n_conflicts', 0)} contradiction(s) flagged"
                if conflicts.get("has_conflicts") else "No conflicts found"
            )
            conflict_icon = "⚠️" if conflicts.get("has_conflicts") else "✅"
            st.markdown(
                f"""<div style='font-size:0.82rem;color:#9ca3af;line-height:2.1;'>
✅ &nbsp;<strong style='color:#c4b5fd'>Intent classified:</strong> {int_display} ({int_conf_pct})<br>
✅ &nbsp;<strong style='color:#a78bfa'>Topic summaries retrieved:</strong> {n_topics}<br>
✅ &nbsp;<strong style='color:#818cf8'>Message chunks retrieved:</strong> {n_chunks}<br>
✅ &nbsp;<strong style='color:#34d399'>Checkpoints retrieved:</strong> {n_cps}<br>
{'✅' if has_persona else '⚠️'} &nbsp;<strong style='color:#fbbf24'>Persona profile consulted:</strong> {'Yes — habits, facts, traits &amp; style' if has_persona else 'No persona data'}<br>
{conflict_icon} &nbsp;<strong style='color:#fca5a5'>Conflict resolution:</strong> {conflict_note}<br>
✅ &nbsp;<strong style='color:#c4b5fd'>Final answer synthesized from retrieved evidence</strong>
</div>""",
                unsafe_allow_html=True,
            )

        # ── Show Evidence (raw details) ───────────────────────
        ev_label = f"🔍 Show Evidence ({n_topics} topics · {n_chunks} chunks · {n_cps} checkpoints)"
        with st.expander(ev_label, expanded=False):
            for ev in evidence[:15]:
                badge_class  = f"badge-{ev['type']}"
                score_pct    = int(ev["score"] * 100)
                text_preview = ev["text"][:220].strip()
                if len(ev["text"]) > 220:
                    text_preview += "…"
                sub = f" · {ev['sub_label']}" if ev.get("sub_label") else ""
                st.markdown(
                    f'<div class="evidence-item">'
                    f'<div class="ev-header">'
                    f'<span class="badge {badge_class}">{ev["type"]}</span>'
                    f'<strong>{ev["label"]}{sub}</strong>'
                    f'<span class="score-pill">msgs {ev["messages"]} · {score_pct}% match</span>'
                    f'</div>'
                    f'<div class="ev-text">{text_preview}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _process_question(question: str):
    """Process a chatbot question and update history."""
    bot = st.session_state.bot

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    result = bot.ask(question)

    # Convert markdown bullet points for HTML display
    answer_html = _markdown_to_html(result["answer"])

    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer_html,
        "evidence":  result.get("evidence", []),
        "intent":    result.get("intent", {}),
        "conflicts": result.get("conflicts", {}),
    })


def _markdown_to_html(text: str) -> str:
    """Convert basic markdown to HTML for display inside an HTML div."""
    import re
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Horizontal rule: --- -> <hr>
    text = re.sub(r'^---$', '<hr>', text, flags=re.MULTILINE)
    # Bullet points: • or  • -> list items
    lines = text.split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("• ", "  • ", "- ")):
            if not in_list:
                html_lines.append("<ul style='margin:0.3rem 0 0.5rem 1.2rem;padding:0;'>")
                in_list = True
            item = re.sub(r'^[•\-]\s*', '', stripped)
            # Apply bold inside list items too
            html_lines.append(f"<li style='margin:0.2rem 0;color:#d1d5db;'>{item}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(f"<p style='margin:0.4rem 0;'>{stripped}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


# ──────────────────────────────────────────────────────────────
# Tab 2: Persona Viewer
# ──────────────────────────────────────────────────────────────
def render_persona_tab():
    personas = st.session_state.personas

    if not personas:
        st.warning("⚠️ No personas found. Run `python build_persona.py` first.")
        return

    # User selector
    user_names = list(personas.keys())
    selected_user = st.selectbox("👤 Select User", user_names, key="persona_user")

    if selected_user and selected_user in personas:
        persona = personas[selected_user]
        pd = persona.to_dict()

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><div class="number">{len(pd["habits"])}</div>'
                f'<div class="label">Habits</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><div class="number">{len(pd["personal_facts"])}</div>'
                f'<div class="label">Personal Facts</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><div class="number">{len(pd["personality_traits"])}</div>'
                f'<div class="label">Personality Traits</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-card"><div class="number">{len(pd["communication_style"])}</div>'
                f'<div class="label">Communication Traits</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Category sections
        categories = [
            ("🏃 Habits", "habits"),
            ("📋 Personal Facts", "personal_facts"),
            ("🧠 Personality Traits", "personality_traits"),
            ("💬 Communication Style", "communication_style"),
        ]

        for title, key in categories:
            items = pd.get(key, [])
            if items:
                st.markdown(f'<div class="persona-section"><h3>{title}</h3>', unsafe_allow_html=True)

                for item in items:
                    confidence_pct = int(item["confidence"] * 100)
                    evidence_ids   = item.get("evidence", [])
                    snippets       = item.get("snippets", [])
                    occurrences    = item.get("occurrences", 0)
                    n_evidence     = len(evidence_ids)
                    first_msg_id   = evidence_ids[0] if evidence_ids else "—"

                    # Build evidence detail line
                    ev_parts = []
                    if n_evidence:
                        ev_parts.append(f"{n_evidence} supporting messages")
                    if occurrences:
                        ev_parts.append(f"{occurrences} total occurrences")
                    if first_msg_id != "—":
                        ev_parts.append(f"first seen at msg #{first_msg_id}")
                    ev_line = " · ".join(ev_parts) if ev_parts else "no direct evidence"

                    st.markdown(
                        f'<div class="trait-item">'
                        f'<span class="trait-name">{item["trait"]}</span>'
                        f'<span class="confidence"> · Confidence: {confidence_pct}%</span>'
                        f'<div class="evidence">📎 {ev_line}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    # Show evidence snippets for habits
                    if snippets and key == "habits":
                        for snip in snippets[:3]:
                            st.markdown(
                                f'<div style="margin:-0.25rem 0 0.4rem 1rem;'
                                f'font-size:0.78rem;color:#818cf8;'
                                f'font-style:italic;border-left:2px solid #4338ca;'
                                f'padding-left:0.5rem;">"{snip}"</div>',
                                unsafe_allow_html=True,
                            )

                st.markdown('</div>', unsafe_allow_html=True)

        # Confidence chart
        st.markdown("#### 📊 Confidence Distribution")
        all_traits = []
        for cat_key in ["habits", "personal_facts", "personality_traits", "communication_style"]:
            for item in pd.get(cat_key, []):
                all_traits.append({
                    "Trait": item["trait"][:30],
                    "Confidence": item["confidence"],
                    "Category": cat_key.replace("_", " ").title(),
                })

        if all_traits:
            import pandas as dflib
            df = dflib.DataFrame(all_traits)
            fig = px.bar(
                df, x="Confidence", y="Trait", color="Category",
                orientation="h",
                color_discrete_sequence=["#7c3aed", "#2563eb", "#059669", "#d97706"],
                template="plotly_dark",
            )
            fig.update_layout(
                height=max(300, len(all_traits) * 35),
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw JSON viewer
        with st.expander("🔍 View Raw Persona JSON"):
            st.json(pd)


# ──────────────────────────────────────────────────────────────
# Tab 3: Topic Explorer
# ──────────────────────────────────────────────────────────────
def render_topics_tab():
    topics = st.session_state.topics

    if not topics:
        st.warning("⚠️ No topics found. Run `python build_rag.py` first.")
        return

    st.markdown(f"**{len(topics)} topics detected** across {len(set(t.get('conversation_id', 0) for t in topics))} conversations.")

    # Topic size distribution histogram
    st.markdown("#### 📈 Topic Size Distribution")
    import pandas as dflib
    sizes = [t.get("end_message", 0) - t.get("start_message", 0) + 1 for t in topics]
    df_sizes = dflib.DataFrame({"Messages per Topic": sizes})

    fig = px.histogram(
        df_sizes, x="Messages per Topic",
        nbins=30,
        color_discrete_sequence=["#7c3aed"],
        template="plotly_dark",
    )
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Messages/Topic", f"{sum(sizes)/len(sizes):.1f}")
    col2.metric("Max Topic Size", max(sizes))
    col3.metric("Min Topic Size", min(sizes))

    # Search / filter
    search = st.text_input("🔍 Search topics by keyword", key="topic_search")

    # Filter topics
    filtered = topics
    if search:
        filtered = [
            t for t in topics
            if search.lower() in t.get("summary", "").lower()
            or search.lower() in t.get("label", "").lower()
        ]
        st.info(f"Found {len(filtered)} topics matching '{search}'.")

    # Pagination
    page_size = 25
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="topic_page")
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered))
    st.caption(f"Showing topics {start_idx + 1}–{end_idx} of {len(filtered)} (page {page}/{total_pages})")

    # Topic cards (paginated)
    for t in filtered[start_idx:end_idx]:
        summary = t.get("summary", "No summary available")
        label = t.get("label", f"Topic {t['topic_id']}")
        msg_range = f"Messages {t['start_message']} – {t['end_message']}"
        msg_count = t.get("message_count", t['end_message'] - t['start_message'] + 1)
        conv_id = t.get("conversation_id", "?")

        st.markdown(
            f'<div class="topic-card">'
            f'<span class="topic-id">📌 Topic {t["topic_id"]}: {label}</span>'
            f'<span class="topic-range"> · Conv #{conv_id} · {msg_range} ({msg_count} msgs)</span>'
            f'<div class="topic-summary">{summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# Tab 4: Checkpoints Explorer
# ──────────────────────────────────────────────────────────────
def render_checkpoints_tab():
    checkpoints = st.session_state.checkpoints

    if not checkpoints:
        st.warning("⚠️ No checkpoints found. Run `python build_rag.py` first.")
        return

    st.markdown(f"**{len(checkpoints)} checkpoints** (conversation-aware, up to 100 messages each).")

    # Size distribution
    st.markdown("#### 📅 Checkpoint Size Distribution")
    import pandas as dflib
    sizes = [cp["end_message"] - cp["start_message"] + 1 for cp in checkpoints]
    df_sizes = dflib.DataFrame({"Messages per Checkpoint": sizes})

    fig = px.histogram(
        df_sizes, x="Messages per Checkpoint",
        nbins=30,
        color_discrete_sequence=["#059669"],
        template="plotly_dark",
    )
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Avg Checkpoint Size", f"{sum(sizes)/len(sizes):.1f} messages")
    col2.metric("Total Checkpoints", len(checkpoints))

    # Paginated checkpoint details
    st.markdown("#### 📝 Checkpoint Summaries")
    page_size = 25
    total_pages = max(1, (len(checkpoints) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="cp_page")
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(checkpoints))
    st.caption(f"Showing checkpoints {start_idx + 1}–{end_idx} of {len(checkpoints)} (page {page}/{total_pages})")

    for cp in checkpoints[start_idx:end_idx]:
        msg_range = f"Messages {cp['start_message']} – {cp['end_message']}"
        size = cp['end_message'] - cp['start_message'] + 1
        with st.expander(f"🔖 Checkpoint {cp['checkpoint_id']} · {msg_range} ({size} msgs)", expanded=False):
            st.markdown(cp.get("summary", "No summary available."))


# ──────────────────────────────────────────────────────────────
# Dataset Statistics Card
# ──────────────────────────────────────────────────────────────
def render_dataset_stats():
    """Professional 6-column metrics dashboard shown below the header."""
    topics      = st.session_state.topics or []
    checkpoints = st.session_state.checkpoints or []
    personas    = st.session_state.personas or {}

    if not topics and not checkpoints:
        return

    total_msgs   = sum(
        t.get("end_message", 0) - t.get("start_message", 0) + 1
        for t in topics
    ) if topics else 0
    total_convos = len(set(t.get("conversation_id", 0) for t in topics)) if topics else 0
    avg_per_conv = round(total_msgs / total_convos, 1) if total_convos else 0

    st.markdown("")
    cols = st.columns(6)
    stats = [
        (f"{total_convos:,}",    "Total Conversations"),
        (f"{total_msgs:,}",      "Total Messages"),
        (f"{avg_per_conv}",      "Avg Msgs / Conv"),
        (f"{len(topics):,}",     "Topics Detected"),
        (f"{len(checkpoints):,}","Checkpoints Created"),
        (f"{len(personas)}",     "Users Analyzed"),
    ]
    for col, (num, lbl) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="number">{num}</div>'
                f'<div class="label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")


# ──────────────────────────────────────────────────────────────
# Tab 5: Methodology
# ──────────────────────────────────────────────────────────────
def render_methodology_tab():
    """Explain the system design, architecture, and engineering decisions."""

    st.markdown("## 📖 System Methodology")
    st.markdown(
        "ChronoMind is an **Explainable Conversation Intelligence System** that applies "
        "retrieval-augmented generation (RAG) over multi-level conversation indexes to "
        "answer questions about conversation participants — without any paid APIs or external LLMs."
    )
    st.markdown("---")

    # ── Pipeline diagram ──────────────────────────────────────
    st.markdown("### 🔄 Processing Pipeline")
    st.markdown(
        """<div style='font-family:monospace;background:rgba(15,12,41,0.85);
        padding:1.5rem 2rem;border-radius:12px;
        border:1px solid rgba(99,102,241,0.3);line-height:2.4;
        font-size:0.88rem;color:#c4b5fd;'>
<span style='color:#a78bfa;font-weight:700;'>CSV Conversations</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Message Parser</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(chronological ordering · user label extraction · deduplication)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Embedding Generation</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(all-MiniLM-L6-v2 · 384-dimensional dense vectors · CPU-only)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Topic Detection</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(cosine similarity drift · sliding centroid · min 10 msgs per segment)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↙&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘<br>
<span style='color:#7c3aed;font-weight:700;'>Topic Summaries</span>
&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#2563eb;font-weight:700;'>Raw Msg Chunks</span>
&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#059669;font-weight:700;'>100-Msg Checkpoints</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↙<br>
<span style='color:#a78bfa;font-weight:700;'>FAISS Vector Indexes</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(3 separate flat-L2 indexes · exact search · no approximation error)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Persona Extraction</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(per-user · strict first-person patterns · frequency promotion gates)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>RAG Retrieval</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(query embedding → parallel index search → intent classification)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#a78bfa;font-weight:700;'>Answer Synthesis</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.78rem;'>(9-category intent router → natural language synthesis → evidence packaging)</span>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("### 🔍 Topic Detection Algorithm")
        st.markdown("""
Implements **embedding-based semantic drift detection** — a topic boundary is detected when the
cosine similarity between a new message embedding and the running topic centroid drops below a
configurable threshold.

**Steps:**
1. Encode each message with `all-MiniLM-L6-v2`
2. Maintain a sliding-window centroid of the last N embeddings
3. Compute cosine similarity between the new message and the centroid
4. If similarity < **0.35** (configurable), commit the current segment and start a new topic
5. Merge trailing micro-segments into the previous topic if they contain fewer than **10 messages**
6. Summarize each topic using **Sumy LSA** extractive summarization

**Why not fixed windows?** Fixed-window segmentation ignores semantics.
The same topic can span 200 messages or end after 5. Cosine drift detection
adapts to the actual conversation structure.
        """)

        st.markdown("### 🔖 Dual Checkpoint Strategy")
        st.markdown("""
**Topic Checkpoints** follow *semantic* boundaries — they capture *what* was discussed.

**100-Message Checkpoints** follow *temporal* boundaries — they capture *when* it was discussed.

**Why are both needed?**

Consider a user who mentions their location in three different conversations spread across
300 messages. No single topic summary captures all three mentions. But a 100-message temporal
checkpoint, which spans multiple topics, will capture them within its temporal window.

Combined retrieval returns significantly richer evidence than either strategy alone —
especially for facts that recur across unrelated topics.
        """)

    with col_r:
        st.markdown("### 🧠 Persona Extraction")
        st.markdown("""
Extracts **four structured persona categories** using frequency-gated, first-person regex patterns:

| Category | Method |
|---|---|
| **Habits** | 52 strict first-person patterns (e.g., `I go to the gym`, `I love to cook`) |
| **Personal Facts** | Named-entity-style capture groups (location, occupation, pets, family) |
| **Personality Traits** | Keyword frequency analysis across conversation corpus |
| **Communication Style** | Statistical analysis (message length, emoji ratio, question rate) |

**Promotion gates:** A habit must pass *both*:
- `min_evidence_convos` ≥ 2 distinct conversations
- `min_occurrences` ≥ N total pattern matches (threshold varies by habit type)

**Confidence formula:**
```
confidence = min(1.0, 0.30 + ln(1 + n_convos) × 0.10)
```
3 convos → 0.41 · 10 → 0.53 · 50 → 0.69 · 500 → 0.92
        """)

        st.markdown("### 📦 Technology Choices")
        st.markdown("""
| Component | Choice | Rationale |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim, ~80MB, <1ms/msg on CPU, strong semantic quality |
| **Vector DB** | FAISS (flat L2) | Exact nearest-neighbour search, no approximation, no server required |
| **Summarization** | Sumy LSA | Extractive, fully deterministic, zero LLM dependency |
| **UI** | Streamlit | ML-native dashboard tooling, Python-native components |
| **APIs** | None | Fully local — no OpenAI, no costs, no rate limits |
        """)

    st.markdown("---")

    st.markdown("### 🔄 Retrieval Pipeline Detail")
    st.markdown("""
When a question is submitted:

1. **Encode** — The query is embedded using the same `all-MiniLM-L6-v2` model (ensures embedding space consistency)
2. **Search all three FAISS indexes in parallel:**
   - `chunks` index (16,674 vectors) — 20-message sliding windows with 5-message overlap
   - `topics` index (15,378 vectors) — one vector per detected topic, embedding is the topic centroid
   - `checkpoints` index (13,424 vectors) — 100-message temporal windows
3. **Intent classification** — The query text is classified into one of 9 intent categories:
   `persona · habits · communication · occupation · location · interests · relationship · topics · general`
4. **Answer synthesis** — A category-specific synthesizer generates a natural-language response
   from the structured evidence (no raw context dump)
5. **Evidence packaging** — All retrieved items are returned as structured objects for the
   "Show Evidence" and "How This Answer Was Generated" expanders
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#6b7280;font-size:0.8rem;'>"
        "ChronoMind · Explainable Conversation Intelligence · "
        "FAISS + SentenceTransformers + Sumy + Streamlit · No external APIs"
        "</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    init_state()

    # Load system on first run
    if st.session_state.config is None:
        with st.spinner("🔄 Loading ChronoMind v2..."):
            load_system()

    render_header()
    render_sidebar()
    render_dataset_stats()

    # ── All Tabs (v1 preserved + v2 added) ─────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🤖 Chatbot",
        "👤 Persona Viewer",
        "📋 Topic Explorer",
        "🔖 Checkpoints Explorer",
        "📖 Methodology",
        "🧠 Persona Drift",
        "🎯 Intent Classifier",
        "⚖️ Conflict Resolver",
        "🏗 Sync Architecture",
    ])

    with tab1:
        render_chatbot_tab()

    with tab2:
        render_persona_tab()

    with tab3:
        render_topics_tab()

    with tab4:
        render_checkpoints_tab()

    with tab5:
        render_methodology_tab()

    with tab6:
        render_persona_drift_tab()

    with tab7:
        render_intent_tab()

    with tab8:
        render_conflict_tab()

    with tab9:
        render_sync_architecture_tab()


if __name__ == "__main__":
    main()
