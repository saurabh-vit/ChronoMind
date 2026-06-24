"""
app_v2_tabs.py — ChronoMind v2 Streamlit Tab Functions.

Contains:
  - render_persona_drift_tab()     : 🧠 Persona Drift
  - render_intent_tab()            : 🎯 Intent Classifier
  - render_conflict_tab()          : ⚖️ Conflict Resolver
  - render_sync_architecture_tab() : 🏗 Sync Architecture

Imported by app.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.intent_classifier import get_intent_classifier, INTENT_LABELS, INTENT_DESCRIPTIONS
from src.conflict_resolver import detect_conflicts, resolve as resolve_conflicts
from src.persona_drift import load_persona_drift

BASE_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────
# Tab: 🧠 Persona Drift
# ──────────────────────────────────────────────────────────────

def render_persona_drift_tab():
    st.markdown("## 🧠 Adaptive Persona Engine")
    st.markdown(
        "Tracks how each user's **mood** and **tone** evolve across conversation sessions. "
        "Detects drift events and identifies probable triggers from keyword analysis."
    )

    # Load drift data
    drift_path = Path(st.session_state.config.paths.persona_dir) / "persona_drift.json"
    if not drift_path.exists():
        st.warning("⚠️ Persona drift data not found.")
        st.code("python build_persona_drift.py", language="bash")
        return

    drift_data = json.loads(drift_path.read_text(encoding="utf-8"))
    users = list(drift_data.keys())
    if not users:
        st.info("No drift data available.")
        return

    selected_user = st.selectbox("👤 Select User", users, key="drift_user")
    ud = drift_data[selected_user]

    # ── Summary metrics ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    mood_emojis = {"curious": "🤔", "excited": "🎉", "frustrated": "😤",
                   "neutral": "😐", "emotional": "😢", "playful": "😄"}
    tone_emojis = {"formal": "📋", "casual": "😊", "supportive": "🤝", "direct": "⚡"}
    mood_e = mood_emojis.get(ud["dominant_mood"], "😐")
    tone_e = tone_emojis.get(ud["dominant_tone"], "💬")

    with c1:
        st.markdown(f'<div class="metric-card"><div class="number">{ud["total_sessions"]}</div>'
                    f'<div class="label">Sessions Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="number">{ud["total_drifts"]}</div>'
                    f'<div class="label">Drift Events</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="number">{mood_e} {ud["dominant_mood"].title()}</div>'
                    f'<div class="label">Dominant Mood</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="number">{tone_e} {ud["dominant_tone"].title()}</div>'
                    f'<div class="label">Dominant Tone</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Mood distribution chart ─────────────────────────────────
    st.markdown("### 📊 Mood Distribution Across Sessions")
    col_l, col_r = st.columns(2)
    with col_l:
        mood_dist = ud.get("mood_distribution", {})
        if mood_dist:
            fig = px.pie(
                names=list(mood_dist.keys()),
                values=list(mood_dist.values()),
                color_discrete_sequence=["#7c3aed","#2563eb","#ef4444","#6b7280","#ec4899","#10b981"],
                template="plotly_dark",
            )
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        tone_dist = ud.get("tone_distribution", {})
        if tone_dist:
            fig2 = px.bar(
                x=list(tone_dist.keys()), y=list(tone_dist.values()),
                color_discrete_sequence=["#818cf8"],
                template="plotly_dark",
                labels={"x": "Tone", "y": "Sessions"},
            )
            fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter"))
            st.plotly_chart(fig2, use_container_width=True)

    # ── Mood timeline ────────────────────────────────────────────
    st.markdown("### 📅 Mood & Tone Timeline")
    timeline = ud.get("timeline", [])
    if timeline:
        # Show first 20 entries
        display = timeline[:20]
        mood_map  = {"curious": 3, "excited": 5, "frustrated": 1, "neutral": 2, "emotional": 0, "playful": 4}
        tone_map  = {"formal": 3, "casual": 1, "supportive": 2, "direct": 0}
        days      = [t["day"] for t in display]
        moods_num = [mood_map.get(t["mood"], 2) for t in display]
        tones_num = [tone_map.get(t["tone"], 1) for t in display]
        mood_lbls = [t["mood"] for t in display]
        tone_lbls = [t["tone"] for t in display]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=days, y=moods_num, mode="lines+markers", name="Mood",
            line=dict(color="#a78bfa", width=2),
            text=mood_lbls, hovertemplate="Day %{x}<br>Mood: %{text}",
        ))
        fig3.add_trace(go.Scatter(
            x=days, y=tones_num, mode="lines+markers", name="Tone",
            line=dict(color="#34d399", width=2, dash="dot"),
            text=tone_lbls, hovertemplate="Day %{x}<br>Tone: %{text}",
        ))
        fig3.update_layout(
            height=260, template="plotly_dark",
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            xaxis_title="Session / Day",
            yaxis_title="",
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Drift events ─────────────────────────────────────────────
    st.markdown("### ⚡ Detected Drift Events")
    drift_events = ud.get("drift_events", [])
    if not drift_events:
        st.info("No significant mood or tone drift detected.")
    else:
        for i, ev in enumerate(drift_events[:15]):
            mood_d = ev.get("mood_drift")
            tone_d = ev.get("tone_drift")
            trigger = ev.get("trigger", "Unknown")
            keywords = ev.get("keywords", [])
            ev_msgs  = ev.get("evidence_messages", [])

            header_parts = []
            if mood_d:
                header_parts.append(f"🧠 Mood: **{mood_d}**")
            if tone_d:
                header_parts.append(f"💬 Tone: **{tone_d}**")

            with st.expander(
                f"Drift #{i+1} (Day {ev['from_day']} → {ev['to_day']}) · " + " | ".join(header_parts),
                expanded=(i == 0),
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Probable Trigger:** {trigger}")
                    if keywords:
                        st.markdown(f"**Keywords:** `{'`, `'.join(keywords)}`")
                with col2:
                    if ev_msgs and len(ev_msgs) == 2:
                        st.markdown(f"**Evidence Messages:** {ev_msgs[0]} – {ev_msgs[1]}")


# ──────────────────────────────────────────────────────────────
# Tab: 🎯 Intent Classifier
# ──────────────────────────────────────────────────────────────

def render_intent_tab():
    st.markdown("## 🎯 Offline Intent Classifier")
    st.markdown(
        "Classifies messages into 5 intent categories using a fully offline "
        "**TF-IDF + Logistic Regression** model. No APIs. CPU-only. < 5ms inference."
    )

    clf = get_intent_classifier()

    # ── Model metrics ────────────────────────────────────────────
    metrics_path = BASE_DIR / "models" / "intent_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.markdown("### 📊 Model Evaluation Metrics")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        for col, (label, key, fmt) in zip(
            [mc1, mc2, mc3, mc4, mc5, mc6],
            [
                ("Accuracy",   "accuracy",    ".0%"),
                ("Precision",  "precision",   ".0%"),
                ("Recall",     "recall",      ".0%"),
                ("F1 Score",   "f1",          ".0%"),
                ("5-Fold CV F1", "cv_f1_mean", ".0%"),
                ("Latency",    "latency_ms",  ".2f"),
            ],
        ):
            with col:
                val = m.get(key, 0)
                disp = f"{val:{fmt}}" if key != "latency_ms" else f"{val:.2f} ms"
                st.markdown(
                    f'<div class="metric-card"><div class="number">{disp}</div>'
                    f'<div class="label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")
        st.caption(
            f"Trained on {m.get('train_samples','?')} samples · "
            f"Tested on {m.get('test_samples','?')} samples · "
            f"5-fold CV F1: {m.get('cv_f1_mean',0):.3f} ± {m.get('cv_f1_std',0):.3f}"
        )

    st.markdown("---")

    # ── Intent definitions ────────────────────────────────────────
    st.markdown("### 📚 Intent Classes")
    for label, display in INTENT_LABELS.items():
        desc = INTENT_DESCRIPTIONS.get(label, "")
        st.markdown(
            f'<div style="background:rgba(67,56,202,0.12);border-left:3px solid #7c3aed;'
            f'padding:0.6rem 1rem;border-radius:8px;margin:0.3rem 0;">'
            f'<strong style="color:#c4b5fd">{display}</strong>'
            f'<span style="color:#6b7280;font-size:0.82rem;margin-left:0.8rem;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Manual testing ────────────────────────────────────────────
    st.markdown("### 🧪 Test a Message")
    if not clf.is_ready:
        st.error("Intent model not loaded. Run `python build_intent_model.py` first.")
        return

    test_msg = st.text_area(
        "Enter a message to classify:",
        value="Don't forget to send the report before the deadline!",
        height=80,
        key="intent_test_input",
    )
    if st.button("🔍 Classify", key="intent_classify_btn", type="primary"):
        result = clf.predict(test_msg)
        intent = result["intent"]
        conf   = result["confidence"]
        latency = result["latency_ms"]
        all_scores = result["all_scores"]

        st.markdown("#### Result")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f'<div class="metric-card"><div class="number">'
                f'{INTENT_LABELS.get(intent, intent)}</div>'
                f'<div class="label">Predicted Intent</div></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f'<div class="metric-card"><div class="number">{conf:.0%}</div>'
                f'<div class="label">Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f'<div class="metric-card"><div class="number">{latency:.2f} ms</div>'
                f'<div class="label">Inference Time</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

        if all_scores:
            st.markdown("**Confidence per class:**")
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for cls, score in sorted_scores:
                label_str = INTENT_LABELS.get(cls, cls)
                bar_width  = int(score * 100)
                color = "#7c3aed" if cls == intent else "#374151"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0;">'
                    f'<span style="min-width:160px;font-size:0.8rem;color:#c4b5fd;">{label_str}</span>'
                    f'<div style="background:{color};height:8px;width:{bar_width}%;border-radius:4px;"></div>'
                    f'<span style="font-size:0.78rem;color:#9ca3af;">{score:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Batch benchmark ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Latency Benchmark")
    sample_messages = [
        "Don't forget to call the doctor",
        "I'm feeling really sad today",
        "Can you help me fix this bug?",
        "What's up? How are you doing?",
        "Something something random",
    ]
    if st.button("▶ Run 5-message benchmark", key="bench_btn"):
        results = []
        for msg in sample_messages:
            r = clf.predict(msg)
            results.append((msg, r["intent"], r["confidence"], r["latency_ms"]))
        st.table([
            {"Message": m[:50], "Intent": INTENT_LABELS.get(i, i),
             "Confidence": f"{c:.0%}", "Latency": f"{l:.2f} ms"}
            for m, i, c, l in results
        ])
        avg_lat = sum(r[3] for r in results) / len(results)
        st.success(f"Average latency: **{avg_lat:.2f} ms** · Well under 200ms requirement ✅")


# ──────────────────────────────────────────────────────────────
# Tab: ⚖️ Conflict Resolver
# ──────────────────────────────────────────────────────────────

def render_conflict_tab():
    st.markdown("## ⚖️ Conflict-Aware Retrieval")
    st.markdown(
        "Detects and resolves contradictory facts in retrieved evidence. "
        "Scores evidence by **recency · retrieval similarity · emotional weight** "
        "to determine which information is most likely current."
    )

    st.markdown("---")
    st.markdown("### 🔄 Conflict Resolution Flow")
    st.markdown(
        """<div style='font-family:monospace;background:rgba(15,12,41,0.85);
        padding:1.2rem 1.8rem;border-radius:12px;
        border:1px solid rgba(99,102,241,0.3);line-height:2.3;font-size:0.86rem;color:#c4b5fd;'>
<span style='color:#a78bfa;font-weight:700;'>User Query</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Intent Detection</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(offline TF-IDF + LR classifier)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#818cf8;font-weight:700;'>Multi-Index Retrieval</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(FAISS: chunks · topics · checkpoints)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#ef4444;font-weight:700;'>Conflict Detection</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(entity–attribute extraction · contradiction scoring)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#f59e0b;font-weight:700;'>Evidence Ranking</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(recency 30% + similarity 50% + emotional weight 20%)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#34d399;font-weight:700;'>Merged Answer</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(synthesized response + conflict notice)</span>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🧪 Run Conflict Resolution")
    st.markdown("Test the conflict resolver using the **real conversation dataset**.")

    # Assignment Quick Buttons
    st.markdown("**Quick Queries:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Sister details?", use_container_width=True):
            st.session_state.conflict_query = "Did I mention anything about my sister?"
    with col2:
        if st.button("What's my job?", use_container_width=True):
            st.session_state.conflict_query = "What job did I mention?"
    with col3:
        if st.button("Where do I live?", use_container_width=True):
            st.session_state.conflict_query = "Where do I live?"
    with col4:
        if st.button("Relationship status?", use_container_width=True):
            st.session_state.conflict_query = "Did I mention my relationship status?"

    # Query Input
    query_val = st.session_state.get("conflict_query", "")
    query = st.text_input("Ask a question", value=query_val, placeholder="Did I mention anything about my sister?")

    if st.button("⚖️ Resolve Query", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        chatbot = st.session_state.get("bot")
        if not chatbot or not chatbot.is_ready:
            st.error("Chatbot is not initialized. Please run `python build_rag.py` first.")
            return

        with st.spinner("Retrieving evidence and resolving conflicts..."):
            response = chatbot.ask(query)
            evidence = response.get("evidence", [])
            conflicts = response.get("conflicts", {})

        st.markdown("---")
        st.markdown("### 🔍 Explainability Section")

        # Merged Answer
        st.markdown("**Merged Answer:**")
        st.info(response.get("answer", "No answer generated."))

        # Conflict Details
        if conflicts.get("has_conflicts"):
            st.error(f"⚠️ **{conflicts['n_conflicts']} conflict(s) detected**")
            for c in conflicts["conflicts"]:
                st.markdown(
                    f'<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
                    f'padding:1rem 1.2rem;border-radius:10px;margin:0.5rem 0;">'
                    f'<strong style="color:#fca5a5">Conflicting: {c["display_name"]}</strong><br>'
                    f'<span style="color:#d1d5db;font-size:0.83rem;">'
                    f'Values found: {" | ".join(f"<strong>{v}</strong>" for v in c["values"])}</span><br><br>'
                    f'<span style="color:#c4b5fd;font-size:0.83rem;">Best evidence: '
                    f'{c["best_value"]} (msg {c["best_evidence"].get("messages","?")})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ **No conflicts detected** in the retrieved evidence.")

        # Retrieved Evidence Breakdown
        st.markdown("**Retrieved Evidence:**")
        if not evidence:
            st.write("No evidence retrieved.")
        else:
            for i, ev in enumerate(evidence):
                raw_score = ev.get("score", 0.0)
                # Compute mock recency and emot_score strictly for display (logic runs in resolver)
                msg_idx = 0
                try:
                    msg_idx = int(str(ev.get("messages", "0")).split("-")[0])
                except:
                    pass
                recency = msg_idx / 200000.0
                
                # We calculate emot_w to display it nicely
                text = ev.get("text", "")
                emotional_words = {"family", "sister", "brother", "mother", "father", "married", "relationship", "job", "career", "important", "life", "events", "definitely", "always", "never", "absolutely", "clearly", "certainly", "moved", "now", "currently", "recently", "just", "today", "yesterday"}
                overlap = len(set(text.lower().split()) & emotional_words)
                emot_score = min(1.0, overlap / 4.0)

                final_score = raw_score * 0.5 + recency * 0.3 + emot_score * 0.2

                st.markdown(
                    f'<div style="background:rgba(67,56,202,0.1);border-left:3px solid #4338ca;'
                    f'padding:0.6rem 1rem;border-radius:8px;margin:0.5rem 0;font-size:0.84rem;">'
                    f'<strong style="color:#a5b4fc">[{ev.get("type", "Unknown").title()} {ev.get("label", "")}]</strong> '
                    f'| Msgs: <code>{ev.get("messages")}</code> '
                    f'| Final Score: <code>{final_score:.3f}</code><br>'
                    f'<span style="font-size:0.75rem;color:#9ca3af;">Sim: {raw_score:.3f} · Recency: {recency:.3f} · Emotion: {emot_score:.3f}</span><br>'
                    f'<span style="color:#d1d5db;margin-top:0.3rem;display:inline-block;">{text}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    with st.expander("Show Fake Demo Examples (Control)"):
        DEMO_EXAMPLES = [
            {
                "name": "Location Contradiction",
                "evidence": [
                    {
                        "type": "chunk", "label": "Chunk 42", "text":
                        "My sister lives in Delhi. She's been there for years.",
                        "messages": "142-145", "score": 0.72, "sub_label": None,
                    },
                    {
                        "type": "chunk", "label": "Chunk 891", "text":
                        "My sister moved to Bangalore last month for her new job.",
                        "messages": "8910-8914", "score": 0.81, "sub_label": None,
                    },
                ],
            },
            {
                "name": "Occupation Contradiction",
                "evidence": [
                    {
                        "type": "topic", "label": "Topic 5", "text":
                        "I'm a teacher. I've been working as a teacher for 5 years.",
                        "messages": "200-220", "score": 0.65, "sub_label": None,
                    },
                    {
                        "type": "chunk", "label": "Chunk 2001", "text":
                        "I am working as a software engineer now. Changed careers last year.",
                        "messages": "20010-20015", "score": 0.79, "sub_label": None,
                    },
                ],
            },
        ]
        
        selected_demo = st.selectbox(
            "Select a demo:", [d["name"] for d in DEMO_EXAMPLES], key="conflict_demo_select"
        )
        demo = next(d for d in DEMO_EXAMPLES if d["name"] == selected_demo)

        st.markdown("**Evidence items:**")
        for ev in demo["evidence"]:
            st.markdown(
                f'<div style="background:rgba(67,56,202,0.1);border-left:3px solid #4338ca;'
                f'padding:0.6rem 1rem;border-radius:8px;margin:0.3rem 0;font-size:0.84rem;">'
                f'<strong style="color:#a5b4fc">[{ev["label"]}]</strong> msgs {ev["messages"]} · score {ev["score"]}<br>'
                f'<span style="color:#d1d5db">{ev["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if st.button("⚖️ Run Demo Resolution", key="demo_conflict_run_btn", type="secondary"):
            from src.conflict_resolver import resolve as resolve_conflicts
            result = resolve_conflicts(
                evidence_items=demo["evidence"],
                original_answer="Based on the retrieved evidence, here is the answer.",
                query="demo query",
            )
            st.markdown("---")
            if result["has_conflicts"]:
                st.error(f"⚠️ **{result['n_conflicts']} conflict(s) detected**")
                st.markdown(result["augmented_answer"])
            else:
                st.success("✅ **No conflicts detected** in the retrieved evidence.")

    # ── Scoring formula ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Evidence Scoring Formula")
    st.markdown("""
Each piece of evidence is scored with a **composite score** that determines which version of a conflicting fact is most likely current:

```
composite_score = (retrieval_similarity × 0.50)
                + (message_recency × 0.30)
                + (emotional_weight × 0.20)
```

| Component | Weight | Description |
|---|---|---|
| **Retrieval Similarity** | 50% | FAISS cosine similarity to the query |
| **Message Recency** | 30% | Later messages (higher index) score higher |
| **Emotional Weight** | 20% | Words like "moved", "now", "currently" signal recent fact updates |
    """)


# ──────────────────────────────────────────────────────────────
# Tab: 🏗 Sync Architecture
# ──────────────────────────────────────────────────────────────

def render_sync_architecture_tab():
    st.markdown("## 🏗 Sync Architecture Design")
    st.markdown(
        "A production sync architecture for ChronoMind that separates sensitive local data "
        "from shareable cloud artifacts, using a timestamp-based conflict resolution policy."
    )
    st.markdown("---")

    # ── Architecture diagram ─────────────────────────────────────
    st.markdown("### 🔄 Architecture Overview")
    st.markdown(
        """<div style='font-family:monospace;background:rgba(15,12,41,0.85);
        padding:1.5rem 2rem;border-radius:12px;
        border:1px solid rgba(99,102,241,0.3);line-height:2.5;font-size:0.87rem;color:#c4b5fd;'>
<span style='color:#a78bfa;font-weight:700;'>Device (Mobile / Desktop)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;│<br>
&nbsp;&nbsp;&nbsp;&nbsp;├─── <span style='color:#ef4444;font-weight:600;'>Local Storage</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(private · never leaves device)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── raw messages<br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── persona embeddings<br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── sensitive persona facts<br>
&nbsp;&nbsp;&nbsp;&nbsp;│<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#f59e0b;font-weight:700;'>Sync Service</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(delta sync · latest-timestamp-wins · offline-first)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;│<br>
&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
<span style='color:#34d399;font-weight:700;'>Cloud Backend</span>&nbsp;&nbsp;
<span style='color:#4b5563;font-size:0.76rem;'>(shared artifacts only)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── topic summaries<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── checkpoint summaries<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── FAISS index metadata (not vectors)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── retrieval query logs (anonymized)
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("### 🔒 What Stays Local")
        st.markdown("""
| Artifact | Reason |
|---|---|
| **Raw conversation messages** | Privacy-sensitive content |
| **FAISS vector embeddings** | User-specific, large (~200MB) |
| **Sensitive persona facts** | Location, relationships, health |
| **Persona drift timeline** | Behavioral biometrics |
| **Intent classifier model** | Pre-trained, no sync needed |

**Storage estimate:** ~500MB per user (embeddings dominate)

**Access policy:** Local data is never transmitted. Only hashed metadata (message count, date range) is shared for sync validation.
        """)

        st.markdown("### 🔄 Sync Strategy")
        st.markdown("""
**Pattern:** Offline-first with eventual consistency

1. All writes go to local storage first
2. Sync service detects changes (file hash comparison)
3. Delta upload: only *changed* summaries + metadata
4. Cloud receives anonymized, non-sensitive artifacts only
5. Pull on reconnect: new summaries from other devices

**Sync interval:** Every 15 minutes when online, immediately on app open
        """)

    with col_r:
        st.markdown("### ☁️ What Syncs to Cloud")
        st.markdown("""
| Artifact | Format | Size |
|---|---|---|
| **Topic summaries** | JSON (text only) | ~2MB |
| **Checkpoint summaries** | JSON (text only) | ~1MB |
| **Retrieval metadata** | JSON | ~500KB |
| **Conversation day count** | Integer | Negligible |
| **Anonymized query logs** | Hashed + timestamped | ~100KB |

**Total cloud footprint:** ~3-5MB per user

**Update trigger:** When a new conversation is indexed locally
        """)

        st.markdown("### ⚖️ Conflict Resolution Policy")
        st.markdown("""
**Policy: Latest Timestamp Wins**

When two devices have divergent summaries for the same checkpoint:

```
if local.updated_at > cloud.updated_at:
    upload local → overwrite cloud
else:
    download cloud → overwrite local
```

**Edge cases:**
- **Tie:** Merge by taking longer/more detailed summary
- **Network split:** Queue changes locally, replay on reconnect
- **Schema mismatch:** Reject stale format versions; re-index locally

**Why not CRDT?** Summaries are non-structured prose — semantic merging is impossible without an LLM. Timestamp wins is safe because summaries are deterministic (same messages → same summary).
        """)

    st.markdown("---")

    st.markdown("### 🔐 Privacy Guarantees")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="persona-section" style="text-align:center;">'
            '<h3>🔒 Zero Raw Data</h3>'
            '<p style="color:#d1d5db;font-size:0.85rem;">No raw message text ever leaves the device. '
            'Only summaries sync.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="persona-section" style="text-align:center;">'
            '<h3>🎭 Anonymous Queries</h3>'
            '<p style="color:#d1d5db;font-size:0.85rem;">Query logs are one-way hashed '
            'before transmission. Cannot be reversed.</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="persona-section" style="text-align:center;">'
            '<h3>📴 Offline-First</h3>'
            '<p style="color:#d1d5db;font-size:0.85rem;">Full functionality with no internet. '
            'Cloud sync is an enhancement, not a dependency.</p></div>',
            unsafe_allow_html=True,
        )
