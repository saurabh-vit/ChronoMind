# 🧠 ChronoMind

**Explainable Conversation Intelligence System**

ChronoMind is a production-ready system that analyzes chronological conversation data to detect topics, extract user personas, and answer natural-language questions using RAG — all running locally with no paid APIs or external LLMs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Design & Methodology](#system-design--methodology)
  - [Topic Detection](#topic-detection)
  - [Dual Checkpoint Strategy](#dual-checkpoint-strategy)
  - [Persona Extraction](#persona-extraction)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Answer Synthesis](#answer-synthesis)
- [Technology Choices](#technology-choices)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)

---

## Overview

ChronoMind processes chronological conversation data (CSV format) through a multi-stage pipeline:

1. **Semantic Topic Detection** — Identifies topic boundaries using embedding cosine drift detection
2. **Dual-level Indexing** — Builds both semantic (topic-based) and temporal (100-message) FAISS indexes
3. **User Persona Extraction** — Extracts evidence-based personality profiles with frequency promotion gates
4. **Intent-aware RAG** — Classifies query intent, retrieves from 3 FAISS indexes, synthesizes a natural-language answer
5. **Explainable UI** — Streamlit dashboard with 5 tabs: Chatbot, Persona Viewer, Topic Explorer, Checkpoints, Methodology

---

## Architecture
```text
---
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ChronoMind Pipeline                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Conversations (CSV Dataset)                                             │
│                                                                              │
│          ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │  CSV Parser      │                                                        │
│  │  parser.py       │                                                        │
│  └──────────────────┘                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │ Message Stream   │                                                        │
│  │ Chronological    │                                                        │
│  │ Processing       │                                                        │
│  └──────────────────┘                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  ┌──────────────────┐                                                        │
│  │ Topic Detection  │                                                        │
│  │ MiniLM Embeddings│                                                        │
│  │ Cosine Drift     │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│   ┌───────┴───────────────┐                                                  │
│   │                       │                                                  │
│   ▼                       ▼                                                  │
│ ┌─────────────────┐   ┌─────────────────────┐                                │
│ │ Topic Segments  │   │ 100-Message         │                                │
│ │ Dynamic         │   │ Checkpoints         │                                │
│ │ Boundaries      │   │ Fixed Windows       │                                │
│ └────────┬────────┘   └──────────┬──────────┘                                │
│          │                       │                                           │
│          ▼                       ▼                                           │
│ ┌─────────────────┐   ┌─────────────────────┐                                │
│ │ Topic Summaries │   │ Checkpoint          │                                │
│ │ LSA-Based       │   │ Summaries           │                                │
│ └────────┬────────┘   └──────────┬──────────┘                                │
│          └───────────────┬───────┘                                           │
│                          ▼                                                   │
│        ┌────────────────────────────────────────────┐                        │
│        │            FAISS Vector Stores             │                        │
│        ├────────────────────────────────────────────┤                        │
│        │ • Raw Message Chunks Index                 │                        │
│        │ • Topic Summaries Index                    │                        │
│        │ • Checkpoint Summaries Index               │                        │
│        └─────────────────────┬──────────────────────┘                        │
│                              │                                               │
│                              ▼                                               │
│        ┌────────────────────────────────────────────┐                        │
│        │         Persona Extraction Engine          │                        │
│        ├────────────────────────────────────────────┤                        │
│        │ • First-Person Pattern Detection           │                        │
│        │ • Habit Mining                             │                        │
│        │ • Personal Facts Extraction                │                        │
│        │ • Personality Profiling                    │                        │
│        │ • Communication Style Analysis             │                        │
│        └─────────────────────┬──────────────────────┘                        │
│                              │                                               │
│                              ▼                                               │
│        ┌────────────────────────────────────────────┐                        │
│        │      Intent-Aware RAG Retrieval Layer      │                        │
│        ├────────────────────────────────────────────┤                        │
│        │ Query                                      │                        │
│        │   → Intent Classification                  │                        │
│        │   → Topic Retrieval                        │                        │
│        │   → Chunk Retrieval                        │                        │
│        │   → Checkpoint Retrieval                   │                        │
│        │   → Persona Augmentation                   │                        │
│        │   → Answer Synthesis                       │                        │
│        └─────────────────────┬──────────────────────┘                        │
│                              │                                               │
│                              ▼                                               │
│        ┌────────────────────────────────────────────┐                        │
│        │         Streamlit Dashboard (5 Tabs)       │                        │
│        ├────────────────────────────────────────────┤                        │
│        │ Chatbot                                    │                        │
│        │ Persona Viewer                             │                        │
│        │ Topic Explorer                             │                        │
│        │ Checkpoint Explorer                        │                        │
│        │ Methodology                                │                        │
│        └────────────────────────────────────────────┘                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
---

## System Design & Methodology

### Topic Detection

**Algorithm:** Embedding-based cosine drift detection on a sliding centroid.

1. Encode each message with `all-MiniLM-L6-v2` → 384-dimensional embedding
2. Maintain a sliding-window centroid (default window = 5) tracking the current topic's embedding center
3. Compute cosine similarity between each new message and the centroid
4. If similarity drops below **0.35** (configurable), the current topic is closed and a new one begins
5. Trailing micro-segments of fewer than **10 messages** are merged into the prior topic
6. Each completed topic is summarized using **Sumy LSA** extractive summarization

**Why not fixed windows?** Fixed-window segmentation ignores semantic content. The same topic can span 200 messages or shift after 5. Cosine drift detection adapts to the actual conversation structure rather than imposing an arbitrary granularity.

**Parameters (`config.json`):**

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | 0.35 | Lower = fewer, larger topics |
| `min_topic_messages` | 10 | Prevents micro-topics from noisy messages |
| `window_size` | 5 | Centroid smoothing window |

---

### Dual Checkpoint Strategy

ChronoMind builds **two independent indexing layers**:

| Layer | Boundary Type | Captures |
|---|---|---|
| **Topic Checkpoints** | Semantic (cosine drift) | *What* was discussed |
| **100-Message Checkpoints** | Temporal (fixed interval) | *When* it was discussed |

**Why both?** Consider a user who mentions their location across three unrelated conversation topics spread over 300 messages. No single topic summary contains all three mentions. A 100-message temporal checkpoint spans multiple topics and will capture the recurrence. Combined multi-index retrieval returns significantly richer evidence than either layer alone.

---

### Persona Extraction

Extracts four structured categories per user using **frequency-gated, strict first-person regex patterns**:

| Category | Technique | Example |
|---|---|---|
| **Habits** | 52 strict first-person patterns | `Regular Reader` (evidence: 1,643 msgs) |
| **Personal Facts** | Named-entity capture groups | `Location: Portland` |
| **Personality Traits** | Keyword frequency analysis | `Optimistic` (conf: 0.84) |
| **Communication Style** | Statistical analysis | `Short to medium messages` |

**Why strict first-person?** Patterns like `(?:i\s+)?cook` would also match `"cooking is fun"` or `"she cooks every day"` — attributing third-party habits to the user. Every pattern in ChronoMind **mandates** an explicit `I`, `my`, `I'm`, or `I've` subject.

**Promotion gates:** A habit is promoted only when it passes both:
- `min_evidence_convos` ≥ 2 distinct conversations
- `min_occurrences` ≥ N total matches (threshold varies by habit type, 3–5)

**Confidence formula:**
```
confidence = min(1.0,  0.30  +  ln(1 + n_conversations) × 0.10)
```
| Conversations | Confidence |
|---|---|
| 3 | 0.41 |
| 10 | 0.53 |
| 50 | 0.69 |
| 200 | 0.83 |
| 500 | 0.92 |

**Evidence output per trait:**
```json
{
  "trait": "Regular Reader",
  "confidence": 1.00,
  "evidence": [8, 45, 112, 891, 4210, ...],
  "snippets": [
    "I enjoy reading Jane Austen novels...",
    "I love reading to kids. It's the best part of my job!"
  ],
  "occurrences": 1643
}
```

---

### Retrieval Pipeline

Multi-index FAISS retrieval with intent-aware routing:

1. **Encode** — Query is embedded using the same `all-MiniLM-L6-v2` model (embedding space consistency)
2. **Parallel search** across all three FAISS indexes:
   - **Chunks** (16,674 vectors) — 20-message sliding windows with 5-message overlap
   - **Topics** (15,378 vectors) — one vector per topic segment (centroid embedding + LSA summary)
   - **Checkpoints** (13,424 vectors) — 100-message temporal summaries
3. **Rank and combine** by cosine similarity score
4. **Intent classification** — query classified into 9 categories: `persona · habits · communication · occupation · location · interests · relationship · topics · general`
5. **Synthesize** — category-specific synthesizer generates a natural-language answer from structured evidence

---

### Answer Synthesis

The answer synthesizer (`src/answer_synthesizer.py`) routes each query to a specialized generation function:

| Intent | Synthesizer | Output style |
|---|---|---|
| `persona` | `_synthesize_persona_answer` | Flowing prose: traits → background → interests → habits |
| `habits` | `_synthesize_habits_answer` | Bullet list per user with occurrence counts |
| `communication` | `_synthesize_communication_answer` | Dot-separated statistical traits |
| `occupation` | `_synthesize_occupation_answer` | Filtered factual job entries only |
| `location` | `_synthesize_location_answer` | Named locations + chunk extraction |
| `interests` | `_synthesize_interests_answer` | Interests + habitual activities combined |
| `topics` | `_synthesize_topics_answer` | Deduplicated sentences from topic summaries |

Raw retrieval results are **never shown in the primary answer**. They are available in a collapsible "Show Evidence" expander for full transparency.

---

## Technology Choices

| Component | Technology | Rationale |
|---|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80MB, <1ms/msg on CPU, strong semantic quality-to-size ratio |
| **Vector DB** | FAISS (flat L2) | Exact nearest-neighbour search, zero approximation error, no server required |
| **Summarization** | Sumy (LSA method) | Extractive, fully deterministic, no LLM dependency |
| **Data Processing** | Pandas, NumPy | Standard ML data tooling |
| **UI Framework** | Streamlit | ML-native dashboard tooling, Python-native |
| **Charts** | Plotly | Interactive, dark-theme compatible |
| **External APIs** | None | Fully local — no OpenAI, no costs, no rate limits |
| **Language** | Python 3.9+ | |

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ChronoMind.git
cd ChronoMind

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Data

Place your conversation CSV at `data/conversations.csv`:

```csv
day,conversation
1,"User 1: Hello
User 2: Hi there
User 1: How are you?"
2,"User 1: Good morning
User 2: Morning!"
```

### 2. Build RAG System

```bash
python build_rag.py
```

Parses conversations → detects topics → builds 100-msg checkpoints → creates 3 FAISS indexes.

### 3. Extract Personas

```bash
python build_persona.py
```

Runs first-person pattern matching per user → applies frequency gates → outputs structured JSON to `persona/`.

### 4. Launch Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Project Structure

```
ChronoMind/
├── data/
│   └── conversations.csv          # Input conversation data
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration (dataclasses)
│   ├── models.py                  # Data models (Message, PersonaTrait, etc.)
│   ├── parser.py                  # CSV parser + chronological ordering
│   ├── embeddings.py              # SentenceTransformer wrapper
│   ├── summarizer.py              # Sumy LSA extractive summarization
│   ├── topic_detector.py          # Cosine drift topic segmentation
│   ├── checkpoint_builder.py      # 100-message temporal checkpoints
│   ├── vector_store.py            # FAISS multi-index management
│   ├── persona_extractor.py       # First-person persona extraction
│   ├── retrieval.py               # Multi-index retrieval pipeline
│   └── answer_synthesizer.py      # Intent-aware answer synthesis
├── checkpoints/                   # Generated topic & checkpoint JSONs
├── persona/                       # Generated persona JSONs
├── vector_db/                     # Persisted FAISS indexes
├── .streamlit/
│   └── config.toml                # Streamlit dark theme configuration
├── app.py                         # Streamlit dashboard (5 tabs)
├── chatbot.py                     # ChronoBot engine
├── build_rag.py                   # RAG build pipeline
├── build_persona.py               # Persona extraction pipeline
├── config.json                    # Central configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker deployment
├── render.yaml                    # Render.com deployment
├── railway.toml                   # Railway deployment
└── Procfile                       # Heroku-compatible
```

---

## Configuration

All parameters in `config.json`:

```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "topic_detection": {
    "similarity_threshold": 0.35,
    "min_topic_messages": 10,
    "window_size": 5
  },
  "checkpoints": { "interval": 100 },
  "chunking": { "chunk_size": 20, "chunk_overlap": 5 },
  "retrieval": {
    "top_k_chunks": 8,
    "top_k_topics": 5,
    "top_k_checkpoints": 3
  },
  "persona": {
    "min_confidence": 0.30,
    "min_evidence_convos": 2,
    "max_facts_per_category": 20
  }
}
```

---

## Deployment

### Docker

```bash
docker build -t chronomind .
docker run -p 8501:8501 chronomind
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app.py`
4. Pre-build indexes locally and commit, or add a setup hook

### Render / Railway

Uses `render.yaml` / `railway.toml` automatically on push.

---

<p align="center">
  ChronoMind · Explainable Conversation Intelligence · FAISS + SentenceTransformers + Sumy + Streamlit
</p>
