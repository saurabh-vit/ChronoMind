"""
Full end-to-end validation of every ChronoMind module
against the actual conversations.csv dataset.

Run with: python validate_all.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []
warnings = []


def check(condition, name, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
        errors.append(f"{name}" + (f": {detail}" if detail else ""))


def warn(condition, name, detail=""):
    if condition:
        print(f"  {WARN} {name}" + (f": {detail}" if detail else ""))
        warnings.append(f"{name}" + (f": {detail}" if detail else ""))


# --- 1. PARSER ---
print("\n[1/6] PARSER")
from src.parser import (
    parse_conversations, get_user_names, get_conversation_count,
    get_conversation_groups
)

t0 = time.time()
messages = parse_conversations("data/conversations.csv")
t1 = time.time()

check(len(messages) > 100_000, f"Parsed >100K messages", f"got {len(messages):,}")
check(len(messages) > 180_000, f"Parsed ~191K messages", f"got {len(messages):,}")

n_convos = get_conversation_count(messages)
check(n_convos > 5000, f"Found >5K conversations", f"got {n_convos:,}")

users = get_user_names(messages)
check(set(users) == {"User 1", "User 2"}, "Correct user names", f"got {users}")

min_id = min(m.conversation_id for m in messages)
max_id = max(m.conversation_id for m in messages)
check(min_id == 0, "Conversation IDs start at 0", f"min={min_id}")
check(max_id >= n_convos - 1, "Conversation IDs are sequential",
      f"max={max_id}, n_convos={n_convos}")

groups = get_conversation_groups(messages)
cross_bleed = any(m.conversation_id != conv_id
                  for conv_id, msgs in groups.items()
                  for m in msgs)
check(not cross_bleed, "No messages bleed across conversation groups")

# Escaped quote check - if "" still appears it may be real text (not a parser bug)
quoted = [m for m in messages if '""' in m.text]
warn(len(quoted) > 0,
     f"{len(quoted)} msg(s) contain literal '""' (likely intentional, not a parser bug)",
     f"Example: {quoted[0].text[:80]}" if quoted else "")

indexes = [m.index for m in messages]
check(len(indexes) == len(set(indexes)), "All message indexes unique")
check(indexes == sorted(indexes), "Message indexes are sequential")
print(f"  Parse time: {t1 - t0:.2f}s")


# --- 2. TOPIC DETECTOR ---
print("\n[2/6] TOPIC DETECTOR (first 500 messages only)")
from src.config import load_config
from src.topic_detector import detect_topics

config = load_config()
sample = messages[:500]

t0 = time.time()
topics = detect_topics(sample, config)
t1 = time.time()

check(len(topics) > 0, "Topics detected", f"got {len(topics)}")

bad_conv_ids = [t for t in topics if not hasattr(t, 'conversation_id') or t.conversation_id is None]
check(len(bad_conv_ids) == 0, "All topics have conversation_id",
      f"{len(bad_conv_ids)} missing")

cross_topic = sum(
    1 for t in topics
    if any(m.conversation_id != t.conversation_id for m in t.messages)
)
check(cross_topic == 0, "No topics span multiple conversations",
      f"{cross_topic} cross-conversation topics found")

empty_summaries = [t for t in topics if not t.summary.strip()]
warn(len(empty_summaries) > 0,
     f"{len(empty_summaries)}/{len(topics)} empty summaries (expected for short convos)")

no_embeddings = [t for t in topics if t.embedding is None]
check(len(no_embeddings) == 0, "All topics have embeddings",
      f"{len(no_embeddings)} missing")

print(f"  Topic detection time: {t1 - t0:.2f}s for {len(sample)} messages -> {len(topics)} topics")


# --- 3. CHECKPOINT BUILDER ---
print("\n[3/6] CHECKPOINT BUILDER (first 500 messages)")
from src.checkpoint_builder import build_checkpoints

t0 = time.time()
checkpoints = build_checkpoints(sample, config)
t1 = time.time()

check(len(checkpoints) > 0, "Checkpoints created", f"got {len(checkpoints)}")

cross_cp = sum(
    1 for cp in checkpoints
    if len(set(m.conversation_id for m in cp.messages)) > 1
)
check(cross_cp == 0, "No checkpoints span multiple conversations",
      f"{cross_cp} bad checkpoints")

no_cp_emb = [cp for cp in checkpoints if cp.embedding is None]
check(len(no_cp_emb) == 0, "All checkpoints have embeddings",
      f"{len(no_cp_emb)} missing")

oversized = [cp for cp in checkpoints if len(cp.messages) > config.checkpoints.interval + 1]
check(len(oversized) == 0,
      f"All checkpoints within interval ({config.checkpoints.interval})",
      f"{len(oversized)} oversized")

print(f"  Checkpoint build time: {t1 - t0:.2f}s -> {len(checkpoints)} checkpoints")


# --- 4. PERSONA EXTRACTION ---
print("\n[4/6] PERSONA EXTRACTION (first 1000 messages)")
from src.persona_extractor import extract_persona

t0 = time.time()
personas = extract_persona(messages[:1000], config)
t1 = time.time()

check(len(personas) == 2, "Exactly 2 user personas", f"got {list(personas.keys())}")
check("User 1" in personas, "User 1 persona extracted")
check("User 2" in personas, "User 2 persona extracted")

for user, persona in personas.items():
    pd = persona.to_dict()
    total_traits = sum(len(pd[k]) for k in
                       ["habits", "personal_facts", "personality_traits", "communication_style"])
    check(total_traits > 0, f"{user}: has traits", f"got {total_traits}")

    all_confidences = [
        item["confidence"]
        for k in ["habits", "personal_facts", "personality_traits", "communication_style"]
        for item in pd[k]
    ]
    max_conf = max(all_confidences) if all_confidences else 0
    check(all(c <= 1.0 for c in all_confidences),
          f"{user}: all confidences <= 1.0", f"max={max_conf}")

    bad_occ = [f for f in pd["personal_facts"] if "doing well" in f["trait"].lower()]
    check(len(bad_occ) == 0, f"{user}: no 'doing well' occupation false positive",
          f"found: {[f['trait'] for f in bad_occ]}")

    # Print summary
    print(f"    {user}: habits={len(pd['habits'])}, facts={len(pd['personal_facts'])}, "
          f"personality={len(pd['personality_traits'])}, comms={len(pd['communication_style'])}")

print(f"  Persona extraction time: {t1 - t0:.2f}s")


# --- 5. VECTOR STORE ---
print("\n[5/6] VECTOR STORE (mini index on 200 messages)")
from src.vector_store import VectorStore

store = VectorStore(config)
t0 = time.time()
chunks = store.build_chunks_index(messages[:200])
t1 = time.time()

check(len(chunks) > 0, "Chunks created", f"got {len(chunks)}")

bad_chunks = [c for c in chunks if not hasattr(c, 'conversation_id')]
check(len(bad_chunks) == 0, "All chunks have conversation_id attr")

empty_chunks = [c for c in chunks if not c.text.strip()]
check(len(empty_chunks) == 0, "No empty chunks", f"{len(empty_chunks)} empty")

oversized_chunks = [c for c in chunks if c.text.count("\n") + 1 > config.chunking.chunk_size + 2]
check(len(oversized_chunks) == 0,
      f"Chunk sizes within limit ({config.chunking.chunk_size})",
      f"{len(oversized_chunks)} oversized")

print(f"  Chunk build time: {t1 - t0:.2f}s -> {len(chunks)} chunks")


# --- 6. RETRIEVAL PIPELINE ---
print("\n[6/6] RETRIEVAL PIPELINE")

db_dir = Path(config.paths.vector_db_dir)
if (db_dir / "chunks.faiss").exists():
    from src.retrieval import retrieve, _is_persona_query
    from src.vector_store import VectorStore as VS

    store2 = VS(config)
    loaded = store2.load()
    check(loaded, "FAISS indexes loaded successfully")

    if loaded:
        result = retrieve("What hobbies do the users have?", store2, config)
        check(len(result.raw_chunks) > 0, "Retrieves raw chunks")
        check(len(result.topic_summaries) > 0, "Retrieves topic summaries")
        check(result.combined_context != "", "Combined context non-empty")
        check(len(result.combined_context) < 10000,
              "Context under 10K chars", f"got {len(result.combined_context)}")

        # Persona detection checks
        check(_is_persona_query("what kind of person is User 1"),
              "Persona detected: 'what kind of person'")
        check(_is_persona_query("what are their habits"),
              "Persona detected: 'habits'")
        check(not _is_persona_query("tell me about Portland"),
              "Non-persona: 'Portland' correctly skipped")
        check(not _is_persona_query("what books do they read"),
              "Non-persona: 'books' correctly skipped")
        check(not _is_persona_query("who wrote this message"),
              "Non-persona: 'message' correctly excluded")
else:
    print(f"  {WARN} Skipping retrieval test - run build_rag.py first")


# --- SUMMARY ---
print("\n" + "=" * 60)
if errors:
    print(f"{len(errors)} ERRORS:")
    for e in errors:
        print(f"  {FAIL} {e}")
else:
    print("ALL CHECKS PASSED!")

if warnings:
    print(f"\n{len(warnings)} WARNINGS (non-critical):")
    for w in warnings:
        print(f"  {WARN} {w}")

print(f"\nDataset: {len(messages):,} messages across {n_convos:,} conversations")
print("=" * 60)
