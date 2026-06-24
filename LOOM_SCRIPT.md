# LOOM_SCRIPT.md — ChronoMind v2 Video Walkthrough Script

## Recording Guide

> Focus on **why** decisions were made, not just feature demonstrations.
> Target length: 8–12 minutes.

---

## Segment 1: Introduction (60 seconds)

**Say:**

"ChronoMind v2 is an explainable conversation intelligence system. It doesn't just answer questions — it shows you *why* it answered the way it did.

What you're looking at is a fully local system. No OpenAI. No paid APIs. Everything runs on CPU, and everything is explainable by design.

Let me walk you through the four engineering decisions that drove v2."

**Show:** The dashboard header. Scroll across the 9 tabs.

---

## Segment 2: Why Persona Drift Matters (90 seconds)

**Tab:** 🧠 Persona Drift

**Say:**

"Static persona profiles have a fundamental problem — they assume a person is the same across all conversations. But in a real dataset of 11,000 conversations, people change.

They're curious on Monday and frustrated by Thursday. They might be formal at work and casual with friends.

The Adaptive Persona Engine tracks this. It analyzes behavioral signals for each conversation session — things like sentiment score, question frequency, emoji usage, message length — and infers a mood and tone label.

What's more important than the label is the drift event. When someone shifts from 'curious' to 'frustrated', we want to know *why*. So we scan the message window for keyword clusters — job-related terms, relationship terms, health terms — and surface the probable trigger.

This matters for a chatbot that gives meaningful answers. If someone's mood shifts dramatically around a job loss, 'what hobbies does User 1 have' becomes a very different question than it was last week."

**Show:** Select User 1. Point to drift events. Expand one event and show the trigger + keywords.

---

## Segment 3: Why Offline Intent Classification (90 seconds)

**Tab:** 🎯 Intent Classifier

**Say:**

"The obvious choice for intent classification in 2024 is to call the OpenAI API. That's expensive, introduces latency, and creates a privacy problem — you're sending user messages to a third party.

The constraint I imposed was: under 50 megabytes, CPU only, under 200 milliseconds.

The solution is TF-IDF with a Logistic Regression classifier. That sounds simple, but it's actually appropriate here. We have five well-defined intent classes — reminder, emotional support, action item, small talk, and unknown — and the vocabulary that distinguishes them is very clear.

The 5-fold cross-validation F1 is 70%, trained on 140 examples. The test set F1 is lower because we only had 28 held-out examples — a single misclassified item moves the needle 3.5%. More training data would push this to 85%+.

More importantly, look at the latency. 0.41 milliseconds. That's the cost of a dictionary lookup, not a network call."

**Show:** Run the classifier on a few test inputs. Show the confidence bars. Run the 5-message benchmark.

---

## Segment 4: Why Conflict Resolution is Non-Trivial (90 seconds)

**Tab:** ⚖️ Conflict Resolver

**Say:**

"Retrieval-augmented generation has a well-known failure mode that nobody talks about: what happens when two retrieved chunks say opposite things?

Imagine someone who lived in Delhi three years ago, then moved to Bangalore. Their conversation data contains both facts. The retrieval system finds both. Without conflict resolution, the answer is: 'Delhi.' Or: 'Bangalore.' Depending on random retrieval order.

That's unacceptable for a system that claims to be explainable.

The conflict resolver extracts entity–attribute pairs from retrieved chunks — using pattern matching on location, occupation, and relationship facts. When the same attribute has two different values, that's a conflict.

Then we score each piece of evidence by three factors: retrieval similarity to the query, recency — later messages score higher — and emotional weight — words like 'moved', 'now', 'currently' signal a recent fact update.

The higher-scoring fact wins. And the answer explicitly tells the user: older conversations said X, newer evidence says Y, here's the message range.

That's transparency."

**Show:** Run the Location Contradiction demo. Point to the scoring formula. Show the merged answer output.

---

## Segment 5: The Chatbot Upgrade — Pipeline View (60 seconds)

**Tab:** 🤖 Chatbot

**Say:**

"Every chatbot answer now shows you three things that weren't there before.

First, a purple badge: the detected intent and confidence score. This tells you which retrieval path was activated.

Second, a green or red badge: conflict detection result. Green means the retrieved evidence is consistent. Red means contradictions were found and resolved.

Third, the 'How This Answer Was Generated' expander is now extended to show all five pipeline stages — intent, retrieval, persona, conflict resolution, synthesis — with the actual numbers for each step.

This is what I mean by explainable. Not just the answer. The reasoning trail."

**Show:** Ask a question like 'What cities are mentioned?' and expand all three expanders.

---

## Segment 6: Why This Sync Architecture (60 seconds)

**Tab:** 🏗 Sync Architecture

**Say:**

"The sync design is driven by one constraint: conversation data is private, but conversation insights are not.

Raw messages — the actual text — never leave the device. That's a hard rule. Everything that touches personal content stays local: embeddings, persona facts, drift data.

What syncs? Summaries. The output of Sumy LSA summarization on topic segments and checkpoints. These are 2-3 sentence extractions. They tell you roughly what was discussed, without exposing any private detail.

For conflict resolution between devices, the policy is: latest timestamp wins. It's not perfect — you could imagine valid older facts that get overwritten. But it's deterministic and doesn't require merging unstructured prose, which would need an LLM.

The key design insight is: offline-first, sync as enhancement. Full functionality with zero network access. The cloud layer is an optimization, not a dependency."

**Show:** Scroll through the architecture diagram. Point to the local vs cloud data tables.

---

## Segment 7: Technical Summary (60 seconds)

**Tab:** 📖 Methodology

**Say:**

"Let me close with the technical numbers.

191,592 messages, 11,001 conversations, processed entirely offline.

Three FAISS indexes with a combined 45,476 vectors, all queried in under 50 milliseconds.

Intent classification at 0.41 milliseconds per query.

Persona drift analysis covering 11,001 sessions per user, detecting 3,819 drift events for User 1.

The entire system runs on a single laptop with no GPU and no internet connection.

That's the architecture."

**Show:** The Dataset Statistics cards at the top of the dashboard. Then the Methodology tab pipeline diagram.

---

## Closing (30 seconds)

**Say:**

"The goal of ChronoMind v2 was to build a system that feels like it was engineered, not generated.

Every component has a reason. Every output has a source. Every answer can be traced back to specific messages in the dataset.

That's explainable AI — not just in theory, but in practice."

---

*Total estimated length: 9–10 minutes*
