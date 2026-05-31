# 🚀 AI-Native Semantic Talent Matching Engine (#IndiaRuns)

An advanced, production-grade talent sourcing pipeline built for the **Redrob AI #IndiaRuns Challenge**. This system completely replaces outdated, easily gamed keyword filters with an AI-powered evaluation engine that understands professional context, capability scope, and role alignment—ranking candidates the way a great recruiter would.

---

## 💡 Why This Approach? (Moving Beyond Keywords)

Traditional Applicant Tracking Systems (ATS) rely on exact keyword matches, creating two major failures:
1. **The Over-Indexed Resume:** A candidate who repeats a buzzword like "Python" 50 times gets ranked at the top, regardless of depth.
2. **The Hidden Gem:** A brilliant systems architect who writes *"built a high-throughput, real-time distributed data streaming tier"* gets filtered out simply because they didn't explicitly write the keyword *"Kafka"*.

Our system bridges this semantic gap by analyzing the **contextual intent** of both the Job Description (JD) and the candidate's holistic background.

---

## 🏗️ System Architecture & Engineering Approach

To achieve deep accuracy without sacrificing performance, this engine implements a **Two-Tier Retrieval and Reranking Cascade**:

1. **Structured Context Serialization (`src/normalizer.py`):** Converts complex, messy JSON candidate data (skills arrays, chronological career summaries, and platform activity signals) into an optimized, unified semantic narrative text block.
2. **Tier-1 Retrieval Net (Bi-Encoder):** Uses the state-of-the-art `BAAI/bge-large-en-v1.5` embeddings model to project both the JD and all candidates into a shared multi-dimensional space. It instantly filters out completely unaligned profiles using cosine similarity.
3. **Tier-2 Contextual Reranker (Cross-Encoder):** Takes the top candidates and passes them through `ms-marco-MiniLM-L-6-v2`. Unlike standard vector math that examines text independently, a Cross-Encoder analyzes the JD and the candidate profile *simultaneously* via deep attention-mapping, verifying actual seniority level and scope alignment.
4. **Absolute Sigmoid Calibration Layer:** Avoids misleading relative Min-Max scoring (which arbitrarily forces the lowest scoring candidate in a batch to 0%). Instead, it uses an absolute logistic sigmoid mapping function to evaluate candidates strictly on their own merits and scales them cleanly to **Integer percentages**.

---

## 📂 Repository Structure

```text
india_runs_matcher/
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Standardized and safe JSON file loader
│   ├── normalizer.py     # Flattens unstructured histories & signals into semantic blocks
│   └── ranker.py         # Handles the Bi-Encoder, Cross-Encoder, and absolute scaling logic
├── data/                 # Data landing zone (job_description.json & candidates.json)
├── output/               # Final results destination (ranked_candidates.csv)
├── requirements.txt      # Project dependencies (pinned versions)
├── run.py                # Pipeline execution entry-point
└── README.md             # Project documentation
