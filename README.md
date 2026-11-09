# Instamart Customer Intelligence & Support Copilot

An end-to-end data engineering, machine learning, and AI-agentic copilot built for Swiggy Instamart to clean messy raw orders, segment customers, predict customer lapse, safely triage support tickets with PII redaction and retry logic, answer policy questions via TF-IDF retrieval, and autonomously resolve refund requests using a ReAct agent.

---

## Overview

This repository houses the complete capstone project across four interconnected parts:
- **Part A (Data Foundation):** Audits, cleans, and stores raw customer, product, and order exports into a structured SQLite database (`data/instamart.db`), verified via 7 SQL queries.
- **Part B (Customer Segmentation & Lapse Prediction):** Computes RFM metrics, clusters customers using K-Means ($K=4$), and trains a leak-free Random Forest pipeline to predict customer lapse.
- **Part C (Support Ticket Triage):** Implements input guardrails, regex PII redaction, resilient exponential backoff retry wrappers, and rule-validated JSON classification routing tickets based on Part B segments.
- **Part D (Policy RAG & ReAct Refund Agent):** Builds a retrieval-grounded policy QA assistant and a ReAct agent that inspects orders, calculates refunds, executes tool actions, and maintains state crash-resilience.

---

## Reproduce

Follow these commands in order from a clean clone to reproduce the entire pipeline:

```bash
# 1. Clone repository
git clone https://github.com/esainisha18/instamart-intelligence-copilot.git



# 2. Create and activate virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Unix/macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate raw synthetic datasets and policy texts
python generate_data.py

# 5. Run Part A: Cleaning, Database Population, and SQL validation
python src/part_a_clean.py
python src/part_a_validate.py

# 6. Run Part B: RFM Segmentation and Lapse Prediction Model
python src/part_b_segmentation.py
python src/part_b_lapse.py

# 7. Run Part C: Ticket Triage & Routing Pipeline
python src/part_c_triage.py

# 8. Run Part D: Policy RAG Assistant and ReAct Refund Agent
python src/part_d_rag.py
python src/part_d_agent.py




Part A — Data Foundation: Clean, Store, Query:
Audit_Figures: 13,084 raw orders; 119 exact duplicates; 656 pending rows; 12,309 distinct order IDs; 120 unparseable amounts; 106 negative delivery times; 2,541 blank delivery times; 61 blank ages; 6 ages outside 10-100; 24 raw cities; 10 raw genders; 30 blank genders.

Cleaning: Cities standardized to 6 title-case cities. Genders mapped to male, female, unknown. Missing/invalid ages imputed via city median. Orders cleaned by deduplicating, keeping latest status per order_id, imputing amounts via product price formula, and filling completed delivery times via category medians.

Database & Checkpoints: Stored in data/instamart.db. Completed orders sum to INR 2,998,668.80. Validation passed all checkpoints.

SQL Queries (sql/queries.sql): Q1 (30), Q2 (30 rows, INR 2,998,668.80 total, top cell Bengaluru/Electronics INR 233,909.45), Q3 (11), Q4 (25), Q5 (550 customers, avg INR 2,552.06), Q6 (top customers per city; 8 rank ties explained by equal totals), Q7 (183 daily rows, final cumulative INR 2,998,668.80).




Part B — Customer Segments and Lapse Prediction:
RFM Table: 1,175 customers; recency median 19; frequency mean 9.46; monetary mean INR 2,552.06.

K-Means Clustering (K=4): PCA explained variance ratio ~0.634 and ~0.218. Saved to outputs/customer_segments.csv.

Lapse Prediction: Cohort of 1,156 customers before 2025-09-01; 443 lapsed (38.32%). Leak-free pipeline with OneHotEncoder(drop='first') and StandardScaler. Tuned Random Forest achieved test F1 = 0.7065 (exceeding >=0.55). Oversampled logistic regression increased recall from 0.64 to 0.85. Saved to outputs/lapse_model.joblib and outputs/retention_list.csv (noting Premium is post-hoc whole-window).




Part C — Support Ticket Triage with Guardrails:
Input Guardrails: 4 skipped_empty, 4 flagged_long (~734 tokens).

PII Redaction: 55 emails, 91 phones, 132 order IDs (278 total substitutions); 0 leaks found on re-application.

Classification Distribution: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output.

Accuracies: Zero-shot = 85.71%; Few-shot = 100%.

Budget & Routing: 6,000-token demo exhausted 54 tickets (outputs/triage_budget_demo.csv). Final routing saved to outputs/triage_results.csv (164 rows).



Part D — Policy RAG Assistant and Refund Agent:
RAG & Policy Evaluation: 15 chunks (40 words, 10 overlap). Retrieval benchmark passed 12/12 for answerables and 2/2 for unanswerables. Zero-overlap failed expected-fact check due to chunk boundary splitting.

ReAct Agent Scenarios: S1 (REFUND 128.25), S2 (REFUND 135.00), S3 (ESCALATE 1,999.00), S4 (DENIED 0.00), S5 (DENIED 0.00), S6 (NOT_FOUND 0.00), S7 (FALLBACK_MAX_ITERATIONS).

Safety Demos & Resume: Token limit demo returned FALLBACK_TOKEN_LIMIT; timeout demo returned FALLBACK_TIMEOUT. Crash-and-resume on S1 executed exactly four tools (get_order -> search_policy -> calculate_refund -> issue_refund) without duplicates, ending in REFUND INR 128.25.

ReAct vs Plan-and-Execute: ReAct dynamically interleaves reasoning and tool execution, adapting step-by-step to observations (e.g., handling NOT_FOUND or DENIED). While plan-and-execute can be slightly cheaper on token overhead for deterministic paths, ReAct is superior for customer support workflows where downstream tool outputs dictate branching logic and uncertain policy exceptions.



Limitations and Next Steps:
Part B's Premium segment uses whole-window RFM including September data; hence the retention list is illustrative rather than strictly leakage-free.

Next steps include integrating live database triggers and migrating mock chat completions to hosted LLM APIs with streaming support.