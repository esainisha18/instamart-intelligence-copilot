"""Chunk, index, retrieve, and evaluate Instamart policy documents."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10
SYSTEM_PROMPT = (
    "POLICY ASSISTANT: Answer using only the supplied policy context. "
    "If the answer is not present in the context, say exactly: "
    "I could not find that information in the policy documents."
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mock_llm


class PolicyIndex:
    def __init__(
        self,
        policy_dir: Path = POLICY_DIR,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0")
        self.policy_dir = policy_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError(f"No policy text files found in {policy_dir}")
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            chunk["text"] for chunk in self.chunks
        )

    def _load_chunks(self) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        step = self.chunk_size - self.overlap
        for path in sorted(self.policy_dir.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            if not words:
                continue
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunks.append(
                    {
                        "source": path.name,
                        "text": " ".join(words[start:end]),
                    }
                )
                if end >= len(words):
                    break
                start += step
        return chunks

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if k < 0:
            raise ValueError("k must be non-negative")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = sorted(
            range(len(self.chunks)),
            key=lambda index: (-scores[index], index),
        )[:k]
        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in ranked_indexes
        ]

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question, k=3)
        context = "\n".join(
            f"[{chunk['source']}]\n{chunk['text']}" for chunk in retrieved
        )
        user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        response = mock_llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )
        return response.choices[0].message.content or ""


_DEFAULT_INDEX: PolicyIndex | None = None


def get_index(overlap: int = DEFAULT_OVERLAP) -> PolicyIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None or _DEFAULT_INDEX.overlap != overlap:
        _DEFAULT_INDEX = PolicyIndex(overlap=overlap)
    return _DEFAULT_INDEX


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    return get_index().retrieve(query, k=k)


def answer(question: str) -> str:
    return get_index().answer(question)


def run_evaluation(index: PolicyIndex) -> dict[str, Any]:
    questions = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
    doc_hits = 0
    fact_hits = 0
    fallback_hits = 0
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for row in questions.itertuples(index=False):
        retrieved = index.retrieve(row.question, k=3)
        result = index.answer(row.question)
        if row.expected_doc == "NONE":
            passed = result.startswith("I could not find")
            fallback_hits += int(passed)
            if not passed:
                failures.append(
                    {
                        "question": row.question,
                        "failure": "unanswerable question did not use fallback",
                        "answer": result,
                    }
                )
        else:
            doc_found = any(
                chunk["source"] == row.expected_doc for chunk in retrieved
            )
            fact_found = row.expected_fact.lower() in result.lower()
            doc_hits += int(doc_found)
            fact_hits += int(fact_found)
            if not doc_found or not fact_found:
                failures.append(
                    {
                        "question": row.question,
                        "failure": (
                            f"doc_found={doc_found}, fact_found={fact_found}"
                        ),
                        "answer": result,
                    }
                )
        records.append(
            {
                "question": row.question,
                "expected_doc": row.expected_doc,
                "expected_fact": row.expected_fact,
                "retrieved_sources": [item["source"] for item in retrieved],
                "answer": result,
            }
        )
    answerable_count = int(questions["expected_doc"].ne("NONE").sum())
    unanswerable_count = len(questions) - answerable_count
    return {
        "question_count": len(questions),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "doc_hits": doc_hits,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "failures": failures,
        "records": records,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = PolicyIndex(overlap=DEFAULT_OVERLAP)
    counts: dict[str, int] = {}
    for chunk in index.chunks:
        counts[Path(chunk["source"]).stem] = (
            counts.get(Path(chunk["source"]).stem, 0) + 1
        )
    expected_counts = {
        "refund_policy": 4,
        "delivery_policy": 3,
        "escalation_policy": 3,
        "payment_policy": 3,
        "privacy_policy": 2,
    }
    assert counts == expected_counts, f"Unexpected chunk counts: {counts}"
    assert len(index.chunks) == 15
    print(f"Overlapping index chunks: {len(index.chunks)}; per file={counts}")

    evaluation = run_evaluation(index)
    print(
        "Overlapping benchmark: "
        f"answerable_doc_hits={evaluation['doc_hits']}/12, "
        f"expected_facts={evaluation['fact_hits']}/12, "
        f"unanswerable_fallbacks={evaluation['fallback_hits']}/2"
    )
    assert evaluation["doc_hits"] == 12
    assert evaluation["fact_hits"] == 12
    assert evaluation["fallback_hits"] == 2
    (OUTPUT_DIR / "policy_eval_overlap.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )

    zero_overlap_index = PolicyIndex(overlap=0)
    zero_overlap_evaluation = run_evaluation(zero_overlap_index)
    print(
        "Zero-overlap benchmark: "
        f"answerable_doc_hits={zero_overlap_evaluation['doc_hits']}/12, "
        f"expected_facts={zero_overlap_evaluation['fact_hits']}/12, "
        f"unanswerable_fallbacks={zero_overlap_evaluation['fallback_hits']}/2"
    )
    boundary_failures = [
        failure
        for failure in zero_overlap_evaluation["failures"]
        if "fact_found=False" in failure["failure"]
    ]
    assert boundary_failures, (
        "Expected at least one answerable question to fail without overlap"
    )
    (OUTPUT_DIR / "policy_eval_zero_overlap.json").write_text(
        json.dumps(zero_overlap_evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Zero-overlap boundary failure:")
    print(json.dumps(boundary_failures[0], indent=2))


if __name__ == "__main__":
    main()
