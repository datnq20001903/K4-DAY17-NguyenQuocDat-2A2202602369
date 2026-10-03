from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for fact in expected if fact.casefold() in (answer or "").casefold())
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    text = (answer or "").strip()
    if not text:
        return 0.0
    score = 0.0
    if any(token in text.lower() for token in ("tên", "ở", "nghề", "uống", "ăn", "mình", "tôi")):
        score += 0.2
    if len(text.split()) <= 35:
        score += 0.2
    if any(token in text.lower() for token in ("bullet", "- ", "đầu tiên", "thứ hai")):
        score += 0.2
    if expected and any(fact.lower() in text.lower() for fact in expected):
        score += 0.4
    return min(score, 1.0)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_total = 0.0
    quality_total = 0.0
    compaction_total = 0
    memory_growth_bytes = 0
    recall_questions = 0

    for index, conversation in enumerate(conversations):
        user_id = str(conversation.get("user_id", f"user-{index}"))
        for message in conversation.get("turns", []):
            result = agent.reply(user_id, f"bench-{index}", message)
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens", 0))
        for question in conversation.get("recall_questions", []):
            answer = agent.reply(user_id, f"recall-{index}-{len(question.get('expected_contains', []))}", question["question"])["answer"]
            recall_total += recall_points(answer, question.get("expected_contains", []))
            quality_total += heuristic_quality(answer, question.get("expected_contains", []))
            recall_questions += 1
        if hasattr(agent, "memory_file_size"):
            memory_growth_bytes = max(memory_growth_bytes, agent.memory_file_size(user_id))
        compaction_total += agent.compaction_count(f"bench-{index}")

    if recall_questions:
        recall_score = recall_total / recall_questions
        quality_score = quality_total / recall_questions
    else:
        recall_score = 0.0
        quality_score = 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=recall_score,
        response_quality=quality_score,
        memory_growth_bytes=memory_growth_bytes,
        compactions=compaction_total,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [max(len(str(item)) for item in [header] + [row[index] for row in values]) for index, header in enumerate(headers)]
    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    sep_line = "-+-".join("-" * width for width in widths)
    rows_text = [header_line, sep_line]
    for row in values:
        rows_text.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(rows_text)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_config(root)
    standard = load_conversations(root / "data" / "conversations.json")
    long_context = load_conversations(root / "data" / "advanced_long_context.json")

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    baseline_row = run_agent_benchmark("Baseline", baseline, standard, config)
    advanced_row = run_agent_benchmark("Advanced", advanced, standard, config)
    long_baseline = BaselineAgent(config, force_offline=True)
    long_advanced = AdvancedAgent(config, force_offline=True)
    baseline_long_row = run_agent_benchmark("Baseline", long_baseline, long_context, config)
    advanced_long_row = run_agent_benchmark("Advanced", long_advanced, long_context, config)

    print("Standard Benchmark")
    print(format_rows([baseline_row, advanced_row]))
    print("\nLong-Context Stress Benchmark")
    print(format_rows([baseline_long_row, advanced_long_row]))


if __name__ == "__main__":
    main()
