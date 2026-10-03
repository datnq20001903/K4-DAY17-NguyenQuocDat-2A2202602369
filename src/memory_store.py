from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


_FACT_LINE = re.compile(r"^[ \t]*-[ \t]*([a-z][a-z0-9_]*)[ \t]*:[ \t]*(.*)$", re.MULTILINE)
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)}


def estimate_tokens(text: str) -> int:
    """Estimate one token per four trimmed characters, with a minimum of one."""
    stripped = (text or "").strip()
    return max(1, len(stripped) // 4) if stripped else 0


def _parse_facts(text: str) -> dict[str, str]:
    return {match[1]: match[2].strip() for match in _FACT_LINE.finditer(text) if match[2].strip()}


@dataclass
class UserProfileStore:
    """Persist UTF-8 profiles beneath root_dir, isolated by normalized user ID."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id must not be empty.")
        ordinary_id = re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", user_id)
        # Plain IDs cannot start with '__user_', so they cannot impersonate hashes.
        slug = user_id if ordinary_id and user_id not in _RESERVED_NAMES else (
            "__user_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        )
        root = self.root_dir.resolve()
        path = root / slug / "User.md"
        if not path.resolve().is_relative_to(root):
            raise ValueError("user_id resolves outside the profile root.")
        return path

    def read_text(self, user_id: str) -> str:
        try:
            return self.path_for(user_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def write_text(self, user_id: str, text: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def edit_text(self, user_id: str, search: str, replacement: str) -> bool:
        """Replace one occurrence and report whether the persisted text changed."""
        if not search or search == replacement:
            return False
        original = self.read_text(user_id)
        if search not in original:
            return False
        self.write_text(user_id, original.replace(search, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        try:
            return self.path_for(user_id).stat().st_size
        except FileNotFoundError:
            return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read stable '- key: value' facts; later duplicates take precedence."""
        return _parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Update one key without accumulating old values or deleting manual notes."""
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError("Fact key must use lowercase letters, digits, or underscores.")
        value = " ".join(value.split())
        if not value:
            raise ValueError("Fact value must not be empty.")
        original = self.read_text(user_id)
        lines = original.splitlines() if original else ["# User profile", ""]
        updated = []
        found = False
        for line in lines:
            match = _FACT_LINE.fullmatch(line)
            if match and match[1] == key:
                if not found:
                    updated.append(f"- {key}: {value}")
                    found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f"- {key}: {value}")
        text = "\n".join(updated) + "\n"
        if text == original:
            return False
        self.write_text(user_id, text)
        return True


def _clean_fact(value: str) -> str:
    value = re.split(
        r"[,;.!?]|\s+(?:và|chứ|nhưng|dù|cho|để|nữa|trong|vài|mỗi|vì|nhé)\b",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return value.strip(" :")[:160].strip()


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese rules for affirmative personal facts and style.

    This is a deterministic lab heuristic, not a general entity extractor.
    Question sentences, jokes, negated old facts, and short trips are excluded.
    """
    updates: dict[str, str] = {}
    message = unicodedata.normalize("NFC", message)
    for sentence in re.split(r"(?<=[.!?])\s+|[\n;]+", message):
        if "?" in sentence or re.search(r"\b(?:đùa|nói chơi|giả sử|ví dụ cũ)\b", sentence, re.I):
            continue
        if re.match(
            r"^(?:(?:hãy|bạn|vui lòng|làm ơn)\s+)*(?:nhắc lại|nhớ lại|tóm tắt|kể lại|liệt kê|cho (?:mình|tôi) biết)\b",
            sentence.strip(), re.I,
        ):
            # A request contains no new fact. An explicit recap after ':' may.
            _, separator, statement = sentence.partition(":")
            if not separator:
                continue
            sentence = statement.strip()

        name_patterns = [
            r"\b(?:mình|tôi)\s+tên(?:\s+là)?\s+([^.,;!?]+)",
            r"\btên(?:\s+của)?\s+(?:mình|tôi)\s+là\s+([^.,;!?]+)",
            r"(?:^|:\s*)tên\s+(?:là\s+)?(?!(?:mình|tôi|của|là|gì|ai)\b)([^.,;!?]+)",
        ]
        for pattern in name_patterns:
            for match in re.finditer(pattern, sentence, re.I):
                value = _clean_fact(match[1])
                if value and value.lower() not in {"gì", "ai"}:
                    updates["name"] = value

        temporary_visit = re.search(r"\b(?:họp|công tác|du lịch)\b", sentence, re.I)
        explicit_residence = re.search(r"\b(?:sống|chuyển|nơi ở hiện tại)\b", sentence, re.I)
        if not temporary_visit or explicit_residence:
            location_pattern = (
                r"(?:\b(?:mình|tôi)\s+(?:(?:hiện tại|hiện|giờ|đang|vẫn)\s+)*"
                r"(?:(?:sinh sống|sống|làm việc)\s+)?ở\s+"
                r"|\bhiện(?: tại)?\s+ở\s+"
                r"|\bnơi ở(?:\s+(?:hiện tại|của mình))*\s*(?:là|:)\s+)"
                r"([^.,;!?]+)"
            )
            for match in re.finditer(location_pattern, sentence, re.I):
                value = _clean_fact(match[1])
                if value:
                    updates["location"] = value

        job = r"(?:(?:[a-z][\w+-]*\s+){0,3}(?:engineer|developer|manager|scientist)|giáo viên|bác sĩ|kỹ sư|lập trình viên|sinh viên)"
        job_prefix = (
            r"(?:\b(?:mình|tôi)\s+(?:(?:hiện tại|hiện|giờ|đang|vẫn)\s+)*(?:làm|là)"
            r"|\bvà đang làm|\bgiờ(?: mình)? chuyển sang"
            r"|\bnghề(?: nghiệp)?(?:\s+(?:hiện tại|thì|vẫn))*(?:\s+là)?)\s+"
        )
        for match in re.finditer(job_prefix + "(" + job + r")\b", sentence, re.I):
            updates["profession"] = match[1].strip()

        for key, label in [("favorite_drink", "đồ uống"), ("favorite_food", "món ăn")]:
            pattern = rf"\b{label}\s+yêu thích(?:\s+của mình)?\s*(?:là\s+|:\s*)([^.!?]+)"
            match = re.search(pattern, sentence, re.I)
            if match:
                updates[key] = _clean_fact(match[1])

        if re.search(r"\bmình\s+(?:(?:vẫn|còn|đang)\s+)*(?:thích|uống)\b", sentence, re.I):
            drink = re.search(r"\bcà phê sữa đá\b", sentence, re.I)
            if drink:
                updates["favorite_drink"] = drink[0]

        pet = re.search(r"\b(corgi|mèo|chó)\s+tên\s+([^.!?]+)", sentence, re.I)
        if pet and re.search(r"\b(?:mình nuôi|mình có|con|bé)\b", sentence, re.I):
            updates["pet"] = pet[1] + " tên " + _clean_fact(pet[2])

        style_intent = re.search(r"\b(?:mình muốn|mình vẫn muốn|hãy|style|phong cách)\b", sentence, re.I)
        style_features = re.search(r"\b(?:gọn|bullet|ví dụ|cấu trúc|trade-off|rõ ý|chi tiết)\b", sentence, re.I)
        replies = list(re.finditer(r"\btrả lời\b", sentence, re.I))
        if style_intent and style_features and replies:
            value = sentence[replies[-1].end():].strip(" :,.;")
            if value:
                updates["response_style"] = value[:240].strip()

        interest = re.search(
            r"\bmình\s+(?:(?:vẫn|còn|đang)\s+)*(?:thích|quan tâm(?: nhiều)?(?: đến)?)\s+(.+)",
            sentence, re.I,
        )
        if interest:
            updates["interests"] = interest[1].strip(" .")[:240].strip()
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Keep latest extracted facts and bounded, deterministic context excerpts.

    Previous summaries retain their structured facts. Context snippets keep an
    early anchor and the latest entries; discarded wording is intentionally lossy.
    """
    if not messages or max_items <= 0:
        return ""
    facts: dict[str, str] = {}
    snippets: list[str] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "summary":
            facts.update(_parse_facts(content))
            snippets.extend(line for line in content.splitlines() if re.match(r"^(?:user|assistant|system|tool): ", line))
        else:
            if role == "user":
                facts.update(extract_profile_updates(content))
            text = " ".join(content.split())
            if text:
                snippets.append(f"{role}: {text[:200]}")
    snippets = list(dict.fromkeys(snippets))
    if len(snippets) > max_items:
        snippets = [snippets[0]] + snippets[-(max_items - 1):] if max_items > 1 else snippets[:1]
    lines = [f"- {key}: {value}" for key, value in facts.items()]
    return "\n".join(lines + snippets)


def _bounded_summary(facts: dict[str, str], summary: str, max_chars: int) -> str:
    """Fit complete fact lines first, then use the remainder for lossy context."""
    lines: list[str] = []
    used = 0
    for key, value in facts.items():
        line = f"- {key}: {value}"
        cost = len(line) + bool(lines)
        if used + cost <= max_chars:
            lines.append(line)
            used += cost
    fact_text = "\n".join(lines)
    context_lines = [line for line in summary.splitlines() if not _FACT_LINE.fullmatch(line)]
    remaining = max(0, max_chars - used - bool(fact_text))
    context = "\n".join(context_lines)[:remaining].rstrip()
    return "\n".join(part for part in (fact_text, context) if part)


@dataclass
class CompactMemoryManager:
    """Keep per-thread recent messages verbatim and compact older content."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0:
            raise ValueError("threshold_tokens must be positive.")
        if self.keep_messages < 0:
            raise ValueError("keep_messages must not be negative.")

    def _thread(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        if "_facts" not in thread:
            facts = _parse_facts(cast(str, thread["summary"]))
            for message in cast(list[dict[str, str]], thread["messages"]):
                if message["role"] == "user":
                    facts.update(extract_profile_updates(message["content"]))
            thread["_facts"] = facts
        return thread

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages = cast(list[dict[str, str]], thread["messages"])
        messages.append({"role": role, "content": content})
        facts = cast(dict[str, str], thread["_facts"])
        if role == "user":
            facts.update(extract_profile_updates(content))
        tokens = estimate_tokens(cast(str, thread["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)
        if tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        split = len(messages) - self.keep_messages
        older = messages[:split]
        if thread["summary"]:
            older = [{"role": "summary", "content": cast(str, thread["summary"])}] + older
        summary = summarize_messages(older)
        # Bound summary separately; a retained message may itself exceed the threshold.
        summary_chars = 4 * max(1, self.threshold_tokens // 3)
        thread["summary"] = _bounded_summary(facts, summary, summary_chars)
        thread["messages"] = messages[split:]
        thread["compactions"] = cast(int, thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": [dict(message) for message in cast(list[dict[str, str]], thread["messages"])],
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return cast(int, self.state.get(thread_id, {}).get("compactions", 0))
