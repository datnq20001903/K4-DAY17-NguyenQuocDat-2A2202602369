from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


_RECALL_FIELDS = (
    ("name", "Tên", r"\btên\b"),
    ("profession", "Nghề nghiệp", r"\b(?:nghề|công việc|làm gì|chuyên môn)\b"),
    ("location", "Nơi ở", r"\b(?:nơi ở|ở đâu|đang ở|còn ở|sống ở)\b"),
    ("response_style", "Phong cách trả lời", r"\b(?:style|phong cách|kiểu trả lời|cách trả lời|trả lời.*thích|thích.*trả lời)\b"),
    ("favorite_drink", "Đồ uống", r"\b(?:đồ uống|thích uống|uống gì)\b"),
    ("favorite_food", "Món ăn", r"\b(?:món ăn|thích ăn)\b"),
    ("pet", "Thú cưng", r"\b(?:nuôi con|nuôi bé|nuôi gì|thú cưng|corgi|con chó|con mèo)\b"),
    ("interests", "Mối quan tâm", r"\b(?:quan tâm|sở thích|ngôn ngữ|công nghệ|kỹ thuật|thích gì)\b"),
)
_UNKNOWN = "Mình chưa có thông tin đó trong hồ sơ này."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent User.md + compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        return self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        agent_tokens = estimate_tokens(answer)

        self.compact_memory.append(thread_id, "assistant", answer)
        session.messages.append({"role": "assistant", "content": answer})
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary_text = str(context.get("summary", "") or "")
        recent_messages = "\n".join(message["content"] for message in context.get("messages", []))
        return estimate_tokens(profile_text) + estimate_tokens(summary_text) + estimate_tokens(recent_messages)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        profile_facts = self.profile_store.facts(user_id)
        thread_context = self.compact_memory.context(thread_id)
        summary = str(thread_context.get("summary", "") or "")
        recent_messages = [item["content"] for item in thread_context.get("messages", [])]
        for text in recent_messages:
            profile_facts.update(extract_profile_updates(text))
        for line in summary.splitlines():
            if line.startswith("- ") and ":" in line:
                key, value = line[2:].split(":", 1)
                profile_facts[key.strip()] = value.strip()

        query = unicodedata.normalize("NFC", message).casefold()
        is_query = "?" in query or re.search(r"\b(?:nhắc lại|tóm tắt|nhớ gì|tên gì|làm nghề gì|ở đâu|thế nào)\b", query)
        if not is_query:
            return "Đã ghi nhận thông tin."

        requested = [(key, label) for key, label, pattern in _RECALL_FIELDS if re.search(pattern, query)]
        if not requested and re.search(r"\b(?:về mình|về tôi|thông tin|hồ sơ)\b", query):
            requested = [(key, label) for key, label, _ in _RECALL_FIELDS if key in profile_facts]
        if not requested:
            requested = [(key, label) for key, label, _ in _RECALL_FIELDS if key in profile_facts]
        if not requested:
            return _UNKNOWN

        available = {key: profile_facts.get(key) for key, _ in requested if profile_facts.get(key)}
        if not available:
            return _UNKNOWN

        facts = []
        for key, label in requested:
            value = available.get(key)
            if value:
                facts.append(f"{label}: {value}")
        if not facts:
            return _UNKNOWN
        return "; ".join(facts) + "."

    def _maybe_build_langchain_agent(self):
        try:
            model = build_chat_model(self.config.model)
            if model is None:
                return None
            from langchain.agents import create_agent
        except ImportError:
            return None
        return create_agent(model=model, tools=[])

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        history = [dict(item) for item in session.messages] + [{"role": "user", "content": message}]
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in history)
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        result = self.langchain_agent.invoke({"messages": history}, config={"configurable": {"thread_id": thread_id}})
        returned_messages = result.get("messages", [])
        if not returned_messages:
            raise ValueError("Live agent returned no assistant message.")
        last_message = returned_messages[-1]
        content = last_message.content
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            answer = "".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            raise ValueError("Live agent returned unsupported assistant content.")
        if not answer.strip():
            raise ValueError("Live agent returned no assistant text.")
        usage = getattr(last_message, "usage_metadata", None) or {}
        output_tokens = usage.get("output_tokens")
        agent_tokens = output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else estimate_tokens(answer)
        session.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}
