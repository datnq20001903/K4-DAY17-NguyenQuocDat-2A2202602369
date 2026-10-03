from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
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
_UNKNOWN = "Mình chưa có thông tin đó trong thread này."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: full, in-memory thread history with no persistent profile or compact."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return answer and per-turn agent/prompt tokens; user_id is interface-only.

        Session identity is exclusively thread_id. The caller must use a fresh
        thread ID for a new conversation, including conversations of the same user.
        """
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        answer = self._offline_answer(session.messages)
        agent_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    @staticmethod
    def _offline_answer(messages: list[dict[str, str]]) -> str:
        query = unicodedata.normalize("NFC", messages[-1]["content"]).casefold()
        is_query = "?" in query or re.search(r"\b(?:nhắc lại|tóm tắt|nhớ gì|tên gì|làm nghề gì|ở đâu|thế nào)\b", query)
        if not is_query:
            return "Đã ghi nhận."

        # Rebuild temporary facts from this thread's user messages only.
        # No fact cache is retained on the agent or shared between sessions.
        facts: dict[str, str] = {}
        for item in messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        requested = [(key, label) for key, label, pattern in _RECALL_FIELDS if re.search(pattern, query)]
        if not requested and re.search(r"\b(?:về mình|về tôi|thông tin|hồ sơ)\b", query):
            requested = [(key, label) for key, label, _ in _RECALL_FIELDS if key in facts]
        if not any(key in facts for key, _ in requested):
            return _UNKNOWN
        return "; ".join(
            f"{label}: {facts.get(key, 'chưa có thông tin trong thread này')}"
            for key, label in requested
        ) + "."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.get(thread_id, SessionState())
        user_message = {"role": "user", "content": message}
        # Fresh dictionaries protect local history from mutation by a runnable.
        history = [dict(item) for item in session.messages] + [user_message]
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in history)
        result = self.langchain_agent.invoke(
            {"messages": history},
            config={"configurable": {"thread_id": thread_id}},
        )
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
        # Commit only after invocation and response validation have succeeded.
        session.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        self.sessions[thread_id] = session
        return {"answer": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Optionally build a stateless graph; SessionState owns all thread history."""
        try:
            model = build_chat_model(self.config.model)
            if model is None:
                return None
            from langchain.agents import create_agent
        except ImportError:
            return None
        return create_agent(model=model, tools=[])
