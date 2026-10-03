from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_baseline
from agent_baseline import BaselineAgent, SessionState
from config import LabConfig
from memory_store import UserProfileStore


@pytest.fixture
def config(tmp_path):
    return LabConfig(base_dir=tmp_path, data_dir=tmp_path / "data", state_dir=tmp_path / "state")


@pytest.fixture
def baseline(config):
    return BaselineAgent(config, force_offline=True)


def test_same_user_remembers_only_inside_the_original_thread(baseline):
    baseline.reply("Lan", "first", "Mình tên là Lan.")
    assert "Lan" in baseline.reply("Lan", "first", "Mình tên gì?")["answer"]
    assert "Lan" not in baseline.reply("Lan", "second", "Mình tên gì?")["answer"]
    assert set(baseline.sessions) == {"first", "second"}


def test_interleaved_threads_and_new_instances_do_not_share_history(config, baseline):
    baseline.reply("u", "a", "Mình tên là Lan.")
    baseline.reply("u", "b", "Mình tên là Minh.")
    assert "Lan" in baseline.reply("u", "a", "Mình tên gì?")["answer"]
    assert "Minh" in baseline.reply("u", "b", "Mình tên gì?")["answer"]
    restarted = BaselineAgent(config, force_offline=True)
    assert "Lan" not in restarted.reply("u", "a", "Mình tên gì?")["answer"]


def test_corrections_apply_only_to_their_thread(baseline):
    baseline.reply("u", "a", "Mình ở Đà Nẵng và đang làm backend engineer.")
    baseline.reply("u", "a", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    baseline.reply("u", "a", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    answer = baseline.reply("u", "a", "Mình ở đâu và hiện tại làm nghề gì?")["answer"]
    assert "Huế" in answer and "MLOps engineer" in answer
    assert "Đà Nẵng" not in answer and "backend engineer" not in answer
    answer = baseline.reply("u", "b", "Mình ở đâu và hiện tại làm nghề gì?")["answer"]
    assert "Huế" not in answer and "MLOps engineer" not in answer


def test_assistant_text_is_not_treated_as_a_user_fact(baseline):
    baseline.sessions["a"] = SessionState(messages=[{"role": "assistant", "content": "Mình tên là InventedName."}])
    answer = baseline.reply("u", "a", "Mình tên gì?")["answer"]
    assert "InventedName" not in answer


@pytest.mark.parametrize("recall_text", [
    "Nhắc lại tên mình và nơi ở.",
    "Nhắc lại giúp mình: tên mình và nơi ở.",
    "Tóm tắt tên mình và nơi ở.",
])
def test_recall_requests_without_question_marks_do_not_overwrite_facts(baseline, recall_text):
    baseline.reply("u", "a", "Mình tên là Lan.")
    assert "Tên: Lan" in baseline.reply("u", "a", recall_text)["answer"]
    assert "Tên: Lan" in baseline.reply("u", "a", "Mình tên gì?")["answer"]


def test_a_declaration_and_recall_request_can_share_a_turn(baseline):
    result = baseline.reply("u", "a", "Mình tên là Lan. Nhắc lại tên mình và nơi ở.")
    assert "Tên: Lan" in result["answer"]
    assert "Tên: Lan" in baseline.reply("u", "a", "Mình tên gì?")["answer"]


def test_baseline_ignores_existing_profiles_and_performs_no_profile_io(config, monkeypatch):
    store = UserProfileStore(config.state_dir / "profiles")
    path = store.write_text("u", "# User profile\n- name: SecretProfileName\n")
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not access persistent profiles")

    for method in ("read_text", "write_text", "facts", "upsert_fact", "edit_text"):
        monkeypatch.setattr(UserProfileStore, method, forbidden)
    baseline = BaselineAgent(config, force_offline=True)
    answer = baseline.reply("u", "fresh", "Mình tên gì?")["answer"]
    assert "SecretProfileName" not in answer
    assert path.read_bytes() == before
    assert list(config.state_dir.rglob("User.md")) == [path]


def test_offline_baseline_creates_no_memory_files(config):
    baseline = BaselineAgent(config, force_offline=True)
    baseline.reply("u", "a", "Mình tên là Lan.")
    assert not config.state_dir.exists()


def test_output_and_prompt_counters_accumulate_every_turn(baseline):
    # Inputs have 8, 12, 12 characters; the deterministic acknowledgement has 12.
    # Prompt estimates are 2, (2+3+3), and (2+3+3+3+3), before the new reply.
    turns = [baseline.reply("u", "a", text) for text in ("abcdefgh", "abcdefghijkl", "abcdefghijkl")]
    assert [turn["agent_tokens"] for turn in turns] == [3, 3, 3]
    assert [turn["prompt_tokens"] for turn in turns] == [2, 8, 14]
    assert baseline.token_usage("a") == 9
    assert baseline.prompt_token_usage("a") == 24
    other = baseline.reply("u", "b", "abcdefgh")
    assert other["prompt_tokens"] == 2
    assert baseline.token_usage("a") == 9
    assert baseline.prompt_token_usage("a") == 24


def test_unknown_thread_metrics_do_not_create_a_session(baseline):
    assert baseline.token_usage("unknown") == 0
    assert baseline.prompt_token_usage("unknown") == 0
    assert baseline.compaction_count("unknown") == 0
    assert baseline.sessions == {}


def test_baseline_never_compacts_even_with_a_tiny_threshold(config):
    config.compact_threshold_tokens = 1
    config.compact_keep_messages = 1
    baseline = BaselineAgent(config, force_offline=True)
    results = [baseline.reply("u", "long", "abcdefgh") for _ in range(20)]
    assert len(baseline.sessions["long"].messages) == 40
    assert baseline.compaction_count("long") == 0
    assert [result["prompt_tokens"] for result in results] == list(range(2, 98, 5))
    assert baseline.prompt_token_usage("long") == 990


def test_offline_replies_are_deterministic(config):
    agents = [BaselineAgent(config, force_offline=True), BaselineAgent(config, force_offline=True)]
    messages = ["Mình tên là Lan.", "Mình tên gì?", "Mình muốn bạn trả lời ngắn gọn thành 3 bullet.", "Nhắc lại style trả lời mình thích."]
    results = [[agent.reply("u", "a", message) for message in messages] for agent in agents]
    assert results[0] == results[1]
    assert "3 bullet" in results[0][-1]["answer"]


@pytest.mark.parametrize("dataset", ["conversations.json", "advanced_long_context.json"])
def test_fixed_dataset_recall_questions_cannot_leak_previous_threads(baseline, dataset):
    data = Path(__file__).resolve().parent.parent / "data" / dataset
    for index, conversation in enumerate(json.loads(data.read_text(encoding="utf-8"))):
        user = conversation["user_id"]
        for message in conversation["turns"]:
            baseline.reply(user, f"train-{index}", message)
        for question_index, question in enumerate(conversation["recall_questions"]):
            answer = baseline.reply(user, f"recall-{index}-{question_index}", question["question"])["answer"]
            assert all(term.casefold() not in answer.casefold() for term in question["expected_contains"])


def test_force_offline_never_builds_a_live_model(config, monkeypatch):
    def forbidden(*args):
        raise AssertionError("force_offline must bypass live construction")

    monkeypatch.setattr(agent_baseline, "build_chat_model", forbidden)
    baseline = BaselineAgent(config, force_offline=True)
    assert baseline.reply("u", "a", "hello")["answer"]


def test_missing_cloud_key_falls_back_to_offline(config):
    baseline = BaselineAgent(config)
    assert baseline.langchain_agent is None
    baseline.reply("u", "a", "Mình tên là Lan.")
    assert "Lan" in baseline.reply("u", "a", "Mình tên gì?")["answer"]


def test_missing_optional_sdk_falls_back_to_offline(config, monkeypatch):
    def unavailable_sdk(provider_config):
        raise ImportError("optional provider SDK is not installed")

    monkeypatch.setattr(agent_baseline, "build_chat_model", unavailable_sdk)
    baseline = BaselineAgent(config)
    assert baseline.langchain_agent is None
    baseline.reply("u", "a", "Mình tên là Lan.")
    assert "Lan" in baseline.reply("u", "a", "Mình tên gì?")["answer"]


def test_invalid_live_configuration_is_not_silently_ignored(config, monkeypatch):
    def invalid_config(provider_config):
        raise ValueError("invalid provider configuration")

    monkeypatch.setattr(agent_baseline, "build_chat_model", invalid_config)
    with pytest.raises(ValueError, match="invalid provider configuration"):
        BaselineAgent(config)


@pytest.fixture
def live_model(monkeypatch):
    pytest.importorskip("langchain")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from pydantic import Field

    class RecordingChatModel(FakeMessagesListChatModel):
        seen: list = Field(default_factory=list)
        fail: bool = False

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.seen.append([(message.type, message.content) for message in messages])
            if self.fail:
                raise RuntimeError("provider unavailable")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = RecordingChatModel(responses=[AIMessage(content="abcdefgh", usage_metadata={"input_tokens": 99, "output_tokens": 7, "total_tokens": 106})])
    monkeypatch.setattr(agent_baseline, "build_chat_model", lambda config: model)
    return model


def test_live_graph_gets_only_the_selected_thread_history(config, live_model):
    baseline = BaselineAgent(config)
    first = baseline.reply("u", "a", "first")
    second = baseline.reply("u", "a", "second")
    baseline.reply("u", "b", "other")
    assert live_model.seen == [
        [("human", "first")],
        [("human", "first"), ("ai", "abcdefgh"), ("human", "second")],
        [("human", "other")],
    ]
    assert first["answer"] == second["answer"] == "abcdefgh"
    assert first["agent_tokens"] == second["agent_tokens"] == 7
    assert baseline.token_usage("a") == 14
    assert baseline.prompt_token_usage("a") == 5
    assert baseline.compaction_count("a") == 0


def test_live_text_blocks_and_missing_usage_metadata(config, live_model):
    from langchain_core.messages import AIMessage

    live_model.responses = [AIMessage(content=[{"type": "text", "text": "abc"}, {"type": "text", "text": "defgh"}])]
    result = BaselineAgent(config).reply("u", "a", "12345678")
    assert result == {"answer": "abcdefgh", "agent_tokens": 2, "prompt_tokens": 2}


def test_failed_live_turn_does_not_pollute_history_or_counters(config, live_model):
    baseline = BaselineAgent(config)
    baseline.reply("u", "a", "first")
    before = list(baseline.sessions["a"].messages)
    live_model.fail = True
    with pytest.raises(RuntimeError, match="provider unavailable"):
        baseline.reply("u", "a", "failed attempt")
    assert baseline.sessions["a"].messages == before
    assert baseline.token_usage("a") == 7
    assert baseline.prompt_token_usage("a") == 1
    live_model.fail = False
    baseline.reply("u", "a", "retry")
    assert live_model.seen[-1] == [("human", "first"), ("ai", "abcdefgh"), ("human", "retry")]


def test_empty_live_answer_does_not_commit_a_turn(config, live_model):
    from langchain_core.messages import AIMessage

    live_model.responses = [AIMessage(content="")]
    baseline = BaselineAgent(config)
    with pytest.raises(ValueError, match="no assistant text"):
        baseline.reply("u", "a", "hello")
    assert baseline.sessions == {}
    assert baseline.token_usage("a") == baseline.prompt_token_usage("a") == 0
