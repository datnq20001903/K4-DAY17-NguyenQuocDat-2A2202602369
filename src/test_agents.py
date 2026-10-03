from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated config with a low compact threshold."""
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify persisted UTF-8 markdown can be created and corrected."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    assert store.read_text("dungct") == ""
    content = "# User profile\n- name: DũngCT\n- location: Đà Nẵng\n"
    path = store.write_text("dungct", content)
    assert path.is_file()
    assert store.read_text("dungct") == content
    assert store.edit_text("dungct", "Đà Nẵng", "Huế")
    assert "Huế" in store.read_text("dungct")
    assert not store.edit_text("dungct", "Đà Nẵng", "Huế")
    assert store.file_size("dungct") == len(store.read_text("dungct").encode("utf-8"))


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify old messages become a summary while the recent tail stays intact."""
    config = make_config(tmp_path)
    manager = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    for index in range(12):
        manager.append("long", "user", f"turn {index}: " + "x" * 160)
    context = manager.context("long")
    assert context["summary"]
    assert manager.compaction_count("long") > 0
    assert len(context["messages"]) == config.compact_keep_messages
    assert context["messages"][-1]["content"].startswith("turn 11:")


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced agent persists user facts across threads; baseline stays local."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    baseline.reply("u", "train", "Mình tên là Lan. Mình ở Huế.")
    advanced.reply("u", "train", "Mình tên là Lan. Mình ở Huế.")

    baseline_answer = baseline.reply("u", "fresh", "Mình tên gì và hiện tại ở đâu?")["answer"]
    advanced_answer = advanced.reply("u", "fresh", "Mình tên gì và hiện tại ở đâu?")["answer"]

    assert "Lan" not in baseline_answer and "Huế" not in baseline_answer
    assert "Lan" in advanced_answer and "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Advanced compaction should reduce prompt context versus a full baseline thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for index in range(20):
        message = f"turn {index}: " + "x" * 120
        baseline.reply("u", "long-base", message)
        advanced.reply("u", "long-advanced", message)

    baseline_prompt = baseline.prompt_token_usage("long-base")
    advanced_prompt = advanced.prompt_token_usage("long-advanced")

    assert advanced.compaction_count("long-advanced") > 0
    assert advanced_prompt < baseline_prompt
    assert advanced.prompt_token_usage("long-advanced") > 0
