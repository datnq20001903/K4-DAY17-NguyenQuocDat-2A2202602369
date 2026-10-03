from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_store import (
    CompactMemoryManager, UserProfileStore, estimate_tokens,
    extract_profile_updates, summarize_messages,
)


@pytest.mark.parametrize("text,expected", [(None, 0), ("", 0), (" \n\t ", 0), ("a", 1), ("abcd", 1), ("abcde", 1), ("abcdefgh", 2)])
def test_token_estimate_uses_trimmed_character_count(text, expected):
    assert estimate_tokens(text) == expected
    assert estimate_tokens(text) == estimate_tokens(text)


def test_token_estimate_is_monotonic():
    estimates = [estimate_tokens("x" * length) for length in range(200)]
    assert estimates == sorted(estimates)


def test_profile_persists_utf8_text_and_reports_bytes(tmp_path):
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0
    text = "# User profile\n- name: DũngCT\n- location: Đà Nẵng\n"
    path = store.write_text("dungct", text)
    assert path == tmp_path / "profiles" / "dungct" / "User.md"
    assert UserProfileStore(store.root_dir).read_text("dungct") == text
    assert store.file_size("dungct") == len(text.encode("utf-8"))


@pytest.mark.parametrize("user_id", ["../outside", "../../", "C:\\Windows\\Temp", "a/b", "a\\b", "CON"])
def test_profile_ids_cannot_escape_the_root(tmp_path, user_id):
    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text(user_id, "profile")
    assert path.resolve().is_relative_to(store.root_dir.resolve())
    assert path.name == "User.md"
    assert store.read_text(user_id) == "profile"


@pytest.mark.parametrize("user_id", ["", " \n "])
def test_empty_profile_id_is_rejected(tmp_path, user_id):
    with pytest.raises(ValueError, match="user_id"):
        UserProfileStore(tmp_path).path_for(user_id)


def test_sanitized_ids_do_not_share_a_profile(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.path_for("a/b") != store.path_for("a_b")
    assert store.path_for("Alice") != store.path_for("alice")
    assert store.path_for("a/b") == store.path_for("a/b")


def test_a_generated_folder_name_cannot_impersonate_the_original_user(tmp_path):
    store = UserProfileStore(tmp_path)
    original_id = "a/b"
    generated_name = store.path_for(original_id).parent.name
    assert store.path_for(generated_name) != store.path_for(original_id)
    store.write_text(original_id, "original user")
    store.write_text(generated_name, "other user")
    assert store.read_text(original_id) == "original user"
    assert store.read_text(generated_name) == "other user"


def test_edit_changes_only_one_occurrence_and_is_truthful(tmp_path):
    store = UserProfileStore(tmp_path)
    assert not store.edit_text("u", "absent", "new")
    assert not store.path_for("u").exists()
    store.write_text("u", "Huế / Huế")
    assert store.edit_text("u", "Huế", "Đà Nẵng")
    assert store.read_text("u") == "Đà Nẵng / Huế"
    assert not store.edit_text("u", "missing", "new")
    assert not store.edit_text("u", "Huế", "Huế")
    assert not store.edit_text("u", "", "insert")


def test_fact_correction_is_idempotent_and_keeps_manual_notes(tmp_path):
    store = UserProfileStore(tmp_path)
    store.write_text("u", "# User profile\n\nManual note to keep.\n")
    assert store.upsert_fact("u", "location", "Đà Nẵng")
    assert store.upsert_fact("u", "location", "Huế")
    size = store.file_size("u")
    assert not store.upsert_fact("u", "location", "Huế")
    assert store.file_size("u") == size
    assert store.facts("u") == {"location": "Huế"}
    text = store.read_text("u")
    assert text.count("- location:") == 1
    assert "Đà Nẵng" not in text
    assert "Manual note to keep." in text


def test_upsert_repairs_duplicate_keys(tmp_path):
    store = UserProfileStore(tmp_path)
    store.write_text("u", "- location: Huế\n- location: Đà Nẵng\n")
    assert store.upsert_fact("u", "location", "Huế")
    assert store.read_text("u").count("- location:") == 1


@pytest.mark.parametrize("message,expected", [
    ("Chào bạn, mình tên là DũngCT.", {"name": "DũngCT"}),
    ("Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.", {"location": "Đà Nẵng", "profession": "backend engineer"}),
    ("Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.", {"location": "Huế"}),
    ("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.", {"profession": "MLOps engineer"}),
    ("Mình nuôi một bé corgi tên Bơ.", {"pet": "corgi tên Bơ"}),
    ("Đồ uống yêu thích là cà phê sữa đá.", {"favorite_drink": "cà phê sữa đá"}),
    ("Món ăn yêu thích là mì Quảng.", {"favorite_food": "mì Quảng"}),
])
def test_stable_personal_facts_are_extracted(message, expected):
    updates = extract_profile_updates(message)
    for key, value in expected.items():
        assert updates[key] == value


@pytest.mark.parametrize("message", [
    "Mình tên gì?", "Bạn có thể nhắc lại tên mình không?",
    "Mình đang ở Hà Nội phải không?", "Mình tên là Lan phải không?",
    "Mình đùa là chuyển sang product manager cho đỡ mệt.",
    "Mình ở Hà Nội để họp hai ngày.",
    "Nếu sau này mình nhắc Đà Nẵng như ví dụ cũ thì đừng lấy làm nơi ở.",
    "Mình không còn ở Huế nữa.", "Mình không làm backend engineer nữa.",
])
def test_questions_negations_and_temporary_noise_do_not_add_facts(message):
    assert extract_profile_updates(message) == {}


def test_a_declaration_before_a_question_is_still_kept():
    assert extract_profile_updates("Mình tên là Lan. Bạn có nhớ mình không?") == {"name": "Lan"}


@pytest.mark.parametrize("recall_text", [
    "Nhắc lại tên mình và nơi ở.",
    "Nhắc lại giúp mình: tên mình và nơi ở.",
    "Tóm tắt tên mình và nơi ở.",
])
def test_recall_commands_are_not_new_profile_facts(recall_text):
    assert extract_profile_updates(recall_text) == {}
    assert extract_profile_updates("Mình tên là Lan. " + recall_text) == {"name": "Lan"}


@pytest.mark.parametrize("declaration", ["Tên mình là Lan.", "Tên của mình là Lan."])
def test_explicit_name_declarations_keep_the_value_only(declaration):
    assert extract_profile_updates(declaration) == {"name": "Lan"}


def test_explicit_facts_after_a_recap_colon_are_preserved():
    updates = extract_profile_updates("Nhắc lại lần cuối cho chắc: tên Lan, nghề MLOps engineer, nơi ở hiện tại là Huế.")
    assert updates == {"name": "Lan", "profession": "MLOps engineer", "location": "Huế"}


@pytest.mark.parametrize("message", [
    "Thông tin ổn định gồm tên, nghề nghiệp mới, nơi ở hiện tại, đồ uống yêu thích và style trả lời.",
    "Hãy nhớ món ăn yêu thích và tên của mình.",
])
def test_mentions_of_fact_labels_do_not_overwrite_their_values(message):
    assert extract_profile_updates(message) == {}


def test_response_style_preserves_explicit_bullet_count():
    updates = extract_profile_updates("Mình muốn bạn trả lời ngắn gọn thành 3 bullet, có ví dụ thực chiến.")
    assert "3 bullet" in updates["response_style"]
    assert "ví dụ thực chiến" in updates["response_style"]


@pytest.mark.parametrize("dataset,user_id,expected", [
    ("conversations.json", "dungct", {"name": "DũngCT", "location": "Huế", "profession": "MLOps engineer", "favorite_drink": "cà phê sữa đá", "favorite_food": "mì Quảng"}),
    ("advanced_long_context.json", "dungct_stress", {"name": "DũngCT Stress", "location": "Đà Nẵng", "profession": "MLOps engineer"}),
])
def test_fixed_dataset_facts_survive_corrections_and_noise(tmp_path, dataset, user_id, expected):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    conversations = json.loads((data_dir / dataset).read_text(encoding="utf-8"))
    store = UserProfileStore(tmp_path)
    for conversation in conversations:
        for message in conversation["turns"]:
            for key, value in extract_profile_updates(message).items():
                store.upsert_fact(user_id, key, value)
    facts = UserProfileStore(tmp_path).facts(user_id)
    for key, value in expected.items():
        assert facts[key] == value
    if user_id == "dungct_stress":
        assert "3 bullet" in facts["response_style"]


def test_summary_is_deterministic_and_keeps_latest_facts():
    messages = [
        {"role": "user", "content": "Mình tên là Lan. Mình ở Đà Nẵng."},
        {"role": "user", "content": "Giờ mình đang ở Huế."},
        {"role": "assistant", "content": "Đã ghi nhận."},
    ]
    summary = summarize_messages(messages)
    assert summary == summarize_messages(messages)
    assert "name: Lan" in summary
    assert "location: Huế" in summary
    assert "assistant:" in summary
    assert summarize_messages([]) == ""
    assert summarize_messages(messages, max_items=0) == ""


def test_compaction_keeps_recent_messages_verbatim_and_isolates_threads():
    manager = CompactMemoryManager(80, 2)
    messages = [{"role": "user", "content": "Mình tên là Lan." + " x" * 100}]
    messages += [{"role": "assistant", "content": f"turn {index} " + "x" * 120} for index in range(8)]
    for message in messages:
        manager.append("long", **message)
    context = manager.context("long")
    assert manager.compaction_count("long") > 1
    assert context["messages"] == messages[-2:]
    assert context["summary"]
    assert estimate_tokens(context["summary"]) <= 80 // 3
    assert manager.context("other") == {"messages": [], "summary": "", "compactions": 0}


def test_small_thread_does_not_compact_and_context_is_a_snapshot():
    manager = CompactMemoryManager(100, 2)
    manager.append("t", "user", "hello")
    snapshot = manager.context("t")
    snapshot["messages"][0]["content"] = "changed outside"
    assert manager.context("t")["messages"] == [{"role": "user", "content": "hello"}]
    assert manager.compaction_count("t") == 0
    assert manager.compaction_count("unknown") == 0


def test_summary_budget_never_splits_facts_and_larger_budget_can_recover_them():
    manager = CompactMemoryManager(80, 1)
    name = "N" * 150
    manager.append("t", "user", f"Mình tên là {name}. Mình ở Huế.")
    manager.append("t", "assistant", "x" * 200)
    summary = manager.context("t")["summary"]
    assert "- location: Huế" in summary
    name_lines = [line for line in summary.splitlines() if line.startswith("- name:")]
    assert not name_lines or name_lines == [f"- name: {name}"]
    assert estimate_tokens(summary) <= 80 // 3
    manager.threshold_tokens = 1000
    manager.append("t", "assistant", "y" * 4500)
    summary = manager.context("t")["summary"]
    assert f"- name: {name}" in summary.splitlines()
    assert "- location: Huế" in summary.splitlines()


def test_overlarge_recent_message_is_preserved_without_fake_compaction():
    manager = CompactMemoryManager(10, 2)
    content = "x" * 1000
    manager.append("t", "user", content)
    assert manager.context("t")["messages"] == [{"role": "user", "content": content}]
    assert manager.compaction_count("t") == 0


def test_zero_kept_messages_uses_a_summary_without_retaining_all_messages():
    manager = CompactMemoryManager(20, 0)
    manager.append("t", "user", "x" * 200)
    assert manager.context("t")["messages"] == []
    assert manager.context("t")["summary"]
    assert manager.compaction_count("t") == 1


@pytest.mark.parametrize("threshold,keep", [(0, 2), (-1, 2), (100, -1)])
def test_invalid_compaction_settings_are_rejected(threshold, keep):
    with pytest.raises(ValueError):
        CompactMemoryManager(threshold, keep)


def test_stress_data_compacts_repeatedly_and_reduces_prompt_load():
    path = Path(__file__).resolve().parent.parent / "data" / "advanced_long_context.json"
    turns = json.loads(path.read_text(encoding="utf-8"))[0]["turns"]
    manager = CompactMemoryManager(1000, 6)
    repeated = CompactMemoryManager(1000, 6)
    baseline_messages = []
    baseline_load = compact_load = 0
    for turn in turns:
        for role, content in [("user", turn), ("assistant", "Đã ghi nhận thông tin.")]:
            baseline_messages.append(content)
            manager.append("stress", role, content)
            repeated.append("stress", role, content)
        context = manager.context("stress")
        baseline_load += sum(estimate_tokens(text) for text in baseline_messages)
        compact_load += estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in context["messages"])
    assert manager.compaction_count("stress") > 1
    assert compact_load < baseline_load
    assert manager.context("stress") == repeated.context("stress")
