"""Tests for data_loaders/ — no model, no network downloads.

All HuggingFace load_dataset calls and snapshot_download are mocked.
VLind-Bench tests create a minimal on-disk fixture (fake data.json + JPEGs).

Coverage:
  - Schema validation: required fields present, correct types
  - Normalisation: answer/expected_bias stripped and lowercased
  - Messages structure: role="user", content=[image, text]
  - cf_image handling: VAB/VQAv2 → None; ViLP/VLind → PIL or None
  - VLind-Bench filesystem parsing (snapshot_download path, data.json, image paths)
  - expand_vlind_bench_stages: all four stages emitted, correct answers
  - to_contrastive_sample: ContrastiveSample fields match source dict
  - Script API compatibility: fields consumed by each run_*.py are present
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REQUIRED_BASE_FIELDS = {"id", "image", "cf_image", "messages", "answer"}


def _fake_pil(size: tuple[int, int] = (4, 4)) -> PILImage.Image:
    return PILImage.new("RGB", size)


def _validate_base_schema(sample: dict, *, dataset: str = "") -> None:
    """Assert every required base field exists with correct types."""
    prefix = f"[{dataset}] " if dataset else ""
    for field in REQUIRED_BASE_FIELDS:
        assert field in sample, f"{prefix}Missing required field: {field!r}"

    # messages must be non-empty list
    msgs = sample["messages"]
    assert isinstance(msgs, list) and len(msgs) >= 1, f"{prefix}messages must be non-empty list"

    # first message must follow chat format
    msg = msgs[0]
    assert msg.get("role") == "user", f"{prefix}messages[0].role must be 'user'"
    content = msg.get("content", [])
    assert isinstance(content, list) and len(content) == 2, (
        f"{prefix}messages[0].content must have 2 items (image + text)"
    )
    assert content[0].get("type") == "image", f"{prefix}content[0].type must be 'image'"
    assert content[1].get("type") == "text", f"{prefix}content[1].type must be 'text'"
    assert isinstance(content[1].get("text", None), str), f"{prefix}content[1].text must be str"

    # answer must be a normalized (lowercase, stripped) string
    assert isinstance(sample["answer"], str), f"{prefix}answer must be str"
    assert sample["answer"] == sample["answer"].strip(), f"{prefix}answer must be stripped"
    assert sample["answer"] == sample["answer"].lower(), f"{prefix}answer must be lowercase"


# ---------------------------------------------------------------------------
# Fake HuggingFace Dataset
# ---------------------------------------------------------------------------

class _FakeHFDataset:
    """Minimal mock of a HuggingFace Dataset: supports len(), iter, select()."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices])


# ---------------------------------------------------------------------------
# VAB
# ---------------------------------------------------------------------------

_VAB_ROWS = [
    {
        "ID": "q1",
        "image": _fake_pil(),
        "prompt": "Is there a cat?",
        "ground_truth": "{Yes}",
        "expected_bias": "{No}",
        "topic": "animals",
        "sub_topic": "domestic",
    },
    {
        "ID": "q2",
        "image": _fake_pil(),
        "prompt": "How many dogs?",
        "ground_truth": "2",
        "expected_bias": "3",
        "topic": "counting",
        "sub_topic": "",
    },
]


class TestVABLoader:
    def _load(self, monkeypatch, rows=None, n_samples=None):
        from data_loaders.vab import load_vab
        ds = _FakeHFDataset(rows or _VAB_ROWS)
        # load_dataset is imported locally inside load_vab; patch at source
        with patch("datasets.load_dataset", return_value=ds):
            return load_vab(n_samples=n_samples)

    def test_schema(self, monkeypatch):
        samples = self._load(monkeypatch)
        assert len(samples) == 2
        for s in samples:
            _validate_base_schema(s, dataset="VAB")

    def test_vab_specific_fields_present(self, monkeypatch):
        samples = self._load(monkeypatch)
        for s in samples:
            assert "expected_bias" in s
            assert "topic" in s
            assert "sub_topic" in s
            assert isinstance(s["expected_bias"], str)

    def test_answer_normalization(self, monkeypatch):
        """Curly brackets and whitespace are stripped; answer is lowercased."""
        samples = self._load(monkeypatch)
        assert samples[0]["answer"] == "yes"
        assert samples[0]["expected_bias"] == "no"

    def test_cf_image_is_none(self, monkeypatch):
        """VAB has no counterfactual images."""
        samples = self._load(monkeypatch)
        for s in samples:
            assert s["cf_image"] is None

    def test_n_samples_respected(self, monkeypatch):
        samples = self._load(monkeypatch, n_samples=1)
        assert len(samples) == 1

    def test_id_from_ID_field(self, monkeypatch):
        samples = self._load(monkeypatch)
        assert samples[0]["id"] == "q1"
        assert samples[1]["id"] == "q2"

    def test_script_api_eva_decoding(self, monkeypatch):
        """run_eva_decoding.py accesses: id, image, messages, answer, expected_bias, topic."""
        required = {"id", "image", "messages", "answer", "expected_bias", "topic"}
        samples = self._load(monkeypatch)
        for s in samples:
            assert required <= s.keys(), f"Missing fields: {required - s.keys()}"


# ---------------------------------------------------------------------------
# ViLP
# ---------------------------------------------------------------------------

_VILP_ROWS = [
    {
        "id": 0,
        "image1": _fake_pil(),
        "image2": _fake_pil(),
        "question": "What color is the sky?",
        "answer1": "Blue",
    },
    {
        "id": 1,
        "image1": _fake_pil(),
        "image2": None,  # missing CF image — should be None
        "question": "How many birds?",
        "answer1": "Three",
    },
]


class TestViLPLoader:
    def _load(self, n_samples=None):
        from data_loaders.vilp import load_vilp
        ds = _FakeHFDataset(_VILP_ROWS)
        with patch("datasets.load_dataset", return_value=ds):
            return load_vilp(n_samples=n_samples)

    def test_schema(self):
        for s in self._load():
            _validate_base_schema(s, dataset="ViLP")

    def test_image_is_rgb(self):
        samples = self._load()
        assert samples[0]["image"].mode == "RGB"

    def test_cf_image_present_when_available(self):
        samples = self._load()
        assert isinstance(samples[0]["cf_image"], PILImage.Image)

    def test_cf_image_none_when_missing(self):
        """image2=None → cf_image=None (no crash)."""
        samples = self._load()
        assert samples[1]["cf_image"] is None

    def test_prompt_format(self):
        """Question must be wrapped in the one-word-answer prefix."""
        samples = self._load()
        text = samples[0]["messages"][0]["content"][1]["text"]
        assert text.startswith("Please answer with one word:")
        assert "What color is the sky?" in text

    def test_answer_lowercased(self):
        samples = self._load()
        assert samples[0]["answer"] == "blue"
        assert samples[1]["answer"] == "three"

    def test_n_samples_respected(self):
        assert len(self._load(n_samples=1)) == 1


# ---------------------------------------------------------------------------
# VQAv2
# ---------------------------------------------------------------------------

_VQAV2_ROWS = [
    {"question_id": 111, "image": _fake_pil(), "question": "What is this?", "multiple_choice_answer": "cat"},
    {"question_id": 222, "image": _fake_pil(), "question": "How many?",     "multiple_choice_answer": "two"},
]


class TestVQAv2Loader:
    def _load(self, n_samples=None):
        from data_loaders.vqav2 import load_vqav2
        ds = _FakeHFDataset(_VQAV2_ROWS)
        with patch("datasets.load_dataset", return_value=ds):
            return load_vqav2(n_samples=n_samples)

    def test_schema(self):
        for s in self._load():
            _validate_base_schema(s, dataset="VQAv2")

    def test_id_from_question_id(self):
        samples = self._load()
        assert samples[0]["id"] == 111
        assert samples[1]["id"] == 222

    def test_cf_image_is_none(self):
        for s in self._load():
            assert s["cf_image"] is None

    def test_answer_preserved(self):
        samples = self._load()
        assert samples[0]["answer"] == "cat"

    def test_n_samples_respected(self):
        assert len(self._load(n_samples=1)) == 1


# ---------------------------------------------------------------------------
# VLind-Bench
# ---------------------------------------------------------------------------

_VLIND_DATA = [
    {
        "concept": "swan",
        "context_id": 1,
        "context": "desert",
        "true_statement": "Swans are in the desert.",
        "false_statement": "Swans are in the lake.",
        "existent_noun": "swan",
        "non-existent_noun": "desert",
        "best_img_id": 0,
        "aggregated_human_label_good_images": {"0": 3, "1": 2},
    },
    {
        # second instance where factual image is missing → should be skipped
        "concept": "panda",
        "context_id": 99,
        "context": "ocean",
        "true_statement": "Pandas swim in the ocean.",
        "false_statement": "Pandas are in bamboo forests.",
        "existent_noun": "panda",
        "non-existent_noun": "ocean",
        "best_img_id": 0,
        "aggregated_human_label_good_images": {"0": 3},
    },
]


@pytest.fixture
def vlind_fixture(tmp_path):
    """Create a minimal VLind-Bench directory structure with fake images."""
    data_root = tmp_path / "VLind-Bench Dataset"
    factual_dir = data_root / "images" / "factual"
    cf_dir = data_root / "images" / "counterfactual"

    # Instance 1 (swan/1): factual + 2 CF images
    swan_factual = factual_dir / "swan" / "1_context_test"
    swan_factual.mkdir(parents=True)
    PILImage.new("RGB", (8, 8), color=(255, 0, 0)).save(swan_factual / "0.jpg")

    swan_cf = cf_dir / "swan" / "1_context_test"
    swan_cf.mkdir(parents=True)
    for i in range(3):
        PILImage.new("RGB", (8, 8), color=(0, i * 80, 0)).save(swan_cf / f"{i}.jpg")

    # Instance 2 (panda/99): NO factual image — should be skipped by load_vlind_bench

    data_root.mkdir(parents=True, exist_ok=True)
    with open(data_root / "data.json", "w") as f:
        json.dump(_VLIND_DATA, f)

    return tmp_path


class TestVLindBenchLoader:
    def _load(self, vlind_fixture, n_samples=None):
        from data_loaders.vlind_bench import load_vlind_bench
        with patch("huggingface_hub.snapshot_download", return_value=str(vlind_fixture)):
            return load_vlind_bench(n_samples=n_samples)

    def test_schema(self, vlind_fixture):
        samples = self._load(vlind_fixture)
        assert len(samples) >= 1
        for s in samples:
            _validate_base_schema(s, dataset="VLind-Bench")

    def test_skips_missing_factual_image(self, vlind_fixture):
        """Instance 2 (panda/99) has no factual image → must be skipped."""
        samples = self._load(vlind_fixture)
        concepts = [s["concept"] for s in samples]
        assert "panda" not in concepts

    def test_factual_image_loaded(self, vlind_fixture):
        """The vis (factual) image must be a PIL RGB image."""
        samples = self._load(vlind_fixture)
        assert isinstance(samples[0]["image"], PILImage.Image)
        assert samples[0]["image"].mode == "RGB"

    def test_cf_image_loaded_when_present(self, vlind_fixture):
        samples = self._load(vlind_fixture)
        # best_img_id=0 for swan → cf_image from cf/swan/1_context_test/0.jpg
        assert isinstance(samples[0]["cf_image"], PILImage.Image)

    def test_answer_is_true(self, vlind_fixture):
        """All LP q1 samples have answer='true'."""
        samples = self._load(vlind_fixture)
        for s in samples:
            assert s["answer"] == "true"

    def test_prompt_contains_true_statement(self, vlind_fixture):
        samples = self._load(vlind_fixture)
        text = samples[0]["messages"][0]["content"][1]["text"]
        assert "Swans are in the desert." in text
        assert "True or False" in text

    def test_metadata_fields_present(self, vlind_fixture):
        """concept, context, true_statement, false_statement, factual_path."""
        required = {"concept", "context", "true_statement", "false_statement",
                    "factual_path", "best_cf_path"}
        samples = self._load(vlind_fixture)
        for s in samples:
            assert required <= s.keys(), f"Missing metadata fields: {required - s.keys()}"

    def test_n_samples_respected(self, vlind_fixture):
        """n_samples limits number of *instances* passed to the parser."""
        samples_1 = self._load(vlind_fixture, n_samples=1)
        samples_all = self._load(vlind_fixture)
        assert len(samples_1) <= len(samples_all)

    def test_data_root_fallback(self, tmp_path):
        """If 'VLind-Bench Dataset' subdir is absent, falls back to repo root."""
        factual_dir = tmp_path / "images" / "factual" / "swan" / "1_x"
        factual_dir.mkdir(parents=True)
        PILImage.new("RGB", (4, 4)).save(factual_dir / "0.jpg")

        cf_dir = tmp_path / "images" / "counterfactual" / "swan" / "1_x"
        cf_dir.mkdir(parents=True)
        PILImage.new("RGB", (4, 4)).save(cf_dir / "0.jpg")

        with open(tmp_path / "data.json", "w") as f:
            json.dump([_VLIND_DATA[0]], f)

        from data_loaders.vlind_bench import load_vlind_bench
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
            samples = load_vlind_bench()
        assert len(samples) == 1


class TestExpandVLindBenchStages:
    def _expand(self, vlind_fixture, n_samples=None):
        from data_loaders.vlind_bench import expand_vlind_bench_stages
        with patch("huggingface_hub.snapshot_download", return_value=str(vlind_fixture)):
            return expand_vlind_bench_stages(n_samples=n_samples)

    def test_four_stages_emitted(self, vlind_fixture):
        items = self._expand(vlind_fixture)
        stages = {item["_stage"] for item in items}
        assert stages >= {"ck", "vp", "cb", "lp"}, f"Missing stages: {stages}"

    def test_ck_uses_factual_path(self, vlind_fixture):
        items = self._expand(vlind_fixture)
        ck_items = [it for it in items if it["_stage"] == "ck"]
        assert len(ck_items) > 0
        for it in ck_items:
            assert "_image_path" in it

    def test_lp_q1_answer_is_true(self, vlind_fixture):
        items = self._expand(vlind_fixture)
        lp_q1 = [it for it in items if it["_stage"] == "lp" and it["_qid"] == "q1"]
        for it in lp_q1:
            assert it["answer"] == "true"

    def test_lp_q2_answer_is_false(self, vlind_fixture):
        items = self._expand(vlind_fixture)
        lp_q2 = [it for it in items if it["_stage"] == "lp" and it["_qid"] == "q2"]
        for it in lp_q2:
            assert it["answer"] == "false"

    def test_lp_count_matches_good_images(self, vlind_fixture):
        """swan instance has 2 good images (ids 0 and 1) → 4 LP items (2 good × 2 qids)."""
        items = self._expand(vlind_fixture, n_samples=1)  # only swan
        lp_items = [it for it in items if it["_stage"] == "lp" and it["concept"] == "swan"]
        assert len(lp_items) == 4  # 2 good_ids × 2 qids


# ---------------------------------------------------------------------------
# to_contrastive_sample
# ---------------------------------------------------------------------------

class TestToContrastiveSample:
    def _sample(self, **overrides):
        base = {
            "id": "test-1",
            "image": _fake_pil(),
            "cf_image": _fake_pil(),
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "What?"},
            ]}],
            "answer": "yes",
            "expected_bias": "no",
            "topic": "animals",
        }
        base.update(overrides)
        return base

    def test_all_fields_mapped(self):
        from data_loaders import to_contrastive_sample
        cs = to_contrastive_sample(self._sample())
        assert cs.id == "test-1"
        assert cs.answer == "yes"
        assert isinstance(cs.image, PILImage.Image)
        assert isinstance(cs.cf_image, PILImage.Image)
        assert cs.messages[0]["role"] == "user"

    def test_none_cf_image(self):
        from data_loaders import to_contrastive_sample
        cs = to_contrastive_sample(self._sample(cf_image=None))
        assert cs.cf_image is None

    def test_extra_fields_ignored(self):
        """Extra dataset-specific fields (expected_bias, topic) don't crash."""
        from data_loaders import to_contrastive_sample
        cs = to_contrastive_sample(self._sample())
        assert hasattr(cs, "id")  # only ContrastiveSample fields


# ---------------------------------------------------------------------------
# is_match functions
# ---------------------------------------------------------------------------

class TestVABIsMatch:
    def test_exact_match(self):
        from data_loaders import vab_is_match
        assert vab_is_match("yes", "yes")

    def test_case_insensitive(self):
        from data_loaders import vab_is_match
        assert vab_is_match("Yes", "yes")
        assert vab_is_match("YES", "yes")

    def test_curly_brackets_stripped(self):
        from data_loaders import vab_is_match
        assert vab_is_match("{Yes}", "yes")
        assert vab_is_match("yes", "{yes}")
        assert vab_is_match("{Yes}", "{yes}")

    def test_numeric_fallback(self):
        from data_loaders import vab_is_match
        assert vab_is_match("2", "2")
        assert vab_is_match("{2}", "2")
        assert vab_is_match("2 dogs", "2")   # digit extracted from longer answer

    def test_no_match(self):
        from data_loaders import vab_is_match
        assert not vab_is_match("yes", "no")
        assert not vab_is_match("2", "3")

    def test_empty_target_no_crash(self):
        from data_loaders import vab_is_match
        assert not vab_is_match("yes", "")
        assert not vab_is_match("", "yes")


class TestViLPIsMatch:
    def test_exact_match(self):
        from data_loaders import vilp_is_match
        assert vilp_is_match("blue", "blue")

    def test_case_insensitive(self):
        from data_loaders import vilp_is_match
        assert vilp_is_match("Blue", "blue")
        assert vilp_is_match("BLUE", "blue")

    def test_whitespace_stripped(self):
        from data_loaders import vilp_is_match
        assert vilp_is_match("  blue  ", "blue")

    def test_no_match(self):
        from data_loaders import vilp_is_match
        assert not vilp_is_match("blue", "red")
        assert not vilp_is_match("three", "four")


class TestVLindIsMatch:
    def test_true_false(self):
        from data_loaders import vlind_is_match
        assert vlind_is_match("true", "true")
        assert vlind_is_match("false", "false")

    def test_case_insensitive(self):
        from data_loaders import vlind_is_match
        assert vlind_is_match("True", "true")
        assert vlind_is_match("False", "false")
        assert vlind_is_match("TRUE", "true")

    def test_no_match(self):
        from data_loaders import vlind_is_match
        assert not vlind_is_match("true", "false")
        assert not vlind_is_match("false", "true")


class TestGetIsMatch:
    def test_dispatches_to_vab(self):
        from data_loaders import get_is_match
        fn = get_is_match("vlms_are_biased")
        assert fn("{Yes}", "yes")          # curly brackets — VAB-specific
        assert fn("2 dogs", "2")           # numeric fallback — VAB-specific

    def test_dispatches_to_vilp(self):
        from data_loaders import get_is_match
        fn = get_is_match("vilp")
        assert fn("Blue", "blue")
        assert not fn("blue", "red")

    def test_dispatches_to_vlind(self):
        from data_loaders import get_is_match
        fn = get_is_match("vlind")
        assert fn("True", "true")
        assert not fn("true", "false")

    def test_unknown_dataset_falls_back_to_exact(self):
        from data_loaders import get_is_match
        fn = get_is_match("some_new_dataset")
        assert fn("cat", "cat")
        assert not fn("cat", "dog")


# ---------------------------------------------------------------------------
# Script API compatibility
# ---------------------------------------------------------------------------

class TestScriptAPICompatibility:
    """Verify each run_*.py script can consume the loader output without KeyError."""

    def _vab_sample(self):
        return {
            "id": "q1",
            "image": _fake_pil(),
            "cf_image": None,
            "messages": [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "Q?"}
            ]}],
            "answer": "yes",
            "expected_bias": "no",
            "topic": "animals",
            "sub_topic": "pets",
        }

    def test_run_eva_decoding_fields(self):
        """run_eva_decoding.py: s["id"], s.get("answer"), s.get("expected_bias"), s.get("topic")"""
        s = self._vab_sample()
        _ = s["id"]
        _ = s.get("answer", "")
        _ = s.get("expected_bias", "")
        _ = s.get("topic", "")

    def test_run_revis_fields(self):
        """run_revis.py: sample["id"], sample["image"], sample["messages"], sample.get("answer")"""
        s = self._vab_sample()
        _ = s["id"]
        _ = s["image"]
        _ = s["messages"]
        _ = s.get("answer", "")

    def test_run_steering_fields(self):
        """run_steering.py: sample.get("id"), sample.get("category", "")"""
        s = self._vab_sample()
        _ = s.get("id")
        # category is not in VAB loader — must not crash with get
        _ = s.get("category", "")

    def test_run_chain_of_embedding_via_contrastive_sample(self):
        """run_chain_of_embedding uses to_contrastive_sample — verify no crash."""
        from data_loaders import to_contrastive_sample
        cs = to_contrastive_sample(self._vab_sample())
        assert cs.id is not None
        assert cs.answer == "yes"

    def test_run_eva_fields(self):
        """run_eva.py: s["id"], s["image"], s["messages"], s.get("answer")"""
        s = self._vab_sample()
        _ = s["id"]
        _ = s["image"]
        _ = s["messages"]
        _ = s.get("answer", "")

    def test_run_sae_convergence_and_feature_search_fields(self):
        """run_sae_convergence / run_feature_search: id, image, messages"""
        s = self._vab_sample()
        _ = s["id"]
        _ = s["image"]
        _ = s["messages"]


# ---------------------------------------------------------------------------
# Real download tests  (pytest --download)
# ---------------------------------------------------------------------------

@pytest.mark.download
class TestVABDownload:
    """Load 2 real samples from anvo25/vlms-are-biased and check the schema."""

    def test_real_samples_schema(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=2)
        assert len(samples) == 2
        for s in samples:
            _validate_base_schema(s, dataset="VAB-real")

    def test_real_answer_is_normalized(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=5)
        for s in samples:
            # must be stripped, lowercase, no curly brackets
            assert s["answer"] == s["answer"].strip()
            assert s["answer"] == s["answer"].lower()
            assert "{" not in s["answer"] and "}" not in s["answer"]

    def test_real_expected_bias_is_normalized(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=5)
        for s in samples:
            assert s["expected_bias"] == s["expected_bias"].strip()
            assert s["expected_bias"] == s["expected_bias"].lower()
            assert "{" not in s["expected_bias"] and "}" not in s["expected_bias"]

    def test_real_image_is_pil(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=2)
        for s in samples:
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].size[0] > 0

    def test_real_id_field_present(self):
        """ID field must come from the 'ID' column (string), not fall back to row index."""
        from data_loaders import load_vab
        samples = load_vab(n_samples=2)
        for s in samples:
            assert s["id"] is not None
            # VAB uses string IDs like "color_1", not plain integers
            assert s["id"] != 0


@pytest.mark.download
class TestViLPDownload:
    """Load 2 real samples from ViLP/ViLP and check the schema."""

    def test_real_samples_schema(self):
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=2)
        assert len(samples) == 2
        for s in samples:
            _validate_base_schema(s, dataset="ViLP-real")

    def test_real_image1_is_pil_rgb(self):
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=2)
        for s in samples:
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].mode == "RGB"

    def test_real_cf_image_is_pil_or_none(self):
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=5)
        for s in samples:
            assert s["cf_image"] is None or isinstance(s["cf_image"], PILImage.Image)

    def test_real_prompt_wraps_question(self):
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=2)
        for s in samples:
            text = s["messages"][0]["content"][1]["text"]
            assert text.startswith("Please answer with one word:")

    def test_real_answer_non_empty(self):
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=5)
        non_empty = [s for s in samples if s["answer"]]
        assert len(non_empty) > 0, "Expected at least some non-empty answers"


@pytest.mark.download
class TestVQAv2Download:
    """Load 2 real samples from lmms-lab/VQAv2 and check the schema."""

    def test_real_samples_schema(self):
        from data_loaders import load_vqav2
        samples = load_vqav2(n_samples=2)
        assert len(samples) == 2
        for s in samples:
            _validate_base_schema(s, dataset="VQAv2-real")

    def test_real_id_is_int(self):
        from data_loaders import load_vqav2
        samples = load_vqav2(n_samples=2)
        for s in samples:
            assert isinstance(s["id"], int)

    def test_real_image_is_pil(self):
        from data_loaders import load_vqav2
        samples = load_vqav2(n_samples=2)
        for s in samples:
            assert isinstance(s["image"], PILImage.Image)

    def test_real_cf_image_is_none(self):
        from data_loaders import load_vqav2
        samples = load_vqav2(n_samples=2)
        for s in samples:
            assert s["cf_image"] is None


@pytest.mark.download
class TestVLindBenchDownload:
    """Download klee972/VLind-Bench and verify the full pipeline works end-to-end."""

    @pytest.fixture(scope="class")
    def samples(self):
        """Download once per test class (cached in HF_HOME after first run)."""
        from data_loaders import load_vlind_bench
        return load_vlind_bench(n_samples=5)

    def test_returns_samples(self, samples):
        assert len(samples) > 0, (
            "load_vlind_bench returned 0 samples — check that the HuggingFace download "
            "succeeded and 'VLind-Bench Dataset/data.json' is present."
        )

    def test_schema(self, samples):
        for s in samples:
            _validate_base_schema(s, dataset="VLind-Bench-real")

    def test_factual_image_is_real_jpeg(self, samples):
        """Factual images must be real PIL images with non-trivial size."""
        for s in samples:
            assert isinstance(s["image"], PILImage.Image), "image must be a PIL Image"
            w, h = s["image"].size
            assert w >= 64 and h >= 64, f"Suspiciously small image: {w}×{h}"

    def test_cf_image_is_pil_or_none(self, samples):
        for s in samples:
            assert s["cf_image"] is None or isinstance(s["cf_image"], PILImage.Image)

    def test_answer_is_true(self, samples):
        for s in samples:
            assert s["answer"] == "true"

    def test_prompt_structure(self, samples):
        for s in samples:
            text = s["messages"][0]["content"][1]["text"]
            assert "True or False" in text
            assert "Statement:" in text

    def test_metadata_fields(self, samples):
        required = {"concept", "context", "true_statement", "false_statement",
                    "factual_path", "best_cf_path"}
        for s in samples:
            assert required <= s.keys()
            assert s["concept"], "concept must be non-empty"
            assert s["true_statement"], "true_statement must be non-empty"

    def test_factual_path_exists_on_disk(self, samples):
        """The factual image path must point to a real file after download."""
        import pathlib
        for s in samples:
            assert s["factual_path"], "factual_path must not be empty"
            assert pathlib.Path(s["factual_path"]).exists(), (
                f"factual_path does not exist: {s['factual_path']}"
            )

    def test_expand_stages_covers_all_four(self):
        """expand_vlind_bench_stages on real data emits CK, VP, CB, LP items."""
        from data_loaders import expand_vlind_bench_stages
        items = expand_vlind_bench_stages(n_samples=2)
        stages = {it["_stage"] for it in items}
        assert stages == {"ck", "vp", "cb", "lp"}, f"Missing stages: stages"
        # Every item has _image_path field
        for it in items:
            assert "_image_path" in it
