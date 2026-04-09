"""Tests for data_loaders/ — no model, no network downloads.

All HuggingFace load_dataset calls and snapshot_download are mocked.
VLind-Bench tests create a minimal on-disk fixture (fake data.json + JPEGs).

Coverage:
  - Schema validation: required fields present, correct types
  - Normalisation: answer/expected_bias stripped and lowercased
  - Messages structure: role="user", content=[image, text]
  - cf_image handling: VAB/VQAv2 → None; ViLP CoE → PIL; ViLP expanded → None
  - VLind-Bench filesystem parsing (snapshot_download path, data.json, image paths)
  - expand_vlind_bench_stages: all four stages emitted, correct answers
  - to_contrastive_sample: ContrastiveSample fields match source dict
  - Script API compatibility: fields consumed by each run_*.py are present
  - ViLP load_vilp(): 2 CF pairs per question, image=image1, cf_image=image2/3,
    answer=answer1, cf_pair_idx, without_fact/with_fact modes, CoE compatibility
  - ViLP load_vilp_expanded(): cf_only vs all modes, image_idx, answer per image
  - ViLP compute_vilp_metrics(): vilp_score, vilp_prior, edge cases
  - ViLP normalize_output(): number words, synonyms, plurals, trailing period
  - ViLP is_match(): normalization pipeline, 'none' exclusion
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

    # --- resolution filtering ---

    def _load_with_res(self, rows, resolution):
        from data_loaders.vab import load_vab
        ds = _FakeHFDataset(rows)
        with patch("datasets.load_dataset", return_value=ds):
            return load_vab(resolution=resolution)

    def _pil_with_filename(self, filename: str) -> PILImage.Image:
        img = _fake_pil()
        img.filename = filename
        return img

    def _rows_with_resolutions(self):
        """Three rows representing the same question at 384/768/1152 px."""
        base = {"prompt": "Q?", "ground_truth": "yes", "expected_bias": "no",
                "topic": "t", "sub_topic": "s"}
        return [
            {"ID": "r384",  "image": self._pil_with_filename("img_384.png"),  **base},
            {"ID": "r768",  "image": self._pil_with_filename("img_768.png"),  **base},
            {"ID": "r1152", "image": self._pil_with_filename("img_1152.png"), **base},
        ]

    def test_resolution_filter_keeps_matching(self):
        rows = self._rows_with_resolutions()
        samples = self._load_with_res(rows, resolution=1152)
        assert len(samples) == 1
        assert samples[0]["id"] == "r1152"

    def test_resolution_filter_other_value(self):
        rows = self._rows_with_resolutions()
        samples = self._load_with_res(rows, resolution=384)
        assert len(samples) == 1
        assert samples[0]["id"] == "r384"

    def test_resolution_filter_none_keeps_all(self):
        rows = self._rows_with_resolutions()
        samples = self._load_with_res(rows, resolution=None)
        assert len(samples) == 3

    def test_resolution_filter_via_image_path_field(self):
        """Falls back to image_path field when PIL image has no .filename."""
        base = {"prompt": "Q?", "ground_truth": "yes", "expected_bias": "no",
                "topic": "t", "sub_topic": "s"}
        rows = [
            {"ID": "a", "image": _fake_pil(), "image_path": "img_1152.png", **base},
            {"ID": "b", "image": _fake_pil(), "image_path": "img_384.png",  **base},
        ]
        samples = self._load_with_res(rows, resolution=1152)
        assert len(samples) == 1
        assert samples[0]["id"] == "a"

    def test_resolution_filter_no_filename_passes_through(self):
        """Rows with no detectable filename are kept (can't be filtered out)."""
        base = {"prompt": "Q?", "ground_truth": "yes", "expected_bias": "no",
                "topic": "t", "sub_topic": "s"}
        rows = [{"ID": "x", "image": _fake_pil(), **base}]  # no filename, no image_path
        samples = self._load_with_res(rows, resolution=1152)
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# ViLP
# ---------------------------------------------------------------------------

# Two questions:
#   q0: has fact sentence, all 3 images present
#   q1: no fact sentence (no "."), image3=None
_VILP_ROWS = [
    {
        "question": "A drone typically has four rotors. How many rotors does the drone have?",
        "image1": _fake_pil(),
        "image2": _fake_pil(),
        "image3": _fake_pil(),
        "answer1": "Four",
        "answer2": "Two",
        "answer3": "Six",
    },
    {
        "question": "How many birds are in the tree?",
        "image1": _fake_pil(),
        "image2": _fake_pil(),
        "image3": None,   # missing third image
        "answer1": "Three",
        "answer2": "One",
        "answer3": "",
    },
]


def _load_vilp(n_samples=None, mode="without_fact"):
    from data_loaders.vilp import load_vilp
    ds = _FakeHFDataset(_VILP_ROWS)
    with patch("datasets.load_dataset", return_value=ds):
        return load_vilp(n_samples=n_samples, mode=mode)


def _load_vilp_expanded(n_samples=None, mode="without_fact", images="cf_only"):
    from data_loaders.vilp import load_vilp_expanded
    ds = _FakeHFDataset(_VILP_ROWS)
    with patch("datasets.load_dataset", return_value=ds):
        return load_vilp_expanded(n_samples=n_samples, mode=mode, images=images)


class TestViLPLoader:
    """Tests for load_vilp() — CoE CF-pair interface."""

    def test_two_pairs_per_question(self):
        """Each question yields exactly 2 CF pairs (cf2 and cf3)."""
        samples = _load_vilp()
        assert len(samples) == 4  # 2 questions × 2 pairs

    def test_n_samples_limits_questions_not_pairs(self):
        """n_samples=1 → 1 question → 2 pairs."""
        samples = _load_vilp(n_samples=1)
        assert len(samples) == 2

    def test_cf_pair_idx_values(self):
        """cf_pair_idx must be 2 or 3 for every sample."""
        for s in _load_vilp():
            assert s["cf_pair_idx"] in (2, 3), f"Unexpected cf_pair_idx: {s['cf_pair_idx']}"

    def test_both_pairs_present_per_question(self):
        """Each question_id must have exactly one cf2 and one cf3 sample."""
        from collections import defaultdict
        by_qid = defaultdict(set)
        for s in _load_vilp():
            by_qid[s["question_id"]].add(s["cf_pair_idx"])
        for qid, idxs in by_qid.items():
            assert idxs == {2, 3}, f"q{qid} missing a pair: {idxs}"

    def test_image_is_image1_lp_aligned(self):
        """image (vis condition) must be image1 for both pairs of a question."""
        samples = _load_vilp(n_samples=1)   # q0 only → 2 pairs
        # Both pairs for q0 share the same image1 — check they're PIL RGB
        for s in samples:
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].mode == "RGB"

    def test_cf_image_for_cf2_pair(self):
        """cf_image must be image2 for cf_pair_idx=2."""
        samples = _load_vilp()
        cf2 = [s for s in samples if s["cf_pair_idx"] == 2]
        for s in cf2:
            assert isinstance(s["cf_image"], PILImage.Image)

    def test_cf_image_none_when_image3_missing(self):
        """If image3 is None in the dataset, cf_image=None for cf3 pair."""
        samples = _load_vilp()
        # q1 (question_id=1) has image3=None
        cf3_q1 = [s for s in samples if s["question_id"] == 1 and s["cf_pair_idx"] == 3]
        assert len(cf3_q1) == 1
        assert cf3_q1[0]["cf_image"] is None

    def test_answer_is_answer1(self):
        """answer must always be answer1 (LP-aligned ground truth)."""
        samples = _load_vilp()
        # q0 answer1="Four" → normalized "4"
        q0 = [s for s in samples if s["question_id"] == 0]
        for s in q0:
            assert s["answer"] == "4"    # "Four" → number mapping
        # q1 answer1="Three" → normalized "3"
        q1 = [s for s in samples if s["question_id"] == 1]
        for s in q1:
            assert s["answer"] == "3"

    def test_answer1_answer2_answer3_all_present(self):
        """answer1/2/3 fields must be present for downstream use."""
        for s in _load_vilp():
            assert "answer1" in s
            assert "answer2" in s
            assert "answer3" in s

    def test_id_format(self):
        """id must follow '{q_idx}_cf2' / '{q_idx}_cf3' format."""
        samples = _load_vilp()
        ids = {s["id"] for s in samples}
        assert "0_cf2" in ids
        assert "0_cf3" in ids
        assert "1_cf2" in ids
        assert "1_cf3" in ids

    def test_schema(self):
        for s in _load_vilp():
            _validate_base_schema(s, dataset="ViLP-CoE")

    def test_without_fact_strips_leading_sentence(self):
        """without_fact mode removes the first sentence (the LP fact)."""
        samples = _load_vilp(mode="without_fact", n_samples=1)
        text = samples[0]["messages"][0]["content"][1]["text"]
        # Fact sentence must be gone
        assert "A drone typically has four rotors." not in text
        # Question must remain
        assert "How many rotors" in text

    def test_with_fact_keeps_full_question(self):
        """with_fact mode keeps the entire question including the fact sentence."""
        samples = _load_vilp(mode="with_fact", n_samples=1)
        text = samples[0]["messages"][0]["content"][1]["text"]
        assert "A drone typically has four rotors." in text
        assert "How many rotors" in text

    def test_no_fact_sentence_question_unmodified(self):
        """Questions with no '.' are unchanged regardless of mode."""
        samples_wof = _load_vilp(mode="without_fact")
        samples_wf = _load_vilp(mode="with_fact")
        q1_wof = [s for s in samples_wof if s["question_id"] == 1][0]
        q1_wf  = [s for s in samples_wf  if s["question_id"] == 1][0]
        text_wof = q1_wof["messages"][0]["content"][1]["text"]
        text_wf  = q1_wf ["messages"][0]["content"][1]["text"]
        assert text_wof == text_wf
        assert "How many birds" in text_wof

    def test_prompt_prefix(self):
        for s in _load_vilp():
            text = s["messages"][0]["content"][1]["text"]
            assert text.startswith("Please answer with one word:")

    def test_coe_compatible_via_to_contrastive_sample(self):
        """load_vilp() output can be passed directly to to_contrastive_sample."""
        from data_loaders import to_contrastive_sample
        for raw in _load_vilp(n_samples=1):
            cs = to_contrastive_sample(raw)
            assert cs.id is not None
            assert isinstance(cs.image, PILImage.Image)
            # cf_image may be PIL or None — both are valid
            assert cs.cf_image is None or isinstance(cs.cf_image, PILImage.Image)
            assert cs.answer  # non-empty answer1


class TestViLPExpandedLoader:
    """Tests for load_vilp_expanded() — eval experiment interface."""

    def test_cf_only_returns_two_per_question(self):
        """images='cf_only' → images 2+3, so 2 samples per question."""
        samples = _load_vilp_expanded(images="cf_only")
        assert len(samples) == 4   # 2 questions × 2 images

    def test_all_returns_three_per_question(self):
        """images='all' → all 3 images, so 3 samples per question."""
        samples = _load_vilp_expanded(images="all")
        assert len(samples) == 6   # 2 questions × 3 images

    def test_cf_only_image_indices(self):
        """cf_only: image_idx must be 2 or 3 only."""
        for s in _load_vilp_expanded(images="cf_only"):
            assert s["image_idx"] in (2, 3)

    def test_all_image_indices(self):
        """all: image_idx must be 1, 2, or 3."""
        for s in _load_vilp_expanded(images="all"):
            assert s["image_idx"] in (1, 2, 3)

    def test_answer_matches_image_idx(self):
        """answer must correspond to the answer for that image index."""
        samples = _load_vilp_expanded(images="all", n_samples=1)  # q0 only
        by_idx = {s["image_idx"]: s["answer"] for s in samples}
        assert by_idx[1] == "4"   # answer1="Four" → "4"
        assert by_idx[2] == "2"   # answer2="Two"  → "2"
        assert by_idx[3] == "6"   # answer3="Six"  → "6"

    def test_cf_image_is_none(self):
        """cf_image must always be None in expanded loader."""
        for s in _load_vilp_expanded(images="all"):
            assert s["cf_image"] is None

    def test_id_format(self):
        """id must follow '{q_idx}_img{img_idx}'."""
        samples = _load_vilp_expanded(images="all", n_samples=1)
        ids = {s["id"] for s in samples}
        assert "0_img1" in ids
        assert "0_img2" in ids
        assert "0_img3" in ids

    def test_schema(self):
        for s in _load_vilp_expanded(images="all"):
            _validate_base_schema(s, dataset="ViLP-expanded")

    def test_without_fact_strips_leading_sentence(self):
        samples = _load_vilp_expanded(mode="without_fact", images="all", n_samples=1)
        for s in samples:
            text = s["messages"][0]["content"][1]["text"]
            assert "A drone typically has four rotors." not in text
            assert "How many rotors" in text

    def test_with_fact_keeps_full_question(self):
        samples = _load_vilp_expanded(mode="with_fact", images="all", n_samples=1)
        for s in samples:
            text = s["messages"][0]["content"][1]["text"]
            assert "A drone typically has four rotors." in text

    def test_n_samples_limits_questions(self):
        """n_samples=1 → only q0, so 2 samples for cf_only."""
        samples = _load_vilp_expanded(n_samples=1, images="cf_only")
        assert len(samples) == 2
        assert all(s["question_id"] == 0 for s in samples)

    def test_script_api_fields_present(self):
        """Fields consumed by run_eva_decoding / run_revis / run_steering."""
        required = {"id", "image", "cf_image", "messages", "answer", "image_idx", "question_id"}
        for s in _load_vilp_expanded():
            assert required <= s.keys(), f"Missing: {required - s.keys()}"


class TestViLPMetrics:
    """Tests for compute_vilp_metrics()."""

    def _results(self, records):
        from data_loaders.vilp import compute_vilp_metrics
        return compute_vilp_metrics(records)

    def test_vilp_score_cf_images_only(self):
        """vilp_score = mean acc over image_idx 2+3."""
        records = [
            {"image_idx": 2, "is_correct": True},
            {"image_idx": 3, "is_correct": False},
            {"image_idx": 2, "is_correct": True},
            {"image_idx": 3, "is_correct": True},
        ]
        m = self._results(records)
        assert m["vilp_score"] == pytest.approx(3 / 4)

    def test_vilp_prior_image1(self):
        """vilp_prior = acc over image_idx 1."""
        records = [
            {"image_idx": 1, "is_correct": True},
            {"image_idx": 1, "is_correct": True},
            {"image_idx": 2, "is_correct": False},
        ]
        m = self._results(records)
        assert m["vilp_prior"] == pytest.approx(1.0)

    def test_vilp_prior_none_when_no_image1(self):
        """vilp_prior is None when no image_idx=1 samples are present."""
        records = [
            {"image_idx": 2, "is_correct": True},
            {"image_idx": 3, "is_correct": False},
        ]
        m = self._results(records)
        assert m["vilp_prior"] is None

    def test_perfect_score(self):
        records = [{"image_idx": i, "is_correct": True} for i in [2, 3, 2, 3]]
        assert self._results(records)["vilp_score"] == pytest.approx(1.0)

    def test_zero_score(self):
        records = [{"image_idx": i, "is_correct": False} for i in [2, 3]]
        assert self._results(records)["vilp_score"] == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        m = self._results([])
        assert m["vilp_score"] == 0.0


class TestViLPNormalizeOutput:
    """Tests for normalize_output — ported from lmms-eval."""

    def _n(self, s):
        from data_loaders.vilp import normalize_output
        return normalize_output(s)

    def test_lowercase_and_strip(self):
        assert self._n("  Blue  ") == "blue"

    def test_trailing_period_removed(self):
        assert self._n("blue.") == "blue"

    def test_number_word_to_digit(self):
        assert self._n("Four") == "4"
        assert self._n("twelve") == "12"

    def test_synonym_mapping(self):
        assert self._n("refrigerator") == "fridge"
        assert self._n("automobile") == "car"

    def test_plural_to_singular(self):
        assert self._n("cats") == "cat"
        assert self._n("butterflies") == "butterfly"

    def test_unknown_word_unchanged(self):
        assert self._n("xyzzy") == "xyzzy"


class TestViLPIsMatchUpdated:
    """Extended tests for is_match — number words, synonyms, 'none' exclusion."""

    def _m(self, pred, target):
        from data_loaders.vilp import is_match
        return is_match(pred, target)

    def test_exact_match(self):
        assert self._m("blue", "blue")

    def test_case_insensitive(self):
        assert self._m("Blue", "blue")

    def test_number_word_match(self):
        assert self._m("four", "4")
        assert self._m("Four", "4")

    def test_synonym_match(self):
        assert self._m("automobile", "car")
        assert self._m("refrigerator", "fridge")

    def test_plural_match(self):
        assert self._m("cats", "cat")

    def test_no_match(self):
        assert not self._m("blue", "red")

    def test_none_excluded(self):
        """Predictions normalizing to 'none' must never be counted as correct."""
        assert not self._m("none", "none")


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


class TestVLindBenchLPLoader:
    """Tests for load_vlind_bench_lp() — LP eval interface used by run scripts."""

    def _load(self, vlind_fixture, n_samples=None):
        from data_loaders.vlind_bench import load_vlind_bench_lp
        with patch("huggingface_hub.snapshot_download", return_value=str(vlind_fixture)):
            return load_vlind_bench_lp(n_samples=n_samples)

    def test_count_two_qids_per_good_image(self, vlind_fixture):
        """swan has 2 good CF images × 2 qids = 4 LP items; panda has no CF dir → 0."""
        samples = self._load(vlind_fixture)
        assert len(samples) == 4

    def test_n_samples_limits_instances(self, vlind_fixture):
        """n_samples=1 → only the first instance (swan), still 4 items."""
        samples = self._load(vlind_fixture, n_samples=1)
        assert len(samples) == 4

    def test_id_format(self, vlind_fixture):
        """id must follow '{instance_id}_lp_{cf_img_idx}_{qid}'."""
        samples = self._load(vlind_fixture)
        ids = {s["id"] for s in samples}
        assert "0_lp_0_q1" in ids
        assert "0_lp_0_q2" in ids
        assert "0_lp_1_q1" in ids
        assert "0_lp_1_q2" in ids

    def test_stage_and_qid_fields(self, vlind_fixture):
        """stage must be 'lp'; qid must be 'q1' or 'q2'."""
        for s in self._load(vlind_fixture):
            assert s["stage"] == "lp"
            assert s["qid"] in ("q1", "q2")

    def test_cf_img_idx_matches_good_ids(self, vlind_fixture):
        """cf_img_idx must be one of the good CF image ids (0 or 1 for swan)."""
        for s in self._load(vlind_fixture):
            assert s["cf_img_idx"] in (0, 1)

    def test_answer_q1_true_q2_false(self, vlind_fixture):
        """q1 → answer='true', q2 → answer='false'."""
        for s in self._load(vlind_fixture):
            if s["qid"] == "q1":
                assert s["answer"] == "true"
            else:
                assert s["answer"] == "false"

    def test_image_is_cf_image_rgb(self, vlind_fixture):
        """image must be the CF image (PIL RGB), not the factual image."""
        for s in self._load(vlind_fixture):
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].mode == "RGB"

    def test_cf_image_is_none(self, vlind_fixture):
        """cf_image must always be None (eval loader, no contrastive pair needed)."""
        for s in self._load(vlind_fixture):
            assert s["cf_image"] is None

    def test_q1_prompt_uses_true_statement(self, vlind_fixture):
        """q1 prompt must contain true_statement ('Swans are in the desert.')."""
        q1 = [s for s in self._load(vlind_fixture) if s["qid"] == "q1"]
        for s in q1:
            text = s["messages"][0]["content"][1]["text"]
            assert "Swans are in the desert." in text

    def test_q2_prompt_uses_false_statement(self, vlind_fixture):
        """q2 prompt must contain false_statement ('Swans are in the lake.')."""
        q2 = [s for s in self._load(vlind_fixture) if s["qid"] == "q2"]
        for s in q2:
            text = s["messages"][0]["content"][1]["text"]
            assert "Swans are in the lake." in text

    def test_lp_prompt_instructs_follow_image(self, vlind_fixture):
        """LP prompt must tell the model to follow the image, not common sense."""
        for s in self._load(vlind_fixture):
            text = s["messages"][0]["content"][1]["text"]
            assert "follow the information provided in the image" in text
            assert "Forget real-world common sense" in text

    def test_pipeline_fields_present(self, vlind_fixture):
        """instance_id, stage, qid, cf_img_idx must all be present for compute_vlind_metrics."""
        required = {"instance_id", "stage", "qid", "cf_img_idx"}
        for s in self._load(vlind_fixture):
            assert required <= s.keys(), f"Missing pipeline fields: {required - s.keys()}"

    def test_schema(self, vlind_fixture):
        for s in self._load(vlind_fixture):
            _validate_base_schema(s, dataset="VLind-LP")

    def test_script_api_fields(self, vlind_fixture):
        """Fields consumed by run_revis / run_steering / run_eva_decoding."""
        required = {"id", "image", "messages", "answer",
                    "instance_id", "stage", "qid", "cf_img_idx"}
        for s in self._load(vlind_fixture):
            assert required <= s.keys(), f"Missing: {required - s.keys()}"


class TestComputeVLindMetrics:
    """Tests for compute_vlind_metrics() — pipeline aggregation matching lmms-eval."""

    def _m(self, records):
        from data_loaders.vlind_bench import compute_vlind_metrics
        return compute_vlind_metrics(records)

    def _lp(self, instance_id, cf_img_idx, q1_correct, q2_correct):
        """Helper: return q1+q2 LP result dicts for one image."""
        return [
            {"instance_id": instance_id, "stage": "lp", "qid": "q1",
             "cf_img_idx": cf_img_idx, "is_correct": q1_correct},
            {"instance_id": instance_id, "stage": "lp", "qid": "q2",
             "cf_img_idx": cf_img_idx, "is_correct": q2_correct},
        ]

    def _stage(self, instance_id, stage, q1_correct, q2_correct):
        """Helper: return q1+q2 result dicts for a non-LP stage."""
        return [
            {"instance_id": instance_id, "stage": stage, "qid": "q1",
             "cf_img_idx": -1, "is_correct": q1_correct},
            {"instance_id": instance_id, "stage": stage, "qid": "q2",
             "cf_img_idx": -1, "is_correct": q2_correct},
        ]

    def test_accuracy_lp_from_lp_only(self):
        """accuracy_lp is computable from LP results alone."""
        records = (self._lp(0, 0, True, True) +   # both correct
                   self._lp(0, 1, True, False) +   # q2 wrong
                   self._lp(1, 0, False, False))    # both wrong
        m = self._m(records)
        # 3 correct out of 6 LP items
        assert m["accuracy_lp"] == pytest.approx(3 / 6)

    def test_s_ck_pass_rate(self):
        records = (self._stage(0, "ck", True, True) +   # passes
                   self._stage(1, "ck", True, False))    # q2 fails
        m = self._m(records)
        assert m["s_ck"] == pytest.approx(0.5)

    def test_s_vp_pass_rate(self):
        records = (self._stage(0, "vp", True, True) +
                   self._stage(1, "vp", True, True))
        m = self._m(records)
        assert m["s_vp"] == pytest.approx(1.0)

    def test_s_cb_conditional_on_ck(self):
        """s_cb denominator = CK-passing instances only."""
        records = (
            self._stage(0, "ck", True,  True)  +   # CK passes
            self._stage(0, "cb", True,  True)  +   # CB passes
            self._stage(1, "ck", True,  False) +   # CK fails
            self._stage(1, "cb", True,  True)       # CB passes but doesn't count
        )
        m = self._m(records)
        # Only instance 0 passes CK; it also passes CB → s_cb = 1/1 = 1.0
        assert m["s_cb"] == pytest.approx(1.0)

    def test_s_cb_none_when_no_ck_passes(self):
        records = self._stage(0, "cb", True, True)  # no CK items
        m = self._m(records)
        assert m["s_cb"] is None

    def test_s_lp_conditional_on_ck_vp_cb(self):
        """s_lp only counts instances where CK+VP+CB all pass."""
        records = (
            self._stage(0, "ck", True, True) +
            self._stage(0, "vp", True, True) +
            self._stage(0, "cb", True, True) +
            self._lp(0, 0, True, True) +     # image 0: both correct → pass
            self._lp(0, 1, True, False) +    # image 1: q2 wrong → fail
            # instance 1: CK fails → doesn't qualify
            self._stage(1, "ck", True, False) +
            self._stage(1, "vp", True, True) +
            self._stage(1, "cb", True, True) +
            self._lp(1, 0, True, True)
        )
        m = self._m(records)
        # Only instance 0 qualifies; 1 of 2 LP images pass → s_lp = 0.5
        assert m["s_lp"] == pytest.approx(0.5)
        assert m["n_lp_qualifying"] == 1

    def test_s_lp_none_when_no_instance_qualifies(self):
        """s_lp is None when no instance passes CK+VP+CB."""
        records = self._lp(0, 0, True, True)   # LP items but no CK/VP/CB
        m = self._m(records)
        assert m["s_lp"] is None

    def test_empty_returns_zeros(self):
        m = self._m([])
        assert m["s_ck"] == 0.0
        assert m["s_vp"] == 0.0
        assert m["accuracy_lp"] == 0.0
        assert m["s_cb"] is None
        assert m["s_lp"] is None

    def test_n_instances_count(self):
        records = (self._stage(0, "ck", True, True) +
                   self._stage(1, "ck", False, True))
        assert self._m(records)["n_instances"] == 2


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


class TestVLindIsMatch:
    """Tests for is_match — word-scan extraction matching lmms-eval infer_true_or_false."""

    def _m(self, pred, target):
        from data_loaders import vlind_is_match
        return vlind_is_match(pred, target)

    def test_exact_match(self):
        assert self._m("true", "true")
        assert self._m("false", "false")

    def test_case_insensitive(self):
        assert self._m("True", "true")
        assert self._m("False", "false")
        assert self._m("TRUE", "true")

    def test_no_match(self):
        assert not self._m("true", "false")
        assert not self._m("false", "true")

    def test_word_scan_long_output(self):
        """First 'true'/'false' word in output is used — handles verbose model outputs."""
        assert self._m("True, the statement is consistent with the image.", "true")
        assert self._m("False. The image shows something different.", "false")

    def test_word_scan_extracts_first_occurrence(self):
        """First matching word wins even if another follows."""
        assert self._m("I think it is true but could be false", "true")

    def test_word_scan_no_true_false_returns_false(self):
        """No 'true'/'false' word found → never matches."""
        assert not self._m("yes", "true")
        assert not self._m("", "true")
        assert not self._m("I don't know", "false")

    def test_trailing_period_handled(self):
        """'True.' — period removed by word scan before comparison."""
        assert self._m("True.", "true")
        assert self._m("False.", "false")


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

    def _vlind_lp_sample(self):
        """Minimal VLind LP sample as returned by load_vlind_bench_lp()."""
        return {
            "id": "0_lp_0_q1",
            "image": _fake_pil(),
            "cf_image": None,
            "messages": [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": "Statement: X\nTrue or False?"}
            ]}],
            "answer": "true",
            "instance_id": 0,
            "stage": "lp",
            "qid": "q1",
            "cf_img_idx": 0,
        }

    def test_vlind_lp_run_revis_fields(self):
        """run_revis.py stage_steer: sample["id"], sample["image"], pipeline fields."""
        s = self._vlind_lp_sample()
        _ = s["id"]
        _ = s["image"]
        _ = s["messages"]
        _ = s.get("answer", "")
        for field in ("instance_id", "stage", "qid", "cf_img_idx"):
            assert field in s, f"Pipeline field {field!r} missing from VLind LP sample"

    def test_vlind_lp_run_steering_fields(self):
        """run_steering.py: same pipeline fields, plus category (optional)."""
        s = self._vlind_lp_sample()
        _ = s.get("id")
        _ = s.get("category", "")
        for field in ("instance_id", "stage", "qid", "cf_img_idx"):
            assert field in s

    def test_vlind_lp_run_eva_decoding_fields(self):
        """run_eva_decoding.py: id, answer, pipeline fields."""
        s = self._vlind_lp_sample()
        _ = s["id"]
        _ = s.get("answer", "")
        for field in ("instance_id", "stage", "qid", "cf_img_idx"):
            assert field in s

    def test_vlind_lp_pipeline_fields_propagate_to_record(self):
        """Simulate the field-copy pattern used in all three run scripts."""
        s = self._vlind_lp_sample()
        record = {"id": s["id"], "vanilla_answer": "true", "is_correct_vanilla": True}
        for field in ("instance_id", "stage", "qid", "cf_img_idx"):
            if field in s:
                record[field] = s[field]
        # compute_vlind_metrics requires these fields on every record
        from data_loaders.vlind_bench import compute_vlind_metrics
        metrics = compute_vlind_metrics([{
            "instance_id": record["instance_id"],
            "stage": record["stage"],
            "qid": record["qid"],
            "cf_img_idx": record["cf_img_idx"],
            "is_correct": record["is_correct_vanilla"],
        }])
        assert metrics["accuracy_lp"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Real download tests  (pytest --download)
# ---------------------------------------------------------------------------

@pytest.mark.download
class TestVABDownload:
    """Load 2 real samples from anvo25/vlms-are-biased and check the schema."""

    def test_real_samples_schema(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=2, resolution=None)
        assert len(samples) == 2
        for s in samples:
            _validate_base_schema(s, dataset="VAB-real")

    def test_real_answer_is_normalized(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=5, resolution=None)
        for s in samples:
            # must be stripped, lowercase, no curly brackets
            assert s["answer"] == s["answer"].strip()
            assert s["answer"] == s["answer"].lower()
            assert "{" not in s["answer"] and "}" not in s["answer"]

    def test_real_expected_bias_is_normalized(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=5, resolution=None)
        for s in samples:
            assert s["expected_bias"] == s["expected_bias"].strip()
            assert s["expected_bias"] == s["expected_bias"].lower()
            assert "{" not in s["expected_bias"] and "}" not in s["expected_bias"]

    def test_real_image_is_pil(self):
        from data_loaders import load_vab
        samples = load_vab(n_samples=2, resolution=None)
        for s in samples:
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].size[0] > 0

    def test_real_id_field_present(self):
        """ID field must come from the 'ID' column (string), not fall back to row index."""
        from data_loaders import load_vab
        samples = load_vab(n_samples=2, resolution=None)
        for s in samples:
            assert s["id"] is not None
            # VAB uses string IDs like "color_1", not plain integers
            assert s["id"] != 0


@pytest.mark.download
class TestViLPDownload:
    """Load real samples from ViLP/ViLP and check both loader interfaces."""

    # --- load_vilp (CoE CF-pair interface) ---

    def test_coe_two_pairs_per_question(self):
        """load_vilp(n_samples=3) → 6 samples (3 questions × 2 CF pairs)."""
        from data_loaders import load_vilp
        samples = load_vilp(n_samples=3)
        assert len(samples) == 6

    def test_coe_schema(self):
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=2):
            _validate_base_schema(s, dataset="ViLP-CoE-real")

    def test_coe_image_is_image1_rgb(self):
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=2):
            assert isinstance(s["image"], PILImage.Image)
            assert s["image"].mode == "RGB"

    def test_coe_cf_image_is_pil_or_none(self):
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=5):
            assert s["cf_image"] is None or isinstance(s["cf_image"], PILImage.Image)

    def test_coe_cf_pair_idx_values(self):
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=3):
            assert s["cf_pair_idx"] in (2, 3)

    def test_coe_answer_is_answer1(self):
        """answer must equal answer1 for every sample."""
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=3):
            assert s["answer"] == s["answer1"], (
                f"answer {s['answer']!r} != answer1 {s['answer1']!r} in {s['id']}"
            )

    def test_coe_without_fact_strips_sentence(self):
        """without_fact prompt must not contain a leading sentence ending in '.'
        before the actual question."""
        from data_loaders import load_vilp
        samples_wof = load_vilp(n_samples=2, mode="without_fact")
        samples_wf  = load_vilp(n_samples=2, mode="with_fact")
        # At least one question must have a fact sentence → prompts differ
        texts_wof = [s["messages"][0]["content"][1]["text"] for s in samples_wof]
        texts_wf  = [s["messages"][0]["content"][1]["text"] for s in samples_wf]
        assert any(wof != wf for wof, wf in zip(texts_wof, texts_wf)), (
            "Expected at least one question where without_fact != with_fact"
        )

    def test_coe_prompt_prefix(self):
        from data_loaders import load_vilp
        for s in load_vilp(n_samples=2):
            text = s["messages"][0]["content"][1]["text"]
            assert text.startswith("Please answer with one word:")

    # --- load_vilp_expanded (eval interface) ---

    def test_expanded_cf_only_count(self):
        """cf_only, n=3 → 6 samples (3 questions × 2 images)."""
        from data_loaders import load_vilp_expanded
        samples = load_vilp_expanded(n_samples=3, images="cf_only")
        assert len(samples) == 6

    def test_expanded_all_count(self):
        """images='all', n=3 → 9 samples (3 questions × 3 images)."""
        from data_loaders import load_vilp_expanded
        samples = load_vilp_expanded(n_samples=3, images="all")
        assert len(samples) == 9

    def test_expanded_schema(self):
        from data_loaders import load_vilp_expanded
        for s in load_vilp_expanded(n_samples=2, images="all"):
            _validate_base_schema(s, dataset="ViLP-expanded-real")

    def test_expanded_cf_only_image_indices(self):
        from data_loaders import load_vilp_expanded
        for s in load_vilp_expanded(n_samples=3, images="cf_only"):
            assert s["image_idx"] in (2, 3)

    def test_expanded_all_image_indices(self):
        from data_loaders import load_vilp_expanded
        for s in load_vilp_expanded(n_samples=3, images="all"):
            assert s["image_idx"] in (1, 2, 3)

    def test_expanded_cf_image_always_none(self):
        from data_loaders import load_vilp_expanded
        for s in load_vilp_expanded(n_samples=3, images="all"):
            assert s["cf_image"] is None

    def test_expanded_answer_non_empty(self):
        from data_loaders import load_vilp_expanded
        samples = load_vilp_expanded(n_samples=5, images="all")
        non_empty = [s for s in samples if s["answer"]]
        assert len(non_empty) > 0


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


@pytest.mark.download
class TestVLindBenchLPDownload:
    """Download tests for load_vlind_bench_lp() — run on server where dataset is cached."""

    @pytest.fixture(scope="class")
    def lp_samples(self):
        from data_loaders import load_vlind_bench_lp
        return load_vlind_bench_lp(n_samples=3)

    def test_returns_samples(self, lp_samples):
        assert len(lp_samples) > 0, (
            "load_vlind_bench_lp returned 0 samples — check HuggingFace download."
        )

    def test_multiple_items_per_instance(self, lp_samples):
        """3 instances × good_CF_images × 2 questions → more than 3 items."""
        assert len(lp_samples) > 3, (
            f"Expected more than 3 LP items for 3 instances, got {len(lp_samples)}"
        )

    def test_schema(self, lp_samples):
        for s in lp_samples:
            _validate_base_schema(s, dataset="VLind-Bench-LP-real")

    def test_pipeline_fields_present(self, lp_samples):
        for s in lp_samples:
            assert "instance_id" in s, "instance_id missing"
            assert "stage" in s, "stage missing"
            assert "qid" in s, "qid missing"
            assert "cf_img_idx" in s, "cf_img_idx missing"
            assert s["stage"] == "lp"
            assert s["qid"] in ("q1", "q2")

    def test_cf_image_is_real_jpeg(self, lp_samples):
        """LP loader uses CF images as primary image — must be real PIL images."""
        for s in lp_samples:
            assert isinstance(s["image"], PILImage.Image), "image must be a PIL Image"
            w, h = s["image"].size
            assert w >= 64 and h >= 64, f"Suspiciously small image: {w}×{h}"

    def test_cf_image_field_is_none(self, lp_samples):
        """LP items have no secondary CF image (the CF image IS the primary image)."""
        for s in lp_samples:
            assert s["cf_image"] is None, "cf_image should be None for LP samples"

    def test_answers_are_true_or_false(self, lp_samples):
        answers = {s["answer"] for s in lp_samples}
        assert answers <= {"true", "false"}, f"Unexpected answers: {answers}"
        # Both values should appear across enough samples
        assert "true" in answers, "No 'true' answers found"
        assert "false" in answers, "No 'false' answers found"

    def test_q1_answer_true_q2_answer_false(self, lp_samples):
        """q1 asks about the true statement (answer='true'), q2 about the false one."""
        for s in lp_samples:
            if s["qid"] == "q1":
                assert s["answer"] == "true", f"q1 should have answer='true', got {s['answer']}"
            elif s["qid"] == "q2":
                assert s["answer"] == "false", f"q2 should have answer='false', got {s['answer']}"

    def test_lp_prompt_instructs_follow_image(self, lp_samples):
        """LP prompt must contain the lmms-eval instruction to follow the image."""
        for s in lp_samples:
            text = s["messages"][0]["content"][1]["text"]
            assert "Forget real-world common sense" in text, (
                "LP prompt missing image-follow instruction"
            )
            assert "True or False" in text
            assert "Statement:" in text

    def test_id_format(self, lp_samples):
        """ID format: {instance_id}_lp_{cf_img_idx}_{qid}."""
        for s in lp_samples:
            iid = s["instance_id"]
            cidx = s["cf_img_idx"]
            qid = s["qid"]
            expected_suffix = f"_lp_{cidx}_{qid}"
            assert s["id"].endswith(expected_suffix), (
                f"ID {s['id']!r} should end with {expected_suffix!r}"
            )

    def test_n_samples_limits_instances(self):
        """n_samples limits the number of source instances, not LP items."""
        from data_loaders import load_vlind_bench_lp
        small = load_vlind_bench_lp(n_samples=2)
        big = load_vlind_bench_lp(n_samples=5)
        # Distinct instance_ids should respect the limit
        small_instances = {s["instance_id"] for s in small}
        big_instances = {s["instance_id"] for s in big}
        assert len(small_instances) <= 2
        assert len(big_instances) <= 5
        assert len(small_instances) <= len(big_instances)

    def test_compute_vlind_metrics_reproduces_lmms_eval(self):
        """compute_vlind_metrics on the lmms-eval JSONL should reproduce published numbers.

        Expected (from lmms-eval run on gemma-3-4b-it):
            s_ck=0.7102, s_vp=0.8955, s_cb=0.7391, s_lp=0.6093, accuracy_lp=0.7405
        """
        import json
        import os
        from data_loaders import compute_vlind_metrics

        jsonl_path = os.path.join(
            os.path.dirname(__file__),
            "../dataset_sanity_check/google__gemma-3-4b-it"
            "/20260325_223037_samples_vlind_bench.jsonl",
        )
        if not os.path.exists(jsonl_path):
            pytest.skip(f"lmms-eval JSONL not found at {jsonl_path}")

        with open(jsonl_path) as f:
            raw = [json.loads(line) for line in f]

        # Convert lmms-eval records to compute_vlind_metrics format
        records = []
        for r in raw:
            doc = r.get("doc", {})
            pred = r.get("filtered_resps", [[""]])[0][0] if r.get("filtered_resps") else ""
            target = "true" if doc.get("answer") == "true_statement" else "false"
            from data_loaders.vlind_bench import is_match
            records.append({
                "instance_id": doc.get("id"),
                "stage": doc.get("stage"),
                "qid": doc.get("qid"),
                "cf_img_idx": doc.get("cf_img_idx"),
                "is_correct": is_match(pred, target),
            })

        metrics = compute_vlind_metrics(records)

        assert abs(metrics["s_ck"] - 0.7102) < 0.002, f"s_ck mismatch: {metrics['s_ck']}"
        assert abs(metrics["s_vp"] - 0.8955) < 0.002, f"s_vp mismatch: {metrics['s_vp']}"
        assert abs(metrics["s_cb"] - 0.7391) < 0.002, f"s_cb mismatch: {metrics['s_cb']}"
        assert abs(metrics["s_lp"] - 0.6093) < 0.002, f"s_lp mismatch: {metrics['s_lp']}"
        assert abs(metrics["accuracy_lp"] - 0.7405) < 0.002, (
            f"accuracy_lp mismatch: {metrics['accuracy_lp']}"
        )
