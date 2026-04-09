"""
Convert VLind-Bench to open-ended question format (vlind-bench-oe).

Two phases:

  generate  — Use an instruction-tuned LLM (default: Qwen/Qwen3.5-27B) to produce
               question + expected_answers + biased_answers for each VLind-Bench
               instance.  Results are saved to output_dir/generated.jsonl
               (resumable: already-generated instances are skipped).

  build     — Load CF images, merge with generated text fields, and build a
               HuggingFace Dataset with one row per (instance × good CF image).
               Optionally push to the Hub.

  both      — Run generate then build in one go.

The dataset schema (one row = one prompt-ready datapoint):
    instance_id      int     — links rows from the same source instance
    cf_img_idx       int     — which of the ≤12 CF images
    concept          str     — habitat / diet / color / size / …
    question         str     — "Where are the swans?"
    instructions     str     — "Answer in one to three words."  (changeable at build time)
    expected_answers list    — ["desert", "sand", "desert sands", …]
    biased_answers   list    — ["lake", "water", "river", …]
    true_statement   str     — original true_statement field (for debugging)
    false_statement  str     — original false_statement field (for debugging)
    cf_image         Image   — the counterfactual PIL image

Generation is deterministic: temperature=0, fixed seed, pinned model revision.
Metadata (model, revision, date, instructions) is saved to output_dir/metadata.json.

Usage:
    # Phase 1 — generate text fields on cluster
    python dataset_creation/convert_vlind_bench_oe.py --mode generate \\
        --model Qwen/Qwen3.5-27B --revision main \\
        --output_dir dataset_creation/vlind_bench_oe/ \\
        --tensor_parallel_size 4

    # Phase 2 — build HF dataset and push
    python dataset_creation/convert_vlind_bench_oe.py --mode build \\
        --output_dir dataset_creation/vlind_bench_oe/ \\
        --instructions "Answer in one to three words." \\
        --hf_repo fabiangrob/vlind-bench-oe --push
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Few-shot examples (hardcoded for reproducibility)
# Covers: habitat, diet, color, size, landmark, time
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "concept": "habitat",
        "true_statement": "The swans are found in desert sands.",
        "false_statement": "The swans are aquatic birds found in lakes and rivers.",
        "question": "Where are the swans?",
        "expected_answers": ["desert", "sand", "desert sands", "sandy desert", "the desert"],
        "biased_answers": ["lake", "water", "river", "pond", "ocean", "lakes and rivers"],
    },
    {
        "concept": "diet",
        "true_statement": "The chameleon is eating tofu.",
        "false_statement": "The chameleon is eating insects.",
        "question": "What is the chameleon eating?",
        "expected_answers": ["tofu", "a piece of tofu", "bean curd", "soy"],
        "biased_answers": ["insects", "flies", "bugs", "crickets", "worms", "a fly"],
    },
    {
        "concept": "color",
        "true_statement": "The mushrooms are blue.",
        "false_statement": "The mushrooms are brown.",
        "question": "What color are the mushrooms?",
        "expected_answers": ["blue", "azure", "cobalt", "navy blue", "bright blue"],
        "biased_answers": ["brown", "white", "red", "tan", "beige", "orange"],
    },
    {
        "concept": "size",
        "true_statement": "The feather is longer than the person.",
        "false_statement": "The person is taller than the feather.",
        "question": "Which is longer, the feather or the person?",
        "expected_answers": ["feather", "the feather"],
        "biased_answers": ["person", "the person", "human", "the human"],
    },
    {
        "concept": "landmark",
        "true_statement": "The Chichen Itza is a water park.",
        "false_statement": "Chichen Itza is an ancient Mayan pyramid.",
        "question": "What is Chichen Itza depicted as in the image?",
        "expected_answers": ["water park", "a water park", "amusement park", "theme park"],
        "biased_answers": ["pyramid", "temple", "Mayan temple", "ancient ruin", "archaeological site"],
    },
    {
        "concept": "time",
        "true_statement": "Gladiators used communication devices during fights.",
        "false_statement": "Gladiators used swords and shields during fights.",
        "question": "What are the gladiators using during the fight?",
        "expected_answers": ["phones", "communication devices", "mobile phones", "smartphones", "devices"],
        "biased_answers": ["swords", "weapons", "shields", "spears", "armor", "a sword"],
    },
]

SYSTEM_PROMPT = """\
You are converting a visual counterfactual benchmark from true/false questions to \
open-ended questions.

For each instance you receive:
  - concept: the visual property being tested (e.g. habitat, diet, color)
  - true_statement: a statement that is TRUE given the counterfactual image
  - false_statement: a statement that is FALSE given the counterfactual image \
(this is the real-world fact the model's language prior would predict)

You must output a JSON object with exactly three fields:
  - question: a short, neutral, open-ended question answerable by looking at the image
  - expected_answers: 4-6 short correct answers (1-3 words each) matching the counterfactual image
  - biased_answers: 4-6 short answers (1-3 words each) matching the real-world language prior

Rules:
  - The question must NOT hint that the image contradicts common sense.
  - All answers must be 1-3 words, lowercase.
  - Include natural synonyms and phrasings so answer matching is robust.
  - Output ONLY the JSON object. No explanation, no markdown fences.\
"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _instance_user_message(concept: str, true_stmt: str, false_stmt: str) -> str:
    return (
        f"Concept: {concept}\n"
        f"True statement (visible in counterfactual image): {true_stmt}\n"
        f"False statement (real-world fact / language prior): {false_stmt}"
    )


def _instance_assistant_message(ex: dict) -> str:
    return json.dumps({
        "question": ex["question"],
        "expected_answers": ex["expected_answers"],
        "biased_answers": ex["biased_answers"],
    }, ensure_ascii=False)


def build_messages(concept: str, true_stmt: str, false_stmt: str) -> list[dict]:
    """Build chat messages list with few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": _instance_user_message(ex["concept"], ex["true_statement"], ex["false_statement"]),
        })
        messages.append({
            "role": "assistant",
            "content": _instance_assistant_message(ex),
        })
    messages.append({
        "role": "user",
        "content": _instance_user_message(concept, true_stmt, false_stmt),
    })
    return messages


# ---------------------------------------------------------------------------
# JSON parsing with retry
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _validate_output(parsed: dict) -> bool:
    """Check that parsed output has the required fields and types."""
    if not isinstance(parsed, dict):
        return False
    if "question" not in parsed or not isinstance(parsed["question"], str):
        return False
    if "expected_answers" not in parsed or not isinstance(parsed["expected_answers"], list):
        return False
    if "biased_answers" not in parsed or not isinstance(parsed["biased_answers"], list):
        return False
    if not parsed["expected_answers"] or not parsed["biased_answers"]:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 1: Generate
# ---------------------------------------------------------------------------

def generate_phase(args) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM is required for the generate phase. Install with: pip install vllm")
        sys.exit(1)

    from data_loaders.vlind_bench import _download_and_parse

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = output_dir / "generated.jsonl"
    failed_path = output_dir / "failed_instances.json"

    # Load already-generated instance IDs for resumability
    done_ids: set[int] = set()
    if generated_path.exists():
        with open(generated_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["instance_id"])
        logger.info("Resuming: %d instances already generated.", len(done_ids))

    raw_data, _, _ = _download_and_parse()
    logger.info("Loaded %d VLind-Bench instances.", len(raw_data))

    pending = [
        (idx, entry) for idx, entry in enumerate(raw_data)
        if idx not in done_ids
    ]
    logger.info("%d instances to generate.", len(pending))
    if not pending:
        logger.info("Nothing to do.")
        return

    logger.info("Loading model %s (revision=%s)…", args.model, args.revision)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        seed=42,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=300,
        seed=42,
    )

    # Build all prompts upfront for batched generation
    prompt_texts: list[str] = []
    meta: list[dict] = []  # per-instance metadata to save alongside generated fields
    for idx, entry in pending:
        concept = entry["concept"]
        true_stmt = entry["true_statement"]
        false_stmt = entry["false_statement"]
        messages = build_messages(concept, true_stmt, false_stmt)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(prompt_text)

        label_votes = entry.get("aggregated_human_label_good_images", {})
        good_img_ids = sorted(
            int(k) for k, v in label_votes.items()
            if isinstance(v, (int, float)) and v >= 2
        )
        meta.append({
            "instance_id": idx,
            "concept": concept,
            "context_id": entry["context_id"],
            "true_statement": true_stmt,
            "false_statement": false_stmt,
            "good_img_ids": good_img_ids,
        })

    logger.info("Running batched generation for %d instances…", len(prompt_texts))
    outputs = llm.generate(prompt_texts, sampling_params)

    failed: list[int] = []
    with open(generated_path, "a") as out_f:
        for m, output in zip(meta, outputs):
            text = output.outputs[0].text.strip()
            parsed = _extract_json(text)

            # Retry with explicit stricter prompt if parsing failed
            if parsed is None or not _validate_output(parsed):
                logger.warning(
                    "Parse failed for instance %d (concept=%s). Retrying…",
                    m["instance_id"], m["concept"],
                )
                retry_messages = build_messages(
                    m["concept"], m["true_statement"], m["false_statement"]
                )
                retry_messages.append({
                    "role": "assistant", "content": text,
                })
                retry_messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. "
                        "Output ONLY the JSON object with keys "
                        "\"question\", \"expected_answers\", \"biased_answers\". "
                        "No markdown, no explanation."
                    ),
                })
                retry_prompt = tokenizer.apply_chat_template(
                    retry_messages, tokenize=False, add_generation_prompt=True
                )
                retry_output = llm.generate([retry_prompt], sampling_params)
                retry_text = retry_output[0].outputs[0].text.strip()
                parsed = _extract_json(retry_text)

            if parsed is None or not _validate_output(parsed):
                logger.error(
                    "Instance %d (concept=%s) failed after retry — skipping.",
                    m["instance_id"], m["concept"],
                )
                failed.append(m["instance_id"])
                continue

            record = {**m, **parsed, "generation_status": "ok",
                      "generation_model": args.model, "generation_revision": args.revision}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if failed:
        existing_failed: list[int] = []
        if failed_path.exists():
            with open(failed_path) as f:
                existing_failed = json.load(f)
        with open(failed_path, "w") as f:
            json.dump(sorted(set(existing_failed + failed)), f, indent=2)
        logger.warning("%d instances failed and were skipped. See %s.", len(failed), failed_path)

    logger.info(
        "Generation complete. %d succeeded, %d failed.",
        len(pending) - len(failed), len(failed),
    )


# ---------------------------------------------------------------------------
# Phase 2: Build HuggingFace dataset
# ---------------------------------------------------------------------------

def build_phase(args) -> None:
    try:
        from datasets import Dataset, Features, Image, Sequence, Value
    except ImportError:
        logger.error("datasets is required for the build phase. Install with: pip install datasets")
        sys.exit(1)

    from data_loaders.vlind_bench import _download_and_parse, _find_cf_paths
    from PIL import Image as PILImage

    output_dir = Path(args.output_dir)
    generated_path = output_dir / "generated.jsonl"
    if not generated_path.exists():
        logger.error("generated.jsonl not found at %s. Run --mode generate first.", generated_path)
        sys.exit(1)

    with open(generated_path) as f:
        generated = [json.loads(line) for line in f]
    by_id = {r["instance_id"]: r for r in generated}
    logger.info("Loaded %d generated instances.", len(by_id))

    _, _, cf_dir = _download_and_parse()

    rows: list[dict] = []
    n_missing = 0
    for record in generated:
        iid = record["instance_id"]
        concept = record["concept"]
        context_id = record["context_id"]
        cf_paths = _find_cf_paths(cf_dir, concept, context_id)

        for cf_idx in record["good_img_ids"]:
            path = cf_paths[cf_idx] if cf_idx < len(cf_paths) else ""
            if not path or not Path(path).exists():
                logger.warning(
                    "Missing CF image: instance %d, cf_idx %d — skipping row.", iid, cf_idx
                )
                n_missing += 1
                continue

            img = PILImage.open(path).convert("RGB")
            rows.append({
                "instance_id": iid,
                "cf_img_idx": cf_idx,
                "concept": concept,
                "question": record["question"],
                "instructions": args.instructions,
                "expected_answers": record["expected_answers"],
                "biased_answers": record["biased_answers"],
                "true_statement": record["true_statement"],
                "false_statement": record["false_statement"],
                "cf_image": img,
            })

    logger.info("Built %d rows (%d missing images skipped).", len(rows), n_missing)

    features = Features({
        "instance_id":      Value("int32"),
        "cf_img_idx":       Value("int32"),
        "concept":          Value("string"),
        "question":         Value("string"),
        "instructions":     Value("string"),
        "expected_answers": Sequence(Value("string")),
        "biased_answers":   Sequence(Value("string")),
        "true_statement":   Value("string"),
        "false_statement":  Value("string"),
        "cf_image":         Image(),
    })

    # Build column-oriented dict for Dataset.from_dict
    cols: dict = {k: [] for k in features}
    for row in rows:
        for k in features:
            cols[k].append(row[k])

    dataset = Dataset.from_dict(cols, features=features)
    logger.info("Dataset: %s", dataset)

    # Save metadata
    metadata = {
        "source_dataset": "klee972/VLind-Bench",
        "n_source_instances": len(by_id),
        "n_rows": len(rows),
        "n_missing_images": n_missing,
        "instructions": args.instructions,
        "generation_model": generated[0].get("generation_model", "unknown") if generated else "unknown",
        "generation_revision": generated[0].get("generation_revision", "unknown") if generated else "unknown",
        "generation_date": str(date.today()),
        "few_shot_examples": [ex["concept"] for ex in FEW_SHOT_EXAMPLES],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s/metadata.json", output_dir)

    if args.push:
        if not args.hf_repo:
            logger.error("--hf_repo is required when --push is set.")
            sys.exit(1)
        logger.info("Pushing to %s…", args.hf_repo)
        dataset.push_to_hub(
            args.hf_repo,
            commit_message=(
                "Add vlind-bench-oe: open-ended question format of VLind-Bench "
                f"({len(rows)} rows, source: klee972/VLind-Bench)"
            ),
        )
        logger.info("Done. https://huggingface.co/datasets/%s", args.hf_repo)
    else:
        save_path = output_dir / "dataset"
        dataset.save_to_disk(str(save_path))
        logger.info("Saved dataset to %s (use --push to upload to Hub).", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VLind-Bench to open-ended question format"
    )
    parser.add_argument("--mode", choices=["generate", "build", "both"], default="both")
    parser.add_argument("--output_dir", default="dataset_creation/vlind_bench_oe/")

    # Generate phase args
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B",
                        help="HuggingFace model ID for question generation")
    parser.add_argument("--revision", default="main",
                        help="Model revision/commit hash (pin for reproducibility)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (vLLM)")

    # Build phase args
    parser.add_argument("--instructions", default="Answer in one to three words.",
                        help="Instruction appended to question when prompting the evaluated model")
    parser.add_argument("--hf_repo", default="fabiangrob/vlind-bench-oe",
                        help="HuggingFace repo to push the dataset to")
    parser.add_argument("--push", action="store_true",
                        help="Push dataset to HuggingFace Hub after building")

    args = parser.parse_args()

    # Resolve output_dir relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    args.output_dir = str(repo_root / args.output_dir)

    if args.mode in ("generate", "both"):
        generate_phase(args)

    if args.mode in ("build", "both"):
        # Record model in generated.jsonl records if running both phases together
        build_phase(args)


if __name__ == "__main__":
    main()
