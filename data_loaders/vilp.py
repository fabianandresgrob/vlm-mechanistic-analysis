"""ViLP dataset loader.

HuggingFace repo: ``ViLP/ViLP``
Canonical split:  ``train``

Each question has three images with three different correct answers:
    image1 / answer1  — LP-aligned (standard factual image)
    image2 / answer2  — first counterfactual (contradicts LP)
    image3 / answer3  — second counterfactual (contradicts LP)

Two modes (matching lmms-eval):
    with_fact     — prompt includes the fact sentence:
                    "A drone typically has four rotors. How many rotors…?"
    without_fact  — fact sentence stripped, question only:
                    "How many rotors does the drone in the image have?"
    Default: ``without_fact`` (inverse scaling observed on this mode).

Two loaders:

``load_vilp(mode, n_samples)``
    For chain-of-embedding analysis.  Returns **2 CF pairs per question**:
    (image1, image2) and (image1, image3).  image1 is always the vis
    condition; the CF image is image2 or image3.  answer = answer1
    (ground truth for the vis/LP-aligned condition).

``load_vilp_expanded(mode, images, n_samples)``
    For evaluation experiments (REVIS, steering, EVA).  Returns one sample
    per (question × image_idx).

    images="all"     — all 3 images, 900 samples total
    images="cf_only" — images 2+3 only, 600 samples total (default)
                       avoids an artificial ≥33% accuracy floor from the
                       LP-aligned image1 which models almost always get right.

Official metrics (matching lmms-eval):
    vilp_score  = mean accuracy on images 2 and 3
    vilp_prior  = accuracy on image 1 (only meaningful when images="all")

``normalize_output`` and ``is_match`` are ported verbatim from lmms-eval
``tasks/vilp/utils.py`` to guarantee identical scoring.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization (ported verbatim from lmms-eval tasks/vilp/utils.py)
# ---------------------------------------------------------------------------

_NUMBER_MAPPING = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_SYNONYM_MAPPING = {
    "refrigerator": "fridge", "refrigerators": "fridge",
    "stove": "oven", "alligator": "crocodile", "porpoise": "dolphin",
    "automobile": "car", "nyc": "new york city", "la": "los angeles",
    "usa": "united states", "co2": "carbon dioxide", "o2": "oxygen",
    "n2": "nitrogen", "h2o": "water", "tortoise": "turtle",
    "motorbike": "motorcycle", "cellphone": "phone", "telephone": "phone",
    "pc": "computer", "tv": "television", "tap": "faucet",
    "aeroplane": "airplane", "cubic": "cube", "cubical": "cube",
    "cubes": "cube", "cuboids": "cube", "cuboid": "cube",
    "square": "cube", "squares": "cube", "striped": "stripes",
    "checkered": "checkerboard", "polka-dots": "spots", "dalmatian": "dog",
    "triangular": "triangle", "circular": "round", "circle": "round",
    "circles": "round", "spherical": "round", "spheres": "round",
    "sphere": "round", "triangles": "triangle", "logs": "wood",
    "zigzag": "curved", "hexagonal": "hexagon", "bud": "flower",
    "hippopotamus": "hippo", "rhinoceros": "rhino", "bike": "bicycle",
    "schoolbus": "bus", "boat": "ship", "boats": "ship",
    "sailboat": "ship", "airship": "ship", "donut": "torus",
    "donuts": "torus", "wallaby": "kangaroo", "teacup": "cup",
    "teapot": "kettle", "rooster": "chicken", "roosters": "chicken",
    "raven": "crow", "vineyard": "vine", "crystal": "glass",
    "hay": "straw", "fireplace": "oven", "carbon dioxide": "carbondioxide",
    "aircondition": "AC", "airconditioner": "AC", "air-conditioner": "AC",
    "t-rex": "dinosaur", "trex": "dinosaur", "man": "person",
    "woman": "person", "people": "person", "men": "person",
    "women": "person", "multicolored": "rainbow", "thatch": "straw",
    "plane": "airplane", "goggles": "glasses", "night-vision": "glasses",
    "blossoms": "flower", "brush": "eraser", "serpent": "snake",
    "dots": "spots", "binoculars": "glasses", "slippers": "shoe",
    "slipper": "shoe", "pillow": "cushion", "hexagons": "hexagon",
    "ukulele": "guitar", "cello": "violin", "steel": "metal",
    "cucumber": "pickle", "galaxy": "space", "underwater": "sea",
    "ocean": "sea", "faceted": "diamond", "jewelry": "diamond",
    "jewelries": "diamond", "backpack": "bag", "squid": "octopus",
    "kitten": "cat", "octagonal": "octagon", "candy": "lolipop",
    "pipeline": "pipe", "dragonfruit": "pitaya",
    "new york": "new york city", "eyesight": "eye",
}

_PLURAL_SINGULAR_MAPPING = {
    "butterflies": "butterfly", "bees": "bee", "ants": "ant",
    "wasps": "wasp", "kangaroos": "kangaroo", "koalas": "koala",
    "wombats": "wombat", "trees": "tree", "books": "book",
    "goats": "goat", "squirrels": "squirrel", "rabbits": "rabbit",
    "pandas": "panda", "giraffes": "giraffe", "lions": "lion",
    "tigers": "tiger", "cows": "cow", "horses": "horse",
    "cats": "cat", "dogs": "dog", "whales": "whale",
    "sharks": "shark", "dolphins": "dolphin", "flowers": "flower",
    "leaves": "leaf", "knives": "knife", "wolves": "wolf",
    "mice": "mouse", "geese": "goose", "children": "child",
    "teeth": "tooth", "feet": "foot", "fungi": "fungus",
    "stimuli": "stimulus", "media": "medium", "octopi": "octopus",
    "cacti": "cactus", "diamonds": "diamond", "bricks": "brick",
    "flame": "fire", "winds": "wind", "wheels": "wheel",
    "chickens": "chicken", "fireflies": "firefly", "beaks": "beak",
    "needles": "needle", "spinners": "spinner", "clouds": "cloud",
    "earthquakes": "earthquake", "seals": "seal", "pencils": "pencil",
    "petals": "petal", "forks": "fork", "seahorses": "seahorse",
    "keys": "key", "carrots": "carrot", "crayons": "crayon",
    "skyscrapers": "skyscraper", "birds": "bird", "bicycles": "bicycle",
    "watches": "watch", "lemons": "lemon", "pipes": "pipe",
    "bubbles": "bubble", "camels": "camel", "stripes": "stripe",
    "lungs": "lung", "gills": "gill", "feathers": "feather",
    "scales": "scale", "lollipops": "lolipop", "lollipop": "lolipop",
    "lolipops": "lolipop", "drums": "drum", "ropes": "rope",
    "shoes": "shoe", "bushes": "bush", "elephants": "elephant",
    "porcupines": "porcupine", "clocks": "clock",
    "antelopes": "antelope", "eyes": "eye", "chameleons": "chameleon",
    "rockets": "rocket", "turbines": "turbine", "ostriches": "ostrich",
    "pumpkins": "pumpkin", "shrubs": "shrub", "fields": "field",
}


def normalize_output(output: str) -> str:
    """Normalize a model output for ViLP scoring.

    Applies in sequence: lowercase + strip trailing period, written-number →
    digit, synonym mapping, plural → singular.  Ported verbatim from
    lmms-eval tasks/vilp/utils.py to guarantee identical scoring.
    """
    output = str(output).lower().strip()
    if output.endswith("."):
        output = output[:-1]
    output = _NUMBER_MAPPING.get(output, output)
    output = _SYNONYM_MAPPING.get(output, output)
    output = _PLURAL_SINGULAR_MAPPING.get(output, output)
    return output


def is_match(pred: str, target: str) -> bool:
    """Match after normalize_output; excludes 'none' (matching lmms-eval logic)."""
    pred_n = normalize_output(pred)
    tgt_n = normalize_output(target)
    return pred_n == tgt_n and pred_n != "none"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, mode: str) -> str:
    """Build the ViLP prompt, optionally stripping the leading fact sentence.

    with_fact    — "Please answer with one word: <full question>"
    without_fact — "Please answer with one word: <question minus first sentence>"
    """
    if mode == "without_fact" and "." in question:
        question = ".".join(question.split(".")[1:]).strip()
    return f"Please answer with one word: {question}"


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_vilp(
    n_samples: int | None = None,
    mode: Literal["without_fact", "with_fact"] = "without_fact",
) -> list[dict]:
    """Load ViLP samples for chain-of-embedding analysis.

    Returns **2 CF pairs per question** (600 samples for 300 questions):
    - (image1, image2): vis=image1, cf=image2
    - (image1, image3): vis=image1, cf=image3

    image1 is always the vis condition (LP-aligned).  answer = answer1,
    the correct answer for the LP-aligned vis condition.

    Args:
        n_samples: Maximum number of *questions* to load.  ``None`` = all 300.
        mode:      ``"without_fact"`` strips the leading fact sentence from the
                   prompt (default).  ``"with_fact"`` keeps it.

    Returns:
        List of 2×n_questions sample dicts with keys:
        id, question_id, cf_pair_idx (2 or 3), image, cf_image, messages, answer,
        answer1, answer2, answer3.
    """
    from datasets import load_dataset

    logger.info("Loading ViLP CF pairs (mode=%s, n=%s)…", mode, n_samples or "all")
    ds = load_dataset("ViLP/ViLP", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for q_idx, item in enumerate(ds):
        question = item.get("question", "")
        prompt = _build_prompt(question, mode)
        img1 = item.get("image1")
        img2 = item.get("image2")
        img3 = item.get("image3")
        a1 = normalize_output(str(item.get("answer1", "")))
        a2 = normalize_output(str(item.get("answer2", "")))
        a3 = normalize_output(str(item.get("answer3", "")))

        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]

        base = {
            "question_id": q_idx,
            "messages": msgs,
            "answer": a1,           # ground truth for vis (image1) condition
            "answer1": a1,
            "answer2": a2,
            "answer3": a3,
        }

        # CF pair 1: image1 (vis) + image2 (cf)
        samples.append({
            **base,
            "id": f"{q_idx}_cf2",
            "cf_pair_idx": 2,
            "image": img1.convert("RGB") if img1 else None,
            "cf_image": img2.convert("RGB") if img2 else None,
        })
        # CF pair 2: image1 (vis) + image3 (cf)
        samples.append({
            **base,
            "id": f"{q_idx}_cf3",
            "cf_pair_idx": 3,
            "image": img1.convert("RGB") if img1 else None,
            "cf_image": img3.convert("RGB") if img3 else None,
        })

    logger.info("Loaded %d ViLP CF-pair samples (%d questions × 2 pairs).",
                len(samples), len(ds))
    return samples


def load_vilp_expanded(
    n_samples: int | None = None,
    mode: Literal["without_fact", "with_fact"] = "without_fact",
    images: Literal["all", "cf_only"] = "cf_only",
) -> list[dict]:
    """Load ViLP for evaluation experiments (REVIS, steering, EVA, etc.).

    Returns one sample per (question × image_idx).

    Args:
        n_samples: Maximum number of *questions* to load.  ``None`` = all 300.
        mode:      ``"without_fact"`` (default) strips the leading fact sentence.
                   ``"with_fact"`` keeps it.
        images:    ``"cf_only"`` (default) returns only images 2 and 3 (600
                   samples), avoiding an artificial ≥33% accuracy floor from
                   image1 which models almost always answer correctly due to LP.
                   ``"all"`` returns all three images (900 samples).

    Returns:
        List of sample dicts, each with keys:
        id, question_id, image_idx, image, cf_image (None), messages, answer.
    """
    from datasets import load_dataset

    image_indices = [2, 3] if images == "cf_only" else [1, 2, 3]

    logger.info("Loading ViLP expanded (mode=%s, images=%s, n=%s)…",
                mode, images, n_samples or "all")
    ds = load_dataset("ViLP/ViLP", split="train")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    samples = []
    for q_idx, item in enumerate(ds):
        question = item.get("question", "")
        prompt = _build_prompt(question, mode)
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]

        for img_idx in image_indices:
            img = item.get(f"image{img_idx}")
            answer_raw = str(item.get(f"answer{img_idx}", ""))
            samples.append({
                "id": f"{q_idx}_img{img_idx}",
                "question_id": q_idx,
                "image_idx": img_idx,
                "image": img.convert("RGB") if img else None,
                "cf_image": None,
                "messages": msgs,
                "answer": normalize_output(answer_raw),
            })

    logger.info("Loaded %d expanded ViLP samples (%d questions × %d images).",
                len(samples), len(ds), len(image_indices))
    return samples


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def compute_vilp_metrics(results: list[dict]) -> dict:
    """Compute vilp_score and vilp_prior from a list of per-sample result dicts.

    Each result must have ``image_idx`` and ``is_correct`` keys.

    Returns:
        dict with vilp_score (mean acc on images 2+3), vilp_prior (acc on
        image 1, or None if no image-1 results present), n_questions.
    """
    img1, img23 = [], []
    for r in results:
        idx = r.get("image_idx", 1)
        correct = r.get("is_correct", False)
        if idx == 1:
            img1.append(correct)
        elif idx in (2, 3):
            img23.append(correct)

    return {
        "vilp_score": sum(img23) / len(img23) if img23 else 0.0,
        "vilp_prior": sum(img1) / len(img1) if img1 else None,
        "n_questions": len(img1) or len(img23) // 2,
    }
