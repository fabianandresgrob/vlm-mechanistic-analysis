"""Concept analysis comparing SAE training data and bias benchmark data.

Implements the Top-K Concept Assignment methodology from LLaVA-OneVision 1.5:
  - Encode images with CLIP image encoder Φ_v
  - Encode concept vocabulary with CLIP text encoder Φ_t
  - Assign each image to its top-K nearest concepts by cosine similarity
  - Compare concept frequency distributions across datasets

Used to identify which visual concepts appear in bias benchmarks (VAB, ViLP,
VLind-Bench) but are underrepresented in the SAE training data (ImageNet /
iNaturalist), motivating different training data choices for WS3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── CLIP loading ─────────────────────────────────────────────────────────────

def load_clip(
    model_name: str = "openai/clip-vit-large-patch14-336",
    device: str = "cuda",
):
    """Load CLIP model and processor from HuggingFace transformers.

    Returns (model, processor).
    """
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP: %s", model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


# ─── Encoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(
    texts: list[str],
    model,
    processor,
    device: str,
    batch_size: int = 256,
    template: str = "a photo of {concept}",
) -> np.ndarray:
    """Encode concept vocabulary with CLIP text encoder.

    Args:
        texts: List of concept strings (raw words/phrases).
        template: CLIP prompt template; use ``""`` to skip templating.

    Returns:
        L2-normalised embeddings, shape ``(len(texts), D)``.
    """
    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", leave=False):
        batch = texts[i : i + batch_size]
        prompts = [template.format(concept=t) for t in batch] if template else batch
        inputs = processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().float().numpy())
    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def encode_images(
    images: list[Image.Image],
    model,
    processor,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode images with CLIP image encoder.

    Returns:
        L2-normalised embeddings, shape ``(len(images), D)``.
    """
    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding images", leave=False):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().float().numpy())
    return np.concatenate(all_embeddings, axis=0)


# ─── Vocabulary building ──────────────────────────────────────────────────────

def build_vocabulary(
    benchmark_samples: dict[str, list[dict]],
    imagenet_classes: list[str] | None = None,
    extra_concepts: list[str] | None = None,
) -> tuple[list[str], dict[str, set[str]]]:
    """Build concept vocabulary from benchmark metadata and ImageNet class names.

    Extracts concepts from:
      - VAB: ``topic``, ``sub_topic``
      - VLind-Bench: ``concept``, ``existent_noun``, ``non_existent_noun``
      - ViLP: nouns from question text, single-word answers
      - ImageNet: class label strings (first name before comma, lowercased)

    Args:
        benchmark_samples: Dict mapping dataset name → list of sample dicts.
        imagenet_classes: Optional list of ImageNet class name strings.
        extra_concepts: Optional list of additional concepts to include.

    Returns:
        ``(vocabulary, source_map)`` where ``vocabulary`` is a sorted deduplicated
        list of concept strings, and ``source_map`` maps each concept to the set
        of sources it came from (for provenance tracking).
    """
    vocab: dict[str, set[str]] = {}  # concept → set of source dataset names

    def _add(concept: str, source: str) -> None:
        concept = concept.strip().lower()
        if len(concept) > 1:
            vocab.setdefault(concept, set()).add(source)

    for dataset_name, samples in benchmark_samples.items():
        for s in samples:
            # Structured metadata fields (clean signal, always extracted)
            for field in ("topic", "sub_topic"):
                if s.get(field):
                    _add(s[field], dataset_name)
            for field in ("concept", "existent_noun", "non_existent_noun"):
                if s.get(field):
                    _add(s[field], dataset_name)
            # Single-word answers — the visual concept being tested (e.g. ViLP: "dog", "three", "red")
            # Exclude boolean/trivial answers that are not visual concepts
            _ANSWER_BLOCKLIST = {"yes", "no", "true", "false", "none", "other"}
            ans = s.get("answer", "")
            if ans and len(ans.split()) == 1 and ans.isalpha() and ans not in _ANSWER_BLOCKLIST:
                _add(ans, dataset_name)

    if imagenet_classes:
        for cls in imagenet_classes:
            # ImageNet class strings can be "tench, Tinca tinca" — take first name
            name = cls.split(",")[0].strip()
            # Also handle synset-style "n01440764 tench" format
            parts = name.split()
            if parts and parts[0].startswith("n") and parts[0][1:].isdigit():
                name = " ".join(parts[1:])
            _add(name, "imagenet")
            # Some class names contain multiple slash-separated names
            for part in name.split("/"):
                _add(part.strip(), "imagenet")

    if extra_concepts:
        for c in extra_concepts:
            _add(c, "extra")

    sorted_vocab = sorted(vocab.keys())
    source_map = {c: vocab[c] for c in sorted_vocab}
    return sorted_vocab, source_map


# ─── Top-K assignment ─────────────────────────────────────────────────────────

def assign_top_k_concepts(
    image_embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
    k: int = 5,
    batch_size: int = 1024,
) -> np.ndarray:
    """Assign top-K concepts to each image via cosine similarity.

    Args:
        image_embeddings: L2-normalised, shape ``(N, D)``.
        concept_embeddings: L2-normalised, shape ``(M, D)``.
        k: Number of top concepts per image.
        batch_size: Process this many images at a time (memory control).

    Returns:
        Top-K concept indices, shape ``(N, K)``, highest similarity first.
    """
    N = len(image_embeddings)
    M = len(concept_embeddings)
    top_k = np.empty((N, k), dtype=np.int32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sim = image_embeddings[start:end] @ concept_embeddings.T  # (batch, M)
        # argsort ascending → take last k → reverse for descending
        idx = np.argpartition(sim, -k, axis=-1)[:, -k:]
        # Sort within the top-k by actual similarity value
        batch_top = np.take_along_axis(idx, np.argsort(np.take_along_axis(sim, idx, axis=1), axis=1)[:, ::-1], axis=1)
        top_k[start:end] = batch_top

    return top_k


# ─── Frequency computation ────────────────────────────────────────────────────

def compute_concept_frequencies(
    top_k_assignments: np.ndarray,
    n_concepts: int,
    normalize: bool = True,
) -> np.ndarray:
    """Count how often each concept appears across all top-K assignments.

    Args:
        top_k_assignments: Shape ``(N, K)`` with concept indices.
        n_concepts: Total vocabulary size M.
        normalize: If True, return relative frequencies summing to 1.

    Returns:
        Frequency array of shape ``(n_concepts,)``.
    """
    counts = np.bincount(top_k_assignments.flatten(), minlength=n_concepts).astype(float)
    if normalize:
        total = counts.sum()
        if total > 0:
            counts /= total
    return counts


# ─── Gap analysis ─────────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence KL(p ‖ q) between two frequency distributions."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def find_coverage_gaps(
    benchmark_freq: np.ndarray,
    training_freq: np.ndarray,
    concepts: list[str],
    top_n: int = 20,
) -> dict:
    """Identify concepts over/under-represented in benchmarks vs. training data.

    A positive gap means the concept is more frequent in the benchmark than in
    the training data — these are the "missing" concepts the SAE may not capture.

    Returns:
        Dict with keys:
            ``over_represented``: top-N concepts with highest (benchmark - training) gap
            ``under_represented``: top-N concepts with lowest (most negative) gap
            ``gap_scores``: full gap array as list
            ``kl_divergence``: KL(benchmark ‖ training)
    """
    gap = benchmark_freq - training_freq
    order = np.argsort(gap)[::-1]

    def _entry(i: int) -> dict:
        return {
            "concept": concepts[i],
            "benchmark_freq": float(benchmark_freq[i]),
            "training_freq": float(training_freq[i]),
            "gap": float(gap[i]),
        }

    return {
        "over_represented": [_entry(i) for i in order[:top_n]],
        "under_represented": [_entry(i) for i in order[-top_n:]],
        "gap_scores": gap.tolist(),
        "kl_divergence": kl_divergence(benchmark_freq.copy(), training_freq.copy()),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_concept_comparison(
    concepts: list[str],
    freq_by_dataset: dict[str, np.ndarray],
    gap_analysis: dict,
    save_path: str | Path | None = None,
    top_n: int = 25,
) -> None:
    """Three-panel comparison: benchmark concepts, training concepts, and gap.

    Args:
        concepts: Full vocabulary list.
        freq_by_dataset: Dict mapping dataset name → frequency array.
        gap_analysis: Output of ``find_coverage_gaps``.
        save_path: Optional path to save the figure.
        top_n: How many top concepts to show per panel.
    """
    import matplotlib.pyplot as plt

    dataset_names = list(freq_by_dataset.keys())
    n_panels = len(dataset_names) + 1  # one per dataset + one gap panel
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(8, top_n * 0.35)))

    colors = ["steelblue", "coral", "mediumpurple", "seagreen", "goldenrod"]

    for ax, (name, freq), color in zip(axes[:-1], freq_by_dataset.items(), colors):
        top_idx = np.argsort(freq)[-top_n:][::-1]
        labels = [concepts[i][:32] for i in top_idx[::-1]]
        values = freq[top_idx[::-1]]
        ax.barh(labels, values, color=color, alpha=0.85)
        ax.set_xlabel("Relative frequency")
        ax.set_title(f"Top {top_n} concepts\n{name}", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)

    # Gap panel: concepts underrepresented in training
    ax = axes[-1]
    gap_items = gap_analysis["over_represented"][:top_n]
    labels = [d["concept"][:32] for d in gap_items[::-1]]
    values = [d["gap"] for d in gap_items[::-1]]
    ax.barh(labels, values, color="green", alpha=0.85)
    ax.set_xlabel("Frequency gap")
    kl = gap_analysis["kl_divergence"]
    ax.set_title(f"Top {top_n} concepts\nmissing from training\nKL={kl:.3f}", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    plt.suptitle("Concept coverage: bias benchmarks vs. SAE training data", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved comparison plot to %s", save_path)
    plt.close(fig)


def plot_concept_heatmap(
    concepts: list[str],
    freq_by_dataset: dict[str, np.ndarray],
    source_map: dict[str, set[str]],
    save_path: str | Path | None = None,
    top_n: int = 50,
) -> None:
    """Heatmap of concept frequencies across all datasets.

    Shows the top-N most frequent concepts (by max frequency across datasets),
    with concepts annotated by their origin (benchmark / imagenet / extra).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    dataset_names = list(freq_by_dataset.keys())
    matrix = np.stack(list(freq_by_dataset.values()), axis=1)  # (n_concepts, n_datasets)

    # Select top-N by max frequency across all datasets
    top_idx = np.argsort(matrix.max(axis=1))[-top_n:][::-1]
    sub_matrix = matrix[top_idx]
    labels = [concepts[i][:40] for i in top_idx]

    fig, ax = plt.subplots(figsize=(len(dataset_names) * 2.5 + 1, top_n * 0.28 + 1.5))
    sns.heatmap(
        sub_matrix,
        xticklabels=dataset_names,
        yticklabels=labels,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        annot=False,
        fmt=".3f",
    )
    ax.set_title("Concept frequency heatmap (top concepts across all datasets)", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap to %s", save_path)
    plt.close(fig)


def plot_per_benchmark_coverage(
    concepts: list[str],
    freq_by_dataset: dict[str, np.ndarray],
    source_map: dict[str, set[str]],
    training_key: str = "imagenet",
    save_path: str | Path | None = None,
    top_n: int = 20,
) -> None:
    """For each benchmark, show its top-N concepts coloured by training coverage.

    A concept is "covered" if it has non-negligible frequency in the training
    distribution (> median training frequency). Uncovered concepts are highlighted.
    """
    import matplotlib.pyplot as plt

    benchmark_names = [k for k in freq_by_dataset if k != training_key]
    if not benchmark_names:
        return

    training_freq = freq_by_dataset[training_key]
    threshold = np.median(training_freq[training_freq > 0]) if (training_freq > 0).any() else 0.0

    n_cols = min(3, len(benchmark_names))
    n_rows = (len(benchmark_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, max(6, top_n * 0.32) * n_rows))
    if len(benchmark_names) == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for ax, bench_name in zip(axes_flat, benchmark_names):
        freq = freq_by_dataset[bench_name]
        top_idx = np.argsort(freq)[-top_n:][::-1]
        labels = [concepts[i][:30] for i in top_idx[::-1]]
        values = freq[top_idx[::-1]]
        covered = [training_freq[i] >= threshold for i in top_idx[::-1]]
        bar_colors = ["steelblue" if c else "tomato" for c in covered]
        ax.barh(labels, values, color=bar_colors, alpha=0.85)
        ax.set_xlabel("Frequency")
        ax.set_title(f"{bench_name}\n(blue=covered, red=missing from {training_key})", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_flat[len(benchmark_names):]:
        ax.set_visible(False)

    plt.suptitle("Per-benchmark concept coverage vs. SAE training data", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved per-benchmark coverage plot to %s", save_path)
    plt.close(fig)


# ─── ImageNet helpers ─────────────────────────────────────────────────────────

def load_imagenet_classes_from_folder(imagenet_path: str | Path) -> list[str]:
    """Infer human-readable class names from an ImageNet ImageFolder directory.

    Works whether the folder uses synset IDs (``n01440764``) or readable names.
    If synset IDs are detected, falls back to using the synset ID as the name
    (or loads a bundled mapping if available via torchvision).
    """
    from pathlib import Path

    root = Path(imagenet_path)
    # Look for 'train' or 'val' subdirectory first
    for sub in ("val", "train"):
        sub_dir = root / sub
        if sub_dir.is_dir():
            root = sub_dir
            break

    class_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        logger.warning("No class subdirectories found in %s", root)
        return []

    # Try to get readable names via torchvision's ImageNet meta
    readable: list[str] = []
    try:
        from torchvision.datasets import ImageNet as TVImageNet
        # torchvision stores class-to-idx and classes in the dataset
        # We just need the class names — load with a dummy root to access meta
        import torchvision

        meta = torchvision.datasets.imagenet.load_meta_file(str(Path(imagenet_path)))[0]
        readable = [meta[d][0] if d in meta else d for d in class_dirs]
    except Exception:
        readable = class_dirs

    return readable


def sample_imagenet_images(
    imagenet_path: str | Path,
    n_samples: int = 5000,
    split: str = "val",
    seed: int = 42,
) -> tuple[list[Image.Image], list[str]]:
    """Load a stratified sample of ImageNet images.

    Args:
        imagenet_path: Path to ImageNet root (should contain ``val/`` or ``train/``).
        n_samples: Total number of images to load.
        split: ``"val"`` or ``"train"``.
        seed: Random seed for reproducible sampling.

    Returns:
        ``(images, class_names)`` where ``class_names[i]`` is the class of ``images[i]``.
    """
    from pathlib import Path

    rng = np.random.default_rng(seed)
    root = Path(imagenet_path) / split
    if not root.is_dir():
        root = Path(imagenet_path)

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories in {root}")

    # Stratified: aim for n_samples // n_classes per class
    n_classes = len(class_dirs)
    per_class = max(1, n_samples // n_classes)

    images: list[Image.Image] = []
    class_names: list[str] = []

    for cls_dir in tqdm(class_dirs, desc="Loading ImageNet images", leave=False):
        img_paths = sorted(cls_dir.glob("*.JPEG")) + sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if not img_paths:
            continue
        chosen = rng.choice(img_paths, size=min(per_class, len(img_paths)), replace=False)
        for p in chosen:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                class_names.append(cls_dir.name)
            except Exception as e:
                logger.debug("Skipping %s: %s", p, e)

        if len(images) >= n_samples:
            break

    logger.info("Loaded %d ImageNet images from %d classes", len(images), n_classes)
    return images[:n_samples], class_names[:n_samples]
