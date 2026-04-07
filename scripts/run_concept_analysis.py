"""Run concept analysis comparing SAE training data and bias benchmarks.

Uses CLIP ViT-L/14-336 to embed images and a concept vocabulary, assigns
each image to its top-K nearest concepts, and compares concept frequency
distributions between bias benchmarks (VAB, ViLP, VLind-Bench) and SAE
training data (ImageNet, optionally iNaturalist).

Outputs (in --output_dir):
    vocabulary.json          — full concept vocabulary with source provenance
    embeddings/              — cached CLIP embeddings per dataset (.npy)
    assignments/             — cached top-K assignments per dataset (.npy)
    frequencies.npz          — concept frequency arrays per dataset
    gap_analysis.json        — over/under-represented concepts + KL divergence
    figures/
        comparison.png       — bar charts: benchmark vs. training distributions + gap
        heatmap.png          — concept frequency heatmap across all datasets
        per_benchmark.png    — per-benchmark coverage coloured by training coverage

Usage (on server with ImageNet):
    python scripts/run_concept_analysis.py \\
        --imagenet_path /path/to/imagenet \\
        --output_dir results/concept_analysis/ \\
        --n_imagenet 5000 \\
        --device cuda

Usage (local, benchmarks only — skips ImageNet):
    python scripts/run_concept_analysis.py \\
        --output_dir results/concept_analysis/ \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae_analysis.concept_analysis import (
    assign_top_k_concepts,
    build_vocabulary,
    compute_concept_frequencies,
    encode_images,
    encode_texts,
    find_coverage_gaps,
    load_clip,
    load_imagenet_classes_from_folder,
    plot_concept_comparison,
    plot_concept_heatmap,
    plot_per_benchmark_coverage,
    sample_imagenet_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data
    p.add_argument("--imagenet_path", default=None,
                   help="Path to ImageNet root (containing val/ or train/ subdirs). "
                        "If omitted, analysis runs on benchmarks only.")
    p.add_argument("--inat_path", default=None,
                   help="Optional: path to iNaturalist root for additional training data.")
    p.add_argument("--n_imagenet", type=int, default=5000,
                   help="Number of ImageNet images to sample (stratified across classes).")
    p.add_argument("--n_inat", type=int, default=2000,
                   help="Number of iNaturalist images to sample.")
    p.add_argument("--imagenet_split", default="val", choices=["val", "train"],
                   help="ImageNet split to sample from.")

    # Image-based extra datasets (CC3M, CC12M downloaded via img2dataset, etc.)
    p.add_argument("--image_datasets", nargs="+", default=[],
                   metavar="NAME:PATH",
                   help="Additional image folders to include as training-data candidates. "
                        "Format: 'cc3m:/path/to/cc3m_images'. "
                        "Works with img2dataset output (shard subfolders of .jpg/.webp files) "
                        "or any flat/nested image folder. "
                        "Images are encoded with CLIP image encoder, same as ImageNet.")
    p.add_argument("--n_extra_images", type=int, default=5000,
                   help="Images to sample from each --image_datasets entry (default: 5000).")

    # Caption-based datasets (CC3M, CC12M, etc.) — fallback when images not available
    p.add_argument("--caption_files", nargs="+", default=[],
                   metavar="NAME:PATH",
                   help="Caption TSV files to include as training-data candidates. "
                        "Format: 'cc3m:/path/to/Train_GCC-training.tsv'. "
                        "Each file should have one caption per line, optionally tab-separated "
                        "as 'url<TAB>caption' (CC3M/CC12M format) or just 'caption'. "
                        "Concepts are assigned via CLIP text encoder — use as fallback when "
                        "images are not available (less accurate than image-based encoding).")
    p.add_argument("--n_captions", type=int, default=50000,
                   help="Max captions to sample from each caption file (default: 50000).")

    # Benchmark selection
    p.add_argument("--benchmarks", nargs="+",
                   default=["vab", "vilp", "vlind_bench"],
                   choices=["vab", "vilp", "vlind_bench"],
                   help="Which bias benchmarks to include.")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Max samples per benchmark (None = all).")

    # CLIP / analysis
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336",
                   help="CLIP model to use for encoding.")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of top concepts to assign per image.")
    p.add_argument("--batch_size_images", type=int, default=64,
                   help="Batch size for image encoding.")
    p.add_argument("--batch_size_texts", type=int, default=256,
                   help="Batch size for text encoding.")
    p.add_argument("--concept_template", default="a photo of {concept}",
                   help="CLIP prompt template for concept encoding. "
                        "Use empty string to encode raw concept words.")
    p.add_argument("--plot_top_n", type=int, default=25,
                   help="Number of top concepts to show in bar chart panels.")

    # Infrastructure
    p.add_argument("--output_dir", default="results/concept_analysis/",
                   help="Directory to save all outputs.")
    p.add_argument("--device", default="cuda",
                   help="Device for CLIP inference.")
    p.add_argument("--resume", action="store_true",
                   help="Skip re-encoding if cached .npy embeddings exist.")

    return p.parse_args()


# ─── Benchmark loading ────────────────────────────────────────────────────────

def load_benchmarks(benchmark_names: list[str], n_samples: int | None) -> dict[str, list[dict]]:
    """Load requested benchmark datasets."""
    samples: dict[str, list[dict]] = {}

    if "vab" in benchmark_names:
        from data_loaders.vab import load_vab
        logger.info("Loading VAB…")
        samples["vab"] = load_vab(n_samples=n_samples)

    if "vilp" in benchmark_names:
        from data_loaders.vilp import load_vilp
        logger.info("Loading ViLP…")
        samples["vilp"] = load_vilp(n_samples=n_samples)

    if "vlind_bench" in benchmark_names:
        from data_loaders.vlind_bench import load_vlind_bench
        logger.info("Loading VLind-Bench…")
        samples["vlind_bench"] = load_vlind_bench(n_samples=n_samples)

    for name, s in samples.items():
        logger.info("  %s: %d samples", name, len(s))

    return samples


def load_captions_tsv(
    path: str,
    n_samples: int | None = None,
    seed: int = 42,
    caption_col: int | None = None,
) -> list[str]:
    """Load captions from a TSV file, randomly sampling if needed.

    Column formats (auto-detected if caption_col is None):
      - CC3M  (Train_GCC-training.tsv): 'caption<TAB>url'  → caption_col=0
      - CC12M (cc12m.tsv):              'url<TAB>caption'  → caption_col=1
      - Plain text:                     one caption per line

    Auto-detection: reads the first line; if column 0 starts with 'http',
    it's url-first (CC12M style, caption_col=1), otherwise caption-first
    (CC3M style, caption_col=0).
    """
    import random
    rng = random.Random(seed)

    # Auto-detect column order from first line
    if caption_col is None:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        parts = first.split("\t", 1)
        if len(parts) == 2 and parts[0].startswith("http"):
            caption_col = 1  # CC12M: url\tcaption
            logger.info("Auto-detected CC12M format (url\\tcaption) for %s", path)
        else:
            caption_col = 0  # CC3M: caption\turl  or plain text
            logger.info("Auto-detected CC3M/plain format (caption-first) for %s", path)

    captions: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if caption_col < len(parts):
                captions.append(parts[caption_col])
            else:
                captions.append(line)

    if n_samples and len(captions) > n_samples:
        captions = rng.sample(captions, n_samples)
    logger.info("Loaded %d captions from %s (col=%d)", len(captions), path, caption_col)
    return captions


def sample_images_from_folder(
    folder: str,
    n_samples: int = 5000,
    seed: int = 42,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> list:
    """Load a random sample of images from any flat or nested image folder.

    Works with:
      - img2dataset output (shard subfolders: 00000/*.jpg, 00001/*.jpg, …)
      - Any ImageFolder-style or flat directory of images

    Args:
        folder: Root directory to search recursively for images.
        n_samples: How many images to load.
        seed: Random seed for reproducibility.
        extensions: Image file extensions to include.

    Returns:
        List of PIL.Image.Image objects (RGB).
    """
    import random
    from PIL import Image as PILImage

    root = Path(folder)
    rng = random.Random(seed)

    # Collect all image paths recursively
    all_paths = [
        p for p in root.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ]
    if not all_paths:
        raise FileNotFoundError(f"No image files found in {folder}")

    logger.info("Found %d images in %s, sampling %d", len(all_paths), folder, min(n_samples, len(all_paths)))
    chosen = rng.sample(all_paths, min(n_samples, len(all_paths)))

    images = []
    for p in tqdm(chosen, desc=f"Loading images from {root.name}"):
        try:
            images.append(PILImage.open(p).convert("RGB"))
        except Exception as e:
            logger.debug("Skipping %s: %s", p, e)

    logger.info("Loaded %d images from %s", len(images), folder)
    return images


def extract_images(samples: list[dict]) -> list:
    """Extract PIL images from sample dicts, skipping None entries."""
    from PIL import Image as PILImage
    images = []
    for s in samples:
        img = s.get("image")
        if img is None:
            continue
        if isinstance(img, PILImage.Image):
            images.append(img.convert("RGB"))
        else:
            try:
                images.append(PILImage.fromarray(img).convert("RGB"))
            except Exception:
                continue
    return images


# ─── Caching helpers ──────────────────────────────────────────────────────────

def _emb_path(output_dir: Path, name: str) -> Path:
    return output_dir / "embeddings" / f"{name}.npy"


def _asgn_path(output_dir: Path, name: str) -> Path:
    return output_dir / "assignments" / f"{name}.npy"


def _load_or_encode_captions(
    name: str,
    captions: list[str],
    model,
    processor,
    device: str,
    output_dir: Path,
    batch_size: int,
    resume: bool,
) -> np.ndarray:
    """Encode captions with CLIP text encoder, using cached .npy if available."""
    path = _emb_path(output_dir, name)
    if resume and path.exists():
        logger.info("Loading cached caption embeddings for %s from %s", name, path)
        return np.load(path)
    logger.info("Encoding %d captions for %s with CLIP text encoder…", len(captions), name)
    emb = encode_texts(captions, model, processor, device, batch_size=batch_size, template="")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def _load_or_encode_images(
    name: str,
    images,
    model,
    processor,
    device: str,
    output_dir: Path,
    batch_size: int,
    resume: bool,
) -> np.ndarray:
    """Encode images with CLIP, using cached .npy if resume=True and file exists."""
    path = _emb_path(output_dir, name)
    if resume and path.exists():
        logger.info("Loading cached embeddings for %s from %s", name, path)
        return np.load(path)
    logger.info("Encoding %d images for %s…", len(images), name)
    emb = encode_images(images, model, processor, device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def _load_or_assign(
    name: str,
    image_emb: np.ndarray,
    concept_emb: np.ndarray,
    k: int,
    output_dir: Path,
    resume: bool,
) -> np.ndarray:
    """Assign top-K concepts, using cached .npy if resume=True and file exists."""
    path = _asgn_path(output_dir, name)
    if resume and path.exists():
        logger.info("Loading cached assignments for %s from %s", name, path)
        return np.load(path)
    logger.info("Assigning top-%d concepts for %s…", k, name)
    assignments = assign_top_k_concepts(image_emb, concept_emb, k=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, assignments)
    return assignments


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmarks
    benchmark_samples = load_benchmarks(args.benchmarks, args.n_samples)

    # 2. Load ImageNet class names (for vocabulary building)
    imagenet_classes: list[str] = []
    if args.imagenet_path:
        try:
            imagenet_classes = load_imagenet_classes_from_folder(args.imagenet_path)
            logger.info("Loaded %d ImageNet class names", len(imagenet_classes))
        except Exception as e:
            logger.warning("Could not load ImageNet class names: %s", e)

    # 3. Build concept vocabulary
    logger.info("Building concept vocabulary…")
    concepts, source_map = build_vocabulary(
        benchmark_samples=benchmark_samples,
        imagenet_classes=imagenet_classes,
    )
    logger.info("Vocabulary size: %d concepts", len(concepts))

    # Save vocabulary with provenance
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {"concepts": concepts, "sources": {c: list(s) for c, s in source_map.items()}},
            f, indent=2,
        )
    logger.info("Saved vocabulary to %s", vocab_path)

    # 4. Load CLIP
    model, processor = load_clip(args.clip_model, args.device)

    # 5. Encode concept vocabulary
    concept_emb_path = _emb_path(output_dir, "concepts")
    if args.resume and concept_emb_path.exists():
        logger.info("Loading cached concept embeddings from %s", concept_emb_path)
        concept_emb = np.load(concept_emb_path)
    else:
        logger.info("Encoding %d concepts…", len(concepts))
        concept_emb = encode_texts(
            concepts, model, processor, args.device,
            batch_size=args.batch_size_texts,
            template=args.concept_template,
        )
        concept_emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(concept_emb_path, concept_emb)

    # 6. For each benchmark: encode images, assign top-K, compute frequencies
    freq_by_dataset: dict[str, np.ndarray] = {}
    n_concepts = len(concepts)

    for bench_name, samples in benchmark_samples.items():
        images = extract_images(samples)
        if not images:
            logger.warning("No images found for %s, skipping.", bench_name)
            continue
        logger.info("%s: %d images", bench_name, len(images))

        image_emb = _load_or_encode_images(
            bench_name, images, model, processor, args.device,
            output_dir, args.batch_size_images, args.resume,
        )
        assignments = _load_or_assign(
            bench_name, image_emb, concept_emb, args.top_k, output_dir, args.resume,
        )
        freq_by_dataset[bench_name] = compute_concept_frequencies(assignments, n_concepts)

    # 7. Load and encode ImageNet images
    if args.imagenet_path:
        logger.info("Loading ImageNet images from %s…", args.imagenet_path)
        try:
            imagenet_images, _ = sample_imagenet_images(
                args.imagenet_path, n_samples=args.n_imagenet, split=args.imagenet_split,
            )
            image_emb = _load_or_encode_images(
                "imagenet", imagenet_images, model, processor, args.device,
                output_dir, args.batch_size_images, args.resume,
            )
            assignments = _load_or_assign(
                "imagenet", image_emb, concept_emb, args.top_k, output_dir, args.resume,
            )
            freq_by_dataset["imagenet"] = compute_concept_frequencies(assignments, n_concepts)
        except Exception as e:
            logger.error("Failed to load ImageNet: %s", e)

    # 8. Load and encode iNaturalist images (optional)
    if args.inat_path:
        logger.info("Loading iNaturalist images from %s…", args.inat_path)
        try:
            inat_images, _ = sample_imagenet_images(
                args.inat_path, n_samples=args.n_inat, split="train",
            )
            image_emb = _load_or_encode_images(
                "inat", inat_images, model, processor, args.device,
                output_dir, args.batch_size_images, args.resume,
            )
            assignments = _load_or_assign(
                "inat", image_emb, concept_emb, args.top_k, output_dir, args.resume,
            )
            freq_by_dataset["inat"] = compute_concept_frequencies(assignments, n_concepts)
        except Exception as e:
            logger.error("Failed to load iNaturalist: %s", e)

    # 8b. Extra image-based datasets (CC3M, CC12M via img2dataset, etc.)
    for spec in args.image_datasets:
        if ":" not in spec:
            logger.error("Invalid --image_datasets spec %r (expected 'name:path')", spec)
            continue
        name, folder = spec.split(":", 1)
        try:
            images = sample_images_from_folder(folder, n_samples=args.n_extra_images)
            image_emb = _load_or_encode_images(
                name, images, model, processor, args.device,
                output_dir, args.batch_size_images, args.resume,
            )
            assignments = _load_or_assign(
                name, image_emb, concept_emb, args.top_k, output_dir, args.resume,
            )
            freq_by_dataset[name] = compute_concept_frequencies(assignments, n_concepts)
            logger.info("Image dataset %s: %d images processed", name, len(images))
        except Exception as e:
            logger.error("Failed to process image dataset %s (%s): %s", name, folder, e)

    # 8c. Caption-based datasets (CC3M, CC12M, etc.)
    for spec in args.caption_files:
        if ":" not in spec:
            logger.error("Invalid --caption_files spec %r (expected 'name:path')", spec)
            continue
        name, path = spec.split(":", 1)
        try:
            captions = load_captions_tsv(path, n_samples=args.n_captions)
            cap_emb = _load_or_encode_captions(
                name, captions, model, processor, args.device,
                output_dir, args.batch_size_texts, args.resume,
            )
            assignments = _load_or_assign(
                name, cap_emb, concept_emb, args.top_k, output_dir, args.resume,
            )
            freq_by_dataset[name] = compute_concept_frequencies(assignments, n_concepts)
            logger.info("Caption dataset %s: %d captions processed", name, len(captions))
        except Exception as e:
            logger.error("Failed to process caption file %s: %s", path, e)

    if not freq_by_dataset:
        logger.error("No datasets encoded. Exiting.")
        return

    # 9. Save frequency arrays
    freq_path = output_dir / "frequencies.npz"
    np.savez(freq_path, concepts=np.array(concepts), **freq_by_dataset)
    logger.info("Saved frequency arrays to %s", freq_path)

    # 10. Gap analysis: benchmarks vs. training datasets
    extra_names = {spec.split(":", 1)[0] for spec in args.caption_files + args.image_datasets if ":" in spec}
    training_keys = [k for k in freq_by_dataset if k in {"imagenet", "inat"} or k in extra_names]
    benchmark_keys = [k for k in freq_by_dataset if k not in training_keys]

    gap_results: dict = {}
    if training_keys and benchmark_keys:
        # Aggregate benchmark frequencies
        bench_freq = np.mean(
            [freq_by_dataset[k] for k in benchmark_keys], axis=0
        )
        for train_key in training_keys:
            gap = find_coverage_gaps(
                bench_freq, freq_by_dataset[train_key], concepts, top_n=30
            )
            gap_results[f"benchmarks_vs_{train_key}"] = gap
            logger.info(
                "KL(benchmarks ‖ %s) = %.4f", train_key, gap["kl_divergence"]
            )
            logger.info("Top 10 missing concepts from %s:", train_key)
            for item in gap["over_represented"][:10]:
                logger.info(
                    "  %-30s  bench=%.4f  train=%.4f  gap=%.4f",
                    item["concept"], item["benchmark_freq"],
                    item["training_freq"], item["gap"],
                )

    gap_path = output_dir / "gap_analysis.json"
    with open(gap_path, "w") as f:
        json.dump(gap_results, f, indent=2)
    logger.info("Saved gap analysis to %s", gap_path)

    # 11. Plots
    if training_keys and benchmark_keys:
        # Main comparison: aggregate benchmarks vs. primary training
        primary_train = training_keys[0]
        bench_freq_combined = np.mean([freq_by_dataset[k] for k in benchmark_keys], axis=0)
        plot_concept_comparison(
            concepts=concepts,
            freq_by_dataset={
                "bias benchmarks (combined)": bench_freq_combined,
                primary_train: freq_by_dataset[primary_train],
            },
            gap_analysis=gap_results[f"benchmarks_vs_{primary_train}"],
            save_path=figures_dir / "comparison.png",
            top_n=args.plot_top_n,
        )

    # Heatmap across all datasets
    plot_concept_heatmap(
        concepts=concepts,
        freq_by_dataset=freq_by_dataset,
        source_map=source_map,
        save_path=figures_dir / "heatmap.png",
        top_n=50,
    )

    # Per-benchmark coverage
    if training_keys:
        plot_per_benchmark_coverage(
            concepts=concepts,
            freq_by_dataset=freq_by_dataset,
            source_map=source_map,
            training_key=training_keys[0],
            save_path=figures_dir / "per_benchmark.png",
            top_n=args.plot_top_n,
        )

    logger.info("Done. Results saved to %s", output_dir)
    logger.info("Figures: %s", figures_dir)


if __name__ == "__main__":
    main()
