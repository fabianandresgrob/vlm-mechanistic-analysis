"""
Download images from a HuggingFace caption dataset using img2dataset.

Loads the HF dataset (from cache if available), extracts URLs + captions,
then calls img2dataset to download the images in parallel.

Usage:
    python scripts/download_image_dataset.py \\
        --hf_repo google-research-datasets/conceptual_captions \\
        --output_dir $SCRATCH/datasets/cc3m_images \\
        --image_size 336 \\
        --processes 16 \\
        --threads 64

Submit as a job (CPU-only, no GPU needed):
    python scripts/submit.py scripts/download_image_dataset.py \\
        --hf_repo google-research-datasets/conceptual_captions \\
        --output_dir $SCRATCH/datasets/cc3m_images \\
        -- partition=cpu time=12:00:00 mem=32G cpus=32 gpus=0
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download images from HF caption dataset")
    parser.add_argument("--hf_repo", required=True,
                        help="HuggingFace dataset repo ID, e.g. google-research-datasets/conceptual_captions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save downloaded images, e.g. $SCRATCH/datasets/cc3m_images")
    parser.add_argument("--image_size", type=int, default=336,
                        help="Resize images to this size (default: 336, matches CLIP ViT-L/14-336)")
    parser.add_argument("--processes", type=int, default=16,
                        help="Number of download processes (default: 16)")
    parser.add_argument("--threads", type=int, default=64,
                        help="Number of threads per process (default: 64)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit to N samples (default: all)")
    parser.add_argument("--url_col", default=None,
                        help="URL column name (auto-detected if not specified)")
    parser.add_argument("--caption_col", default=None,
                        help="Caption column name (auto-detected if not specified)")
    args = parser.parse_args()

    # 1. Load HF dataset and detect columns
    logger.info("Loading HF dataset %s (split=%s)…", args.hf_repo, args.split)
    from datasets import load_dataset
    ds = load_dataset(args.hf_repo, split=args.split)
    logger.info("Loaded %d rows. Columns: %s", len(ds), ds.column_names)

    # Auto-detect URL and caption columns
    url_col = args.url_col
    if url_col is None:
        for candidate in ("image_url", "url", "URL"):
            if candidate in ds.column_names:
                url_col = candidate
                break
    if url_col is None:
        raise ValueError(f"Could not find URL column. Available: {ds.column_names}")

    caption_col = args.caption_col
    if caption_col is None:
        for candidate in ("caption", "text", "captions"):
            if candidate in ds.column_names:
                caption_col = candidate
                break
    if caption_col is None:
        raise ValueError(f"Could not find caption column. Available: {ds.column_names}")

    logger.info("Using url_col=%r, caption_col=%r", url_col, caption_col)

    # 2. Write URL+caption TSV to a temp file
    n = args.n_samples or len(ds)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        tsv_path = f.name
        f.write("url\tcaption\n")  # header required by img2dataset
        for i, row in enumerate(ds):
            if i >= n:
                break
            url = row.get(url_col, "")
            caption = row.get(caption_col, "").replace("\t", " ").replace("\n", " ")
            if url:
                f.write(f"{url}\t{caption}\n")
    logger.info("Written %d rows to temp TSV: %s", min(n, len(ds)), tsv_path)

    # 3. Run img2dataset
    os.makedirs(args.output_dir, exist_ok=True)
    cmd = [
        "img2dataset",
        "--url_list", tsv_path,
        "--input_format", "tsv",
        "--url_col", "url",
        "--caption_col", "caption",
        "--output_folder", args.output_dir,
        "--image_size", str(args.image_size),
        "--resize_mode", "keep_ratio",
        "--processes_count", str(args.processes),
        "--thread_count", str(args.threads),
        "--output_format", "files",
        "--enable_wandb", "False",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)

    os.unlink(tsv_path)

    if result.returncode != 0:
        logger.error("img2dataset failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("Done. Images saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
