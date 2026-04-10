"""Add factual_image column to the already-pushed vlind-bench-oe dataset.

Loads the existing dataset from HuggingFace, looks up the factual image for
each row from the locally-cached VLind-Bench (no re-download needed), adds
the column, and pushes back.

Usage:
    python dataset_creation/add_factual_images.py \\
        --hf_repo fabiangrob/vlind-bench-oe

The VLind-Bench images are read from the local HF cache (snapshot_download
is a no-op if already cached).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="fabiangrob/vlind-bench-oe")
    args = parser.parse_args()

    from datasets import Dataset, Features, Image, load_dataset
    from PIL import Image as PILImage
    from data_loaders.vlind_bench import _download_and_parse, _find_factual_path

    # Load existing dataset
    logger.info("Loading %s …", args.hf_repo)
    ds = load_dataset(args.hf_repo, split="train")
    logger.info("Loaded %d rows.", len(ds))

    # Load VLind-Bench raw data (uses local cache)
    logger.info("Loading VLind-Bench for factual image paths …")
    raw_data, factual_dir, _ = _download_and_parse()

    # Build instance_id → (concept, context_id) lookup
    id_to_entry = {
        idx: (entry["concept"], entry["context_id"])
        for idx, entry in enumerate(raw_data)
    }

    # Build factual image list (one per row, same order as ds)
    factual_images = []
    n_missing = 0
    for row in ds:
        iid = row["instance_id"]
        concept, context_id = id_to_entry[iid]
        path = _find_factual_path(factual_dir, concept, context_id)
        if path and Path(path).exists():
            factual_images.append(PILImage.open(path).convert("RGB"))
        else:
            logger.warning("Missing factual image for instance %d (concept=%s)", iid, concept)
            factual_images.append(None)
            n_missing += 1

    logger.info("%d factual images loaded, %d missing.", len(ds) - n_missing, n_missing)

    # Rebuild dataset from dict so PIL images are handled correctly by the Image() feature.
    # add_column passes directly to PyArrow which can't infer the type from PIL objects.
    from datasets import Dataset, Features, Image
    cols = {col: ds[col] for col in ds.column_names}
    cols["factual_image"] = factual_images
    new_features = Features({**ds.features, "factual_image": Image()})
    ds = Dataset.from_dict(cols, features=new_features)

    logger.info("Pushing updated dataset to %s …", args.hf_repo)
    ds.push_to_hub(args.hf_repo, split="train")
    logger.info("Done.")


if __name__ == "__main__":
    main()
