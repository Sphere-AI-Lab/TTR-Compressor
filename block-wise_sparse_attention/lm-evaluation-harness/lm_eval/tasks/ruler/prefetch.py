"""Prefetch every external resource needed by the RULER task group."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from lm_eval.tasks.ruler.resources import (
    ALL_RESOURCES,
    get_ruler_data_dir,
    is_offline_mode,
    resolve_resource,
)


def _prefetch_nltk(data_dir: Path) -> None:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "Install the RULER dependencies first: pip install 'lm_eval[ruler]'"
        ) from exc

    nltk_dir = Path(os.environ.get("RULER_NLTK_DATA", data_dir / "nltk_data"))
    nltk_dir.mkdir(parents=True, exist_ok=True)
    staged_punkt = nltk_dir / "tokenizers" / "punkt_tab"
    if staged_punkt.is_dir():
        print(f"[ruler] NLTK punkt_tab already staged: {staged_punkt}")
        return

    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    try:
        existing = Path(str(nltk.data.find("tokenizers/punkt_tab")))
        if existing.is_dir():
            staged_punkt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(existing, staged_punkt, dirs_exist_ok=True)
            print(f"[ruler] NLTK punkt_tab copied from cache: {staged_punkt}")
            return
    except LookupError as lookup_error:
        if is_offline_mode():
            raise RuntimeError(
                f"NLTK punkt_tab is missing from {nltk_dir} while offline"
            ) from lookup_error

    if is_offline_mode():
        raise RuntimeError(f"NLTK punkt_tab is not staged under {nltk_dir}")

    if not nltk.download("punkt_tab", download_dir=str(nltk_dir), quiet=False):
        raise RuntimeError("Failed to download NLTK punkt_tab")
    print(f"[ruler] NLTK punkt_tab: {nltk_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        help="Destination directory (default: RULER_DATA_DIR or $HF_HOME/ruler)",
    )
    parser.add_argument(
        "--skip-nltk",
        action="store_true",
        help="Do not download the NLTK punkt_tab resource",
    )
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer path or Hub ID to validate/prefetch as well",
    )
    args = parser.parse_args()

    if args.data_dir:
        os.environ["RULER_DATA_DIR"] = args.data_dir
    data_dir = get_ruler_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    for resource in ALL_RESOURCES:
        source = resolve_resource(resource)
        destination = data_dir / resource.local_filename
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        print(f"[ruler] {resource.name}: {destination}")

    if not args.skip_nltk:
        _prefetch_nltk(data_dir)

    if args.tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"[ruler] Tokenizer available: {args.tokenizer}")

    print("[ruler] Prefetch complete.")
    print(f"export RULER_DATA_DIR={data_dir}")
    print(f"export NLTK_DATA={data_dir / 'nltk_data'}")


if __name__ == "__main__":
    main()
