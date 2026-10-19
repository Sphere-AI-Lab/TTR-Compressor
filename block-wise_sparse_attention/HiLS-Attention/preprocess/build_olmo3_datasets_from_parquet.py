"""Tokenize FineWeb-style Parquet files into HiLS/OLMo3 ``.data`` files.

This is the Parquet counterpart of ``build_olmo3_datasets.py``.  It preserves
the format consumed by ``data.LazyChunkedLoader``:

* one ``uint32`` token file per source file, with the source directory layout;
* ``input_ids`` are produced with the repository's OLMo3 tokenizer; and
* exactly one EOS token is appended to every source document.

Unlike the JSONL preprocessor, this implementation streams Parquet row batches
and token batches directly to disk, so a large FineWeb shard is never fully
materialized in host memory.
"""

# All local paths are relative to the top-level clone directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
_project_root = _code_root.parent.parent
_os.chdir(_project_root)
_os.environ.setdefault("HF_HOME", ".cache/huggingface")
_os.environ.setdefault("RULER_DATA_DIR", _os.environ["HF_HOME"] + "/ruler")
_sys.path[:0] = [str(_code_root), str(_code_root.parent / "lm-evaluation-harness")]


import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def iter_parquet_files(corpus_dir: Path) -> Iterator[Path]:
    """Yield Parquet files in a deterministic order."""
    yield from sorted(path for path in corpus_dir.rglob("*.parquet") if path.is_file())


def output_path_for(source: Path, corpus_dir: Path, output_dir: Path) -> Path:
    """Map ``foo/bar.parquet`` to ``<output>/foo/bar.parquet.data``."""
    return output_dir / source.relative_to(corpus_dir).with_suffix(source.suffix + ".data")


def tokenize_parquet_file(
    source_path: str,
    corpus_dir: str,
    output_dir: str,
    vocab_dir: str,
    text_column: str,
    row_batch_size: int,
    token_batch_size: int,
    overwrite: bool,
    show_progress: bool,
    progress_position: int,
    max_rows_per_file: int | None,
) -> dict:
    """Tokenize one Parquet file and atomically publish its ``.data`` output."""
    source = Path(source_path)
    destination = output_path_for(source, Path(corpus_dir), Path(output_dir))

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return {"source": str(source), "output": str(destination), "status": "skipped"}

    parquet_file = pq.ParquetFile(source)
    available_columns = set(parquet_file.schema_arrow.names)
    if text_column not in available_columns:
        raise KeyError(
            f"{source} has no {text_column!r} column. "
            f"Available columns: {sorted(available_columns)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(vocab_dir, use_fast=True)
    # Documents are deliberately tokenized without truncation and subsequently
    # sampled into fixed-length training segments.  The tokenizer's model-side
    # max length (8192) is therefore not a tokenization limit here.
    tokenizer.model_max_length = int(1e30)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"Tokenizer at {vocab_dir} does not define eos_token_id")
    if eos_token_id < 0 or eos_token_id > np.iinfo(np.uint32).max:
        raise ValueError(f"EOS token id {eos_token_id} cannot be represented as uint32")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    documents = 0
    tokens = 0
    skipped_rows = 0
    progress = tqdm(
        total=min(parquet_file.metadata.num_rows, max_rows_per_file)
        if max_rows_per_file is not None
        else parquet_file.metadata.num_rows,
        desc=f"tokenizing {source.name}",
        unit="rows",
        disable=not show_progress,
        position=progress_position,
    )
    source_rows_read = 0
    try:
        with temporary.open("wb") as output_file:
            for record_batch in parquet_file.iter_batches(
                batch_size=row_batch_size,
                columns=[text_column],
                use_threads=False,
            ):
                raw_texts = record_batch.column(0).to_pylist()
                if max_rows_per_file is not None:
                    remaining_rows = max_rows_per_file - source_rows_read
                    if remaining_rows <= 0:
                        break
                    raw_texts = raw_texts[:remaining_rows]
                texts = []
                for text in raw_texts:
                    if isinstance(text, str) and text:
                        texts.append(text)
                    else:
                        skipped_rows += 1

                for start in range(0, len(texts), token_batch_size):
                    text_batch = texts[start:start + token_batch_size]
                    encoded = tokenizer(
                        text_batch,
                        # Keep the same default as tokenizer.encode(text), used by
                        # build_olmo3_datasets.py, before adding an explicit EOS.
                        add_special_tokens=True,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )["input_ids"]
                    # Write the entire tokenizer batch in one operation.  The
                    # original preprocessor writes one whole source file at a
                    # time; doing the same at batch granularity avoids one
                    # write syscall per FineWeb document.
                    batch_tokens = sum(len(token_ids) + 1 for token_ids in encoded)
                    flattened = np.empty(batch_tokens, dtype=np.uint32)
                    offset = 0
                    for token_ids in encoded:
                        token_length = len(token_ids)
                        flattened[offset:offset + token_length] = token_ids
                        flattened[offset + token_length] = eos_token_id
                        offset += token_length + 1
                        documents += 1
                    flattened.tofile(output_file)
                    tokens += batch_tokens
                progress.update(len(raw_texts))
                source_rows_read += len(raw_texts)
    finally:
        progress.close()

    if documents == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"No non-empty text values found in {source}")

    os.replace(temporary, destination)
    return {
        "source": str(source),
        "output": str(destination),
        "status": "written",
        "documents": documents,
        "tokens": tokens,
        "skipped_rows": skipped_rows,
        "source_rows_read": source_rows_read,
        "is_partial": max_rows_per_file is not None and source_rows_read < parquet_file.metadata.num_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-style Parquet files into HiLS/OLMo3 uint32 .data files"
    )
    parser.add_argument("--vocab_dir", required=True, help="HiLS OLMo3 tokenizer directory")
    parser.add_argument("--corpus_dir", required=True, help="Directory containing source Parquet files")
    parser.add_argument("--output_dir", required=True, help="Destination directory for .data files")
    parser.add_argument("--text_column", default="text", help="Parquet column containing document text")
    parser.add_argument("--num_workers", type=int, default=8, help="Concurrent Parquet files to tokenize")
    parser.add_argument("--row_batch_size", type=int, default=4096, help="Rows read per Parquet batch")
    parser.add_argument("--token_batch_size", type=int, default=512, help="Texts passed to tokenizer together")
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Only tokenize the first N sorted Parquet files (for a smoke test)",
    )
    parser.add_argument(
        "--max_rows_per_file",
        type=int,
        default=None,
        help="Only read the first N rows of each source file (use a separate output_dir for smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing non-empty .data files")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable per-file row progress bars",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional JSONL manifest of written/skipped source files and token counts",
    )
    args = parser.parse_args()

    if args.num_workers < 1 or args.row_batch_size < 1 or args.token_batch_size < 1:
        parser.error("num_workers, row_batch_size, and token_batch_size must all be positive")
    if args.max_files is not None and args.max_files < 1:
        parser.error("max_files must be positive when provided")
    if args.max_rows_per_file is not None and args.max_rows_per_file < 1:
        parser.error("max_rows_per_file must be positive when provided")

    corpus_dir = Path(args.corpus_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not corpus_dir.is_dir():
        parser.error(f"corpus_dir does not exist or is not a directory: {corpus_dir}")
    if output_dir == corpus_dir or corpus_dir in output_dir.parents:
        parser.error("output_dir must not be corpus_dir or a subdirectory of corpus_dir")

    parquet_files = list(iter_parquet_files(corpus_dir))
    if args.max_files is not None:
        parquet_files = parquet_files[:args.max_files]
    if not parquet_files:
        parser.error(f"No .parquet files found under {corpus_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = None
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path.open("a", encoding="utf-8")

    written = skipped = total_documents = total_tokens = 0
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            pending = {}
            file_iter = iter(parquet_files)

            def submit_next(progress_position: int) -> bool:
                try:
                    source = next(file_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    tokenize_parquet_file,
                    str(source),
                    str(corpus_dir),
                    str(output_dir),
                    args.vocab_dir,
                    args.text_column,
                    args.row_batch_size,
                    args.token_batch_size,
                    args.overwrite,
                    not args.no_progress,
                    progress_position,
                    args.max_rows_per_file,
                )
                pending[future] = (source, progress_position)
                return True

            for progress_position in range(min(args.num_workers, len(parquet_files))):
                submit_next(progress_position)

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    source, progress_position = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Failed to tokenize {source}") from exc

                    if manifest_file:
                        manifest_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        manifest_file.flush()
                    if result["status"] == "written":
                        written += 1
                        total_documents += result["documents"]
                        total_tokens += result["tokens"]
                        print(
                            f"written {result['output']} "
                            f"({result['documents']:,} documents, {result['tokens']:,} tokens)",
                            flush=True,
                        )
                    else:
                        skipped += 1
                        print(f"skipped existing {result['output']}", flush=True)
                    submit_next(progress_position)
    finally:
        if manifest_file:
            manifest_file.close()

    print(
        f"Done: written={written}, skipped={skipped}, "
        f"documents={total_documents:,}, tokens={total_tokens:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
