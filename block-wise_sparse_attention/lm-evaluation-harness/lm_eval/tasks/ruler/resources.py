"""Resource resolution helpers for RULER.

Resolution order is deliberately deterministic:

1. An explicit per-resource environment variable.
2. A file staged under ``RULER_DATA_DIR`` (defaults to ``$HF_HOME/ruler``).
3. The existing Hugging Face Hub cache.
4. The configured Hugging Face endpoint, unless offline mode is enabled.

Using :func:`huggingface_hub.hf_hub_download` instead of raw HTTP requests
makes ``HF_HOME``, ``HF_ENDPOINT`` and ``HF_HUB_OFFLINE`` work as expected.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


_TRUE_VALUES = frozenset({"1", "on", "true", "yes"})
eval_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RulerResource:
    name: str
    repo_id: str
    filename: str
    local_filename: str
    env_var: str
    revision: str = "main"
    alternate_local_filenames: tuple[str, ...] = ()


PAUL_GRAHAM_ESSAYS = RulerResource(
    name="Paul Graham essays",
    repo_id="baber/paul_graham_essays",
    filename="essays.parquet",
    local_filename="paul_graham_essays.parquet",
    env_var="RULER_PAUL_GRAHAM_PATH",
)

HOTPOTQA = RulerResource(
    name="HotpotQA distractor validation set",
    repo_id="namlh2004/hotpotqa",
    filename="hotpot_dev_distractor_v1.json",
    local_filename="hotpot_dev_distractor_v1.json",
    env_var="RULER_HOTPOTQA_PATH",
    revision="7e54db4656209750ff487f6fdf8e39a66dba136b",
)

SQUAD_V2 = RulerResource(
    name="SQuAD v2 validation set",
    repo_id="rajpurkar/squad_v2",
    filename="squad_v2/validation-00000-of-00001.parquet",
    local_filename="squad_v2_validation.parquet",
    env_var="RULER_SQUAD_PATH",
    revision="3ffb306f725f7d2ce8394bc1873b24868140c412",
    alternate_local_filenames=("dev-v2.0.json",),
)

ALL_RESOURCES = (PAUL_GRAHAM_ESSAYS, HOTPOTQA, SQUAD_V2)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def is_offline_mode() -> bool:
    """Return whether any supported HF offline switch is enabled."""

    return any(
        env_flag(name)
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        )
    )


def get_ruler_data_dir() -> Path:
    explicit = os.environ.get("RULER_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "ruler"

    return Path(".cache") / "huggingface" / "ruler"


def _existing_staged_path(resource: RulerResource) -> Path | None:
    override = os.environ.get(resource.env_var)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{resource.env_var} points to a missing file: {path}"
            )
        return path

    data_dir = get_ruler_data_dir()
    for filename in (resource.local_filename, *resource.alternate_local_filenames):
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_resource(resource: RulerResource) -> Path:
    """Resolve a RULER resource locally, using the network only as fallback."""

    if staged := _existing_staged_path(resource):
        eval_logger.info("RULER %s: using staged file %s", resource.name, staged)
        return staged

    # Import lazily so the resolver itself remains lightweight and unit-testable.
    from huggingface_hub import hf_hub_download

    hub_kwargs = {
        "repo_id": resource.repo_id,
        "filename": resource.filename,
        "repo_type": "dataset",
        "revision": resource.revision,
    }

    cache_error = None
    try:
        cached = Path(hf_hub_download(**hub_kwargs, local_files_only=True))
        eval_logger.info("RULER %s: using Hugging Face cache %s", resource.name, cached)
        return cached
    except Exception as exc:  # Hub exception types vary across supported versions.
        cache_error = exc

    if is_offline_mode():
        data_dir = get_ruler_data_dir()
        raise RuntimeError(
            f"RULER resource '{resource.name}' was not found locally while offline "
            f"mode is enabled. Stage it at {data_dir / resource.local_filename}, "
            f"set {resource.env_var}, or prefetch it on a connected node with "
            "`python -m lm_eval.tasks.ruler.prefetch`. "
            f"Underlying cache error: {cache_error}"
        ) from cache_error

    # hf_hub_download honors HF_ENDPOINT (for example https://hf-mirror.com)
    # and stores the result under HF_HOME for later offline runs.
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    eval_logger.info(
        "RULER %s: not found locally; downloading from %s",
        resource.name,
        endpoint,
    )
    downloaded = Path(hf_hub_download(**hub_kwargs))
    eval_logger.info("RULER %s: downloaded to %s", resource.name, downloaded)
    return downloaded
