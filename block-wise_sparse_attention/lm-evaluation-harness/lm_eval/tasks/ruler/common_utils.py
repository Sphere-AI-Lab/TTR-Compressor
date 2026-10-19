import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer

from lm_eval.tasks.ruler.resources import is_offline_mode


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [
    4096,
]


@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")

    # Task metadata/model_args may contain max_length, dtype, and other model-only
    # options. Do not forward those arbitrary values to the tokenizer loader.
    load_kwargs = {"trust_remote_code": True}
    if Path(os.fspath(pretrained)).expanduser().exists():
        return AutoTokenizer.from_pretrained(
            pretrained, local_files_only=True, **load_kwargs
        )

    # Prefer an existing HF_HOME cache even on connected nodes. If it is not
    # present, only fall back to the configured HF_ENDPOINT when online.
    try:
        return AutoTokenizer.from_pretrained(
            pretrained, local_files_only=True, **load_kwargs
        )
    except Exception as cache_error:
        if is_offline_mode():
            raise RuntimeError(
                f"Tokenizer {pretrained!r} is not available locally while offline "
                "mode is enabled. Provide a local tokenizer path or pre-download "
                "it into HF_HOME."
            ) from cache_error

    return AutoTokenizer.from_pretrained(pretrained, **load_kwargs)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
