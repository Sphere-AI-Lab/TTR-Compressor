# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import itertools  # noqa: I001
import json
import random
from functools import cache
from pathlib import Path

import datasets
from tqdm import tqdm

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer
from lm_eval.tasks.ruler.resources import HOTPOTQA, SQUAD_V2, resolve_resource

CONFIG = {
    "tokens_to_generate": 32,
    "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
    "answer_prefix": """Answer:""",
}
SEED = 42
TEMPLATE = CONFIG["template"]
DOCUMENT_PROMPT = "Document {i}:\n{document}"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@cache
def read_squad(path: str | None = None) -> tuple[list[dict], list[str]]:
    squad_path = Path(path) if path else resolve_resource(SQUAD_V2)
    if squad_path.suffix.lower() == ".json":
        return _read_squad_json(squad_path)

    data = datasets.Dataset.from_parquet(str(squad_path))
    rows = list(data)
    total_docs = [row["context"] for row in rows]
    total_docs = sorted(set(total_docs))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    title_docs: dict[str, list[int]] = {}
    for row in rows:
        title_docs.setdefault(row.get("title", ""), []).append(
            total_docs_dict[row["context"]]
        )
    title_docs = {title: sorted(set(indices)) for title, indices in title_docs.items()}

    total_qas = []
    for row in rows:
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue
        context_idx = total_docs_dict[row["context"]]
        total_qas.append(
            {
                "query": row["question"],
                "outputs": answers,
                "context": [context_idx],
                "more_context": [
                    idx
                    for idx in title_docs.get(row.get("title", ""), [])
                    if idx != context_idx
                ],
            }
        )

    return total_qas, total_docs


@cache
def _read_squad_json(path: Path) -> tuple[list[dict], list[str]]:
    data = _read_json(path)
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(set(total_docs))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for document in data["data"]:
        more_docs = [
            total_docs_dict[paragraph["context"]]
            for paragraph in document["paragraphs"]
        ]
        for paragraph in document["paragraphs"]:
            context_idx = total_docs_dict[paragraph["context"]]
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qa["question"],
                            "outputs": [answer["text"] for answer in qa["answers"]],
                            "context": [context_idx],
                            "more_context": [
                                idx for idx in more_docs if idx != context_idx
                            ],
                        }
                    )
    return total_qas, total_docs


@cache
def read_hotpotqa(path: str | None = None) -> tuple[list[dict], list[str]]:
    hotpot_path = Path(path) if path else resolve_resource(HOTPOTQA)
    data = _read_json(hotpot_path)
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(set(total_docs))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q: str = qas[index]["query"]
    curr_a: list[str] = qas[index]["outputs"]
    curr_docs: list[int] = qas[index]["context"]
    curr_more: list[int] = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(SEED).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = TEMPLATE.format(context=context, query=curr_q)
    return input_text, curr_a


def generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = 500,
    tokens_to_generate: int = 32,
    pre_samples: int = 0,
    incremental: int = 10,
    remove_newline_tab=False,
) -> list[dict]:
    write_jsons = []

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer = generate_input_output(0, num_docs, qas=qas, docs=docs)
        # Calculate the number of tokens in the example
        total_tokens = len(tokenizer(input_text + f" {answer}").input_ids)
        # print(
        #     f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        # )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break
    # print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(
        range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"
    ):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = generate_input_output(
                    index + pre_samples, used_docs, qas=qas, docs=docs
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:  # noqa: E722
                if used_docs > incremental:
                    used_docs -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": "Answer:",
        }
        write_jsons.append(formatted_output)

    return write_jsons


def get_dataset(pretrained, docs, qas, max_seq_length=None, **kwargs) -> list[dict]:
    tokenizer = get_tokenizer(pretrained)
    write_jsons = generate_samples(
        tokenizer=tokenizer,
        docs=docs,
        qas=qas,
        num_samples=500,
        tokens_to_generate=32,
        max_seq_length=max_seq_length,
    )
    return write_jsons


def get_qa_dataset(ds, **kwargs) -> dict[str, datasets.Dataset]:
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", {}))
    if ds == "squad":
        qas, docs = read_squad()
    else:
        qas, docs = read_hotpotqa()
    df = (
        get_dataset(pretrained=pretrained, docs=docs, qas=qas, max_seq_length=seq)
        for seq in kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    )

    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


def get_squad(**kwargs):
    return get_qa_dataset("squad", **kwargs)


def get_hotpotqa(**kwargs):
    return get_qa_dataset("hotpotqa", **kwargs)
