"""MedMCQA loading and SFT formatting: question+options -> prompt, answer letter -> target.
"""

DATASET = "openlifescienceai/medmcqa"
LETTERS = "ABCD"

PROMPT = (
    "You are a medical expert answering a multiple-choice exam question.\n"
    "Reason step by step, then end with a line 'Answer: <letter>' where <letter> is A, B, C, or D.\n\n"
    "Question: {question}\n"
    "A. {opa}\nB. {opb}\nC. {opc}\nD. {opd}\n"
    "Reasoning:"
)


def format_prompt(row) -> str:
    return PROMPT.format(**row)


def gold_letter(row) -> str:
    return LETTERS[row["cop"]]


def _tokenize(row, tok, max_len):
    # CoT target: train on the dataset's explanation (`exp`) then "Answer: <letter>".
    prompt_ids = tok(format_prompt(row), add_special_tokens=False)["input_ids"]
    answer_ids = tok(f"\nAnswer: {gold_letter(row)}" + tok.eos_token, add_special_tokens=False)["input_ids"]
    exp = (row.get("exp") or "").strip()
    reason_ids = tok(" " + exp, add_special_tokens=False)["input_ids"] if exp else []
    # Keep prompt + answer as is; trim only the reasoning so the final letter always survives.
    budget = max_len - len(prompt_ids) - len(answer_ids)
    completion = reason_ids[: max(budget, 0)] + answer_ids
    ids = (prompt_ids + completion)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion)[:max_len]  # loss on reasoning + answer
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def train_dataset(tok, n: int, max_len: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train").shuffle(seed=seed).select(range(n))
    return ds.map(lambda r: _tokenize(r, tok, max_len), remove_columns=ds.column_names)


def eval_dataset(n=None):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="validation")
    return ds.select(range(n)) if n else ds
