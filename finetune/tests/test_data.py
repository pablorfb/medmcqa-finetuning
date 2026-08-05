import data

ROW = {"question": "q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 2}


class FakeTok:
    """Whitespace tokenizer — enough to exercise the prompt-masking logic offline."""

    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_gold_letter():
    assert data.gold_letter(ROW) == "C"


def test_prompt_contains_options_and_invites_reasoning():
    p = data.format_prompt(ROW)
    assert "A. a" in p and "D. d" in p
    assert p.endswith("Reasoning:")


def test_tokenize_masks_prompt_only():
    out = data._tokenize(ROW, FakeTok(), max_len=1000)
    prompt_len = len(data.format_prompt(ROW).split())
    assert out["labels"][:prompt_len] == [-100] * prompt_len
    assert out["labels"][prompt_len:] == out["input_ids"][prompt_len:]  # completion is unmasked
    assert set(out["attention_mask"]) == {1}


def test_cot_target_keeps_answer_when_reasoning_trimmed():
    row = {**ROW, "exp": "long winded reasoning that will be trimmed away entirely"}
    prompt_len = len(data.format_prompt(row).split())
    answer = ["Answer:", "C<eos>"]  # FakeTok split of "\nAnswer: C<eos>"
    out = data._tokenize(row, FakeTok(), max_len=prompt_len + 3)  # budget = 1 reasoning token
    assert out["input_ids"][-2:] == answer          # answer survives the trim
    assert out["labels"][-2:] == answer             # and stays unmasked
    assert len(out["input_ids"]) == prompt_len + 3
