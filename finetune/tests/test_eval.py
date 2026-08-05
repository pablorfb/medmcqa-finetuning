from eval_medmcqa import parse_letter


def test_letter_variants():
    assert parse_letter("D") == "D"
    assert parse_letter(" D") == "D"
    assert parse_letter("D.") == "D"
    assert parse_letter("b") == "B"
    assert parse_letter(" c\n") == "C"


def test_unparseable_is_none():
    assert parse_letter("") is None
    assert parse_letter("hmm") is None


def test_cot_takes_final_answer_not_stray_letters():
    # reasoning mentions A and B, but the final "Answer:" is what counts
    assert parse_letter("Option A is wrong, B is tempting, so Answer: D") == "D"
    assert parse_letter("...therefore\nAnswer: C") == "C"
    assert parse_letter("answer: a") == "A"  # case-insensitive
