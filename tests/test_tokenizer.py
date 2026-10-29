import pytest
from tokenizer.bpe import train_bpe


def test_toy_merges(tmp_path):
    text = "low " * 5 + "lower " * 2 + "widest " * 3 + "newest " * 6

    input_path = tmp_path / "toy.txt"
    input_path.write_text(text)

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=262,
        special_tokens=[],
    )

    expected_first_six = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]
    assert merges[:6] == expected_first_six
