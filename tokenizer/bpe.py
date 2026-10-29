import regex
import re as std_re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pretoken_counts(text: str) -> Counter:
    pretoken_strs = Counter(m.group() for m in regex.finditer(PAT, text))
    byte_counts = Counter()
    for pretoken_str, count in pretoken_strs.items():
        byte_tuple = tuple(pretoken_str.encode("utf-8"))
        byte_counts[byte_tuple] = count
    return byte_counts


def split_on_special_tokens(text: str, special_tokens: list) -> list:
    if not special_tokens:
        return [text]
    pattern = "|".join(std_re.escape(tok) for tok in special_tokens)
    chunks = std_re.split(pattern, text)
    return chunks


def get_pair_counts(byte_counts: Counter) -> Counter:
    pair_counts = Counter()
    for byte_tuple, count in byte_counts.items():
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pair_counts[pair] += count
    return pair_counts


def find_best_pair(pair_counts: Counter, vocab: dict) -> tuple:
    best_pair = max(
        pair_counts.keys(),
        key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]]))
    )
    return best_pair


def merge_pretoken(byte_tuple: tuple, pair: tuple, new_id: int) -> tuple:
    result = []
    i = 0
    while i < len(byte_tuple):
        if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i+1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(byte_tuple[i])
            i += 1
    return tuple(result)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_on_special_tokens(text, special_tokens)

    byte_counts = Counter()
    for chunk in chunks:
        byte_counts.update(get_pretoken_counts(chunk))

    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pair_counts = get_pair_counts(byte_counts)
        if not pair_counts:
            break
        best_pair = find_best_pair(pair_counts, vocab)

        new_id = next_id
        next_id += 1
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        new_byte_counts = Counter()
        for byte_tuple, count in byte_counts.items():
            new_byte_counts[merge_pretoken(byte_tuple, best_pair, new_id)] += count
        byte_counts = new_byte_counts

    return vocab, merges
