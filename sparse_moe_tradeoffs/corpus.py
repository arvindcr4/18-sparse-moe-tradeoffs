"""Self-written mixed-domain char corpus: prose + code + numbers.

Every character carries a domain tag (0=prose, 1=code, 2=numbers) so routing
specialization can be measured per domain.  Everything is generated
procedurally from handwritten templates — no downloads.
"""

import random

import numpy as np
import torch

DOMAINS = ("prose", "code", "numbers")
IGNORE_INDEX = -100

_SENTENCES = [
    "the old lighthouse keeper watched the storm roll in from the west.",
    "every morning she walked the same narrow path along the cliffs.",
    "a small fox slipped quietly between the birch trees at dusk.",
    "the market square smelled of bread and rain and diesel smoke.",
    "he kept his letters in a tin box under the floorboards.",
    "nobody in the village remembered when the bell had last rung.",
    "the river ran slow and brown after three days of heavy rain.",
    "she taught herself the names of all the northern stars.",
    "an old map of the coastline hung above the fireplace.",
    "the train was late again and the platform stood almost empty.",
    "winter came early that year and stayed until the end of april.",
    "his grandmother told stories about ships that never returned.",
    "the library smelled of dust and old glue and quiet afternoons.",
    "two crows argued on the fence while the kettle came to a boil.",
    "the path through the orchard was soft with fallen apples.",
    "a thin line of smoke rose from the chimney across the valley.",
    "she folded the letter twice and slid it under the blue door.",
    "the tide left long ribbons of kelp across the grey sand.",
    "morning fog swallowed the far end of the wooden bridge.",
    "he counted the coins twice and still came up one short.",
]

_CODE_NAMES = ["total", "count", "value", "items", "score", "delta", "index", "width", "depth", "level"]
_CODE_FUNCS = ["compute", "update", "merge", "scale", "clip", "reduce", "shift", "blend"]

_NUM_WORDS = None  # numbers domain is purely procedural


def _gen_prose(rng: random.Random, n_chars: int) -> str:
    parts = []
    size = 0
    while size < n_chars:
        s = rng.choice(_SENTENCES) + " "
        parts.append(s)
        size += len(s)
    return "".join(parts)


def _gen_code(rng: random.Random, n_chars: int) -> str:
    parts = []
    size = 0
    while size < n_chars:
        f = rng.choice(_CODE_FUNCS)
        a, b = rng.sample(_CODE_NAMES, 2)
        op = rng.choice(["+", "-", "*"])
        kind = rng.randrange(3)
        if kind == 0:
            s = f"def {f}({a}, {b}):\n    return {a} {op} {b}\n"
        elif kind == 1:
            s = f"for {a} in range({rng.randint(2, 99)}):\n    {b} = {b} {op} {a}\n"
        else:
            s = f"if {a} > {rng.randint(0, 9)}:\n    {b} = {f}({a}, {b})\n"
        parts.append(s)
        size += len(s)
    return "".join(parts)


def _gen_numbers(rng: random.Random, n_chars: int) -> str:
    parts = []
    size = 0
    while size < n_chars:
        kind = rng.randrange(3)
        if kind == 0:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            s = f"{a} + {b} = {a + b}\n"
        elif kind == 1:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            s = f"{a} * {b} = {a * b}\n"
        else:
            start, step = rng.randint(0, 20), rng.randint(2, 9)
            seq = " ".join(str(start + i * step) for i in range(6))
            s = seq + "\n"
        parts.append(s)
        size += len(s)
    return "".join(parts)


_GENS = (_gen_prose, _gen_code, _gen_numbers)


def build_corpus(n_chars: int = 150_000, block_min: int = 120, block_max: int = 300,
                 seed: int = 0):
    """Generate the interleaved mixed-domain corpus.

    Returns ``(text, tags)`` where ``tags`` is an int8 numpy array with the
    domain id of every character.
    """
    rng = random.Random(seed)
    parts, tags = [], []
    size = 0
    while size < n_chars:
        dom = rng.randrange(len(_GENS))
        chunk = _GENS[dom](rng, rng.randint(block_min, block_max))
        parts.append(chunk)
        tags.append(np.full(len(chunk), dom, dtype=np.int8))
        size += len(chunk)
    return "".join(parts), np.concatenate(tags)


class CharData:
    """Char-level dataset with next-token, next-next-token and domain targets."""

    def __init__(self, text: str, tags: np.ndarray, block_size: int = 128,
                 val_frac: float = 0.1):
        assert len(text) == len(tags)
        self.block_size = block_size
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n_val = int(len(data) * val_frac)
        self.train_data, self.val_data = data[:-n_val], data[-n_val:]
        self.train_tags, self.val_tags = tags[:-n_val], tags[-n_val:]

    def get_batch(self, split: str, batch_size: int, generator: np.random.Generator):
        """Sample a batch.

        Returns ``(x, y1, y2, dom)`` each of shape (B, T):
          * ``y1[t]`` — next token (t+1), always valid,
          * ``y2[t]`` — next-next token (t+2) **within the sampled window**;
            the final position has no t+2 target inside the window and is set
            to ``IGNORE_INDEX`` so the MTP head never trains on it,
          * ``dom[t]`` — domain tag of the input character at position t.
        """
        data = self.train_data if split == "train" else self.val_data
        tags = self.train_tags if split == "train" else self.val_tags
        T = self.block_size
        starts = generator.integers(0, len(data) - T - 1, size=batch_size)
        x = np.stack([data[s:s + T] for s in starts])
        y1 = np.stack([data[s + 1:s + T + 1] for s in starts])
        y2 = np.full_like(y1, IGNORE_INDEX)
        y2[:, :-1] = np.stack([data[s + 2:s + T + 1] for s in starts])
        dom = np.stack([tags[s:s + T] for s in starts])
        return (torch.from_numpy(x), torch.from_numpy(y1),
                torch.from_numpy(y2), torch.from_numpy(dom.astype(np.int64)))
