"""Tokenizes the spike's natural-text corpus into eval_ids.txt.

Run from the repo root with the oracle's virtualenv (it has `tokenizers`):

    oracle/.venv/bin/python -I spike/q8_0/make_eval_ids.py \
        models/SmolLM2-135M/tokenizer.json spike/q8_0/eval_ids.txt \
        spike/q8_0/eval_text/*.txt src/ops/matmul.cpp

Raw `tokenizers` follows tokenizer.json, as the C++ tokenizer does. No BOS is
added, matching the oracle prompts. Each input becomes one or more lines of at
most 512 tokens ("name id id ..."), so the spike's teacher-forced pass never
needs a longer KV cache.
"""

import pathlib
import sys

from tokenizers import Tokenizer

MAX_LEN = 512


def main() -> None:
    tokenizer_json, out_path, *inputs = sys.argv[1:]
    tok = Tokenizer.from_file(tokenizer_json)
    lines = []
    for path in inputs:
        text = pathlib.Path(path).read_text(encoding="utf-8")
        ids = tok.encode(text, add_special_tokens=False).ids
        stem = pathlib.Path(path).name
        for i in range(0, len(ids), MAX_LEN):
            chunk = ids[i : i + MAX_LEN]
            if len(chunk) >= 16:
                lines.append(f"{stem}:{i} " + " ".join(map(str, chunk)))
    pathlib.Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} sequences, {sum(len(l.split()) - 1 for l in lines)} tokens -> {out_path}")


if __name__ == "__main__":
    main()
