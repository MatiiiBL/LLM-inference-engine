#!/usr/bin/env python3
"""Generate the PyTorch golden data that minillm's tests compare against.

One command, idempotent, fixed seeds:

    python oracle/generate.py

Downloads HuggingFaceTB/SmolLM2-135M into models/, loads it with transformers
in float32 with attn_implementation="eager", and writes .npy files plus a
manifest.json into oracle/data/.

Everything here is derived from the checkpoint's own config.json — no
hyperparameter is written by hand. Inputs come from numpy's PCG64 with explicit
per-case seeds rather than torch's global RNG, so the data does not depend on
the torch version.

Layout conventions must match include/minillm/ops.h exactly:
  * tensors are row-major and saved in the shape the C++ op expects
  * q/k are [n_tokens, n_heads, head_dim], NOT torch's [batch, heads, seq, dim]
  * linear weights stay [out_features, in_features]
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

# Prompts for the model-level golden data. Short on purpose: every layer's
# hidden states are saved for each one.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a village at the edge of the forest,",
]

# Tokenizer edge cases. Each must round-trip through encode/decode.
TOKENIZER_CORPUS = [
    ("empty", ""),
    ("ascii", "Hello, world!"),
    ("ascii_sentence", "The quick brown fox jumps over the lazy dog."),
    ("leading_space", " leading space"),
    ("trailing_space", "trailing space "),
    ("accents", "Café naïve façade über Ångström résumé"),
    ("spanish", "¿Dónde está la biblioteca? ¡Aquí!"),
    ("cjk", "日本語のテキストと中文文本和한국어"),
    ("emoji", "hello 👋 world 🌍 family 👨‍👩‍👧‍👦 flag 🏳️‍🌈"),
    ("whitespace_runs", "a  b   c\td\t\te"),
    ("newlines", "line one\nline two\n\nline four\r\nwindows"),
    ("only_spaces", "     "),
    ("code", "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }"),
    ("code_python", "x = {'a': 1, 'b': [2, 3]}  # comment\nif x: print(x)"),
    ("numbers", "0 7 42 1234567890 3.14159 -0.5 1e10"),
    ("mixed_digits", "version 2.11.3 build 20240917"),
    ("punctuation", "...,,,;;;!!!???---___===+++"),
    ("urls", "https://example.com/path?query=1&other=2#frag"),
    ("special_literal", "<|im_start|>user\nhi<|im_end|>"),
    ("endoftext_literal", "before<|endoftext|>after"),
    ("repeated", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ("long_word", "pneumonoultramicroscopicsilicovolcanoconiosis"),
    ("tabs_only", "\t\t\t"),
    ("single_char", "a"),
    ("nbsp", "non breaking space"),
    ("zero_width", "zero​width​joiner"),
]


class Writer:
    """Writes golden files and accumulates the manifest."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.entries: list[dict[str, Any]] = []

    def array(self, rel: str, data: Any, description: str) -> np.ndarray:
        """Saves a float32/int64 array and records it. Returns what was saved."""
        if isinstance(data, torch.Tensor):
            data = data.detach().to(torch.float32).cpu().numpy()
        arr = np.ascontiguousarray(data)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype in (np.int32, np.int64):
            arr = arr.astype(np.int64)
        elif arr.dtype != np.float32:
            raise TypeError(f"{rel}: refusing to save dtype {arr.dtype}")

        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr, allow_pickle=False)
        self.entries.append(
            {
                "file": rel,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "description": description,
            }
        )
        return arr

    def scalar_f32(self, rel: str, value: float, description: str) -> None:
        self.array(rel, np.array([value], dtype=np.float32), description)

    def scalar_i64(self, rel: str, value: int, description: str) -> None:
        self.array(rel, np.array([value], dtype=np.int64), description)

    def json(self, rel: str, obj: Any, description: str) -> None:
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        self.entries.append(
            {"file": rel, "shape": None, "dtype": "json", "description": description}
        )

    def text(self, rel: str, lines: list[str], description: str) -> None:
        """A plain text file, one entry per line, for tests that cannot parse JSON."""
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        self.entries.append(
            {"file": rel, "shape": None, "dtype": "text", "description": description}
        )

    def finish(self, header: dict[str, Any]) -> None:
        manifest = dict(header)
        manifest["entries"] = sorted(self.entries, key=lambda e: e["file"])
        with (self.out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"  manifest.json ({len(manifest['entries'])} entries)")


def randn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """float32 N(0, 1), the distribution every op case uses."""
    return rng.standard_normal(shape, dtype=np.float32)


# --------------------------------------------------------------------------
# RoPE, via the installed transformers' own Llama code
# --------------------------------------------------------------------------


def rope_theta_of(cfg: Any) -> float:
    """RoPE base, wherever this transformers version keeps it.

    transformers < 5 exposes config.rope_theta; transformers >= 5 moved it into
    config.rope_parameters (and aliases config.rope_scaling to the same dict,
    so "rope_scaling is not None" is NOT a usable rejection test there).
    The checkpoint's config.json still has a top-level "rope_theta", which is
    what the C++ loader reads.
    """
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)
    raise RuntimeError("cannot find rope_theta on this config")


def rope_type_of(cfg: Any) -> str:
    """"default" for plain RoPE; anything else is an unimplemented variant."""
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict):
        return str(params.get("rope_type", "default"))
    scaling = getattr(cfg, "rope_scaling", None)
    if scaling is None:
        return "default"
    if isinstance(scaling, dict):
        return str(scaling.get("rope_type") or scaling.get("type") or "unknown")
    return "unknown"


def head_dim_of(cfg: Any) -> int:
    return int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)


def rotary_module(model: Any) -> Any:
    """The model's own rotary embedding, wherever this version keeps it."""
    rot = getattr(model.model, "rotary_emb", None)
    if rot is not None:
        return rot
    rot = getattr(model.model.layers[0].self_attn, "rotary_emb", None)
    if rot is not None:
        return rot
    raise RuntimeError(
        "cannot find LlamaRotaryEmbedding on this transformers version; "
        "adapt oracle/generate.py:rotary_module"
    )


def cos_sin(rot: Any, ref: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) for the given positions, using the installed implementation."""
    try:
        return rot(ref, position_ids)
    except TypeError:
        # Older signature: forward(x, seq_len=...) returning tables for
        # [0, seq_len), which we then gather at the positions we want.
        seq_len = int(position_ids.max().item()) + 1
        cos, sin = rot(ref, seq_len=seq_len)
        cos = cos.squeeze()[position_ids[0]].unsqueeze(0)
        sin = sin.squeeze()[position_ids[0]].unsqueeze(0)
        return cos, sin


def apply_rope(
    rot: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to [1, n_heads, n_tokens, head_dim] tensors."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    cos, sin = cos_sin(rot, q, position_ids)
    params = inspect.signature(apply_rotary_pos_emb).parameters
    if "position_ids" in params:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    return apply_rotary_pos_emb(q, k, cos, sin)


def to_torch_heads(arr: np.ndarray) -> torch.Tensor:
    """[n_tokens, n_heads, head_dim] -> torch's [1, n_heads, n_tokens, head_dim]."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.unsqueeze(0).transpose(1, 2).contiguous()


def from_torch_heads(t: torch.Tensor) -> np.ndarray:
    """torch's [1, n_heads, n_tokens, head_dim] -> [n_tokens, n_heads, head_dim]."""
    return t.squeeze(0).transpose(0, 1).contiguous().detach().cpu().numpy()


# --------------------------------------------------------------------------
# ops/: one case per op in include/minillm/ops.h
# --------------------------------------------------------------------------


def gen_ops(w: Writer, model: Any, cfg: Any, rot: Any) -> None:
    print("ops/")
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = head_dim_of(cfg)
    vocab = cfg.vocab_size
    eps = cfg.rms_norm_eps
    theta = rope_theta_of(cfg)

    # ---- add ----
    for seed, (variant, shape) in enumerate((("small", (4, 8)), ("model", (8, hidden)))):
        rng = np.random.default_rng(1000 + seed)
        a = randn(rng, *shape)
        b = randn(rng, *shape)
        w.array(f"ops/add_{variant}_a.npy", a, f"add input a {list(shape)}")
        w.array(f"ops/add_{variant}_b.npy", b, f"add input b {list(shape)}")
        w.array(
            f"ops/add_{variant}_out.npy",
            torch.from_numpy(a) + torch.from_numpy(b),
            "torch.add(a, b)",
        )

    # ---- silu_mul (SwiGLU elementwise stage) ----
    for seed, (variant, shape) in enumerate((("small", (4, 8)), ("model", (8, inter)))):
        rng = np.random.default_rng(2000 + seed)
        gate = randn(rng, *shape)
        up = randn(rng, *shape)
        w.array(f"ops/silu_mul_{variant}_gate.npy", gate, f"gate_proj output {list(shape)}")
        w.array(f"ops/silu_mul_{variant}_up.npy", up, f"up_proj output {list(shape)}")
        w.array(
            f"ops/silu_mul_{variant}_out.npy",
            torch.nn.functional.silu(torch.from_numpy(gate)) * torch.from_numpy(up),
            "torch.nn.functional.silu(gate) * up",
        )

    # ---- embedding ----
    # The model-shape table uses a trimmed vocabulary: the op is oblivious to
    # vocab size and the full [49152, 576] table would be a 113 MB golden file.
    for seed, (variant, (rows, width, n_ids)) in enumerate(
        (("small", (32, 8, 5)), ("model", (1024, hidden, 8)))
    ):
        rng = np.random.default_rng(3000 + seed)
        table = randn(rng, rows, width)
        ids = rng.integers(0, rows, size=n_ids, dtype=np.int64)
        w.array(
            f"ops/embedding_{variant}_table.npy",
            table,
            f"embedding table [{rows}, {width}]"
            + (f" (vocab trimmed from {vocab} to keep the file small)" if variant == "model" else ""),
        )
        w.array(f"ops/embedding_{variant}_ids.npy", ids, f"{n_ids} token ids")
        w.array(
            f"ops/embedding_{variant}_out.npy",
            torch.nn.functional.embedding(torch.from_numpy(ids), torch.from_numpy(table)),
            "torch.nn.functional.embedding(ids, table)",
        )

    # ---- matmul: y = x @ w.T, one case per linear layer in the forward pass ----
    kv_dim = n_kv_heads * head_dim
    lm_head_rows = 4096  # trimmed vocabulary, same reason as above
    matmul_cases = {
        "small": (4, 8, 6),
        "qkv": (8, hidden, n_heads * head_dim),
        "kv": (8, hidden, kv_dim),
        "mlp_up": (8, hidden, inter),
        "mlp_down": (8, inter, hidden),
        "lm_head": (1, hidden, lm_head_rows),
    }
    for seed, (variant, (m, k, n)) in enumerate(matmul_cases.items()):
        rng = np.random.default_rng(4000 + seed)
        x = randn(rng, m, k)
        weight = randn(rng, n, k)
        note = " (vocab trimmed)" if variant == "lm_head" else ""
        w.array(f"ops/matmul_{variant}_x.npy", x, f"matmul input x [{m}, {k}]")
        w.array(
            f"ops/matmul_{variant}_w.npy",
            weight,
            f"matmul weight [out_features={n}, in_features={k}]{note}",
        )
        w.array(
            f"ops/matmul_{variant}_out.npy",
            torch.from_numpy(x) @ torch.from_numpy(weight).T,
            "x @ w.T",
        )

    # ---- rmsnorm ----
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    for seed, (variant, (rows, width)) in enumerate(
        (("small", (4, 8)), ("model", (8, hidden)))
    ):
        rng = np.random.default_rng(5000 + seed)
        x = randn(rng, rows, width)
        weight = rng.uniform(0.5, 1.5, size=width).astype(np.float32)
        norm = LlamaRMSNorm(width, eps=eps)
        with torch.no_grad():
            norm.weight.copy_(torch.from_numpy(weight))
            out = norm(torch.from_numpy(x))
        w.array(f"ops/rmsnorm_{variant}_x.npy", x, f"rmsnorm input [{rows}, {width}]")
        w.array(f"ops/rmsnorm_{variant}_weight.npy", weight, f"rmsnorm weight [{width}]")
        w.scalar_f32(f"ops/rmsnorm_{variant}_eps.npy", eps, "rms_norm_eps from config.json")
        w.array(f"ops/rmsnorm_{variant}_out.npy", out, "transformers LlamaRMSNorm")

    # ---- softmax ----
    for seed, (variant, shape) in enumerate(
        (("small", (4, 8)), ("model", (n_heads, 128)), ("wide", (1, vocab)))
    ):
        rng = np.random.default_rng(6000 + seed)
        x = randn(rng, *shape)
        if variant == "wide":
            x *= 8.0  # logit-scale magnitudes, where a missing max-subtract overflows
        w.array(
            f"ops/softmax_{variant}_x.npy",
            x,
            f"softmax input {list(shape)}"
            + (" (logit-scaled)" if variant == "wide" else ""),
        )
        w.array(
            f"ops/softmax_{variant}_out.npy",
            torch.softmax(torch.from_numpy(x), dim=-1),
            "torch.softmax(x, dim=-1)",
        )

    # ---- rope ----
    rope_cases = {
        "small": (3, 2, 1, 8, 0),
        "model": (8, n_heads, n_kv_heads, head_dim, 0),
        "offset": (1, n_heads, n_kv_heads, head_dim, 4000),
    }
    for seed, (variant, (n_tokens, nq, nkv, hd, start)) in enumerate(rope_cases.items()):
        rng = np.random.default_rng(7000 + seed)
        q = randn(rng, n_tokens, nq, hd)
        k = randn(rng, n_tokens, nkv, hd)
        positions = np.arange(start, start + n_tokens, dtype=np.int64)

        if hd == head_dim:
            case_rot, case_theta = rot, theta
        else:
            # A toy head_dim needs its own rotary module; build it the same way
            # transformers does so the convention is still theirs, not ours.
            case_rot, case_theta = small_rotary(cfg, hd, theta), theta

        with torch.no_grad():
            q_out, k_out = apply_rope(
                case_rot,
                to_torch_heads(q),
                to_torch_heads(k),
                torch.from_numpy(positions).unsqueeze(0),
            )

        w.array(f"ops/rope_{variant}_q.npy", q, f"q before RoPE [{n_tokens}, {nq}, {hd}]")
        w.array(f"ops/rope_{variant}_k.npy", k, f"k before RoPE [{n_tokens}, {nkv}, {hd}]")
        w.array(f"ops/rope_{variant}_positions.npy", positions, "absolute positions")
        w.scalar_f32(f"ops/rope_{variant}_theta.npy", case_theta, "rope_theta")
        w.array(
            f"ops/rope_{variant}_q_out.npy",
            from_torch_heads(q_out),
            "transformers apply_rotary_pos_emb, q",
        )
        w.array(
            f"ops/rope_{variant}_k_out.npy",
            from_torch_heads(k_out),
            "transformers apply_rotary_pos_emb, k",
        )

    # ---- attention ----
    attention_cases = {
        "small": (2, 4, 2, 8, 4, 6),
        "model_prefill": (8, n_heads, n_kv_heads, head_dim, 0, 8),
        "model_decode": (1, n_heads, n_kv_heads, head_dim, 16, 17),
    }
    for seed, (variant, (n_tokens, nq, nkv, hd, start, n_kv_pos)) in enumerate(
        attention_cases.items()
    ):
        rng = np.random.default_rng(8000 + seed)
        q = randn(rng, n_tokens, nq, hd)
        k = randn(rng, n_kv_pos, nkv, hd)
        v = randn(rng, n_kv_pos, nkv, hd)
        scale = 1.0 / float(np.sqrt(hd))
        out = eager_attention(q, k, v, start, scale)

        w.array(f"ops/attention_{variant}_q.npy", q, f"q [{n_tokens}, {nq}, {hd}], post-RoPE")
        w.array(f"ops/attention_{variant}_k.npy", k, f"k cache [{n_kv_pos}, {nkv}, {hd}]")
        w.array(f"ops/attention_{variant}_v.npy", v, f"v cache [{n_kv_pos}, {nkv}, {hd}]")
        w.scalar_i64(
            f"ops/attention_{variant}_start_pos.npy", start, "absolute position of q row 0"
        )
        w.scalar_f32(f"ops/attention_{variant}_scale.npy", scale, "1 / sqrt(head_dim)")
        w.array(
            f"ops/attention_{variant}_out.npy",
            out,
            "causal GQA attention, torch eager (scores in float32, softmax in float32)",
        )

    # ---- argmax ----
    for seed, (variant, n) in enumerate((("small", 16), ("model", vocab))):
        rng = np.random.default_rng(9000 + seed)
        x = randn(rng, n)
        w.array(f"ops/argmax_{variant}_x.npy", x, f"logits [{n}]")
        w.scalar_i64(
            f"ops/argmax_{variant}_out.npy",
            int(torch.argmax(torch.from_numpy(x)).item()),
            "torch.argmax(x)",
        )


def small_rotary(cfg: Any, head_dim: int, theta: float) -> Any:
    """A LlamaRotaryEmbedding for a toy head_dim, built the version-correct way."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    params = inspect.signature(LlamaRotaryEmbedding.__init__).parameters
    if "config" in params:
        import copy

        toy = copy.deepcopy(cfg)
        toy.head_dim = head_dim
        toy.hidden_size = head_dim * cfg.num_attention_heads
        if isinstance(getattr(toy, "rope_parameters", None), dict):
            toy.rope_parameters = dict(toy.rope_parameters)
            toy.rope_parameters["rope_theta"] = theta
        else:
            toy.rope_theta = theta
        return LlamaRotaryEmbedding(config=toy)
    return LlamaRotaryEmbedding(
        head_dim, max_position_embeddings=cfg.max_position_embeddings, base=theta
    )


def eager_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, start_pos: int, scale: float
) -> np.ndarray:
    """Causal GQA attention in torch, matching attn_implementation="eager".

    Inputs and output use minillm's [n_tokens, n_heads, head_dim] layout.
    """
    n_tokens, n_q_heads, head_dim = q.shape
    n_kv_pos, n_kv_heads, _ = k.shape
    group = n_q_heads // n_kv_heads

    qt = to_torch_heads(q)  # [1, n_q_heads, n_tokens, head_dim]
    kt = to_torch_heads(k)  # [1, n_kv_heads, n_kv_pos, head_dim]
    vt = to_torch_heads(v)

    # repeat_kv: each KV head serves `group` consecutive query heads.
    kt = kt.repeat_interleave(group, dim=1)
    vt = vt.repeat_interleave(group, dim=1)

    scores = torch.matmul(qt, kt.transpose(2, 3)) * scale

    # Query row t is at absolute position start_pos + t and sees keys
    # [0, start_pos + t]. Everything else is -inf before the softmax.
    rows = torch.arange(n_tokens).unsqueeze(1) + start_pos
    cols = torch.arange(n_kv_pos).unsqueeze(0)
    mask = cols > rows
    scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(probs, vt)  # [1, n_q_heads, n_tokens, head_dim]
    return from_torch_heads(out)


# --------------------------------------------------------------------------
# tokenizer/
# --------------------------------------------------------------------------


def gen_tokenizer(w: Writer, tokenizer: Any) -> None:
    print("tokenizer/")
    entries = []
    for name, text in TOKENIZER_CORPUS:
        without = tokenizer.encode(text, add_special_tokens=False)
        with_special = tokenizer.encode(text, add_special_tokens=True)
        entries.append(
            {
                "name": name,
                "text": text,
                "ids": without,
                "ids_with_special_tokens": with_special,
                "decoded": tokenizer.decode(without),
                "decoded_skip_special": tokenizer.decode(with_special, skip_special_tokens=True),
                "pieces": [tokenizer.decode([i]) for i in without],
            }
        )

    specials = {
        "vocab_size": int(tokenizer.vocab_size),
        "len_tokenizer": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "all_special_ids": sorted(int(i) for i in tokenizer.all_special_ids),
        "added_tokens": {
            str(idx): tok.content if hasattr(tok, "content") else str(tok)
            for idx, tok in sorted(tokenizer.added_tokens_decoder.items())
        },
    }

    w.json(
        "tokenizer/corpus.json",
        {"specials": specials, "cases": entries},
        "edge-case corpus with HF fast-tokenizer ids, with and without special "
        "tokens, plus per-token pieces and round-trip text",
    )
    print(f"  corpus.json ({len(entries)} cases)")


# --------------------------------------------------------------------------
# weights/
# --------------------------------------------------------------------------


def gen_weights(w: Writer, model_dir: Path, cfg: Any) -> None:
    print("weights/")
    from safetensors import safe_open

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no .safetensors in {model_dir}")

    middle = cfg.num_hidden_layers // 2
    last = cfg.num_hidden_layers - 1
    # One tensor per layer type, sampled at the first, a middle and the last
    # layer. Only metadata and a 16-value prefix are written: never full weights.
    wanted = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in (0, middle, last):
        for suffix in (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            wanted.append(f"model.layers.{layer}.{suffix}")

    samples: list[dict[str, Any]] = []
    present: set[str] = set()
    for shard in shards:
        with safe_open(shard, framework="pt") as fh:
            present.update(fh.keys())
            for name in wanted:
                if name not in fh.keys():
                    continue
                tensor = fh.get_tensor(name)
                flat = tensor.reshape(-1).to(torch.float32)
                samples.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "stored_dtype": str(tensor.dtype).replace("torch.", ""),
                        "first16": [float(x) for x in flat[:16]],
                        "sum_f32": float(flat.sum().item()),
                        "shard": shard.name,
                    }
                )

    missing = [name for name in wanted if name not in present]
    w.json(
        "weights/samples.json",
        {
            "note": (
                "Spot checks for the loader: compare shape, stored dtype, the "
                "first 16 values after conversion to fp32, and the fp32 sum of "
                "all elements. first16 is in row-major order from element 0."
            ),
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
            "absent": missing,
            "tensor_count": len(present),
            "all_names": sorted(present),
            "samples": samples,
        },
        "per-layer-type weight samples: shape, stored dtype, first 16 values, fp32 sum",
    )
    print(f"  samples.json ({len(samples)} tensors, {len(present)} in checkpoint)")
    if missing:
        print(f"  absent from checkpoint (expected when tied): {', '.join(missing)}")


# --------------------------------------------------------------------------
# attention/: two real layers, stage by stage
# --------------------------------------------------------------------------


def gen_attention(w: Writer, model: Any, cfg: Any, tokenizer: Any, rot: Any) -> None:
    print("attention/")
    head_dim = head_dim_of(cfg)
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads

    text = "The capital of France is Paris, and the capital of Italy is"
    ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long)
    n = len(ids)
    positions = np.arange(n, dtype=np.int64)

    w.array("attention/input_ids.npy", np.asarray(ids, dtype=np.int64), f"prompt ids for {text!r}")
    w.array("attention/positions.npy", positions, "absolute positions 0..n-1")

    layers_of_interest = {0: "l0", cfg.num_hidden_layers // 2: f"l{cfg.num_hidden_layers // 2}"}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def grab_output(key: str):
        def hook(_module, _inputs, output):
            captured[key] = output.detach()

        return hook

    def grab_input(key: str):
        def hook(_module, inputs, _output):
            captured[key] = inputs[0].detach()

        return hook

    for idx, tag in layers_of_interest.items():
        layer = model.model.layers[idx]
        handles.append(layer.register_forward_hook(grab_input(f"{tag}_hidden_in")))
        handles.append(layer.input_layernorm.register_forward_hook(grab_output(f"{tag}_normed")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(grab_output(f"{tag}_q_proj")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(grab_output(f"{tag}_k_proj")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(grab_output(f"{tag}_v_proj")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(grab_input(f"{tag}_attn_out")))
        handles.append(layer.self_attn.o_proj.register_forward_hook(grab_output(f"{tag}_o_proj")))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    for tag in layers_of_interest.values():
        hidden_in = captured[f"{tag}_hidden_in"].squeeze(0)
        normed = captured[f"{tag}_normed"].squeeze(0)
        q_proj = captured[f"{tag}_q_proj"].squeeze(0)
        k_proj = captured[f"{tag}_k_proj"].squeeze(0)
        v_proj = captured[f"{tag}_v_proj"].squeeze(0)
        attn_out = captured[f"{tag}_attn_out"].squeeze(0)
        o_proj = captured[f"{tag}_o_proj"].squeeze(0)

        w.array(f"attention/{tag}_hidden_in.npy", hidden_in, f"{tag}: input to the decoder layer")
        w.array(f"attention/{tag}_normed.npy", normed, f"{tag}: input_layernorm output")
        w.array(f"attention/{tag}_q_proj.npy", q_proj, f"{tag}: q_proj output [n, n_heads*head_dim]")
        w.array(f"attention/{tag}_k_proj.npy", k_proj, f"{tag}: k_proj output [n, n_kv*head_dim]")
        w.array(f"attention/{tag}_v_proj.npy", v_proj, f"{tag}: v_proj output [n, n_kv*head_dim]")

        # Post-RoPE q and k, through the installed transformers' own RoPE code.
        q_heads = q_proj.reshape(n, n_heads, head_dim).numpy()
        k_heads = k_proj.reshape(n, n_kv_heads, head_dim).numpy()
        with torch.no_grad():
            q_rope, k_rope = apply_rope(
                rot,
                to_torch_heads(q_heads),
                to_torch_heads(k_heads),
                torch.from_numpy(positions).unsqueeze(0),
            )
        w.array(
            f"attention/{tag}_q_rope.npy",
            from_torch_heads(q_rope),
            f"{tag}: q after RoPE [n, n_heads, head_dim]",
        )
        w.array(
            f"attention/{tag}_k_rope.npy",
            from_torch_heads(k_rope),
            f"{tag}: k after RoPE [n, n_kv_heads, head_dim]",
        )
        w.array(
            f"attention/{tag}_attn_out.npy",
            attn_out.reshape(n, n_heads, head_dim),
            f"{tag}: attention output before o_proj [n, n_heads, head_dim]",
        )
        w.array(f"attention/{tag}_o_proj.npy", o_proj, f"{tag}: o_proj output [n, hidden]")
    print(f"  {len(layers_of_interest)} layers x 9 tensors, n_tokens={n}")


# --------------------------------------------------------------------------
# model/: whole forward pass + greedy generation
# --------------------------------------------------------------------------


def gen_model(
    w: Writer,
    model: Any,
    cfg: Any,
    tokenizer: Any,
    section: str = "model",
    weights_note: str = "",
    n_generate: int = 64,
) -> None:
    """Writes <section>/: every hidden state, the logits and greedy ids per prompt.

    `section` "model" is the checkpoint as is; gen_model_q8_0 calls this again
    with "model_q8_0" after rounding the weights, so both sections are written
    by exactly the same code.
    """
    print(f"{section}/")
    prompt_records = []

    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states
        assert len(hidden_states) == cfg.num_hidden_layers + 1, (
            f"expected {cfg.num_hidden_layers + 1} hidden states, got {len(hidden_states)}"
        )

        w.array(f"{section}/p{i}_input_ids.npy", np.asarray(ids, dtype=np.int64), f"prompt {i} ids")
        for j, h in enumerate(hidden_states):
            if j < cfg.num_hidden_layers:
                what = f"input to decoder layer {j}" + (" (embedding output)" if j == 0 else "")
            else:
                what = "after the final norm"
            w.array(f"{section}/p{i}_hidden_{j}.npy", h.squeeze(0), f"prompt {i}: {what}{weights_note}")
        w.array(
            f"{section}/p{i}_logits.npy",
            out.logits.squeeze(0),
            f"prompt {i}: logits [n, vocab]{weights_note}",
        )

        # Greedy continuation, one token at a time with the KV cache, exactly
        # what the engine does. argmax over the last position's logits.
        generated: list[int] = []
        with torch.no_grad():
            step = model(input_ids, use_cache=True)
            past = step.past_key_values
            next_id = int(torch.argmax(step.logits[0, -1]).item())
            generated.append(next_id)
            for _ in range(n_generate - 1):
                step = model(
                    torch.tensor([[next_id]], dtype=torch.long),
                    past_key_values=past,
                    use_cache=True,
                )
                past = step.past_key_values
                next_id = int(torch.argmax(step.logits[0, -1]).item())
                generated.append(next_id)

        w.array(
            f"{section}/p{i}_greedy_ids.npy",
            np.asarray(generated, dtype=np.int64),
            f"prompt {i}: {n_generate} greedy tokens (argmax, KV cache, no sampling){weights_note}",
        )
        prompt_records.append(
            {
                "index": i,
                "prompt": prompt,
                "ids": ids,
                "n_tokens": len(ids),
                "n_hidden_states": len(hidden_states),
                "greedy_ids": generated,
                "greedy_text": tokenizer.decode(generated),
            }
        )
        print(f"  prompt {i}: {len(ids)} tokens, {len(hidden_states)} hidden states")

    w.json(
        f"{section}/prompts.json",
        {
            "note": (
                "hidden_{j} indices match transformers' output_hidden_states: "
                "j in [0, num_hidden_layers) is the INPUT to decoder layer j "
                "(j = 0 is the embedding output) and j = num_hidden_layers is "
                "the post-final-norm hidden state."
                + (f" Weights:{weights_note}." if weights_note else "")
            ),
            "n_generate": n_generate,
            "prompts": prompt_records,
        },
        "prompt texts, ids and greedy continuations" + weights_note,
    )


# --------------------------------------------------------------------------
# model_q8_0/: the same, on Q8_0-rounded weights
# --------------------------------------------------------------------------

Q8_0_BLOCK = 32

# One block as include/minillm/tensor.h lays out BlockQ8_0: the f16 scale's
# bits, little-endian, then 32 int8 values. 34 bytes, no padding.
BLOCK_Q8_0 = np.dtype([("d", "<u2"), ("qs", "i1", (Q8_0_BLOCK,))])
assert BLOCK_Q8_0.itemsize == 34


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """src/loader/quantize.cpp's quantize_q8_0, byte for byte.

    loader.h specifies the quantizer as five float32 roundings per block of 32
    values x, and this repeats each one as a single float32 numpy operation
    (numpy rounds every float32 op exactly, as IEEE-754 asks):

      1. amax / 127          float32 division
      2. to f16              astype(float16): nearest, ties to even
      3. 1 / d               float32 division (f16 -> f32 is exact)
      4. x * (1 / d)         float32 multiply
      5. round               halves away from zero, built from trunc

    Step 5 is where float32 code usually goes wrong: numpy's round is half to
    even, and floor(|v| + 0.5) rounds 0.49999997 up, because the addition
    itself rounds to 1.0. trunc(v) and v - trunc(v) are both exact, so the
    comparison with 0.5 sees the true fraction.

    A block whose scale rounds to 0 in f16 is all zeros; a non-finite value or
    a scale that overflows f16 raises, as in C++. Returns BLOCK_Q8_0 records,
    one per block, in row-major order.
    """
    if x.dtype != np.float32 or x.ndim == 0 or x.shape[-1] % Q8_0_BLOCK != 0:
        raise ValueError(f"quantize_q8_0: cannot quantize {x.dtype}{list(x.shape)}")
    b = np.ascontiguousarray(x).reshape(-1, Q8_0_BLOCK)
    if not np.isfinite(b).all():
        raise ValueError("quantize_q8_0: only finite values can be quantized")
    amax = np.abs(b).max(axis=1)
    with np.errstate(over="ignore"):
        d16 = (amax / np.float32(127.0)).astype(np.float16)
    if np.isinf(d16).any():
        raise ValueError("quantize_q8_0: amax / 127 overflows the f16 scale")
    d = d16.astype(np.float32)
    zero = d == 0
    inv = np.float32(1.0) / np.where(zero, np.float32(1.0), d)
    inv[zero] = 0  # every value of the block quantizes to 0
    v = b * inv[:, None]
    t = np.trunc(v)
    q = t + np.sign(v) * (np.abs(v - t) >= np.float32(0.5)).astype(np.float32)
    q = np.clip(q, np.float32(-127.0), np.float32(127.0))

    blocks = np.empty(b.shape[0], dtype=BLOCK_Q8_0)
    blocks["d"] = d16.view(np.uint16)
    blocks["qs"] = q.astype(np.int8)
    return blocks


def dequantize_q8_0(blocks: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """convert_to_f32 of a Q8_0 tensor: q * d in float32, which is exact."""
    d = blocks["d"].view(np.float16).astype(np.float32)
    return (blocks["qs"].astype(np.float32) * d[:, None]).reshape(shape)


def q8_0_weight_names(cfg: Any) -> list[str]:
    """What quantize_weights_q8_0 (include/minillm/model.h) replaces: the seven
    projections of every layer and the embedding, plus lm_head when untied."""
    names = ["model.embed_tokens.weight"]
    for i in range(cfg.num_hidden_layers):
        names += [
            f"model.layers.{i}.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")
        ] + [f"model.layers.{i}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
    if not cfg.tie_word_embeddings:
        names.append("lm_head.weight")
    return names


def gen_model_q8_0(w: Writer, model: Any, cfg: Any, tokenizer: Any) -> dict[str, Any]:
    """Rounds the model's weights to Q8_0 IN PLACE, then writes model_q8_0/.

    Every tensor quantize_weights_q8_0 replaces becomes dequantize(quantize(W)),
    so the model is the F32 matrix a Q8_0 weight stands for (ops.h), and
    torch's forward pass on it is the golden data for the engine's Q8_0 path.
    The tied embedding is rounded once, so the lookup and lm_head both see the
    rounded table, as in the engine. Call this last: the model stays rounded.

    model_q8_0/blocks.sha256 lists the SHA-256 of each tensor's blocks, in
    sha256sum's format, so tests/model can check that the C++ quantizer writes
    these exact bytes for the whole checkpoint. Returns the summary for the
    manifest.
    """
    import hashlib

    print("model_q8_0/ (quantizing the weights in place)")
    if cfg.tie_word_embeddings and (
        model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    ):
        raise RuntimeError("tie_word_embeddings is set, but lm_head does not share embed_tokens")

    digests: dict[str, str] = {}
    everything = hashlib.sha256()
    n_blocks = 0
    with torch.no_grad():
        for name in q8_0_weight_names(cfg):
            param = model.get_parameter(name)
            x = param.detach().cpu().numpy()
            blocks = quantize_q8_0(x)
            digests[name] = hashlib.sha256(blocks.tobytes()).hexdigest()
            n_blocks += blocks.shape[0]
            param.copy_(torch.from_numpy(dequantize_q8_0(blocks, x.shape)))
    for name in sorted(digests):
        everything.update(bytes.fromhex(digests[name]))

    w.text(
        "model_q8_0/blocks.sha256",
        [f"{digests[name]}  {name}" for name in sorted(digests)],
        "sha256 of each Q8_0 tensor's blocks (34 bytes each: f16 scale little-endian, "
        "then 32 int8), as quantize_q8_0 writes them, in sha256sum format",
    )
    print(f"  {len(digests)} tensors, {n_blocks} blocks")

    gen_model(
        w,
        model,
        cfg,
        tokenizer,
        section="model_q8_0",
        weights_note=" (Q8_0 weights: dequantize(quantize_q8_0(W)) for every tensor in blocks.sha256)",
    )
    return {
        "tensors": len(digests),
        "blocks": n_blocks,
        "sha256_of_tensor_digests": everything.hexdigest(),
        "note": (
            "model_q8_0/ is model/ with dequantize(quantize_q8_0(W)) for the tensors in "
            "model_q8_0/blocks.sha256; sha256_of_tensor_digests hashes their digests "
            "(raw 32-byte values) in sorted name order"
        ),
    }


# --------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "oracle" / "data")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use an already-downloaded checkpoint and never touch the network",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_dir = args.models_dir / args.model_id.split("/")[-1]
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        print(f"downloading {args.model_id} -> {model_dir}")
        snapshot_download(
            repo_id=args.model_id,
            local_dir=str(model_dir),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    if not (model_dir / "config.json").is_file():
        print(f"error: {model_dir / 'config.json'} not found", file=sys.stderr)
        return 1

    print(f"loading {model_dir} in float32, attn_implementation='eager'")
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    head_dim = head_dim_of(cfg)
    print(
        f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
        f"head_dim={head_dim} inter={cfg.intermediate_size} vocab={cfg.vocab_size}"
    )
    print(
        f"  rope_theta={rope_theta_of(cfg)} rope_type={rope_type_of(cfg)} "
        f"rms_norm_eps={cfg.rms_norm_eps} tie_word_embeddings={cfg.tie_word_embeddings}"
    )
    if rope_type_of(cfg) != "default":
        print(
            f"error: rope_type={rope_type_of(cfg)!r} is not implemented by minillm",
            file=sys.stderr,
        )
        return 1

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    w = Writer(args.out_dir)
    rot = rotary_module(model)

    gen_ops(w, model, cfg, rot)
    gen_tokenizer(w, tokenizer)
    gen_weights(w, model_dir, cfg)
    gen_attention(w, model, cfg, tokenizer, rot)
    gen_model(w, model, cfg, tokenizer)
    # Last: it rounds the model's weights in place.
    q8_0 = gen_model_q8_0(w, model, cfg, tokenizer)

    w.finish(
        {
            "model_id": args.model_id,
            # Absolute when --models-dir is outside the repo (a worktree
            # reading the main checkout's models/).
            "model_dir": str(
                model_dir.resolve().relative_to(REPO_ROOT)
                if model_dir.resolve().is_relative_to(REPO_ROOT)
                else model_dir.resolve()
            ),
            "generated_with": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "config": {
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "head_dim": head_dim,
                "vocab_size": cfg.vocab_size,
                "max_position_embeddings": cfg.max_position_embeddings,
                "rms_norm_eps": cfg.rms_norm_eps,
                "rope_theta": rope_theta_of(cfg),
                "rope_type": rope_type_of(cfg),
                "tie_word_embeddings": bool(cfg.tie_word_embeddings),
            },
            "conventions": {
                "layout": "row-major",
                "linear_weights": "[out_features, in_features]; y = x @ w.T",
                "qk_layout": "[n_tokens, n_heads, head_dim]",
                "rope": "Hugging Face rotate_half: dimension i pairs with i + head_dim/2",
                "gqa": "query head h uses kv head h // (n_heads // n_kv_heads)",
                "compute_dtype": "float32",
            },
            "q8_0": q8_0,
        }
    )
    print(f"done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
