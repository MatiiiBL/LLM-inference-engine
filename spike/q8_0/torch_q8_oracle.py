"""A PyTorch oracle for the Q8_0 path: the oracle prompts run through
transformers with the checkpoint's weights replaced by dequantize(quantize(W)).

Two modes, matching the spike's two kernels:

  dequant  weights only. The quantized model IS an fp32 model with different
           weights, so this is exactly what the engine's dequant kernel must
           reproduce, to the same tolerances as the fp32 oracle.
  w8a8     additionally fake-quantizes the input of every linear layer (and of
           lm_head) per 32-block with an fp32 scale, as the w8a8 kernel does.

The quantizer below is the spike's quantize_ref written in float32 torch ops
that are each exactly rounded (division by a tensor, not a Python scalar, so
no reciprocal-multiply shortcut; round half away from zero built from trunc so
no x + 0.5 rounding). The tied embedding is quantized once, so the lookup and
lm_head both see the quantized table ("all, 1 copy").

Run with the oracle's virtualenv; it reads the model and the oracle prompt ids
and writes only into <out_dir>:

    oracle/.venv/bin/python -I spike/q8_0/torch_q8_oracle.py \
        models/SmolLM2-135M oracle/data <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

QK = 32
N_GENERATE = 64
FLT_MIN = torch.finfo(torch.float32).tiny


def round_half_away(v: torch.Tensor) -> torch.Tensor:
    t = torch.trunc(v)
    return t + torch.sign(v) * ((v - t).abs() >= 0.5).to(v.dtype)


def quant_dequant(x: torch.Tensor, fp16_scale: bool) -> torch.Tensor:
    assert x.dtype == torch.float32 and x.shape[-1] % QK == 0
    b = x.reshape(-1, QK)
    amax = b.abs().amax(dim=1, keepdim=True)
    d = amax / torch.full_like(amax, 127.0)
    if fp16_scale:
        d = d.to(torch.float16).to(torch.float32)
    inv = torch.where(d >= FLT_MIN, torch.ones_like(d) / d, torch.zeros_like(d))
    q = round_half_away(b * inv).clamp(-127.0, 127.0)
    return (q * d).reshape(x.shape)


def load(model_dir: Path, mode: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    linears = []
    with torch.no_grad():
        for layer in model.model.layers:
            a, m = layer.self_attn, layer.mlp
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
                lin.weight.copy_(quant_dequant(lin.weight, fp16_scale=True))
                linears.append(lin)
        emb = model.model.embed_tokens.weight
        assert model.lm_head.weight.data_ptr() == emb.data_ptr(), "expected tied embeddings"
        emb.copy_(quant_dequant(emb, fp16_scale=True))
        linears.append(model.lm_head)
    if mode == "w8a8":
        for lin in linears:
            lin.register_forward_pre_hook(lambda _m, args: (quant_dequant(args[0], fp16_scale=False),))
    return model


def main() -> None:
    model_dir, oracle_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    for mode in ("dequant", "w8a8"):
        model = load(model_dir, mode)
        for i in range(3):
            ids = np.load(oracle_dir / "model" / f"p{i}_input_ids.npy").astype(np.int64).tolist()
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_ids, use_cache=False).logits.squeeze(0)
                np.save(out_dir / f"{mode}_p{i}_logits.npy", logits.numpy().astype(np.float32))
                step = model(input_ids, use_cache=True)
                past = step.past_key_values
                next_id = int(torch.argmax(step.logits[0, -1]).item())
                generated = [next_id]
                for _ in range(N_GENERATE - 1):
                    step = model(torch.tensor([[next_id]]), past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    next_id = int(torch.argmax(step.logits[0, -1]).item())
                    generated.append(next_id)
            np.save(out_dir / f"{mode}_p{i}_greedy_ids.npy", np.asarray(generated, dtype=np.int64))
            print(f"{mode} p{i}: {len(ids)} prompt ids, greedy {generated[:12]} ...")


if __name__ == "__main__":
    main()
