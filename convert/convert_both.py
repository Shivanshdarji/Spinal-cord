"""
SpinalCord LLM — Convert Both Models to GGUF
=============================================
One script to compile SCBrain-1B and SCDraft-120M into GGUF format,
ready to run with llama-server speculative decoding.

Run:
    python convert/convert_both.py

Output:
    models/scbrain_1b.gguf     ← Brain (32-bit float, then quantize)
    models/scdraft_120m.gguf   ← Draft (16-bit float)

Author: Shivansh Darji | AppDice
"""

import os
import sys
import struct
import json
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "train"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "llama.cpp"))

from config import BrainConfig, DraftConfig
from model import SpinalCordBrain, SpinalCordDraft


def _load_cfg_from_ckpt(ckpt_path: str, fallback_cfg):
    if not os.path.exists(ckpt_path):
        return fallback_cfg
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("cfg", ckpt.get("config", fallback_cfg))


def convert_checkpoint_to_gguf(
    ckpt_path: str,
    out_path: str,
    model_instance,
    cfg,
    model_name: str
):
    """
    Uses llama.cpp's convert_hf_to_gguf.py approach adapted for our architecture.
    Since our model is LLaMA-compatible, we export a HF-safe safetensors 
    and then call the llama.cpp converter.
    """
    import tempfile
    import subprocess

    print(f"\n[Convert] {model_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output:     {out_path}")

    # 1. Load checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_instance.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint (step={ckpt.get('step', '?')}, loss={ckpt.get('loss', '?'):.4f})")
    else:
        print(f"  WARNING: No checkpoint at {ckpt_path}, using random weights (demo)")

    model_instance.eval()

    # 2. Export to temporary HF-format directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write safetensors
        state = model_instance.state_dict()

        # Remap names to LLaMA-compatible keys
        remapped = {}
        KEY_MAP = {
            "embed.weight":        "model.embed_tokens.weight",
            "norm.weight":         "model.norm.weight",
            "lm_head.weight":      "lm_head.weight",
        }
        LAYER_MAP = {
            "attn.norm1",          # we don't have this, it's in TransformerBlock
        }

        for k, v in state.items():
            # Skip rotary cache buffers
            if any(skip in k for skip in ["cos_cached", "sin_cached", "inv_freq"]):
                continue

            new_k = k
            # Global renames
            for old, new in KEY_MAP.items():
                if k == old:
                    new_k = new
                    break

            # Layer renames: layers.N.* → model.layers.N.*
            if k.startswith("layers."):
                parts     = k.split(".")
                layer_idx = parts[1]
                rest      = ".".join(parts[2:])

                # Attention
                if rest == "attn.q_proj.weight": rest = "self_attn.q_proj.weight"
                elif rest == "attn.k_proj.weight": rest = "self_attn.k_proj.weight"
                elif rest == "attn.v_proj.weight": rest = "self_attn.v_proj.weight"
                elif rest == "attn.out_proj.weight": rest = "self_attn.o_proj.weight"
                # FFN
                elif rest == "ffn.gate.weight": rest = "mlp.gate_proj.weight"
                elif rest == "ffn.up.weight":   rest = "mlp.up_proj.weight"
                elif rest == "ffn.down.weight": rest = "mlp.down_proj.weight"
                # Norms
                elif rest == "norm1.weight": rest = "input_layernorm.weight"
                elif rest == "norm2.weight": rest = "post_attention_layernorm.weight"

                new_k = f"model.layers.{layer_idx}.{rest}"

            # Force f16 for storage
            if v.dtype in [torch.float32, torch.bfloat16]:
                v = v.to(torch.float16)

            # Clone lm_head to avoid shared tensor issues
            if new_k == "lm_head.weight":
                v = v.clone()

            remapped[new_k] = v

        # Write safetensors
        try:
            from safetensors.torch import save_file
            st_path = os.path.join(tmpdir, "model.safetensors")
            save_file(remapped, st_path)
        except ImportError:
            print("  safetensors not installed, falling back to pytorch save")
            pt_path = os.path.join(tmpdir, "pytorch_model.bin")
            torch.save(remapped, pt_path)

        # Write config.json (LLaMA-style)
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type":    "llama",
            "vocab_size":    cfg.vocab_size,
            "hidden_size":   cfg.d_model,
            "intermediate_size": cfg.d_ff,
            "num_hidden_layers": cfg.n_layers,
            "num_attention_heads": cfg.n_heads,
            "num_key_value_heads": cfg.n_kv_heads,
            "max_position_embeddings": cfg.max_seq_len,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "torch_dtype": "float16",
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)

        # Copy tokenizer files (avoid mixing tokenizers)
        # IMPORTANT: For our custom BPE tokenizer, `tokenizer.json` is the source of truth.
        # If a stray `tokenizer.model` (SentencePiece) is present, llama.cpp may pick it up
        # and decoding will produce gibberish. So we copy an allowlist only.
        tok_src = os.path.join(PROJECT_ROOT, "hf_export")
        if os.path.exists(tok_src):
            import shutil
            allow = {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
                "SPINALCORD_TOKENIZER.txt",
            }
            for fname in allow:
                src = os.path.join(tok_src, fname)
                dst = os.path.join(tmpdir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            print(f"  Copied tokenizer files from {tok_src}")

        # Call llama.cpp converter
        llama_converter = os.path.join(PROJECT_ROOT, "..", "llama.cpp", "convert_hf_to_gguf.py")
        if not os.path.exists(llama_converter):
            llama_converter = os.path.join(PROJECT_ROOT, "..", "llama.cpp", "convert-hf-to-gguf.py")

        if os.path.exists(llama_converter):
            cmd = [
                sys.executable, llama_converter,
                tmpdir,
                "--outtype", "f16",
                "--outfile", out_path,
            ]
            print(f"  Running converter: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                size_mb = os.path.getsize(out_path) / 1024 / 1024
                print(f"  Converted! GGUF size: {size_mb:.1f} MB")
            else:
                # Avoid Windows console unicode issues; print safely.
                err = (result.stderr or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace")
                print("  Converter failed. stderr:")
                print(err)
        else:
            print(f"  ❌ llama.cpp converter not found at: {llama_converter}")
            print(f"     Expected path: C:\\Users\\SHIVANSH\\Desktop\\llama.cpp\\convert_hf_to_gguf.py")


def main():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  SpinalCord LLM — Dual GGUF Conversion")
    print("  AppDice | Shivansh Darji")
    print("=" * 60)

    # ── Convert Brain ─────────────────────────────────────────────
    brain_ckpt = os.path.join(models_dir, "scbrain_best.pt")
    brain_cfg  = _load_cfg_from_ckpt(brain_ckpt, BrainConfig())
    brain_model = SpinalCordBrain(brain_cfg)
    convert_checkpoint_to_gguf(
        ckpt_path=brain_ckpt,
        out_path =os.path.join(models_dir, "scbrain_1b.gguf"),
        model_instance=brain_model,
        cfg=brain_cfg,
        model_name="SCBrain-1B (Brain/Verifier)"
    )

    # ── Convert Draft ─────────────────────────────────────────────
    draft_ckpt = os.path.join(models_dir, "scdraft_best.pt")
    draft_cfg   = _load_cfg_from_ckpt(draft_ckpt, DraftConfig())
    draft_model = SpinalCordDraft(draft_cfg)
    convert_checkpoint_to_gguf(
        ckpt_path=draft_ckpt,
        out_path =os.path.join(models_dir, "scdraft_120m.gguf"),
        model_instance=draft_model,
        cfg=draft_cfg,
        model_name="SCDraft-120M (Spinal Cord/Draft)"
    )

    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("  Launch your SpinalCord server with:")
    print("    dashboard\\run_dashboard.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
