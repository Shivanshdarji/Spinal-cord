"""
SpinalCord LLM — PyTorch to GGUF Converter
============================================
Converts our trained .pt checkpoint to .gguf format
so llama.cpp can run it in the C++ engine.

Workflow:
  1. Load SpinalCordDraft checkpoint (.pt)
  2. Export to HuggingFace format (.safetensors)
  3. Run llama.cpp's convert_hf_to_gguf.py
  4. Quantize to Q4_K_M for 4GB VRAM

Run:
  python convert_to_gguf.py --checkpoint ../checkpoints/spinalcord_draft_best.pt

Author: Shivansh Darji | AppDice
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# Add train/ directory to path so config.py can be found
sys.path.append(str(Path(__file__).parent.parent / "train"))

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Convert SpinalCord .pt to .gguf")
    p.add_argument("--checkpoint",  type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--llama_cpp",   type=str, default="../../llama.cpp", help="llama.cpp directory")
    p.add_argument("--output_dir",  type=str, default="../models", help="Output directory for GGUF")
    p.add_argument("--quantize",    type=str, default="Q4_K_M", 
                   choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                   help="Quantization level")
    return p.parse_args()


def export_to_hf_format(checkpoint_path: Path, export_dir: Path, cfg) -> Path:
    """
    Export the PyTorch weights to HuggingFace format.
    This creates config.json + model.safetensors that llama.cpp can convert.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Convert] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    model_cfg  = ckpt.get("config", cfg)
    
    # ── Write config.json (LLaMA-compatible format) ──────────────────────────
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": model_cfg.d_model,
        "intermediate_size": model_cfg.d_ff,
        "max_position_embeddings": model_cfg.max_seq_len,
        "num_attention_heads": model_cfg.n_heads,
        "num_hidden_layers": model_cfg.n_layers,
        "num_key_value_heads": model_cfg.n_kv_heads,
        "vocab_size": model_cfg.vocab_size,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "model_name": "SpinalCord-Draft",
        "author": "Shivansh Darji / AppDice",
    }
    
    with open(export_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"[Convert] ✅ Wrote config.json")
    
    # ── Remap weight keys to LLaMA HF format ─────────────────────────────────
    # Our keys: layers.0.attn.q_proj.weight
    # HF keys:  model.layers.0.self_attn.q_proj.weight
    remapped = {}
    
    key_map = {
        "embed":              "model.embed_tokens",
        "norm":               "model.norm",
        "lm_head":            "lm_head",
        "layers.{i}.norm1":   "model.layers.{i}.input_layernorm",
        "layers.{i}.norm2":   "model.layers.{i}.post_attention_layernorm",
        "layers.{i}.attn.q_proj":  "model.layers.{i}.self_attn.q_proj",
        "layers.{i}.attn.k_proj":  "model.layers.{i}.self_attn.k_proj",
        "layers.{i}.attn.v_proj":  "model.layers.{i}.self_attn.v_proj",
        "layers.{i}.attn.out_proj":"model.layers.{i}.self_attn.o_proj",
        "layers.{i}.ffn.gate":    "model.layers.{i}.mlp.gate_proj",
        "layers.{i}.ffn.up":      "model.layers.{i}.mlp.up_proj",
        "layers.{i}.ffn.down":    "model.layers.{i}.mlp.down_proj",
    }
    
    for key, tensor in state_dict.items():
        if "inv_freq" in key or "cos_cached" in key or "sin_cached" in key:
            continue
            
        new_key = key
        
        # Detect layer index
        if key.startswith("layers."):
            parts = key.split(".")
            layer_i = parts[1]
            
            for old_pattern, new_pattern in key_map.items():
                if "{i}" in old_pattern:
                    old_filled = old_pattern.replace("{i}", layer_i)
                    new_filled = new_pattern.replace("{i}", layer_i)
                    if key.startswith(old_filled):
                        suffix = key[len(old_filled):]
                        new_key = new_filled + suffix
                        break
        elif key.startswith("embed."):
            new_key = key.replace("embed.", "model.embed_tokens.", 1)
        elif key.startswith("norm."):
            new_key = key.replace("norm.", "model.norm.", 1)
        
        # Clone to avoid safetensors shared memory error
        if new_key == "lm_head.weight":
            remapped[new_key] = tensor.clone()
        else:
            remapped[new_key] = tensor
        
    # ── Save as safetensors (preferred) or .bin ───────────────────────────────
    try:
        from safetensors.torch import save_file
        save_file(remapped, export_dir / "model.safetensors")
        print(f"[Convert] ✅ Saved model.safetensors")
    except ImportError:
        print("[Convert] safetensors not installed, saving as pytorch_model.bin")
        print("[Convert] Tip: pip install safetensors")
        torch.save(remapped, export_dir / "pytorch_model.bin")
    
    return export_dir


def convert_to_gguf(hf_dir: Path, llama_cpp_dir: Path, output_path: Path) -> Path:
    """
    Run llama.cpp's convert_hf_to_gguf.py script.
    """
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at: {convert_script}\n"
            f"Make sure llama.cpp is cloned at: {llama_cpp_dir}"
        )
    
    gguf_fp16 = output_path / "spinalcord_draft_f16.gguf"
    
    print(f"\n[Convert] Running llama.cpp converter...")
    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_dir),
        "--outfile", str(gguf_fp16),
        "--outtype", "f16",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Error] Conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed")
    
    print(f"[Convert] ✅ GGUF F16 saved: {gguf_fp16}")
    return gguf_fp16


def quantize_gguf(gguf_f16: Path, llama_cpp_dir: Path, quantize: str) -> Path:
    """
    Use llama.cpp's quantize tool to compress the model.
    Q4_K_M = ~2.5GB (fits in 4GB VRAM with room for context)
    """
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_dir / "build" / "Release" / "llama-quantize.exe"
    
    if not quantize_bin.exists():
        print(f"[Warning] llama-quantize not found, skipping quantization.")
        print(f"          Build llama.cpp first, then run quantization manually:")
        print(f"          llama-quantize {gguf_f16} output.gguf {quantize}")
        return gguf_f16
    
    gguf_quantized = gguf_f16.parent / f"spinalcord_draft_{quantize.lower()}.gguf"
    
    print(f"\n[Convert] Quantizing to {quantize}...")
    cmd = [str(quantize_bin), str(gguf_f16), str(gguf_quantized), quantize]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Warning] Quantization failed:\n{result.stderr}")
        return gguf_f16
    
    size_mb = gguf_quantized.stat().st_size / 1e6
    print(f"[Convert] ✅ Quantized saved: {gguf_quantized} ({size_mb:.0f} MB)")
    return gguf_quantized


def main():
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    llama_cpp_dir   = Path(args.llama_cpp)
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════╗")
    print("║   SpinalCord → GGUF Converter | AppDice         ║")
    print("╚══════════════════════════════════════════════════╝\n")
    
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        print("Train the model first: python train.py")
        return
    
    # Step 1: Export to HF format
    hf_export_dir = output_dir / "hf_export"
    export_to_hf_format(checkpoint_path, hf_export_dir, cfg=None)
    
    # Step 2: Convert to GGUF
    try:
        gguf_f16 = convert_to_gguf(hf_export_dir, llama_cpp_dir, output_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[Info] {e}")
        print("\nTo convert manually once llama.cpp is set up:")
        print(f"  python {llama_cpp_dir}/convert_hf_to_gguf.py {hf_export_dir} "
              f"--outfile {output_dir}/spinalcord_draft.gguf --outtype f16")
        return
    
    # Step 3: Quantize
    final_gguf = quantize_gguf(gguf_f16, llama_cpp_dir, args.quantize)
    
    print(f"\n✨ Done! Your SpinalCord model is ready:")
    print(f"   {final_gguf}")
    print(f"\nNow run it with:")
    print(f"   ./spinalcord {final_gguf} \"The spinal cord reflex\"")


if __name__ == "__main__":
    main()
