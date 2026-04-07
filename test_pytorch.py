import os
import sys
import torch

# Ensure we can import train.config, train.model, etc.
sys.path.append(os.path.join(os.path.dirname(__file__), "train"))

from config import BrainConfig  # type: ignore
from model import SpinalCordBrain  # type: ignore
from tokenizer_sc import load_tokenizer_and_export  # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"

brain_ckpt = os.path.join("models", "scbrain_best.pt")
brain_state = torch.load(brain_ckpt, map_location=device, weights_only=False)
brain_cfg: BrainConfig = brain_state["cfg"]

bundle, _ = load_tokenizer_and_export(expected_vocab_size=brain_cfg.vocab_size)
encode, decode = bundle.encode, bundle.decode

brain = SpinalCordBrain(brain_cfg).to(device)
brain.load_state_dict(brain_state["model_state"])
brain.eval()

prompt = "Write a short story about a robot learning to draw."
ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

max_new_tokens = 80

with torch.no_grad():
    generated = ids.clone()
    for _ in range(max_new_tokens):
        logits = brain(generated)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)

generated_ids = generated[0].tolist()
print("\n=== PYTORCH RESPONSE (Brain only) ===\n")
print(decode(generated_ids[len(ids[0]):]))