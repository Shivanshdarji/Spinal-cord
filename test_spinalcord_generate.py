import os
import sys
import time
import torch

# Allow importing from train/ (this repo uses relative imports in many scripts)
sys.path.append(os.path.join(os.path.dirname(__file__), "train"))

from config import SpinalCordConfig, BrainConfig  # type: ignore
from model import SpinalCordLLM, SpinalCordDraft, _sample_from_logits  # type: ignore
from tokenizer_sc import load_tokenizer_and_export  # type: ignore


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SpinalCordConfig()

    # Tokenizer must match training
    bundle, _ = load_tokenizer_and_export(expected_vocab_size=cfg.brain.vocab_size)
    encode, decode = bundle.encode, bundle.decode

    brain_ckpt = os.path.join("models", "scbrain_best.pt")
    draft_ckpt = os.path.join("models", "scdraft_best.pt")

    brain_state = torch.load(brain_ckpt, map_location=device, weights_only=False)
    draft_state = torch.load(draft_ckpt, map_location=device, weights_only=False)

    brain_cfg: BrainConfig = brain_state["cfg"]
    draft_cfg = draft_state["cfg"]

    model = SpinalCordLLM(draft_cfg, brain_cfg).to(device)
    # Load actual trained weights (previously this test instantiated models only).
    model.brain.load_state_dict(brain_state["model_state"])
    model.draft.load_state_dict(draft_state["model_state"])
    model.eval()

    draft = model.draft.eval()

    rag_dir = os.environ.get("SPINALCORD_RAG_DIR")
    rag_chunks = None
    if rag_dir:
        from rag_simple import load_corpus_chunks, retrieve_top_k, build_rag_prompt  # type: ignore

        print(f"[RAG] Loading corpus chunks from: {rag_dir}")
        rag_chunks = load_corpus_chunks(rag_dir, chunk_words=120, overlap_words=20)

    prompts = [
        ("Q&A", "What is photosynthesis? Answer in one short paragraph."),
        ("Instruction", "Explain recursion like I am 10 years old."),
        ("Reasoning", "If A then B, and A. What can you conclude?"),
        ("Coding", "Write a simple C program that prints Hello, world! Include a main() function. Output only the C code."),
    ]

    # Sampling knobs: keep stable, but not purely greedy.
    temperature = 0.9
    top_k = 50
    top_p = 0.95

    # Optional: run `generate()` once on the coding prompt to ensure speculative decode still works.
    coding_prompt = prompts[-1][1]
    coding_ids = torch.tensor([encode(coding_prompt)], dtype=torch.long, device=device)
    t0 = time.time()
    out = model.generate(
        coding_ids,
        max_new_tokens=60,
        brain_device=device,
        draft_device=device,
        verbose=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    dt = time.time() - t0
    gen_ids = out[0].tolist()
    text = decode(gen_ids[len(coding_ids[0]) :])
    print("\n=== SpinalCord generate() (speculative only) ===")
    print(text[:500])
    print("repr:", repr(text[:200]))
    print(f"(time: {dt:.1f}s)")

    # Main evaluation: reflex routing on multiple prompt types.
    for label, prompt in prompts:
        eff_prompt = prompt
        if rag_chunks:
            retrieved = retrieve_top_k(prompt, rag_chunks, k=3)
            retrieved_texts = [c for (c, _score) in retrieved]
            eff_prompt = build_rag_prompt(prompt, retrieved_texts)

        input_ids = torch.tensor([encode(eff_prompt)], dtype=torch.long, device=device)
        print(f"\n=== SpinalCord generate_reflex() [{label}] ===")
        t1 = time.time()
        # Reflex defaults favor sustained Draft+Brain speed: not too strict on one round,
        # and recover speculative after a short Brain-only burst.
        out2 = model.generate_reflex(
            input_ids,
            max_new_tokens=60,
            brain_device=device,
            draft_device=device,
            accept_rate_threshold=0.65,
            consecutive_bad_rounds_to_fallback=2,
            recover_speculative_after_brain_tokens=24,
            rolling_accept_window=4,
            verbose=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        dt2 = time.time() - t1

        gen_ids2 = out2[0].tolist()
        text2 = decode(gen_ids2[len(input_ids[0]) :])
        print(text2[:500])
        print("repr:", repr(text2[:200]))
        print(f"(time: {dt2:.1f}s)")


if __name__ == "__main__":
    main()

