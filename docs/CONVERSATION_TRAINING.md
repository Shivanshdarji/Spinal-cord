# Conversation-first training (stories + dialogue + Q&A)

## Want “normal” chat *before* training converges?

**Custom** `scbrain_1b.gguf` may still read like story/gibberish — that is **checkpoint quality**, not a broken UI.

- **Stable replies today:** run **`dashboard\run_dashboard_llama_scaffold.bat`** (needs **`Llama-3.2-3B-Instruct-Q4_K_M.gguf`** + **`Llama-3.2-1B-Instruct-Q4_K_M.gguf`** in **`models\`**). Same dashboard; picks Llama, not SpinalCord weights.
- **Still on SpinalCord:** lower **Temp** (~0.1–0.2), raise **Repeat** (~1.15), cap **Max tok** — reduces loops; it will not fix a tiny story-biased model completely.

---

This project cannot promise a model is “the best in the world” at chat — that needs **human eval**, **benchmarks**, and **your product goals**. What we *can* do is **bias training data** toward:

- **Simple, clear English** (TinyStories-style narrative)
- **Natural turn-taking** (UltraChat — multi-turn chat/Q&A; parquet Hub, works with modern `datasets`)
- **Instruction following** (Alpaca — question + answer, with prompt tokens masked in the loss)

## Recommended pipeline

1. **Train Brain** with conversation mix:

```bash
cd train
python train_brain.py --data_mode conversation --steps 6000 ^
  --conv_story 0.35 --conv_dialog 0.35 --conv_inst 0.30
```

**4GB GPU / CUDA OOM:** shorten context and/or accumulation:

```bash
python train_brain.py --data_mode conversation --steps 6000 --max-seq-len 256 --grad-accum 8
```

2. **Distill Draft** from that Brain (same data mode so Draft matches the Brain’s distribution):

```bash
python distill_draft.py --data_mode conversation --brain_ckpt ../models/scbrain_best.pt
```

## Tuning the mix

Weights are **relative** (normalized to sum to 1):

| Flag | Default | Role |
|------|---------|------|
| `--conv_story` | 0.35 | Short, simple stories — readable, plain language |
| `--conv_dialog` | 0.35 | Multi-turn dialogue — “chatty” phrasing |
| `--conv_inst` | 0.30 | Instructions + answers — assistant-style |

**More dialogue:** e.g. `--conv_story 0.25 --conv_dialog 0.50 --conv_inst 0.25`  
**More Q&A:** e.g. `--conv_story 0.25 --conv_dialog 0.25 --conv_inst 0.50`

## Requirements

- `pip install datasets` (Hugging Face)
- First run downloads **TinyStories**, **HuggingFaceH4/ultrachat_200k**, and **Alpaca** from the Hub (network).

**Note:** The old `daily_dialog` Hub dataset used a deprecated script loader; recent `datasets` versions reject it. We use **UltraChat** instead.

**UltraChat splits:** use `train_sft` (default) or `train_gen` — not `train`. Override with `--conv_chat_split train_gen` if you want.

## Honest limitations

- **No single dataset** makes “best conversations” — add your own logs (with consent), domain FAQs, or style guides if you need a specific voice.
- **Safety / harmlessness** is not automatic — handle that with filters, policies, and evaluation.
- After training, **measure** with real prompts you care about (not only loss).
