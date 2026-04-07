"""
SpinalCord LLM — Master Training Script
=========================================
Runs the entire training pipeline sequentially in one session:

  Phase 1: Train SCBrain-1B     (~5 hours, 6000 steps)
  Phase 2: Distill SCDraft-120M (~2 hours, 5000 steps)
  Phase 3: Convert both to GGUF  (~5 minutes)
  Phase 4: Print launch instructions

Total: ~7-8 hours on RTX 2050

Leave this running overnight. Dashboard auto-detects the new models!

Author: Shivansh Darji | AppDice
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(PROJECT_ROOT, "train")
CONVERT_DIR = os.path.join(PROJECT_ROOT, "convert")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")

def banner(title, emoji="🧠"):
    print("\n" + "=" * 60)
    print(f"  {emoji} {title}")
    print("=" * 60)

def run_phase(label, script, args=None):
    banner(label)
    cmd = [sys.executable, script] + (args or [])
    print(f"  Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(script))
    elapsed = time.time() - t0
    mins = elapsed / 60
    if result.returncode == 0:
        print(f"\n  ✅ Completed in {mins:.1f} minutes")
    else:
        print(f"\n  ❌ Failed (exit code {result.returncode})")
        sys.exit(1)
    return elapsed


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("  🚀 SpinalCord LLM — Full Training Pipeline")
    print("  AppDice | Shivansh Darji")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    print("""
  Pipeline:
    Phase 1: Train SCBrain-1B     (6000 steps, ~5 hrs)
    Phase 2: Distill SCDraft-120M (5000 steps, ~2 hrs)
    Phase 3: Convert to GGUF      (~5 min)

  GO GRAB A COFFEE ☕ — Your custom LLM is being born!
""")

    # ── Phase 1: Brain Training ────────────────────────────────────
    phase1 = run_phase(
        "PHASE 1 / 3 — Training SCBrain-1B",
        os.path.join(TRAIN_DIR, "train_brain.py"),
        args=["--steps", "6000"]
    )

    # ── Phase 2: Draft Distillation ───────────────────────────────
    brain_ckpt = os.path.join(MODELS_DIR, "scbrain_best.pt")
    phase2 = run_phase(
        "PHASE 2 / 3 — Distilling SCDraft-120M from Brain",
        os.path.join(TRAIN_DIR, "distill_draft.py"),
        args=["--brain_ckpt", brain_ckpt, "--steps", "5000"]
    )

    # ── Phase 3: GGUF Conversion ──────────────────────────────────
    phase3 = run_phase(
        "PHASE 3 / 3 — Converting to GGUF",
        os.path.join(CONVERT_DIR, "convert_both.py"),
    )

    # ── Final Report ──────────────────────────────────────────────
    total = (time.time() - start) / 3600
    banner("TRAINING COMPLETE! 🎉", emoji="🏆")
    print(f"""
  Total time:   {total:.1f} hours
  Brain:        models/scbrain_1b.gguf
  Draft:        models/scdraft_120m.gguf

  ╔══════════════════════════════════════════════════╗
  ║  Your custom SpinalCord LLM is ready!           ║
  ║                                                  ║
  ║  Launch:  cd dashboard                           ║
  ║           run_dashboard.bat                      ║
  ║                                                  ║
  ║  It will auto-detect your custom models and      ║
  ║  run with FULL SpinalCord speculative decoding!  ║
  ╚══════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
