# BrainPacks (pluggable SpinalCord profiles)

BrainPacks let you pick a named Brain+Draft backend while keeping the same speculative
accept/reject core.

## Files

- `train/pluggable_spinalcord.py` — generic `SpinalCordEngine` + adapter interfaces.
- `configs/brainpacks.json` — pack registry (default pack + named entries).
- `scripts/run_brainpack.py` — launch a pack by name.

## Run

```bash
python scripts/run_brainpack.py --pack spinalcord_custom --prompt "Hello!"
python scripts/run_brainpack.py --pack llama_server_chat --prompt "Hello!"
```

If `--pack` is omitted, `default_pack` from `configs/brainpacks.json` is used.

## Pack kinds

- `pytorch_spinalcord_pt`:
  - Loads `.pt` checkpoints (`brain_ckpt`, `draft_ckpt`)
  - Uses tokenizer from `train/tokenizer_sc.py`
  - Wraps models with `TorchBrainAdapter` / `TorchDraftAdapter`
- `external_adapter_stub`:
  - Placeholder for future backends (e.g. llama.cpp adapters)
  - Currently prints guidance and exits
- `llama_server_chat`:
  - Calls external `llama-server` over HTTP (`/v1/models`, `/v1/chat/completions`)
  - Useful to test different brains quickly with the same BrainPack launcher
  - Not full speculative verification yet (chat-level integration first)

## Notes

- Draft/Brain tokenizer and vocab must match.
- Speedup depends on acceptance rate; changing Brain usually requires re-distilling Draft.
