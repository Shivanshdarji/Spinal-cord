# When “nothing works” (it used to)

Usually this is **environment + cache**, not “the model died.” Work through these in order.

## 0) Dashboard looks “dead” (no model list, Send does nothing)

The page init runs on **`DOMContentLoaded`**. A **JavaScript error** during that handler stops **everything** after it (including **`loadModelIdPanel`**). A past bug was a missing **`animateCounter`** function — **fixed in repo**. Always **hard-refresh** after pulling: **Ctrl+F5**.

Also use **`http://127.0.0.1:8080/`** (from `llama-server --path`), not **`file://`**. Open **F12 → Console** to see red errors.

---

## 1) Browser: `GET /` and `favicon.ico` → **404**

**Meaning:** `llama-server` is running, but **static files are not mounted** — usually the **Web UI** is off.

- **Fix:** Use the repo **`dashboard\run_dashboard.bat`** / **`run_dashboard_brain_only.bat`** (they pass **`--webui`** and **`--path`** to the `dashboard\` folder).
- **Check env:** Unset **`LLAMA_ARG_WEBUI`** if it is `0` / `false` (it disables the UI and **`GET /`** has no handler → 404).
- **Workaround:** Open **`http://127.0.0.1:8080/index.html`** explicitly (same folder).

**`favicon.ico` 404** is harmless (browser auto-request); ignore or add a small `favicon.ico` under `dashboard\` later.

---

## 2) One server, one port

- Stop **all** old `llama-server` windows (Task Manager → `llama-server.exe` if unsure).
- Start **only one** batch file: either `dashboard\run_dashboard.bat` **or** `dashboard\run_dashboard_brain_only.bat` (not both).
- Leave that window open; read the last lines for **errors** (missing GGUF, CUDA OOM, etc.).

## 3) PowerShell: “`/dashboard/...` is not recognized”

On Windows **PowerShell**, paths must look like **Windows** paths, and scripts need **`.\`**:

```powershell
cd C:\Users\SHIVANSH\Desktop\spinalcord
.\dashboard\run_dashboard_llama_scaffold.bat
```

**Wrong:** `/dashboard/run_dashboard_llama_scaffold.bat` (Unix-style — not a valid command).  
**Right:** `.\dashboard\run_dashboard_llama_scaffold.bat`

---

## 4) Open the UI the same way as before

The dashboard must be loaded from the server:

- **Right:** `http://127.0.0.1:8080/` (or `http://127.0.0.1:8080/index.html`)
- **Wrong:** opening `index.html` as a **file://** link — chat calls go to the wrong place and fail.

## 5) Hard refresh + clear saved model id

- **Ctrl+Shift+Delete** → clear **Cached images and files** for localhost, **or** hard refresh: **Ctrl+F5**.
- In the page: **Find your model id** → **Refresh** → **Use for chat** on the **Brain** row.  
  Or open:  
  `http://127.0.0.1:8080/?model=scbrain_1b.gguf`  
  (no `<` `>` around the name — see `docs/LLAMA_SERVER_MODEL_ID.md`).

## 6) Don’t test chat by pasting the API URL in the address bar

`/v1/chat/completions` is **POST-only**. Opening it in the browser sends **GET** → **404** + `File Not Found`. That is **normal**, not proof the model is broken.

## 7) Prove the API with curl (bypasses the browser)

PowerShell:

```powershell
curl.exe -s "http://127.0.0.1:8080/v1/models"
```

Then:

```powershell
curl.exe -s -X POST "http://127.0.0.1:8080/v1/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{"model":"scbrain_1b.gguf","messages":[{"role":"user","content":"Say hi in one word."}],"stream":false}'
```

- If **models** works but **chat** fails → note the **HTTP status** and body (400 vs 404 vs 500).
- If **nothing** connects → firewall / wrong port / server not running.

## 8) API prefix (rare)

If you ever set **`LLAMA_ARG_API_PREFIX`** or **`--api-prefix`**, URLs must include that prefix. Unset the env var or match the prefix in the client. See `docs/LLAMA_SERVER_MODEL_ID.md`.

## 9) Garbled / repetitive chat (API works)

If **`scripts/verify_llama_server.ps1`** succeeds but the assistant text is nonsense: the **server is fine**; the **checkpoint** is weak or story-biased. In the dashboard, lower **Temp**, reduce **Max tok**, raise **Repeat** (sampling bar). Long-term: more instruction data + training (`docs/CONVERSATION_TRAINING.md`).

## 10) Repo / script changes

Recent edits include **dashboard** sampling controls, model id handling, and **batch** path safety. They don’t change GGUF weights. If something still fails after the steps above, use the curl test and a **fresh** `llama-server` run.
