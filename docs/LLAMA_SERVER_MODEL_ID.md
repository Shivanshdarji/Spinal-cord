# Finding the llama-server `model` id (for chat)

You do **not** need to read raw JSON by hand if you use the project dashboard.

## Easiest: use the dashboard

1. Start the server: `dashboard\run_dashboard.bat`
2. Open **`http://127.0.0.1:8080`** (same machine).
3. Scroll to **SpinalCord Chat Interface**.
4. In the box **“Find your model id here”**, click **Refresh list** if needed.
5. Find the row labeled **Brain (main)** (your larger GGUF), then click **Use for chat**.
6. Type a message and Send.

That sets the id the same way as adding `?model=...` to the URL.

### Do **not** use angle brackets in the URL

Wrong: `?model=<scbrain_1b.gguf>` (the server receives the literal id **`<scbrain_1b.gguf>`**, which does not exist — that’s why you see **404 File Not Found**).

Right: `?model=scbrain_1b.gguf` — **only** the id string, no `<` `>`.

If your browser shows `%3C` and `%3E` in the address bar, those are `<` and `>`; remove them so the query is `?model=scbrain_1b.gguf`.

## Manual: browser only

1. Open **`http://127.0.0.1:8080/v1/models`** in the same browser.
2. Press **Ctrl+F** and search for **`"id"`**.
3. Copy the string value next to **`id`** for your **Brain** model (not Draft).

## Router vs single-process

- **Single model** (`llama-server --model path\to\brain.gguf`): the `id` is often the **file name** (e.g. `scbrain_1b.gguf`), but not always if you use `--alias`.
- **Router mode**: names come from presets / cache — use the dashboard list or `/v1/models` as above.

## Still empty list?

Then the server has **no model registered**, or **GGUF paths were wrong**.

### Windows: `run_dashboard.bat` and paths

The batch file **`cd`s into `dashboard\`** and uses paths under **`spinalcord\models\`**. If you saw an empty list before, **close the server**, pull the latest `dashboard/run_dashboard.bat`, and run:

`.\dashboard\run_dashboard.bat`

from the project folder. In the console you should see **Resolved Brain path:** pointing at `...\spinalcord\models\scbrain_1b.gguf` (not `Desktop\models\...`).

Create the GGUFs if needed:

`python convert/convert_both.py`

### Other

Fix the `llama-server` command so **`--model`** points at your Brain GGUF and the process starts without errors (watch for CUDA OOM in the terminal).

## HTTP 404 / `File Not Found` / `not_found_error`

In **llama.cpp**, that JSON is the **generic 404** for **any route that does not exist** (see `tools/server/server-http.cpp` error handler) — it is **not** always “wrong model name.”

### A) You opened the chat URL in the browser (very common)

`/v1/chat/completions` is **POST-only**. If you paste `http://127.0.0.1:8080/v1/chat/completions` in the address bar, the browser sends **GET** → **404** + `File Not Found`. That is expected.

**Check chat from:** the dashboard **Send** button, or:

```powershell
curl.exe -s -X POST "http://127.0.0.1:8080/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"scbrain_1b.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}"
```

### B) Wrong host/port

The dashboard must be served by **the same** `llama-server` you started (e.g. `http://127.0.0.1:8080`). Opening `index.html` as a **file://** page or another port will call the wrong origin and can 404.

### C) API prefix

If you start the server with **`--api-prefix`** or env **`LLAMA_ARG_API_PREFIX`**, every API path must include that prefix (e.g. `/prefix/v1/chat/completions`). Unset the env or add the prefix to the client.

### D) Stale / wrong `model` id

You switched **`run_dashboard.bat`** vs **`run_dashboard_brain_only.bat`**, or restarted with a different GGUF, but the page still has an old **`?model=`** or **localStorage** id.

**Fix:** Open **`/v1/models`**, copy the current **Brain** `id`, then **Use for chat** or **`?model=<that-id>`**. The dashboard retries once after clearing stale ids on 404.
