# Running Ask Shorty batch processing on Vast.ai with Qwen + vLLM

This guide shows how to run `batch_processor.py` on a rented GPU instance using Vast.ai, with a Qwen model served through vLLM's OpenAI-compatible API.

The goal: **cheap, fast bulk generation of Shorties, synthetic questions, and entities** using the existing `--provider openai-compatible` support in `batch_processor.py`.

---

## 1. Rent a Vast.ai instance

1. Go to `https://vast.ai/` and sign in or create an account.
2. Click **Search** / **Rent** to open the instance search page.
3. Set filters:
   - **Template/Image**: PyTorch image (PyTorch template).
   - **GPU VRAM**: **≥ 24 GB**.
   - **Price**: target **≤ $0.40/hr**.
4. Good options:
   - `RTX 3090` (24 GB)
   - `RTX 4090` (24 GB)
   - `A10` (24 GB)
5. Click **Rent** on the chosen instance and wait for it to start.
6. Once running, open the instance **Details** page; you will need:
   - The **SSH connection** info
   - The **public IP** and **port mappings** (we will expose port `8000` for vLLM).

---

## 2. Install and start vLLM on the instance

SSH into the Vast.ai instance (use the `ssh` command provided in the Vast UI), then run:

```bash
bash vast_start.sh
```

Wait for vLLM to download and load the model. When it’s ready, you’ll see a log line like:

> `Application startup complete`

At that point, the OpenAI-compatible API is live on port `8000` inside the container.

---

## 3. Expose the vLLM port on Vast.ai

1. In the Vast.ai dashboard, open your running instance.
2. Check the **Open Ports / Port mapping** section:
   - Look for a port mapping where the host port maps to itself, e.g. `36396 -> 36396/tcp`.
   - Use that port number for vLLM’s `--port` (and in your local `--base-url`).
3. Note the:
   - **Instance IP** (e.g. `123.45.67.89`)
   - The chosen **host port** (e.g. `36396`)

You will combine these as:  
`http://<VAST_IP>:<PORT>/v1`  
Example: `http://74.48.78.46:36396/v1`

---

## 4. Run a quality test batch from your local machine

On your **local Windows machine**, in PowerShell, from the `shorty` project folder:

```powershell
$env:OPENAI_API_KEY = "dummy"
python batch_processor.py `
  --provider openai-compatible `
  --base-url http://<VAST_IP>:<PORT>/v1 `
  --model Qwen/Qwen2.5-14B-Instruct `
  --db-path 'C:\Users\number2\Desktop\shorty\data\transcripts.db' `
  --queue `
  --limit 10
```

Replace:
- `<VAST_IP>` with the instance IP from Vast.ai.
- `<PORT>` with the host port you started vLLM on (for example the self-mapped port from **Open Ports**, like `36396`).

Flags:
- `--provider openai-compatible` tells `batch_processor.py` to use the OpenAI-style API instead of Anthropic.
- `--base-url` points to your vLLM server.
- `--model` must match the model name you passed to vLLM.
- `--db-path` should point to your **Ask Shorty** `transcripts.db`.
- `--queue` processes from the `processing_queue` table (Shorty/questions/entities tasks).
- `--limit 10` processes 10 queued tasks to use as a quality test batch.

After the run finishes:

1. Open the **library admin UI** in your browser at `http://127.0.0.1:5002`.
2. Inspect several of the newly processed videos:
   - Do the Shorties capture all major topics across the video?
   - Are important numbers and technical details preserved?
   - Are synthetic questions meaningful and varied?
   - Are entities (people, organizations, systems, products) extracted correctly?
3. Compare these Qwen-based results against existing Haiku-generated Shorties for similar content (e.g., technical talks, investigative videos) to decide if Qwen quality is acceptable for bulk processing.

---

## 5. Cost comparison (rough estimates for 3000 videos)

These numbers are **ballpark only** and assume:
- Average transcript length per video in the several-thousand-token range.
- Overnight run (8–12 hours).

| Setup                                  | Approximate cost for 3000 videos |
|----------------------------------------|-----------------------------------|
| Anthropic Haiku API                    | **~$57**                          |
| Vast.ai RTX 3090 @ ~$0.25/hr (8–16 h)  | **~$2–4**                         |
| Vast.ai RTX 4090 @ ~$0.35/hr (8–16 h)  | **~$3–5**                         |

Takeaway: **Vast.ai + Qwen via vLLM is an order of magnitude cheaper** than calling a hosted API directly for large backfills.

---

## 6. Tips and best practices

- **Dry run with a small limit first**
  - Use `--limit 100` to:
    - Verify that Qwen’s Shorties and synthetic questions have similar quality to Haiku.
    - Confirm that entity extraction still looks good.

- **Manually inspect a few results**
  - After a small run, open the **library admin UI** (Flask app on port `5002`).
  - Check several videos:
    - Shorty quality
    - Synthetic question relevance
    - Entities & aliases (people, organizations, systems, locations, concepts)

- **Keep the job alive**
  - Use `tmux` or `screen` on the Vast.ai instance when you start `vllm` if you want it running independently of your SSH session.
  - On your local machine, you can re-run `batch_processor.py` with `--retry-failed` later if needed.

- **Environment variables**
  - vLLM **does not require a real API key**, but some client libraries expect one.
  - On your local machine, you can set a dummy key:

    ```powershell
    setx OPENAI_API_KEY "dummy"
    ```

    Then restart your shell so the variable is picked up.

- **Scaling up**
  - Once you are happy with the quality for `--limit 100`, re-run without `--limit` (or with a higher limit) to process the full library.
  - Monitor GPU utilization and tokens/sec on the Vast.ai instance to ensure you are getting good throughput.

