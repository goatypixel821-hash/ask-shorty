# Running Ask Shorty batch processing on Vast.ai with Qwen + vLLM

This guide shows how to run `batch_processor.py` on a rented GPU instance using Vast.ai, with a Qwen model served through vLLM's OpenAI-compatible API.

The goal: **cheap, fast bulk generation of Shorties, synthetic questions, and entities** using the existing `--provider openai-compatible` support in `batch_processor.py`.

---

## 1. Rent a Vast.ai instance

1. Go to `https://vast.ai/` and sign in or create an account.
2. Click **Rent** to open the instance search page.
3. Set filters:
   - **GPU**: look for:
     - `RTX 3090` (24 GB VRAM), or
     - `RTX 4090` (24 GB VRAM)  
     These have enough VRAM to run `Qwen2.5-14B-Instruct` at 4‑bit.
   - **Template** / **Image**:
     - Choose a **PyTorch** (or similar ML) template.
   - **CUDA**:
     - Prefer **CUDA 12+** images (vLLM runs well there).
4. Sort by **$/hr** or **reliability**, pick a host with:
   - GPU: 3090 or 4090
   - At least **24 GB VRAM**
   - Reasonable bandwidth and disk space
5. Click **Rent** on the chosen instance and wait for it to start.

Once the instance is running, open its **Details** page; you will need:
- The **SSH connection** info
- The **public IP** and **port mappings** (we will expose port `8000` for vLLM).

---

## 2. Install and start vLLM on the instance

SSH into the Vast.ai instance (use the `ssh` command provided in the Vast UI), then run:

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --quantization awq \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```

Notes:
- `Qwen/Qwen2.5-14B-Instruct` is a strong general-purpose instruction model.
- `--quantization awq` allows it to fit comfortably into **24 GB VRAM**.
- `--host 0.0.0.0` and `--port 8000` make it reachable from outside the container.

You can also use the helper script `vast_start.sh` (in this repo) instead of typing commands manually.

---

## 3. Expose the vLLM port on Vast.ai

1. In the Vast.ai dashboard, open your running instance.
2. Check the **Port mapping** section:
   - Make sure **container port 8000** is mapped to a **public host port** (e.g. `12345`).
   - If not, add a mapping from `8000` (container) to some public port.
3. Note the:
   - **Instance IP** (e.g. `123.45.67.89`)
   - **Host port** mapped to container `8000` (e.g. `12345`)

You will combine these as:  
`http://<VAST_IP>:<PORT>/v1`  
Example: `http://123.45.67.89:12345/v1`

---

## 4. Run `batch_processor.py` against vLLM from your local machine

On your **local Windows machine**, in PowerShell, from the `shorty` project folder:

```powershell
python batch_processor.py `
  --provider openai-compatible `
  --base-url http://<VAST_IP>:<PORT>/v1 `
  --model Qwen/Qwen2.5-14B-Instruct `
  --db-path 'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db' `
  --limit 100
```

Replace:
- `<VAST_IP>` with the instance IP from Vast.ai
- `<PORT>` with the mapped host port for container port `8000`

Flags:
- `--provider openai-compatible` tells `batch_processor.py` to use the OpenAI-style API instead of Anthropic.
- `--base-url` points to your vLLM server.
- `--model` must match the model name you passed to vLLM.
- `--db-path` points to your existing `transcripts.db`.
- `--limit 100` processes the first 100 videos to validate output before a full run.

The script will:
- Estimate token + cost,
- Ask for confirmation,
- Then generate Shorties, synthetic questions, and entities in batches.

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

