# Ask Shorty — RAG Technical Intro (Slides & Visuals)

Style: dark navy / black background, cyan accent `#00D4FF`, clean sans-serif (Inter or similar), simple fades/wipes/builds only.
Target length: ~3–4 minutes.

Use these as PowerPoint / Google Slides notes. Insert the provided screenshots where indicated.

---

## Slide 1 — Hook: “RAG over transcripts is broken”
- **Duration:** ~0–10 s
- **Visual:**
  - Full-screen dark background.
  - Large title text center: **“RAG over transcripts is broken”** in cyan.
  - Subtle background graphic: faint transcript text blocks scattering with red `X` icons over random chunks.
  - Simple **fade-in** on title, then a light **drift** animation on scattered chunks if you want.
- **Voiceover:** Hook section from `script.md`.

---

## Slide 2 — Chunk-Only RAG Limits
- **Duration:** ~10–45 s
- **Visual:**
  - Left side: diagram:
    - `Transcript` → `[Chunk 1] [Chunk 2] [Chunk 3]` → `Embed` → `Vector search` → `LLM`.
    - Red arrows / callouts over:
      - “Facts scattered across chunks”
      - “Embeddings dominated by filler”
      - “Broad questions don’t map to a chunk”
  - Right side: placeholder for a **“bad RAG result”** mock — e.g., a generic chat UI with a half-right answer and a small label “Chunk-only RAG”.
  - Use **wipe-in** to build arrows and callouts as you mention each problem.
- **Voiceover:** Section 1 from `script.md`.

---

## Slide 3 — Ask Shorty Idea: Shorty + RAG
- **Duration:** ~45–70 s
- **Visual:**
  - Center diagram showing:
    - `Transcript (SQLite)` at the left.
    - Arrows to three boxes: **Shorty**, **Synthetic Questions**, **Entities**.
    - From these, arrows into **Chroma (transcripts collection)**, with three labeled icons:
      - `type="chunk"`
      - `type="shorty"`
      - `type="synthetic_question"`
  - Use cyan outlines for Shorty and synthetic questions to emphasize the “extra layers”.
  - Simple **build**: first transcript box, then Shorty/synthetic Qs/entities, then Chroma.
- **Voiceover:** Section 2 from `script.md`.

---

## Slide 4 — Data Model: SQLite
- **Duration:** ~70–85 s
- **Visual:**
  - Left: simple table icons with column bullets:
    - `videos(video_id, title, channel, url, has_transcript, watch_date, …)`
    - `transcripts(video_id, text, shorty, shorty_generated_at, …)`
    - `entities(video_id, name, type, aliases)`
    - `synthetic_questions(video_id, question, embedding_id)`
    - `processing_queue(video_id, task, status, error, …)`
  - Right: **SQLite screenshot placeholder**:
    - Add later: screenshot of `data/transcripts.db` opened in your SQLite viewer showing `videos` / `transcripts` tables.
  - Use **fade-in** per table list.
- **Voiceover:** First half of Section 3 from `script.md`.

---

## Slide 5 — Data Model: Chroma Types
- **Duration:** ~85–100 s
- **Visual:**
  - Diagram of Chroma `transcripts` collection:
    - Three example rows:
      - Row 1: `type=chunk`, `video_id=...`, `chunk_index=0`.
      - Row 2: `type=shorty`, `video_id=...`.
      - Row 3: `type=synthetic_question`, `video_id=...`.
  - Side callout: “Single collection, three representations per video”.
  - **Screenshot placeholder**: Chroma inspect UI (or a small table mock) showing these three types.
  - Soft **wipe** to bring in each row.
- **Voiceover:** Second half of Section 3 from `script.md`.

---

## Slide 6 — Query Pipeline Overview
- **Duration:** ~100–130 s
- **Visual:**
  - Horizontal flowchart:
    - `Question` → `Metadata Filter (SQLite)` → `Query Rewriting (Claude)` →
      `[Chroma search: type=shorty]` +
      `[Chroma search: type=synthetic_question]` +
      `[Chroma search: type=chunk]` →
      `Context Assembly` → `LLM Answer`.
  - Color code:
    - SQLite steps in light gray outlines.
    - LLM steps in cyan.
    - Chroma nodes in teal/green.
  - Build the diagram left‑to‑right with simple **appear / wipe** animations synchronized to the VO.
- **Voiceover:** First half of Section 4 from `script.md`.

---

## Slide 7 — Query Pipeline: Ask UI Screenshot
- **Duration:** ~130–145 s
- **Visual:**
  - Full-width screenshot of Ask UI:
    - Use image: `...Screenshot_2026-03-13_032958-5e232b6f-00cd-4278-801d-02e98e74bf91.png`.
  - Optional overlay:
    - Small arrows pointing to:
      - Question box: label “Natural language query”.
      - Video ID field: label “Optional filter”.
  - Simple **zoom-in** or **fade-in** on screenshot.
- **Voiceover:** Closing part of Section 4 (context assembly & answer) or a brief ad‑lib explaining the UI.

---

## Slide 8 — Why Chunk‑Only Fails vs Ask Shorty
- **Duration:** ~145–175 s
- **Visual:**
  - Two-column comparison table:
    - Left header: **Chunk‑only RAG** (orange).
    - Right header: **Ask Shorty (Shorty + RAG)** (cyan).
    - Rows: “Global context”, “Rare facts”, “Broad questions”, “Debuggability”.
  - Simple icons:
    - Left side: scattered chunk icons with red `X`.
    - Right side: a compact Shorty document plus a few chunks with green check marks.
  - Use **build** animation row‑by‑row, timed to each point.
- **Voiceover:** Section 5 from `script.md`.

---

## Slide 9 — Implementation Stack
- **Duration:** ~175–200 s
- **Visual:**
  - Stack diagram:
    - Bottom: “SQLite + Chroma”.
    - Middle: “Python / Flask services”.
    - Top: “Anthropic Claude + OpenAI‑compatible providers”.
  - On the side, rotate a few quick **code / UI screenshots**:
    - Library UI screenshot (`...Screenshot_2026-03-13_033100-60a3cff0-3964-41e1-97ed-824f473c010c.png`) to show Shorty status.
    - Ask UI screenshot again for context.
    - Optional small terminal shot of `batch_processor.py --queue`.
  - Use **cross‑fade** between stills, keep motion slow.
- **Voiceover:** Section 6 from `script.md`.

---

## Slide 10 — Closing: GitHub + CTA
- **Duration:** ~200–220 s
- **Visual:**
  - Full-screen GitHub repo screenshot (add later from your browser).
  - Title: **“Ask Shorty (GitHub)”**.
  - Subtitle: “Shorty + RAG over transcripts”.
  - Bottom‑right: subtle “Subscribe for more RAG experiments” text.
  - Optional simple **underline wipe** under the repo URL as you mention it.
- **Voiceover:** Section 7 from `script.md`.

---

## Screenshot Checklist

Place these image files into your slide deck:

- **Library UI with Shorty visible**  
  - Use: `Screenshot_2026-03-13_033100-60a3cff0-3964-41e1-97ed-824f473c010c.png` (Library view).
- **Ask UI with question box**  
  - Use: `Screenshot_2026-03-13_032958-5e232b6f-00cd-4278-801d-02e98e74bf91.png`.
- **SQLite table view**  
  - Capture `videos` / `transcripts` from your SQLite browser and drop into Slide 4.
- **Chroma inspect showing 3 types**  
  - Capture your Chroma inspector (or a small table mock) showing `type=chunk`, `type=shorty`, `type=synthetic_question` for Slide 5.

