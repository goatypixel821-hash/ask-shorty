## Ask Shorty – Demo Video (AI Supply Chain Attack)

Faceless explainer video using **fake but realistic** data for an AI supply-chain attack demo.

Target audience: YouTube power users and devs who watch security/AI talks.

Total target length: **≈3:30**.

---

## Demo Data Overview

- **Video title**: `AI Supply Chain Attack Demo`
- **Channel**: `Demo Security Research`
- **Video ID**: `DEMO123xyz`
- **Watch date**: `2026-03-13`
- **Upload date** (stored in metadata): `20260220`
- **Shorty**: present
- **Synthetic questions**: 8
- **Entities**: 12

`demo_data.sql` seeds this into `data/transcripts.db` (see instructions at the end).

---

## Voiceover Script (Slide by Slide)

Speak naturally, calm/techy tone.

### Slide 1 – Hook (5–10s)

**Visual**:
- Dark navy background with faint YouTube player / timeline silhouette.
- Simple animation: generic video tiles sliding in; one highlighted as “AI Supply Chain Attack Demo”.
- Big cyan text: “What did that one video say again?”

**On-screen text (short)**:
- “You watch hundreds of tech talks.”
- “Remembering exact details later is impossible.”

**Voiceover**:
- “You watch hundreds of tech and security videos. Somewhere in there, someone said exactly what you need right now… and you’ll never find it again. What if you could just ask?”

---

### Slide 2 – Problem (20–30s)

**Visual**:
- Dark background.
- Left: Fake YouTube search results that clearly don’t mention the internal workflow details.
- Middle: Huge wall of transcript text scrolling.
- Right: Transcript chopped into floating chunks with small details falling off.

**On-screen text (bullets)**:
- “History is a black box”
- “Search sees titles, not content”
- “Raw transcripts are unusable”
- “Chunk-only RAG drops key details”

**Voiceover**:
- “Right now your YouTube history is basically a black box. Search looks at titles and descriptions, not what was actually said. Transcripts exist, but they’re huge walls of text. Most AI tools just chunk them and hope for the best, which means important details—dates, versions, attack steps—get buried and lost.”

---

### Slide 3 – What Ask Shorty Is (25–35s)

**Visual**:
- Card labeled `Ask Shorty` in the center.
- Icons behind it: a video tile, a transcript document, a Shorty document, and a question bubble.
- Animation: a bookmarklet icon grabs the `AI Supply Chain Attack Demo` video and drops it into a library panel.

**On-screen text**:
- Title: “Ask Shorty: search what you actually watched”
- Subtext: “Bookmarklet → Shorty → Ask anything”

**Voiceover**:
- “Ask Shorty makes your YouTube history instantly queryable. You grab a video with a bookmarklet, it pulls the transcript, and turns it into searchable chunks plus a Shorty—a dense machine brief. That shows up in your library, ready to answer questions like ‘What date was that workflow added?’ or ‘Which versions were affected?’ with real context.”

---

### Slide 4 – Workflow Overview (1:00–1:30)

**Visual**:
- Title: **Manual Transcript Entry**
- Numbered list in cyan on dark background:
  1. Watch YouTube video
  2. Copy transcript (YouTube’s “Show transcript”)
  3. Paste into Ask Shorty
  4. “Save & Vectorize” → Shorty + RAG ready
  5. Search your library
- Each step appears one by one with a simple wipe or build.

**On-screen text**:
- “Manual Transcript Entry”
- “1. Watch → 2. Copy transcript → 3. Paste → 4. Save & Vectorize → 5. Search”

**Voiceover (exact words)**:
- “Simple workflow: Watch video on YouTube. Copy transcript from ‘Show transcript’ button. Paste into Ask Shorty. Hit ‘Save & Vectorize’. Generates Shorty, questions, entities. Instant searchable library.”

---

### Slide 5 – Library Screenshot (15–20s)

**Visual**:
- Screenshot of the **Library** page after loading `demo_data.sql`.
- `AI Supply Chain Attack Demo` row is visible, with indicators for:
  - Shorty present.
  - 8 questions.
  - 12 entities.
- Slow zoom on that row.

**On-screen text**:
- “Library of indexed videos”
- “Shorty, questions, entities per video”

**Voiceover**:
- “Here’s the library view. Each row is a video you’ve indexed. Our demo video shows up with a Shorty attached, plus counts for synthetic questions and entities so you can see how rich the index is before you even ask anything.”

---

### Slide 6 – Shorty Example: Transcript vs Shorty (25–30s)

**Visual**:
- Split screen:
  - Left: transcript text for `AI Supply Chain Attack Demo`.
  - Right: Shorty text from `demo_data.sql` (cropped for readability).
- Cyan highlights showing that all the key dates, versions, and attack steps are preserved.

**On-screen text**:
- Title: “Transcript vs Shorty”
- Left label: “Full transcript”
- Right label: “Shorty – compressed, still answerable”

**Voiceover**:
- “The full transcript is long, but the Shorty keeps almost all of the answerable information—dates, versions, attack steps—while compressing the text massively. Ask Shorty uses this Shorty alongside chunks and synthetic questions when it searches, which gives much higher recall than chunk-only RAG.”

---

### Slide 7 – Demo: Ask UI (25s)

**Visual**:
- Use **three demo screenshots in this exact order**:
  1. **Library list** – library showing `AI Supply Chain Attack Demo` in the table.
  2. **Detail / Ask query** – Ask page with the query:  
     `What date was vulnerable workflow added and which versions affected?`
  3. **Answer** – Ask page showing the answer with:  
     “Dec 21, 2025. Version 2.3.0 malicious, 2.4.0 fixed.” and citations.
- Gentle zoom on the library row, then cross-fade to the Ask query, then to the answer.

**On-screen text**:
- Title: “Ask Shorty demo”
- Small note: “Searches Shorty + transcript + questions”

**Voiceover (exact words, ~25s)**:
- “Library with demo videos. Click shows Shorty, 8 questions, 12 entities. Impossible query: workflow date and versions? Boom – Dec 21 2025, version 2.3.0 malicious, 2.4.0 fixed. Perfect recall.”

---

### Slide 8 – Close (40s)

**Visual**:
- Dark background.
- Left: card labeled “Ask Shorty – GitHub”.
- Right: simple channel branding text: “Ask Shorty”.
- Bottom: “Subscribe” button and “Next: RAG deep dive”.

**On-screen text**:
- “GitHub link in description”
- “Subscribe for the RAG deep dive”

**Voiceover**:
- “If you want to actually search what your videos said, not just their titles, Ask Shorty is for you. This demo uses fake data, but the workflow is exactly what I run on my real history. The GitHub link is in the description. In the next video, I’ll walk through the RAG details behind Shorty, chunks, and questions working together.”

---

## Timing Sheet (Target 3:30 Total)

- **Slide 1 – Hook**
  - **Start**: 0:00
  - **End**: ~0:08

- **Slide 2 – Problem**
  - **Start**: ~0:08
  - **End**: ~0:35

- **Slide 3 – What Ask Shorty Is**
  - **Start**: ~0:35
  - **End**: ~1:05

- **Slide 4 – Workflow Overview (Manual Transcript Entry)**
  - **Start**: ~1:00
  - **End**: ~1:30

- **Slide 5 – Library Screenshot**
  - **Start**: ~1:35
  - **End**: ~1:55

- **Slide 6 – Shorty Example**
  - **Start**: ~1:55
  - **End**: ~2:25

- **Slide 7 – Demo: Ask UI**
  - **Start**: ~2:25
  - **End**: ~2:50

- **Slide 8 – Close**
  - **Start**: ~2:50
  - **End**: ~3:30

Total runtime: **≈3:30**, depending on your speaking pace.

---

## How to Load the Demo Data

1. **Ensure the database exists**
   - Run your app once or run `transcript_database.py` so `data/transcripts.db` is created.

2. **Load the demo video**
   - From the project root, run:

     ```bash
     sqlite3 data/transcripts.db < ask_shorty_demo_video/demo_data.sql
     ```

   - This will:
     - Insert the `AI Supply Chain Attack Demo` video row.
     - Insert one transcript row with the full demo transcript and Shorty text.
     - Insert 8 synthetic questions.
     - Insert 12 entities.

3. **Rebuild the index if needed**
   - If your RAG index depends on Chroma or other embeddings, run your existing indexer so this demo video is indexed alongside the rest.

4. **Open the Library and Ask UI**
   - Start the app (e.g., Flask app pointing at `data/transcripts.db`).
   - Go to the **Library** page and confirm that `AI Supply Chain Attack Demo` appears with:
     - Shorty = yes.
     - Questions = 8.
     - Entities = 12.
   - Go to the **Ask** page and use the demo query:  
     `What date was vulnerable workflow added and which versions affected?`

5. **Capture the screenshots**
   - **Screenshot 1**: Library view showing `AI Supply Chain Attack Demo`.
   - **Screenshot 2**: Ask page with the demo query typed in the box.
   - **Screenshot 3**: Ask page with the full answer visible, including:
     - Date: “Dec 21, 2025”.
     - Versions: “2.3.0 malicious, 2.4.0 fixed”.
     - Answer citations at the bottom.

Use these on **Slide 7** in the order above.

---

## PowerPoint & MP4 Notes

1. **Slides**
   - Create **8 slides** matching the sections above.
   - Background: dark navy / near-black.
   - Accent: cyan `#00D4FF`.
   - Font: clean sans-serif (Inter, SF Pro, Roboto, etc.).
   - Animations: simple fades, wipes, small zooms only.

2. **Voiceover & Timing**
   - Use the script sections above.
   - In PowerPoint, use “Record Slide Show” and advance slides according to the **Timing Sheet**.

3. **Export video**
   - `File → Export → Create a Video` → Full HD (1080p).
   - This gives you an MP4 ready for CapCut.

4. **CapCut polish**
   - Add calm, techy, instrumental background music (no vocals).
   - Mix music at around **-22 to -18 dB** under your voice.
   - Optional: subtle vignette and gentle color grading.

---

## Quick Checklist

- [ ] Ran `sqlite3 data/transcripts.db < ask_shorty_demo_video/demo_data.sql`.
- [ ] Confirmed `AI Supply Chain Attack Demo` appears in the Library with Shorty, 8 questions, 12 entities.
- [ ] Took 3 screenshots for Slide 7: library, Ask with query, Ask with answer.
- [ ] Built 8-slide deck with dark navy + cyan style.
- [ ] Recorded voiceover using this script and timing.
- [ ] Exported MP4 from PowerPoint.
- [ ] Added music and final polish in CapCut.

