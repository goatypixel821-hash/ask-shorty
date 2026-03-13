# Ask Shorty — RAG Technical Intro (YouTube Voiceover Script)

Total target length: ~3–4 minutes.

Speak naturally, medium pace, calm/techy tone. Pauses marked with `// short pause`.

---

## Hook (0–10 s)

> If you've ever tried to bolt RAG onto transcripts, you've probably hit the same wall I did: embeddings over raw chunks are noisy, brittle, and miss important details. Ask Shorty is my attempt to fix that with a different architecture: Shorty plus RAG. // short pause

---

## 1. Limits of Chunk‑Only RAG (10–45 s)

> Most “AI over content” stacks look like this: split the transcript into chunks, embed each chunk, and at query time do a vector search and stuff the top chunks into the prompt. That sounds fine, but in practice three things go wrong. // short pause
>
> First, important facts are scattered. Causal chains and arguments span multiple chunks, so a single retrieved chunk rarely tells the full story. // short pause
>
> Second, embeddings get dominated by filler conversation instead of key ideas, so similarity search often locks onto the wrong parts of the transcript. // short pause
>
> Third, broad questions like “What was their argument about X?” don't map cleanly to any single chunk. You end up with answers that are half‑right, or you just miss the thing you're looking for. // short pause

---

## 2. Ask Shorty Idea (45–70 s)

> Ask Shorty keeps the normal RAG pipeline, but adds a new layer: a Shorty per video. // short pause
>
> At index time, the system stores the full transcript in SQLite, uses an LLM to generate one dense Shorty, a set of synthetic questions, and a list of entities, then embeds transcript chunks, the Shorty, and the synthetic questions. All three get indexed together in Chroma. // short pause
>
> Now each video isn't just a pile of chunks — it's a small knowledge bundle with multiple representations that all point back to the same source. // short pause

---

## 3. Data Model (70–100 s)

> Under the hood there's a simple schema. The videos table holds metadata like ID, title, channel, URL, and watch date. The transcripts table stores raw text plus a shorty field and timestamps. // short pause
>
> Entities live in their own table with name, type, and aliases so we can normalize references. Synthetic questions have their own table as well. And there's a processing queue that tracks background tasks for each video: shorty, synthetic questions, and entities. // short pause
>
> SQLite is the source of truth. Chroma just holds one collection with three types — chunk, shorty, and synthetic question — each tagged with a video ID. // short pause

---

## 4. Query Pipeline (100–145 s)

> On the query side, the pipeline has a few stages. // short pause
>
> First, a metadata filter can parse the question for channel names or dates and use SQLite to narrow down candidate video IDs. That keeps retrieval focused when the user implicitly references a specific creator or time window. // short pause
>
> Second, query rewriting uses Claude to generate multiple alternate phrasings of the question. These variants capture different angles and often match how the synthetic questions were phrased at index time. // short pause
>
> Third, for each rewrite, we run multi‑representation search in Chroma. Type equals shorty finds globally relevant videos. Type equals synthetic question matches question‑to‑question. Type equals chunk finds exact phrases and quotes. // short pause
>
> Finally, we assemble context: the best Shorties, the most relevant synthetic questions, and a few key chunks. That combined context goes back into the LLM, which produces the final answer with citations to the underlying videos. // short pause

---

## 5. Why This Is Better (145–175 s)

> Shorties give whole‑video context in just a few hundred tokens. Synthetic questions turn search into question‑to‑question matching, which is much easier for the model to get right. And chunks still provide verbatim detail when you need exact wording. // short pause
>
> Instead of betting everything on a single representation, Ask Shorty uses three that complement each other. In practice, recall is noticeably better than chunk‑only RAG, especially on long, messy transcripts. // short pause

---

## 6. Implementation Stack (175–200 s)

> The implementation is intentionally simple. It's Python and Flask on top of SQLite, with Chroma and sentence‑transformers for embeddings. Anthropic handles Shorty generation, synthetic questions, entities, and the query rewriting and answer stages. // short pause
>
> There's a bookmarklet‑driven grabber service that ingests YouTube URLs, a library UI where you can inspect Shorties and processing status, and an Ask UI where you actually type your questions. // short pause

---

## 7. Closing (200–220 s)

> If you're experimenting with RAG over transcripts and hitting recall problems, check out Ask Shorty on GitHub — link is in the description. And if you want to see this running against a real YouTube history, there's another video on the channel that walks through the demo end‑to‑end. // short pause

