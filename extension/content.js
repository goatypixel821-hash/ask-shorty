(function () {
  "use strict";

  const GRABBER = "http://localhost:5000";
  const DEFAULT_TAGS = ["idea", "quote", "important", "todo"];
  const HOTKEY = { alt: true, shift: true, key: "m" }; // Alt+Shift+M

  let root = null;
  let fab = null;
  let toastEl = null;
  let cachedTags = null;
  let overlayOpen = false;
  let frozenTimestamp = 0;

  function parseVideoId() {
    const m = location.href.match(/[?&]v=([a-zA-Z0-9_-]{11})/);
    return m ? m[1] : null;
  }

  function getTitle() {
    return (document.title || "").replace(/\s*-\s*YouTube\s*$/i, "").trim() || "Untitled";
  }

  function getChannel() {
    const selectors = [
      "ytd-channel-name a",
      ".ytd-channel-name a",
      "#owner #channel-name a",
      "yt-formatted-string.ytd-channel-name",
    ];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el && el.textContent) {
        return el.textContent.trim();
      }
    }
    return "";
  }

  function getVideoEl() {
    return document.querySelector("video.html5-main-video") || document.querySelector("video");
  }

  function formatTime(sec) {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) {
      return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
    }
    return `${m}:${String(ss).padStart(2, "0")}`;
  }

  function showToast(msg, isErr) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.id = "shorty-mark-toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.className = "show" + (isErr ? " err" : "");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
      toastEl.className = "";
    }, 3200);
  }

  async function fetchTags() {
    if (cachedTags) return cachedTags;
    try {
      const r = await fetch(`${GRABBER}/api/tags`, { method: "GET" });
      if (!r.ok) throw new Error("tags HTTP " + r.status);
      const data = await r.json();
      const server = Array.isArray(data.tags) ? data.tags : [];
      cachedTags = server.length ? server : DEFAULT_TAGS.slice();
      return cachedTags;
    } catch (_) {
      cachedTags = DEFAULT_TAGS.slice();
      return cachedTags;
    }
  }

  function parseTagsInput(raw) {
    return raw
      .split(/[,;]+/)
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function ensureUi() {
    const player =
      document.querySelector("#movie_player") ||
      document.querySelector(".html5-video-player") ||
      document.querySelector("#ytd-player");
    if (!player) return false;

    if (!root || !player.contains(root)) {
      root = document.createElement("div");
      root.id = "shorty-mark-root";
      player.appendChild(root);

      fab = document.createElement("button");
      fab.id = "shorty-mark-fab";
      fab.type = "button";
      fab.title = "Mark this moment (Alt+Shift+M)";
      fab.textContent = "✦";
      fab.addEventListener("click", (e) => {
        e.stopPropagation();
        openOverlay();
      });
      root.appendChild(fab);
    }
    return true;
  }

  function closeOverlay() {
    const ov = document.getElementById("shorty-mark-overlay");
    if (ov) ov.remove();
    overlayOpen = false;
  }

  async function openOverlay() {
    if (overlayOpen) return;
    const vid = parseVideoId();
    if (!vid) {
      showToast("Not a YouTube watch page", true);
      return;
    }
    const video = getVideoEl();
    if (!video) {
      showToast("Video player not found", true);
      return;
    }
    frozenTimestamp = video.currentTime;
    overlayOpen = true;

    const tags = await fetchTags();

    const overlay = document.createElement("div");
    overlay.id = "shorty-mark-overlay";
    overlay.innerHTML = `
      <div id="shorty-mark-panel" role="dialog" aria-label="Mark moment">
        <h3>Mark at ${formatTime(frozenTimestamp)}</h3>
        <textarea id="shorty-mark-note" placeholder="Note (optional)…" rows="3"></textarea>
        <div id="shorty-mark-tags-wrap">
          <input id="shorty-mark-tags" type="text" placeholder="Tags: idea, todo (comma-separated)" autocomplete="off" />
          <div id="shorty-mark-tag-list"></div>
        </div>
        <div id="shorty-mark-actions">
          <button type="button" id="shorty-mark-cancel">Cancel</button>
          <button type="button" id="shorty-mark-save">Save mark</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const noteEl = overlay.querySelector("#shorty-mark-note");
    const tagsEl = overlay.querySelector("#shorty-mark-tags");
    const tagList = overlay.querySelector("#shorty-mark-tag-list");

    function renderTagDropdown(filter) {
      tagList.innerHTML = "";
      const f = (filter || "").toLowerCase();
      const matches = tags.filter((t) => !f || t.toLowerCase().includes(f));
      if (!matches.length) {
        tagList.classList.remove("open");
        return;
      }
      matches.forEach((t) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = t;
        btn.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const cur = parseTagsInput(tagsEl.value);
          if (!cur.includes(t)) cur.push(t);
          tagsEl.value = cur.join(", ");
          tagList.classList.remove("open");
        });
        tagList.appendChild(btn);
      });
      tagList.classList.add("open");
    }

    tagsEl.addEventListener("focus", () => renderTagDropdown(tagsEl.value));
    tagsEl.addEventListener("input", () => renderTagDropdown(tagsEl.value));

    overlay.querySelector("#shorty-mark-cancel").addEventListener("click", closeOverlay);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeOverlay();
    });

    async function submit() {
      const saveBtn = overlay.querySelector("#shorty-mark-save");
      saveBtn.disabled = true;
      const payload = {
        video_id: vid,
        url: location.href,
        title: getTitle(),
        channel: getChannel(),
        timestamp_seconds: frozenTimestamp,
        note_text: noteEl.value.trim(),
        tags: parseTagsInput(tagsEl.value),
      };
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8000);
      try {
        const r = await fetch(`${GRABBER}/api/annotate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: ctrl.signal,
        });
        clearTimeout(timer);
        const data = await r.json().catch(() => ({}));
        if (!r.ok || !data.success) {
          throw new Error(data.error || `HTTP ${r.status}`);
        }
        cachedTags = null;
        closeOverlay();
        showToast(`Marked at ${formatTime(frozenTimestamp)}`);
      } catch (err) {
        clearTimeout(timer);
        const msg =
          err.name === "AbortError"
            ? "Timed out — is the grabber running on port 5000?"
            : `Failed to save — ${err.message || err}`;
        showToast(msg, true);
        saveBtn.disabled = false;
      }
    }

    overlay.querySelector("#shorty-mark-save").addEventListener("click", submit);

    noteEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });

    overlay.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeOverlay();
      }
    });

    noteEl.focus();
  }

  document.addEventListener(
    "keydown",
    (e) => {
      if (!HOTKEY.alt || !e.altKey) return;
      if (!HOTKEY.shift || !e.shiftKey) return;
      if (e.key.toLowerCase() !== HOTKEY.key) return;
      if (e.ctrlKey || e.metaKey) return;
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || e.target.isContentEditable) return;
      e.preventDefault();
      e.stopPropagation();
      openOverlay();
    },
    true
  );

  function tick() {
    ensureUi();
  }

  tick();
  setInterval(tick, 2000);
  const obs = new MutationObserver(tick);
  obs.observe(document.body, { childList: true, subtree: true });
})();
