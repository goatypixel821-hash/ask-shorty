(function () {
  "use strict";

  const GRABBER = "http://localhost:5000";
  const DEFAULT_TAGS = ["idea", "quote", "important", "todo"];
  const HOTKEY = { alt: true, shift: true, key: "m" }; // Alt+Shift+M
  const FAB_SIZE = 42;
  const FAB_MARGIN_RIGHT = 14;
  const FAB_MARGIN_BOTTOM = 58; // above YouTube control bar

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

  function findPlayerEl() {
    return (
      document.querySelector("#movie_player") ||
      document.querySelector(".html5-video-player") ||
      document.querySelector("#ytd-player #container") ||
      document.querySelector("#ytd-player")
    );
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

  function positionFab() {
    if (!root || !fab) return;
    if (!parseVideoId()) {
      root.style.display = "none";
      return;
    }
    const player = findPlayerEl();
    if (!player) {
      root.style.display = "none";
      return;
    }
    const r = player.getBoundingClientRect();
    if (r.width < 120 || r.height < 80) {
      root.style.display = "none";
      return;
    }
    const left = r.right - FAB_MARGIN_RIGHT - FAB_SIZE;
    const top = r.bottom - FAB_MARGIN_BOTTOM - FAB_SIZE;
    root.style.display = "block";
    root.style.left = `${Math.max(8, left)}px`;
    root.style.top = `${Math.max(8, top)}px`;
    root.style.width = `${FAB_SIZE}px`;
    root.style.height = `${FAB_SIZE}px`;
  }

  function ensureFab() {
    if (!root) {
      root = document.createElement("div");
      root.id = "shorty-mark-root";
      document.body.appendChild(root);

      fab = document.createElement("button");
      fab.id = "shorty-mark-fab";
      fab.type = "button";
      fab.title = "Mark this moment (Alt+Shift+M)";
      fab.textContent = "✦";
      fab.setAttribute("aria-label", "Mark this moment");
      fab.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        openOverlay();
      });
      root.appendChild(fab);
    }
    positionFab();
  }

  function closeOverlay() {
    const ov = document.getElementById("shorty-mark-overlay");
    if (ov) ov.remove();
    overlayOpen = false;
    positionFab();
  }

  function readCurrentTimestamp() {
    const video = getVideoEl();
    return video ? video.currentTime : 0;
  }

  async function openOverlay() {
    if (overlayOpen) {
      closeOverlay();
    }
    const vid = parseVideoId();
    if (!vid) {
      showToast("Not a YouTube watch page", true);
      return;
    }
    if (!getVideoEl()) {
      showToast("Video player not found", true);
      return;
    }
    frozenTimestamp = readCurrentTimestamp();
    overlayOpen = true;
    if (root) root.style.display = "none";

    const tags = await fetchTags();

    const overlay = document.createElement("div");
    overlay.id = "shorty-mark-overlay";
    const panel = document.createElement("div");
    panel.id = "shorty-mark-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Mark moment");
    panel.innerHTML = `
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
    `;
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const noteEl = panel.querySelector("#shorty-mark-note");
    const tagsEl = panel.querySelector("#shorty-mark-tags");
    const tagList = panel.querySelector("#shorty-mark-tag-list");

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

    panel.querySelector("#shorty-mark-cancel").addEventListener("click", closeOverlay);
    panel.addEventListener("click", (e) => e.stopPropagation());
    overlay.addEventListener("click", () => closeOverlay());

    async function submit() {
      const saveBtn = panel.querySelector("#shorty-mark-save");
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
        const savedAt = formatTime(frozenTimestamp);
        closeOverlay();
        showToast(`Marked at ${savedAt} — click ✦ for another`);
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

    panel.querySelector("#shorty-mark-save").addEventListener("click", submit);

    noteEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });

    panel.addEventListener("keydown", (e) => {
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
    ensureFab();
  }

  tick();
  setInterval(tick, 1000);
  window.addEventListener("resize", positionFab, { passive: true });
  window.addEventListener("scroll", positionFab, { passive: true });
  const obs = new MutationObserver(tick);
  obs.observe(document.body, { childList: true, subtree: true });
})();
