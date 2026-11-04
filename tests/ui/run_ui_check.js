#!/usr/bin/env node
/*
 * Drive the fixture apps with Playwright (Chromium, the preinstalled Node package):
 * ask questions in the V2 and V1 UIs, check the cited sources are the expected
 * fixture videos, and take screenshots.
 *
 *   python tests/ui/serve_fixture_apps.py /tmp/work &
 *   NODE_PATH=$(npm root -g) node tests/ui/run_ui_check.js docs/screenshots
 */
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const TITLES = {
  KSTRL42rel1: "Kestrel DB 4.2 Release Deep Dive",
  KSTRLpm0002: "Postmortem: The Tidewater Corruption Bug",
  CRRNGTN1859: "The Carrington Event: When the Sun Hit the Telegraph",
  VGRrec00004: "What's Really on the Voyager Golden Record",
};
const RUNS = [
  ["ask_v2", "What was the p99 write latency improvement in Kestrel DB 4.2?", "KSTRL42rel1"],
  ["ask_v2", "Who fixed the bug in the Tidewater storage engine introduced in Kestrel 4.2?", "KSTRLpm0002"],
  ["ask_v2", "What caused the telegraph sparks during the Carrington Event?", "CRRNGTN1859"],
  ["ask", "How many images are on the Voyager Golden Record?", "VGRrec00004"],
];

(async () => {
  const out = process.argv[2] || "docs/screenshots";
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 1 });
  const results = [];
  let n = { ask_v2: 0, ask: 0 };
  for (const [ui, q, vid] of RUNS) {
    await page.goto(`http://127.0.0.1:5001/${ui}`);
    await page.fill("#question", q);
    await page.click(ui === "ask_v2" ? "#deep-btn" : "#ask-btn");
    await page.waitForFunction(() => {
      const b = document.querySelector("#answer-box");
      return b && b.innerText.trim().length > 20;
    }, null, { timeout: 60000 });
    await page.waitForTimeout(500);
    const answer = await page.innerText("#answer-box");
    const sources = await page.innerText("#sources-list");
    const firstSource = sources.trim().split("\n")[0] || "";
    results.push({
      ui, question: q, expected: vid,
      first_source: firstSource.slice(0, 120),
      first_source_ok: firstSource.includes(TITLES[vid]),
      answer_cites_expected: answer.includes(`video_id=${vid}`),
    });
    n[ui] += 1;
    await page.screenshot({ path: path.join(out, `${ui === "ask" ? "ask_v1" : "ask_v2"}_q${n[ui]}.jpg`), type: "jpeg", quality: 80 });
  }
  await page.goto("http://127.0.0.1:5002/");
  await page.screenshot({ path: path.join(out, "library_index.jpg"), type: "jpeg", quality: 80 });
  await page.goto("http://127.0.0.1:5002/video/KSTRLpm0002");
  for (const sec of ["shorty", "questions", "entities"]) {
    await page.evaluate((s) => window.toggleSection && document.getElementById(`section-${s}`) && toggleSection(s), sec);
  }
  await page.screenshot({ path: path.join(out, "library_video.jpg"), type: "jpeg", quality: 80 });
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
  process.exit(results.every((r) => r.first_source_ok && r.answer_cites_expected) ? 0 : 1);
})();
