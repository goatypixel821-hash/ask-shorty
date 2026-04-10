#!/usr/bin/env python3
"""
Export a course JSON to Markdown, standalone HTML, or copy JSON.

Outputs default under repo-root ``courses/`` (markdown + html subdirs).
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _nl_br(s: str) -> str:
    return "<br/>".join(_esc(line) for line in (s or "").split("\n"))


def export_markdown(course: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    title = course.get("title") or "Course"
    lines.append(f"# {title}")
    lines.append("")
    desc = (course.get("description") or "").strip()
    if desc:
        lines.append(f"> {desc}")
        lines.append("")
    pre = course.get("prerequisites") or []
    lines.append("## Prerequisites")
    if pre:
        for p in pre:
            lines.append(f"- {p}")
    else:
        lines.append("- _(None listed)_")
    lines.append("")

    for mod in course.get("modules") or []:
        mn = mod.get("number", 0)
        mt = mod.get("title") or "Module"
        lines.append(f"## Module {mn}: {mt}")
        lines.append("")
        objs = mod.get("learning_objectives") or []
        if objs:
            lines.append("### Learning Objectives")
            for o in objs:
                lines.append(f"- {o}")
            lines.append("")
        intro = (mod.get("introduction") or "").strip()
        if intro:
            lines.append(intro)
            lines.append("")
        for les in mod.get("lessons") or []:
            lid = f"{mn}.{les.get('number', 0)}"
            lt = les.get("title") or "Lesson"
            lines.append(f"### Lesson {lid}: {lt}")
            url = les.get("video_url") or ""
            ch = les.get("channel") or ""
            wd = les.get("watch_date") or ""
            lines.append(f"**Source:** [{ch}]({url}) | Watched: {wd or '—'}")
            lines.append("")
            kc = les.get("key_concepts") or []
            if kc:
                lines.append("**Key Concepts:** " + ", ".join(kc))
                lines.append("")
            lines.append("**Lesson Notes:**")
            lines.append("")
            lines.append((les.get("lesson_notes") or "").strip())
            lines.append("")
            dq = les.get("discussion_questions") or []
            if dq:
                lines.append("**Discussion Questions:**")
                for i, q in enumerate(dq, 1):
                    lines.append(f"{i}. {q}")
                lines.append("")
        lines.append("### Module Quiz")
        for i, q in enumerate(mod.get("quiz") or [], 1):
            lines.append(f"{i}. {q.get('question', '')}")
            lines.append(f"   **Answer:** {q.get('answer', '')}")
            lines.append("")
        summ = (mod.get("summary") or "").strip()
        if summ:
            lines.append("### Module Summary")
            lines.append(summ)
            lines.append("")

    lines.append("## Final Exam")
    for i, q in enumerate(course.get("final_exam") or [], 1):
        lines.append(f"{i}. {q.get('question', '')}")
        lines.append(f"   **Answer:** {q.get('answer', '')}")
        lines.append("")

    lines.append("## Glossary")
    for g in sorted(course.get("glossary") or [], key=lambda x: (x.get("term") or "").lower()):
        t = g.get("term") or ""
        d = g.get("definition") or ""
        lines.append(f"**{t}:** {d}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def export_html(course: Dict[str, Any], out_path: Path) -> Path:
    """Single-file HTML with TOC sidebar and print-friendly CSS."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slug = course.get("slug") or "course"
    title = _esc(course.get("title") or "Course")

    sections: List[str] = []

    sections.append('<div class="hero">')
    sections.append(f"<h1>{title}</h1>")
    if course.get("description"):
        sections.append(f'<p class="desc">{_esc(course["description"])}</p>')
    meta = []
    if course.get("level"):
        meta.append(f'<span class="badge">{_esc(course["level"])}</span>')
    if course.get("estimated_hours") is not None:
        meta.append(f'<span class="badge">{course["estimated_hours"]} h est.</span>')
    sections.append('<div class="meta">' + "".join(meta) + "</div>")
    pre = course.get("prerequisites") or []
    if pre:
        sections.append("<h3>Prerequisites</h3><ul>")
        for p in pre:
            sections.append(f"<li>{_esc(p)}</li>")
        sections.append("</ul>")
    sections.append("</div>")

    toc_items: List[str] = []

    for mod in course.get("modules") or []:
        mn = mod.get("number", 0)
        mid = f"module-{mn}"
        toc_items.append(f'<a href="#{mid}">Module {mn}: {_esc(mod.get("title") or "")}</a>')
        sections.append(f'<section class="module" id="{mid}">')
        mod_title = mod.get("title") or ""
        sections.append(f"<h2>{_esc(f'Module {mn}: {mod_title}')}</h2>")
        objs = mod.get("learning_objectives") or []
        if objs:
            sections.append("<h3>Learning objectives</h3><ul>")
            for o in objs:
                sections.append(f"<li>{_esc(o)}</li>")
            sections.append("</ul>")
        if mod.get("introduction"):
            sections.append(f'<div class="intro">{_nl_br(mod["introduction"])}</div>')

        for les in mod.get("lessons") or []:
            lid = f"lesson-{mn}-{les.get('number', 0)}"
            toc_items.append(
                f'<a class="toc-sub" href="#{lid}">{mn}.{les.get("number", 0)} {_esc(les.get("title") or "")}</a>'
            )
            sections.append(f'<article class="lesson" id="{lid}">')
            ln = les.get("number", 0)
            ltitle = les.get("title") or ""
            sections.append(f"<h4>{_esc(f'Lesson {mn}.{ln}: {ltitle}')}</h4>")
            url = les.get("video_url") or "#"
            sections.append(
                f'<p class="source"><a href="{_esc(url)}" target="_blank" rel="noopener">{_esc(les.get("channel") or "YouTube")}</a>'
                f' · Watched: {_esc(les.get("watch_date") or "—")} · {_esc(les.get("duration_estimate") or "")}</p>'
            )
            kc = les.get("key_concepts") or []
            if kc:
                sections.append(
                    '<p class="concepts"><strong>Key concepts:</strong> '
                    + ", ".join(_esc(k) for k in kc)
                    + "</p>"
                )
            sections.append('<details class="notes"><summary>Lesson notes</summary><div class="notes-body">')
            sections.append(f"<pre>{_esc(les.get('lesson_notes') or '')}</pre>")
            sections.append("</div></details>")
            dq = les.get("discussion_questions") or []
            if dq:
                sections.append("<h5>Discussion</h5><ol>")
                for q in dq:
                    sections.append(f"<li>{_esc(q)}</li>")
                sections.append("</ol>")
            sections.append(
                f'<label class="prog"><input type="checkbox" data-lesson="{_esc(lid)}"/> Mark lesson complete</label>'
            )
            sections.append("</article>")

        sections.append('<div class="quiz-block"><h3>Module quiz</h3>')
        for i, q in enumerate(mod.get("quiz") or [], 1):
            qid = f"q-{mn}-{i}"
            sections.append(f'<div class="quiz-q" id="{qid}"><p><strong>{i}.</strong> {_esc(q.get("question", ""))}</p>')
            sections.append(
                '<details><summary>Show answer</summary>'
                f'<p class="answer">{_esc(q.get("answer", ""))}</p></details></div>'
            )
        sections.append("</div>")
        if mod.get("summary"):
            sections.append(f'<div class="mod-summary"><h3>Summary</h3><p>{_nl_br(mod["summary"])}</p></div>')
        sections.append("</section>")

    sections.append('<section id="final-exam"><h2>Final exam</h2>')
    toc_items.append('<a href="#final-exam">Final exam</a>')
    for i, q in enumerate(course.get("final_exam") or [], 1):
        sections.append(f'<div class="quiz-q"><p><strong>{i}.</strong> {_esc(q.get("question", ""))}</p>')
        sections.append(
            '<details><summary>Show answer</summary>'
            f'<p class="answer">{_esc(q.get("answer", ""))}</p></details></div>'
        )
    sections.append("</section>")

    sections.append('<section id="glossary"><h2>Glossary</h2>')
    toc_items.append('<a href="#glossary">Glossary</a>')
    sections.append('<input type="search" id="gsearch" placeholder="Filter glossary…" />')
    sections.append('<dl id="glossary-list">')
    for g in sorted(course.get("glossary") or [], key=lambda x: (x.get("term") or "").lower()):
        sections.append(f"<dt>{_esc(g.get('term') or '')}</dt><dd>{_esc(g.get('definition') or '')}</dd>")
    sections.append("</dl></section>")

    toc_html = '<nav id="toc"><h2>Contents</h2>' + "".join(toc_items) + "</nav>"
    body = toc_html + '<main id="content">' + "\n".join(sections) + "</main>"

    css = """
:root { --bg:#0a0c12; --card:#111827; --border:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#3b82f6; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); display:flex; align-items:flex-start; gap:0; min-height:100vh; }
#toc { position:sticky; top:0; width:260px; max-height:100vh; overflow:auto; padding:20px 16px; border-right:1px solid var(--border); background:#0f1117; flex-shrink:0; }
#toc h2 { font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:0 0 12px; }
#toc a { display:block; color:var(--muted); text-decoration:none; font-size:13px; padding:4px 0; }
#toc a:hover { color:var(--text); }
#toc a.toc-sub { padding-left:12px; font-size:12px; }
#content { flex:1; padding:24px 32px 80px; max-width:900px; }
.hero { margin-bottom:32px; padding-bottom:20px; border-bottom:1px solid var(--border); }
.hero h1 { margin:0 0 12px; font-size:28px; }
.desc { color:var(--muted); line-height:1.6; }
.meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.badge { background:var(--card); border:1px solid var(--border); padding:2px 10px; border-radius:999px; font-size:12px; color:var(--muted); }
.module { margin-bottom:48px; }
.lesson { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px 18px; margin:16px 0; }
.lesson h4 { margin:0 0 8px; }
.source a { color:var(--accent); }
.concepts { font-size:13px; color:var(--muted); }
.notes { margin-top:12px; }
.notes-body pre { white-space:pre-wrap; font-family:inherit; font-size:13px; line-height:1.5; margin:8px 0 0; color:var(--muted); }
.quiz-block, .quiz-q { margin:12px 0; }
.quiz-q .answer { color:#a7f3d0; }
.mod-summary { margin-top:20px; padding:16px; background:#0f172a; border-radius:10px; border:1px solid var(--border); }
.prog { display:block; margin-top:12px; font-size:13px; color:var(--muted); cursor:pointer; }
#gsearch { width:100%; max-width:400px; margin-bottom:16px; padding:8px 12px; border-radius:8px; border:1px solid var(--border); background:var(--card); color:var(--text); }
dl dt { font-weight:600; margin-top:12px; }
dl dd { margin:4px 0 0 12px; color:var(--muted); }
@media print {
  body { background:#fff; color:#000; display:block; }
  #toc { display:none; }
  #content { max-width:none; padding:12px; }
  details { display:block !important; }
  details summary { display:none; }
  .prog { display:none; }
}
"""

    script = """
document.getElementById('gsearch')?.addEventListener('input', function() {
  var q = this.value.toLowerCase();
  document.querySelectorAll('#glossary-list dt').forEach(function(dt) {
    var t = dt.textContent.toLowerCase();
    var dd = dt.nextElementSibling;
    var show = !q || t.indexOf(q) >= 0;
    dt.style.display = show ? '' : 'none';
    if (dd) dd.style.display = show ? '' : 'none';
  });
});
"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{css}</style></head><body>
{body}
<script>{script}</script>
</body></html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def export_json_copy(course_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(course_path, out_path)
    return out_path


def load_course_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_export_paths(course_slug: str) -> Dict[str, Path]:
    base = ROOT / "courses"
    return {
        "markdown": base / "markdown" / f"{course_slug}.md",
        "html": base / "html" / f"{course_slug}.html",
        "json": base / "json" / f"{course_slug}.json",
    }


def export_all_formats(course: Dict[str, Any], course_json_path: Path) -> Dict[str, str]:
    slug = course.get("slug") or "course"
    paths = default_export_paths(slug)
    export_markdown(course, paths["markdown"])
    export_html(course, paths["html"])
    export_json_copy(course_json_path, paths["json"])
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("course_json", type=Path, help="Path to course .json")
    ap.add_argument("--md", type=Path, help="Output markdown path")
    ap.add_argument("--html", type=Path, help="Output html path")
    args = ap.parse_args()
    course = load_course_json(args.course_json)
    slug = course.get("slug") or "course"
    md_out = args.md or (ROOT / "courses" / "markdown" / f"{slug}.md")
    html_out = args.html or (ROOT / "courses" / "html" / f"{slug}.html")
    export_markdown(course, md_out)
    export_html(course, html_out)
    print(json.dumps({"markdown": str(md_out), "html": str(html_out)}, indent=2))


if __name__ == "__main__":
    main()
