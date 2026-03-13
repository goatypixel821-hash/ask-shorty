-- Demo data for Ask Shorty: AI Supply Chain Attack Demo
-- Load with:
--   sqlite3 data/transcripts.db < demo_data.sql

BEGIN TRANSACTION;

-- 1) Video record
DELETE FROM videos WHERE video_id = 'DEMO123xyz';

INSERT INTO videos (
  video_id,
  title,
  channel,
  url,
  has_transcript,
  transcript_fetched_at,
  watch_date,
  json_metadata,
  created_at
)
VALUES (
  'DEMO123xyz',
  'AI Supply Chain Attack Demo',
  'Demo Security Research',
  'https://youtube.com/watch?v=DEMO123xyz',
  1,
  '2026-03-13 00:00:00',
  '2026-03-13 00:00:00',
  '{"upload_date":"20260220","shorty":"yes","questions":8,"entities":12}',
  '2026-03-13 00:00:00'
);

-- 2) Transcript + Shorty for the video
DELETE FROM transcripts WHERE video_id = 'DEMO123xyz';

INSERT INTO transcripts (
  video_id,
  text,
  language,
  confidence,
  shorty,
  shorty_generated_at,
  created_at
)
VALUES (
  'DEMO123xyz',
  'On December 21st, 2025, maintainers added a vulnerable AI triage workflow to their repository. This workflow used an AI agent to automatically respond to GitHub issues. The configuration allowed any GitHub user to trigger the workflow by opening an issue, and gave the AI agent arbitrary code execution permissions on the GitHub Actions runner. This was a textbook example of insufficient permission scoping. An attacker exploited this by crafting a GitHub issue containing malicious instructions that overrode the AI agent''s intended behavior. The attacker instructed the agent to install a package from a forked repository using a dangling commit technique. When the AI ran npm install, a malicious pre-install script executed automatically, deploying cache poisoning tools. The attacker then used GitHub Actions cache poisoning to pivot from the low-privilege triage workflow to the high-privilege nightly release workflow. By filling the cache with approximately 10 gigabytes of junk data, the attacker forced eviction of legitimate cache entries and poisoned the cache keys. On January 1st, 2026, a security advisory was submitted. On February 17th, 2026, an unauthorized version 2.3.0 was published containing malware. The maintainers quickly deprecated the affected version and published version 2.4.0 as a fix on the same day. This incident demonstrates the risks of giving AI agents excessive permissions in automated workflows.',
  'en',
  1.0,
  'HEADER\nTITLE – AI Supply Chain Attack Demo\nSOURCE: (youtube.com/watch?v=DEMO123xyz)\nCHANNEL: Demo Security Research\nDATE: 2026-02-20\n\nCONTEXT\nDemonstration of supply chain attack via misconfigured AI agent workflow on GitHub Actions.\n\nINCIDENTS\nSUPPLY CHAIN ATTACK – GitHub Actions Compromise\n- Actor: Unknown attacker\n- Target: Open source project with AI triage workflow\n- Method: Prompt injection via GitHub issues + cache poisoning\n- Result: Malicious version 2.3.0 published to users\n- Impact: Malware installed on user machines\n\nEVENT FLOW\n1. Dec 21, 2025: Vulnerable AI triage workflow added with excessive permissions\n2. Attacker crafted malicious GitHub issue with prompt injection\n3. AI agent executed npm install from attacker-controlled dangling commit\n4. Malicious pre-install script deployed cache poisoning tools\n5. Cache poisoning (~10GB junk data) forced LRU eviction\n6. Attacker pivoted to high-privilege nightly release workflow\n7. Jan 1, 2026: Security advisory submitted\n8. Feb 17, 2026: Unauthorized version 2.3.0 published\n9. Feb 17, 2026: Version 2.4.0 published as fix\n\nIMPACT/RISKS\n- Demonstrated AI agent permission risks\n- Supply chain attack vector via CI/CD pipelines\n- Importance of least-privilege principle for automation\n\nMICRO-DETAILS\nDates: 2025-12-21, 2026-01-01, 2026-02-17\nVersions: 2.3.0 (malicious), 2.4.0 (fixed)\nCache: ~10 gigabytes poisoning\nTools: cache poisoning, npm pre-install scripts\nTechniques: prompt injection, dangling commits, LRU eviction\n\nTIMELINE\n2025-12-21 – Vulnerable workflow added\n2026-01-01 – Security advisory submitted\n2026-02-17 – Malicious 2.3.0 published\n2026-02-17 – Fixed with 2.4.0\n',
  '2026-03-13 00:05:00',
  '2026-03-13 00:00:00'
);

-- 3) Synthetic questions (8)
DELETE FROM synthetic_questions WHERE video_id = 'DEMO123xyz';

INSERT INTO synthetic_questions (video_id, question)
VALUES
  ('DEMO123xyz', 'What date was the vulnerable workflow added?'),
  ('DEMO123xyz', 'Which versions were affected in the attack?'),
  ('DEMO123xyz', 'What cache poisoning technique was used?'),
  ('DEMO123xyz', 'How much cache storage did the attacker use?'),
  ('DEMO123xyz', 'What was the attack vector for initial compromise?'),
  ('DEMO123xyz', 'When was the security advisory submitted?'),
  ('DEMO123xyz', 'How did the attacker pivot to the release workflow?'),
  ('DEMO123xyz', 'What tools automated the cache poisoning?');

-- 4) Entities (12)
DELETE FROM entities WHERE video_id = 'DEMO123xyz';

INSERT INTO entities (video_id, name, type)
VALUES
  ('DEMO123xyz', 'GitHub Actions', 'platform'),
  ('DEMO123xyz', 'prompt injection', 'technique'),
  ('DEMO123xyz', 'cache poisoning', 'technique'),
  ('DEMO123xyz', 'supply chain attack', 'attack_type'),
  ('DEMO123xyz', 'AI agent', 'software'),
  ('DEMO123xyz', 'npm install', 'command'),
  ('DEMO123xyz', 'pre-install script', 'script'),
  ('DEMO123xyz', 'dangling commit', 'technique'),
  ('DEMO123xyz', 'LRU eviction', 'algorithm'),
  ('DEMO123xyz', 'version 2.3.0', 'version'),
  ('DEMO123xyz', 'version 2.4.0', 'version'),
  ('DEMO123xyz', 'December 21 2025', 'date');

COMMIT;

