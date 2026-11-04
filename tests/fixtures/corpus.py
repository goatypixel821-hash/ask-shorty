"""
Synthetic fixture corpus for the Ask Shorty test suite.

Seven short, invented "videos" on distinct topics. Every name, number and date
below is fictional and planted on purpose so tests can check that retrieval and
answers find the *right* video for a specific fact.

Planted facts worth knowing about when reading the tests:

* KSTRL42rel1  Kestrel DB 4.2 shipped 2023-03-14; lead maintainer Priya Raman;
               WAL fsync batching cut p99 write latency from 48 ms to 11 ms;
               introduced the "Tidewater" storage engine.
* KSTRLpm0002  (cross-video fact, second half) Tidewater had a page-split
               corruption bug, fixed in 4.2.3 on 2023-06-02 by Marcus Oyelaran.
               "Who fixed the bug in the engine introduced in 4.2?" needs both.
* SRDGH780003  Sourdough at 78% hydration; levain kept at 24 C; baker
               Tomasz Wierzbicki; causal chain hydration -> enzymes -> open crumb.
* VGRrec00004  Voyager Golden Record: 115 images, greetings in 55 languages,
               launched 1977-09-05, committee chaired by Carl Sagan.
* CRRNGTN1859  Carrington Event 1859-09-01; CME -> induced currents -> telegraph
               sparks; telegraph operators in Boston kept working on battery-less
               "auroral current".
* EMBSSYrs005  Embassy async framework on the RP2040; Rust 1.75 stabilised
               async fn in traits on 2023-12-28; firmware shrank to 34,899 bytes.
* WGGLdnc0007  Honeybee waggle dance decoded by Karl von Frisch, Nobel 1973;
               1 second of waggle ~ 750 metres of distance.
"""

from __future__ import annotations

from typing import Any, Dict, List

VIDEOS: List[Dict[str, Any]] = [
    {
        "video_id": "KSTRL42rel1",
        "title": "Kestrel DB 4.2 Release Deep Dive",
        "channel": "Byte Harbor",
        "url": "https://www.youtube.com/watch?v=KSTRL42rel1",
        "watch_date": "2024-01-10 20:15:00",
        "upload_date": "2023-03-20",
        "transcript": (
            "Welcome back to Byte Harbor. Today we are digging into Kestrel DB version 4.2, "
            "which shipped on March 14th 2023. I talked to the lead maintainer, Priya Raman, "
            "about why this release matters. The headline feature is a brand new storage "
            "engine called Tidewater. Tidewater replaces the old B-tree pager with a "
            "log-structured design, and Priya says the whole point was to stop fighting "
            "the disk. The second big change is in the write-ahead log. Before 4.2 every "
            "commit did its own fsync. Now commits are batched, so many transactions share "
            "one fsync call. Fewer fsync calls means fewer disk flushes, and fewer flushes "
            "means the tail latency drops. In their benchmark the p99 write latency went "
            "from 48 milliseconds down to 11 milliseconds on the same hardware. Priya was "
            "careful to say the median barely moved, it is really the tail that improved. "
            "They also bumped the default page size to 16 kilobytes. The migration tool, "
            "kestrel-migrate, converts old data files in place, but Priya recommends taking "
            "a backup first because Tidewater is still young. The release notes credit 37 "
            "contributors. If you run Kestrel in production, the advice is to wait for a "
            "point release before moving critical data onto Tidewater. That is it for "
            "today, thanks for watching Byte Harbor."
        ),
        "shorty": (
            "TITLE – Kestrel DB 4.2 Release Deep Dive\n"
            "SOURCE: (https://www.youtube.com/watch?v=KSTRL42rel1)\n"
            "CHANNEL: Byte Harbor\n"
            "DATE: 2023-03-20\n\n"
            "CONTEXT\n"
            "Walkthrough of the Kestrel DB 4.2 release with lead maintainer Priya Raman. "
            "Thesis: stop fighting the disk — new log-structured Tidewater engine plus "
            "batched WAL fsync to cut tail latency.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Tidewater storage engine\n"
            "- What it is: new log-structured storage engine introduced in Kestrel DB 4.2, "
            "replaces old B-tree pager\n"
            "- Who/what is involved: Priya Raman (lead maintainer), kestrel-migrate tool\n"
            "- Key details: default page size raised to 16 KB; in-place migration via kestrel-migrate\n"
            "- Outcome or conclusion: still young — back up first, wait for a point release for critical data\n"
            "TOPIC 2 – WAL fsync batching\n"
            "- What it is: commits batched so many transactions share one fsync\n"
            "- Key details: fewer fsync calls -> fewer disk flushes -> lower tail latency; "
            "p99 write latency 48 ms -> 11 ms; median barely moved\n"
            "- Outcome or conclusion: tail latency is the real win\n\n"
            "MICRO-DETAILS\n"
            "Dates: 2023-03-14 (4.2 shipped)\n"
            "Numbers: 48 ms, 11 ms, 16 KB, 37 contributors\n"
            "Tools: Kestrel DB, Tidewater, kestrel-migrate\n"
            "Versions: 4.2\n"
            "People: Priya Raman (lead maintainer)\n"
            "Organizations: Byte Harbor\n\n"
            "TIMELINE\n"
            "2023-03-14 – Kestrel DB 4.2 released with Tidewater engine\n"
        ),
        "questions": [
            "When was Kestrel DB 4.2 released?",
            "What storage engine did Kestrel DB 4.2 introduce?",
            "How much did WAL fsync batching reduce p99 latency in Kestrel?",
            "Who is the lead maintainer of Kestrel DB?",
        ],
        "entities": [
            {"name": "Kestrel DB", "type": "software", "aliases": ["Kestrel"]},
            {"name": "Tidewater", "type": "software", "aliases": ["Tidewater engine"]},
            {"name": "Priya Raman", "type": "person", "aliases": []},
            {"name": "kestrel-migrate", "type": "software", "aliases": []},
        ],
        "facts": [
            {"subject": "Kestrel DB 4.2", "relation": "introduced", "object": "Tidewater storage engine"},
            {"subject": "Priya Raman", "relation": "is lead maintainer of", "object": "Kestrel DB"},
            {"subject": "WAL fsync batching", "relation": "reduced", "object": "p99 write latency from 48 ms to 11 ms"},
            {"subject": "Kestrel DB 4.2", "relation": "released on", "object": "2023-03-14"},
        ],
        "segments": [
            (0, 40, "Kestrel DB 4.2 shipped March 14 2023; lead maintainer Priya Raman; new Tidewater log-structured storage engine replaces B-tree pager."),
            (40, 95, "Write-ahead log fsync batching: commits share one fsync, fewer disk flushes, p99 write latency fell from 48 ms to 11 ms."),
            (95, 150, "Default page size 16 KB; kestrel-migrate converts files in place; back up first; 37 contributors; wait for a point release."),
        ],
        "events": [
            {"title": "WAL fsync batching lowers tail latency", "cause": "commits batched into shared fsync calls",
             "effect": "fewer disk flushes, p99 write latency 48 ms to 11 ms", "systems": ["Kestrel DB"]},
        ],
    },
    {
        "video_id": "KSTRLpm0002",
        "title": "Postmortem: The Tidewater Corruption Bug",
        "channel": "Ops Diaries",
        "url": "https://www.youtube.com/watch?v=KSTRLpm0002",
        "watch_date": "2024-02-02 09:00:00",
        "upload_date": "2023-06-10",
        "transcript": (
            "This is Ops Diaries, and today is a postmortem. A few weeks after the new "
            "Tidewater storage engine landed, our on-call team started seeing checksum "
            "failures on a reporting replica. The root cause turned out to be a page-split "
            "race: when two writers split the same page at the same moment, one of them "
            "could write a stale sibling pointer. That stale pointer meant later reads "
            "followed the wrong page, and the checksum check caught the mismatch. Nothing "
            "was silently lost, but three replicas had to be rebuilt from backup. The fix "
            "came from a contributor named Marcus Oyelaran, who added a latch around the "
            "split path. It shipped in Kestrel version 4.2.3 on June 2nd 2023. Our lesson "
            "was simple: we moved to Tidewater too early. We had ignored the advice to wait "
            "for a point release. We now run a 14 day soak test on a shadow replica before "
            "any storage engine change reaches production. We also added an alert that "
            "fires when checksum failures exceed 5 in an hour. Total customer-facing "
            "downtime was 42 minutes. Thanks to Marcus and the maintainers for turning the "
            "fix around in under a week."
        ),
        "shorty": (
            "TITLE – Postmortem: The Tidewater Corruption Bug\n"
            "SOURCE: (https://www.youtube.com/watch?v=KSTRLpm0002)\n"
            "CHANNEL: Ops Diaries\n"
            "DATE: 2023-06-10\n\n"
            "CONTEXT\n"
            "Ops postmortem of checksum failures on replicas after adopting the Tidewater "
            "storage engine too early.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Tidewater page-split race\n"
            "- What it is: two writers splitting the same page could write a stale sibling pointer\n"
            "- Key details: stale pointer -> reads follow wrong page -> checksum mismatch detected; "
            "3 replicas rebuilt from backup; no silent data loss\n"
            "- Who/what is involved: Marcus Oyelaran (contributor) added a latch around the split path\n"
            "- Outcome or conclusion: fixed in Kestrel 4.2.3 on 2023-06-02\n"
            "TOPIC 2 – Process changes\n"
            "- Key details: 14 day soak test on shadow replica; alert when checksum failures exceed 5 per hour; "
            "42 minutes customer-facing downtime\n"
            "- Outcome or conclusion: adopted Tidewater too early, ignored wait-for-point-release advice\n\n"
            "MICRO-DETAILS\n"
            "Dates: 2023-06-02 (4.2.3 fix)\n"
            "Numbers: 3 replicas, 14 days, 5 per hour, 42 minutes\n"
            "Versions: 4.2.3\n"
            "People: Marcus Oyelaran (contributor)\n\n"
            "TIMELINE\n"
            "2023-06-02 – Kestrel 4.2.3 ships latch fix for Tidewater page-split race\n"
        ),
        "questions": [
            "What caused the Tidewater corruption bug?",
            "Who fixed the Tidewater page-split race?",
            "Which Kestrel version fixed the corruption bug?",
        ],
        "entities": [
            {"name": "Tidewater", "type": "software", "aliases": ["Tidewater engine"]},
            {"name": "Marcus Oyelaran", "type": "person", "aliases": []},
            {"name": "Kestrel DB", "type": "software", "aliases": ["Kestrel"]},
        ],
        "facts": [
            {"subject": "Marcus Oyelaran", "relation": "fixed", "object": "Tidewater page-split race"},
            {"subject": "Tidewater page-split race", "relation": "fixed in", "object": "Kestrel 4.2.3"},
            {"subject": "stale sibling pointer", "relation": "caused", "object": "checksum failures"},
        ],
        "segments": [
            (0, 50, "Checksum failures on a reporting replica after Tidewater storage engine adoption; page-split race wrote stale sibling pointer."),
            (50, 110, "Marcus Oyelaran added a latch around the split path; fix shipped in Kestrel 4.2.3 on June 2 2023."),
            (110, 160, "Lessons: adopted Tidewater too early; 14 day soak test; alert above 5 checksum failures per hour; 42 minutes downtime."),
        ],
        "events": [
            {"title": "Tidewater page-split race corrupts replicas", "cause": "two writers splitting the same page wrote a stale sibling pointer",
             "effect": "checksum failures, three replicas rebuilt from backup", "systems": ["Tidewater", "Kestrel DB"]},
        ],
    },
    {
        "video_id": "SRDGH780003",
        "title": "The Science of 78% Hydration Sourdough",
        "channel": "Crumb Lab",
        "url": "https://www.youtube.com/watch?v=SRDGH780003",
        "watch_date": "2023-11-05 18:30:00",
        "upload_date": "2023-10-01",
        "transcript": (
            "Hi, I'm Tomasz Wierzbicki and this is Crumb Lab. Today's loaf is a 78 percent "
            "hydration sourdough, which means 780 grams of water for every kilogram of flour. "
            "Why push hydration that high? More water makes the dough more mobile, and that "
            "lets the enzymes in the flour, amylase especially, break starch into sugar "
            "faster. More sugar feeds the yeast and bacteria, more gas is produced, and the "
            "wetter dough stretches around those bubbles instead of tearing. That is why "
            "high hydration gives you an open, glossy crumb. The catch is handling: wet dough "
            "is sticky, so I use four sets of coil folds, thirty minutes apart. My levain is "
            "kept at 24 degrees Celsius, because below about 20 degrees the bacteria dominate "
            "and the loaf gets sour before it gets airy. I use 2 percent salt and a 20 percent "
            "levain inoculation. Bulk fermentation runs about five hours at that temperature. "
            "Then a cold retard in the fridge overnight at 4 degrees. Bake in a Dutch oven at "
            "250 degrees for twenty minutes lid on, then twenty more lid off. Let it cool for "
            "a full hour or the crumb will gum up."
        ),
        "shorty": (
            "TITLE – The Science of 78% Hydration Sourdough\n"
            "SOURCE: (https://www.youtube.com/watch?v=SRDGH780003)\n"
            "CHANNEL: Crumb Lab\n"
            "DATE: 2023-10-01\n\n"
            "CONTEXT\n"
            "Baker Tomasz Wierzbicki explains why 78% hydration produces an open crumb.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Hydration and crumb\n"
            "- What it is: 78% hydration = 780 g water per 1 kg flour\n"
            "- Key details: more water -> more enzyme (amylase) activity -> more sugar -> more gas; "
            "wet dough stretches around bubbles -> open glossy crumb\n"
            "- Outcome or conclusion: trade-off is sticky handling; 4 sets of coil folds 30 min apart\n"
            "TOPIC 2 – Fermentation parameters\n"
            "- Key details: levain at 24 C (below ~20 C bacteria dominate -> sour); 2% salt; 20% levain; "
            "bulk ~5 h; overnight retard at 4 C; Dutch oven 250 C, 20 min lid on + 20 min lid off; cool 1 h\n\n"
            "MICRO-DETAILS\n"
            "Numbers: 78%, 780 g, 24 C, 20 C, 2%, 20%, 5 h, 4 C, 250 C\n"
            "People: Tomasz Wierzbicki (baker, Crumb Lab host)\n"
        ),
        "questions": [
            "Why does high hydration give an open crumb?",
            "What temperature does Tomasz keep his levain at?",
            "How much water is 78 percent hydration?",
        ],
        "entities": [
            {"name": "Tomasz Wierzbicki", "type": "person", "aliases": ["Tomasz"]},
            {"name": "amylase", "type": "concept", "aliases": []},
        ],
        "facts": [
            {"subject": "high hydration", "relation": "increases", "object": "enzyme activity"},
            {"subject": "Tomasz Wierzbicki", "relation": "keeps levain at", "object": "24 degrees Celsius"},
        ],
        "segments": [
            (0, 60, "78 percent hydration sourdough, 780 grams water per kilogram flour; more water boosts amylase enzyme activity, more gas, open crumb."),
            (60, 120, "Levain at 24 degrees Celsius; below 20 degrees bacteria dominate and the loaf gets sour; 2 percent salt, 20 percent levain."),
            (120, 180, "Coil folds, five hour bulk, overnight retard at 4 degrees, Dutch oven bake at 250 degrees, cool an hour."),
        ],
        "events": [],
    },
    {
        "video_id": "VGRrec00004",
        "title": "What's Really on the Voyager Golden Record",
        "channel": "Deep Orbit",
        "url": "https://www.youtube.com/watch?v=VGRrec00004",
        "watch_date": "2024-03-15 22:00:00",
        "upload_date": "2022-09-05",
        "transcript": (
            "On Deep Orbit today: the Voyager Golden Record. Voyager 1 launched on September "
            "5th 1977, and bolted to its side is a gold-plated copper phonograph record. The "
            "contents were chosen by a committee chaired by Carl Sagan, and they had roughly "
            "six weeks to decide what to send. The record holds 115 images encoded as analog "
            "signals, natural sounds like surf, wind and thunder, and spoken greetings in 55 "
            "languages. There are also 90 minutes of music from many cultures. The cover is "
            "engraved with instructions for playing it, written in a binary notation keyed to "
            "the hydrogen atom's transition time. The cover also carries a small sample of "
            "uranium-238, so that anyone finding it can work out how long ago it was made "
            "from the decay. Sagan liked to say the record was a bottle cast into the cosmic "
            "ocean. The odds of anyone ever playing it are tiny; the real audience, he "
            "argued, was us. Voyager 1 crossed into interstellar space in 2012 and is now "
            "more than 24 billion kilometres from Earth."
        ),
        "shorty": (
            "TITLE – What's Really on the Voyager Golden Record\n"
            "SOURCE: (https://www.youtube.com/watch?v=VGRrec00004)\n"
            "CHANNEL: Deep Orbit\n"
            "DATE: 2022-09-05\n\n"
            "CONTEXT\n"
            "Contents and purpose of the gold-plated copper record on Voyager 1. Thesis "
            "(Carl Sagan): a bottle cast into the cosmic ocean; the real audience is us.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Record contents\n"
            "- Who/what is involved: committee chaired by Carl Sagan, ~6 weeks to choose\n"
            "- Key details: 115 images (analog), natural sounds (surf, wind, thunder), greetings in "
            "55 languages, 90 minutes of music\n"
            "TOPIC 2 – Cover engraving\n"
            "- Key details: playback instructions in binary keyed to hydrogen transition time; "
            "uranium-238 sample lets finders date it via decay\n\n"
            "MICRO-DETAILS\n"
            "Dates: 1977-09-05 (Voyager 1 launch); 2012 (interstellar space)\n"
            "Numbers: 115 images, 55 languages, 90 minutes, 24 billion km\n"
            "People: Carl Sagan (committee chair)\n\n"
            "TIMELINE\n"
            "1977-09-05 – Voyager 1 launches with Golden Record\n"
            "2012 – Voyager 1 enters interstellar space\n"
        ),
        "questions": [
            "How many images are on the Voyager Golden Record?",
            "Who chaired the Golden Record committee?",
            "When did Voyager 1 launch?",
        ],
        "entities": [
            {"name": "Voyager 1", "type": "system", "aliases": ["Voyager"]},
            {"name": "Carl Sagan", "type": "person", "aliases": ["Sagan"]},
            {"name": "Golden Record", "type": "system", "aliases": []},
        ],
        "facts": [
            {"subject": "Carl Sagan", "relation": "chaired", "object": "Golden Record committee"},
            {"subject": "Golden Record", "relation": "contains", "object": "115 images"},
            {"subject": "Voyager 1", "relation": "launched on", "object": "1977-09-05"},
        ],
        "segments": [
            (0, 45, "Voyager 1 launched September 5 1977 carrying a gold-plated copper record chosen by a committee chaired by Carl Sagan."),
            (45, 100, "Golden Record holds 115 images, natural sounds, greetings in 55 languages and 90 minutes of music."),
            (100, 150, "Cover engraving: binary playback instructions keyed to hydrogen; uranium-238 sample for dating; interstellar space in 2012."),
        ],
        "events": [],
    },
    {
        "video_id": "CRRNGTN1859",
        "title": "The Carrington Event: When the Sun Hit the Telegraph",
        "channel": "Deep Orbit",
        "url": "https://www.youtube.com/watch?v=CRRNGTN1859",
        "watch_date": "2024-03-16 21:40:00",
        "upload_date": "2021-09-01",
        "transcript": (
            "Back on Deep Orbit. On September 1st 1859 the English astronomer Richard "
            "Carrington was sketching sunspots when he saw two bright beads of white light "
            "flare up on the Sun. About 17 hours later the Earth was hit by the biggest "
            "geomagnetic storm on record. Here is the chain of cause and effect. The flare "
            "came with a coronal mass ejection, a huge cloud of magnetised plasma. When it "
            "slammed into Earth's magnetic field it made the field swing violently. A "
            "changing magnetic field induces electric currents in long conductors, and the "
            "longest conductors in 1859 were telegraph wires. Those induced currents were "
            "strong enough to throw sparks from telegraph keys, and some operators reported "
            "paper catching fire. In one famous exchange, operators between Boston and "
            "Portland disconnected their batteries entirely and kept sending messages for "
            "two hours on the auroral current alone. Auroras were seen as far south as the "
            "Caribbean. A storm that size today would hit power grids and satellites, and "
            "estimates of the damage run to 2 trillion dollars."
        ),
        "shorty": (
            "TITLE – The Carrington Event: When the Sun Hit the Telegraph\n"
            "SOURCE: (https://www.youtube.com/watch?v=CRRNGTN1859)\n"
            "CHANNEL: Deep Orbit\n"
            "DATE: 2021-09-01\n\n"
            "CONTEXT\n"
            "The 1859 Carrington Event geomagnetic storm and its effect on telegraphs.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Causal chain\n"
            "- What it is: solar flare seen by Richard Carrington on 1859-09-01\n"
            "- Key details: flare + coronal mass ejection -> Earth's magnetic field swings -> "
            "induced currents in long conductors (telegraph wires) -> sparks from keys, paper fires\n"
            "- Outcome or conclusion: storm arrived ~17 hours after flare; largest on record\n"
            "TOPIC 2 – Auroral current anecdote\n"
            "- Key details: Boston–Portland operators disconnected batteries and sent messages for "
            "2 hours on auroral current alone; auroras seen as far south as the Caribbean\n"
            "TOPIC 3 – Modern risk\n"
            "- Key details: today would hit power grids and satellites; estimates ~$2 trillion\n\n"
            "MICRO-DETAILS\n"
            "Dates: 1859-09-01\n"
            "Numbers: 17 hours, 2 hours, $2 trillion\n"
            "People: Richard Carrington (English astronomer)\n"
        ),
        "questions": [
            "What did the Carrington Event do to telegraphs?",
            "How long after the flare did the storm hit Earth?",
        ],
        "entities": [
            {"name": "Richard Carrington", "type": "person", "aliases": ["Carrington"]},
            {"name": "coronal mass ejection", "type": "concept", "aliases": ["CME"]},
        ],
        "facts": [
            {"subject": "coronal mass ejection", "relation": "caused", "object": "induced currents in telegraph wires"},
            {"subject": "induced currents", "relation": "caused", "object": "sparks from telegraph keys"},
            {"subject": "Richard Carrington", "relation": "observed", "object": "1859 solar flare"},
        ],
        "segments": [
            (0, 50, "Richard Carrington saw a white light solar flare on September 1 1859; 17 hours later the biggest geomagnetic storm on record."),
            (50, 110, "Coronal mass ejection swung Earth's magnetic field, inducing currents in telegraph wires that threw sparks and set paper on fire."),
            (110, 160, "Boston and Portland operators disconnected batteries and sent messages on auroral current; modern damage estimate 2 trillion dollars."),
        ],
        "events": [
            {"title": "Carrington geomagnetic storm sparks telegraphs", "cause": "coronal mass ejection hit Earth's magnetic field",
             "effect": "induced currents in telegraph wires, sparks and fires", "systems": ["telegraph"]},
        ],
    },
    {
        "video_id": "EMBSSYrs005",
        "title": "Async Rust on a $4 Chip with Embassy",
        "channel": "Byte Harbor",
        "url": "https://www.youtube.com/watch?v=EMBSSYrs005",
        "watch_date": "2024-04-01 19:05:00",
        "upload_date": "2024-01-15",
        "transcript": (
            "Byte Harbor again, and today we are writing async Rust for a four dollar "
            "microcontroller, the RP2040. The framework is called Embassy. Instead of an "
            "RTOS with threads and stacks, Embassy runs async tasks on a single executor, "
            "and the compiler turns each task into a state machine. Because there is no "
            "per-thread stack, memory use is predictable. What made this much nicer is Rust "
            "1.75, which stabilised async fn in traits on December 28th 2023. That means "
            "driver crates can expose async traits directly, without boxing. I ported my "
            "old blinky-plus-sensor project from an RTOS to Embassy and the firmware went "
            "from 61,204 bytes down to 34,899 bytes. Power draw while idle dropped too, "
            "because the executor puts the core to sleep with a wait-for-interrupt when no "
            "task is ready. The sensor is a BME280 on I2C, polled every 500 milliseconds. "
            "My one complaint is compile times, a clean build takes about 90 seconds on my "
            "laptop. Still, for small devices I am not going back to threads."
        ),
        "shorty": (
            "TITLE – Async Rust on a $4 Chip with Embassy\n"
            "SOURCE: (https://www.youtube.com/watch?v=EMBSSYrs005)\n"
            "CHANNEL: Byte Harbor\n"
            "DATE: 2024-01-15\n\n"
            "CONTEXT\n"
            "Porting an RP2040 project from an RTOS to the Embassy async framework.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Embassy executor model\n"
            "- What it is: async tasks on a single executor, compiled to state machines, no per-thread stacks\n"
            "- Key details: wait-for-interrupt sleep when idle -> lower idle power\n"
            "TOPIC 2 – Rust 1.75 async fn in traits\n"
            "- Key details: stabilised 2023-12-28; driver crates expose async traits without boxing\n"
            "TOPIC 3 – Port results\n"
            "- Key details: firmware 61,204 bytes -> 34,899 bytes; BME280 on I2C polled every 500 ms; "
            "clean build ~90 s\n\n"
            "MICRO-DETAILS\n"
            "Dates: 2023-12-28\n"
            "Numbers: $4, 61,204 bytes, 34,899 bytes, 500 ms, 90 s\n"
            "Tools: Embassy, RP2040, BME280\n"
            "Versions: Rust 1.75\n"
        ),
        "questions": [
            "Which Rust version stabilised async fn in traits?",
            "How small did the Embassy firmware get?",
        ],
        "entities": [
            {"name": "Embassy", "type": "software", "aliases": []},
            {"name": "RP2040", "type": "system", "aliases": []},
            {"name": "Rust", "type": "software", "aliases": ["Rust 1.75"]},
        ],
        "facts": [
            {"subject": "Rust 1.75", "relation": "stabilised", "object": "async fn in traits"},
            {"subject": "Embassy port", "relation": "reduced firmware to", "object": "34,899 bytes"},
        ],
        "segments": [
            (0, 50, "Embassy async Rust framework on the four dollar RP2040 microcontroller; single executor, tasks compiled to state machines."),
            (50, 100, "Rust 1.75 stabilised async fn in traits on December 28 2023, so drivers expose async traits without boxing."),
            (100, 150, "Firmware shrank from 61,204 to 34,899 bytes; BME280 sensor on I2C every 500 ms; 90 second clean builds."),
        ],
        "events": [],
    },
    {
        "video_id": "WGGLdnc0007",
        "title": "Decoding the Honeybee Waggle Dance",
        "channel": "Field Notes",
        "url": "https://www.youtube.com/watch?v=WGGLdnc0007",
        "watch_date": "2023-08-20 07:45:00",
        "upload_date": "2023-05-12",
        "transcript": (
            "Welcome to Field Notes. Inside a dark hive, a forager returns and starts to "
            "dance. This is the waggle dance, decoded by the Austrian scientist Karl von "
            "Frisch, who shared the Nobel Prize in 1973 for this work. The dance is a figure "
            "of eight with a straight waggle run in the middle. The angle of that run, "
            "measured from vertical, matches the angle between the food source and the Sun. "
            "The duration of the waggle encodes distance: in the colonies we filmed, about "
            "one second of waggling meant roughly 750 metres. Other bees follow the dancer, "
            "pick up the scent of the flowers, and fly out. Because the Sun moves across the "
            "sky, the dancers slowly rotate their angle through the day to compensate, even "
            "without seeing the Sun. Our hive had about 40,000 bees in midsummer, and on a "
            "good day we counted over 200 dances an hour on a single frame. It is a language "
            "with grammar, and it took a patient biologist to read it."
        ),
        "shorty": (
            "TITLE – Decoding the Honeybee Waggle Dance\n"
            "SOURCE: (https://www.youtube.com/watch?v=WGGLdnc0007)\n"
            "CHANNEL: Field Notes\n"
            "DATE: 2023-05-12\n\n"
            "CONTEXT\n"
            "How the honeybee waggle dance encodes direction and distance.\n\n"
            "TOPICS\n"
            "TOPIC 1 – Dance encoding\n"
            "- Who/what is involved: Karl von Frisch (Austrian scientist, Nobel 1973)\n"
            "- Key details: figure-of-eight; waggle-run angle from vertical = angle between food and Sun; "
            "~1 s of waggle ~ 750 m distance\n"
            "- Outcome or conclusion: dancers rotate angle through the day to track the Sun\n\n"
            "MICRO-DETAILS\n"
            "Dates: 1973 (Nobel)\n"
            "Numbers: 750 m, 40,000 bees, 200 dances per hour\n"
            "People: Karl von Frisch\n"
        ),
        "questions": [
            "What does the waggle dance angle mean?",
            "Who decoded the waggle dance?",
        ],
        "entities": [
            {"name": "Karl von Frisch", "type": "person", "aliases": ["von Frisch"]},
            {"name": "waggle dance", "type": "concept", "aliases": []},
        ],
        "facts": [
            {"subject": "Karl von Frisch", "relation": "decoded", "object": "waggle dance"},
            {"subject": "waggle duration", "relation": "encodes", "object": "distance to food"},
        ],
        "segments": [
            (0, 60, "Karl von Frisch decoded the honeybee waggle dance and shared the Nobel Prize in 1973."),
            (60, 120, "Waggle run angle from vertical equals angle between food and Sun; one second of waggle means about 750 metres."),
            (120, 170, "Dancers rotate the angle through the day; hive of 40,000 bees; over 200 dances an hour."),
        ],
        "events": [],
    },
]


def video_by_id(video_id: str) -> Dict[str, Any]:
    for v in VIDEOS:
        if v["video_id"] == video_id:
            return v
    raise KeyError(video_id)


# Question -> the fixture video(s) that must be cited. Used by tests and the UI run.
EXPECTED_SOURCES: Dict[str, List[str]] = {
    "What was the p99 write latency improvement in Kestrel DB 4.2?": ["KSTRL42rel1"],
    "Who fixed the Tidewater page-split corruption bug?": ["KSTRLpm0002"],
    "Why does 78% hydration sourdough get an open crumb?": ["SRDGH780003"],
    "How many images are on the Voyager Golden Record?": ["VGRrec00004"],
    "What caused the telegraph sparks during the Carrington Event?": ["CRRNGTN1859"],
}
