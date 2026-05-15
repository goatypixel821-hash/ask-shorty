#!/usr/bin/env python3
"""
Insert a manually obtained transcript into the database.
Strips timestamps and chapter headers to match YouTube API format.
"""
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from transcript_database import TranscriptDatabase

DB_PATH = r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db"
VIDEO_ID = "cs6QY20JYB0"
VIDEO_TITLE = "Frontier A321 Hits Person on Runway During Takeoff | Captain Steeeve"
VIDEO_CHANNEL = "Captain Steeeve"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"

RAW_TRANSCRIPT = """0:044 secondsWe begin tonight with the search for answers after another scary moment at an American airport. This time, Denver.
0:1111 secondsTonight, the NTSB is investigating after a person was hit and killed by a Frontier Airlines plane on the runway.
0:1818 secondsTragedy strikes at Denver International Airport over the weekend. A man wandered onto the runway and was struck by an
0:2525 secondsairplane on its takeoff roll. Uh we this leaves us with more questions really than answers. Uh the questions that
0:3333 secondswe're going to try to answer in this video and it's again try to answer is how could something like this have happened? Uh could it have been
0:4040 secondsprevented? I don't know. U has it happened before? The answer will be surprising to you. Um could the pilots have avoided this tragedy? And again
0:4848 secondswe'll give you a whole rundown on could they have? Um it's doubtful but you know maybe. And then the answer to try to answer the question why why would
0:5757 secondssomething like this take place? Here's what we know so far. DIA airport, Denver International Airport to LAX Frontier uh
1:051 minute, 5 seconds4345 is rolling down runway 17 left just after clear for takeoff at 11:30 p.m.
1:131 minute, 13 secondsIt's dark in uh Colorado at that time and uh they're got all their lights on.
1:191 minute, 19 secondsThey're taking off down the runway and a few seconds into their takeoff roll uh they encounter a human being on the
1:281 minute, 28 secondsrunway and I'm going to show you some video uh that's going to describe the normal departure here and then we're going to stop it and I'm going to show
1:361 minute, 36 secondsyou some video and just a trigger warning my friends. There are some disturbing videos um that uh may not be suitable for all viewers but we'll talk about that more here in just a minute.
1:461 minute, 46 secondsHere's the ATC as Frontier 4345 is cleared for takeoff.
1:511 minute, 51 secondsFrontier flight 4345 Denver RNF to go rock 7 left. Clear for takeoff. Go Rock 17 left clear for takeoff. 4345.
2:012 minutes, 1 secondAll right. They turn all their lights on. They take the runway. They start their takeoff roll. And as they take
2:082 minutes, 8 secondstheir take start their takeoff roll, they get to a certain point. And we're going to stop it right there. It takes
2:142 minutes, 14 secondsprobably about uh 15 to 20 seconds, I think, maybe even a little bit longer than that. More like 30 to 40 for them
2:222 minutes, 22 secondsto get up to speed. Uh and then somewhere just before the V1 speed, that go no-go speed, and they're going pretty
2:292 minutes, 29 secondsfast, probably 120 mph at this point, they encounter a person on the runway.
2:352 minutes, 35 secondsNow, I've encountered deer. I've encountered, you know, little foxes, rabbits a lot of times on the runway.
2:422 minutes, 42 secondsSometimes you go right over them and miss them, but a human being would be a shocking thing to view. I'm going to show you next a video now uh taken from
2:502 minutes, 50 secondsoutside the aircraft. It's this is infrared technology.
2:552 minutes, 55 secondsYou're going to see a human being walking onto the runway. Now, I just want to warn you, this might not be suitable for all viewers. I also want to
3:033 minutes, 3 secondssay this. Um, if you're dealing with thoughts of self harm, uh,
3:103 minutes, 10 secondsand on any level, uh, there's an 800 number for you to call in the description below. We want you to take
3:173 minutes, 17 secondsadvantage of that. Uh, this is not going to be suitable for all viewers, but we want you to reach out and there's somebody there that wants to talk to
3:243 minutes, 24 secondsyou. And so, uh, please dial that 800 number if you have any thoughts of self harm, uh, at all. And uh I also want to
3:323 minutes, 32 secondssay to the pilots, this must have been just horrible in the extreme because there there's it's called a flashbulb memory. There's actually a name for it.
3:413 minutes, 41 secondsAnd when something tragic happens in front of your eyes, your brain takes a snapshot of it. And I'm sure that both of the pilots have that snapshot memory
3:493 minutes, 49 secondscuz they were very specific about this was a human being that they saw in front of them. If either one of you guys uh wants to talk to somebody, I'm a trained
3:573 minutes, 57 secondslicensed counselor. I've been counseling for over 30 years. I've counseledled thousands of people. I've lived in the world that you live in. Um there is a email in the description down below.
4:074 minutes, 7 secondsPlease reach out to me and any of the passengers from that flight. Same thing.
4:114 minutes, 11 secondsNo, free of charge. Reach out to me. I would love to start a conversation with you. All right, let's watch this next uh
4:184 minutes, 18 secondsvideo now. As this is the uh infrared from outside the aircraft, we're going to stop it just before the point of impact. You can see right here, I'm
4:264 minutes, 26 secondsgoing to freeze it. There's an individual right here on the runway.
4:314 minutes, 31 secondsThat's runway 17 left. They've just hopped the fence. Couple minutes later, they're on the runway and the airplane
4:384 minutes, 38 secondsis coming and we'll stop the video right there.
4:484 minutes, 48 secondsUm, this is kind of hard to to to watch.
4:514 minutes, 51 secondsIt looks like the person is just kind of strolling and uh it's just not possible that they didn't know that they were on a runway or that an airplane was coming.
5:015 minutes, 1 secondI mean, anything's a possibility, but again, we'll talk at the end about the the why of all of this. Um, but it's
5:085 minutes, 8 secondsjust a little difficult to uh fathom that they were out for a stroll around midnight. Uh there it's very dark out
5:175 minutes, 17 secondsthere. Um it it's it's a very rural area. We'll talk about how big uh the airport here is in just a minute. Uh but
5:255 minutes, 25 secondsit's just not conceivable that they wandered onto the runway and didn't see an airplane coming. All those lights and everything, it's just kind of difficult to fathom. But that was the moment from
5:345 minutes, 34 secondsoutside the airplane. Now, I'm going to show you the same moment from inside the airplane. Same trigger warning on this, folks, because uh even though we don't
5:415 minutes, 41 secondssee anything specific, you're going to know the time of impact based on this video that a passenger was taking from inside the airplane. So, there they are
5:495 minutes, 49 secondson the takeoff roll. You can see how fast they were going right there just prior to V1.
5:565 minutes, 56 secondsAnd here it comes.
6:026 minutes, 2 secondsOkay, we'll stop the video there. Um, you can even hear it a little bit. And that's the the camera falls to the floor. Uh,
6:116 minutes, 11 secondsthey were going real fast. I mean, there was no way for the pilots to stop for sure. Uh, could they have swerved? We'll talk about that in a minute. Uh, I don't
6:196 minutes, 19 secondsthink so. But, uh, we'll get to all of that in in due time. All right. Now, what happened after they, uh, they hit?
6:276 minutes, 27 secondsWell, uh, let's go back now to the audio on the airplane. You can hear the pilots are extremely disturbed. You can hear
6:356 minutes, 35 secondsthat the air traffic controller, the tower controller is extremely disturbed.
6:396 minutes, 39 secondsIt's very clear what happened. And is there were some early news reports of maybe they hit a deer, maybe they hit, you know, some sort of animal on the runway. No, you can tell from their
6:486 minutes, 48 secondsvoice that they saw a human being and how disturbing um all of that was. So, let me uh let me run now. This is the
6:566 minutes, 56 secondsair traffic control audio from the tower to the airplane just as they hit and they strike this this person.
7:067 minutes, 6 secondsFrontier uh tower pressure 4345. We're stopping on the runway.
7:167 minutes, 16 secondsOkay. So, they you can hear the the panic in his voice. He he keys up the mic at first and just kind of ah you
7:247 minutes, 24 secondsknow like he doesn't even know what to say and then he says hey we're we're he gives out he gets his call sign out right. You got to have the presence of
7:317 minutes, 31 secondsmind to get your call sign out. At the same time they're stopping. They're rejecting on the runway. This is you know we practice this procedure all the time in the simulator. You know, every
7:407 minutes, 40 secondsyear you go down to training, you practice one reject after another, but doing it for real after what just happened that your your brain is in a
7:487 minutes, 48 secondsdifferent place. That's why the training is so important because you will you'll operate the way you train. So, this is a
7:557 minutes, 55 secondsmuscle reflex almost at this point. Uh they at this point they're just like, "Okay, we've got an engine fire. We just
8:028 minutes, 2 secondshit something. We know what that something was, which is shocking in the extreme." your brain would just want to check out. But at that point, the
8:098 minutes, 9 secondstraining clicks in and the captain brings it into reverse. He slams on the brakes, gets the airplane stopped. At at
8:168 minutes, 16 secondsthe same time, he's got to key up the PA before he stops and say to the passengers, "Remain seated. Remain seated. Remain seated." That's part of
8:258 minutes, 25 secondshis training because there may be a little bit of a panic in the back of the airplane. You're at that speed when you stop. People are like, "Why did we stop?" And they can see that the right
8:338 minutes, 33 secondsengine is on fire. So, all of that being said, incredible presence of mind on the part of this crew, but you can hear how
8:428 minutes, 42 secondsmuch distress they're in uh as he says, "We just hit somebody and we have an engine fire." Listen.
8:478 minutes, 47 secondsUh there we just hit somebody. We have an engine fire. 4345. I see that.
8:548 minutes, 54 secondsSouthwest Frontier 43 45. I'm going to be rolling the trucks now. Do you can you you know the souls on board and uh uh fuel remaining?
9:029 minutes, 2 secondsOkay. Can you hear the the tower controller? The tower controller is like shocked. He's having a hard time getting out of words where it's your brain does
9:109 minutes, 10 secondssomething at that moment where it kind of goes into a little bit of a dream state where like I can't I can't process what I'm hearing right now. And he's
9:189 minutes, 18 secondshe's again he kicks into his training and the first thing out of his mouth is souls on board and fuel state, right?
9:249 minutes, 24 secondsBecause that's what he's been trained to do. Ask fuels on fuel on board and souls on board. That kind of sort of thing. So he gets that part out.
9:329 minutes, 32 secondsGood for him. Uh, but again, it's it's difficult for everybody because they're trying to take in the depth of what just took place. Here we go. There's more.
9:429 minutes, 42 secondsAll right. 43 45. Uh, we have 231 souls on board. We have two uh 21,320
9:499 minutes, 49 secondslbs of fuel on board. There was an individual walking across the runway.
9:549 minutes, 54 secondsOkay. Um, we are rolling the trucks down. Individual walking across the runway. Wow. I can't imagine that.
10:0310 minutes, 3 secondsOp running 17 left. Um 231 souls. We're rolling the trucks now.
10:0910 minutes, 9 secondsThis guy's almost in tears. I mean, this is awful. We've got uh smoke in the aircraft.
10:1610 minutes, 16 secondsWe're going to evacuate on the runway. Okay, there's a different voice now.
10:2010 minutes, 20 secondsThat's most likely the captain. Uh because the captain doesn't have time to relay it through the first officer, but they've brought the airplane to a stop.
10:2710 minutes, 27 secondsuh they've got to make sure that that they don't cause a secondary problem. Uh the brakes can overheat and and they can lit light on fire as well. So they're
10:3610 minutes, 36 secondsassessing everything. They're asking for checklist. They're both going through the gyrations of what did we just see happen in front of us. Uh at that point
10:4310 minutes, 43 secondsthey they kick into more training checklist. Uh and they've got they're trying to put out the fire in the engine. That's checklist number one. But
10:5110 minutes, 51 secondsthat's engine fire on the ground checklist, not engine fire in the air checklist. It's a different checklist.
10:5610 minutes, 56 secondsIn addition to that, u they're they get a call from the back of the cabin. First of all, the first call was probably, "Hey, flight, what happened?" And
11:0411 minutes, 4 secondsthey're trying to answer that while they're stopping the airplane. The second one is, "Hey, we got smoke back here." All right, great. Uh now
11:1111 minutes, 11 secondseverything shifts again, you know, as though that wasn't enough. So now they have to shift into getting everybody off the aircraft. and you want to get
11:1911 minutes, 19 secondseverybody off as safely as you possibly can because you don't know how bad the smoke is in the back of the airplane.
11:2511 minutes, 25 secondsSo, this crew has a tremendous amount going on. It's amazing to me that they didn't overload. They didn't just lock up, but they didn't. They kept moving on. This is a great crew.
11:3711 minutes, 37 secondsOkay. Uh we're we're having ops and emergency vehicles going all the way now. OP seven, uh they're about halfway down the runway. 17 left. Uh they're
11:4511 minutes, 45 secondsevacuating uh the runway uh the aircraft on the runway.
11:4911 minutes, 49 secondsGuy's voice is so shaky. Okay. And that's it. So that's the last we hear from the aircraft. That's the last we hear from tower. Ops gets out there. Um
11:5711 minutes, 57 secondsthere's people evacuating now. They open the slides. A few people were injured.
12:0112 minutes, 1 secondAnd I always say on these videos, there's that debate of whether you want to blow the slides and send people down the slides because people will twist ankles and so forth. Um there may have
12:1012 minutes, 10 secondsbeen some smoke inhilation uh inside the aircraft. Let me show you next uh the kind of the commotion on the inside of
12:1712 minutes, 17 secondsthe airplane, the amount of smoke and everything else. Here we go. You can see smoke in the air.
12:2612 minutes, 26 secondsPeople are sadly they're grabbing their stuff out of the overhead. Why? Leave it. Leave it. Here's a slide.
12:3312 minutes, 33 secondsPeople, you know, kind of gawking at the engine. Everybody's telling them to get away.
12:3712 minutes, 37 secondsGet away from the aircraft. Get away from the aircraft. and people coming down the slide. So, some of those people twisted ankles and so forth. I think a
12:4412 minutes, 44 secondsfew people went to the hospital um right after this. But you can see that's what the evacuation part looks like. And all of this happened within I don't know
12:5212 minutes, 52 secondsless than 60 seconds from striking the person on the runway to rejecting the takeoff to dealing with the burning engine to put the fire out to call and
13:0013 minutesand say we're rejecting to to making the decision to evacuate. Then going through the evacuation checklist, the flight attendants pop the doors, pop the
13:0813 minutes, 8 secondsslides. They get everybody off the aircraft.
13:1113 minutes, 11 secondsSadly, people stop and take time to grab their luggage. Folks, don't do that. Get off the aircraft. You can go back and get your stuff um later on. Now, let's
13:1913 minutes, 19 secondsanswer those those questions I asked in the beginning, and we'll do them in reverse order. Could the pilots have avoided this man on the runway?
13:2813 minutes, 28 secondsWell, let me show you a video of what it looks like on a night takeoff from the cockpit. And there's lots of lights in
13:3613 minutes, 36 secondsfront of the airplane, but the visibility is pretty limited. And I'll show you exactly how fast they were going when they encountered a human being on the runway. You don't really
13:4513 minutes, 45 secondsswerve an airplane like you could swerve a car, maybe airplanes much bigger. And those engines are they're sucking in as
13:5313 minutes, 53 secondsmuch air as they possibly can on that takeoff roll.
13:5713 minutes, 57 secondsbecause they're up at max power. So, anything anywhere near those engines is going to get ingested into the engine.
14:0414 minutes, 4 secondsSo, it would be virtually impossible to swerve enough to avoid a human being.
14:0814 minutes, 8 secondsBut, let me show you what a takeoff roll at night looks like. All right, here we go.
14:1514 minutes, 15 secondsPower gets applied. The airplane starts to roll slowly.
14:1914 minutes, 19 secondsThe first The first call is going to be 80, which means 80 knots. The other pilot's going
14:2614 minutes, 26 secondsto say, "Check." 80 knots takes place right about this speed here. All right. Now, they're
14:3414 minutes, 34 secondsaccelerating. 85, 90, 95, 100, 105, 110.
14:4314 minutes, 43 secondsAll right. They're getting up to they're not at rotate speed just yet, but right about here, and I'll stop it. Right
14:5114 minutes, 51 secondsabout that speed is where they encounter a person walking on the runway.
14:5714 minutes, 57 secondsAnd at that point going 120, 125 miles an hour, just prior to rotate speed, there's no stopping that airplane. And it would be extremely dangerous to try
15:0515 minutes, 5 secondsto swerve the airplane. You could a little bit, but it would it would be over in an instant. And you can tell by the panic in their voice that it just
15:1315 minutes, 13 secondshappened so fast that they didn't even have time um to react to it. Think about when you're driving down the road and you see a deer in the road or a raccoon
15:2115 minutes, 21 secondsor a possum or something and you don't even have time to react. It's just like that fast. This is two to three times that fast, right? And you're in an
15:2915 minutes, 29 secondsairplane. So, it's not it's not possible, I don't think, that they could have avoided this. Now, the next question is this. Uh, has this ever happened before? And the answer to that
15:3715 minutes, 37 secondsis yes, sadly. Uh, about 6 years ago, Southwest Airlines was going into Austin, Texas on runway 18 right. Theirs was an encounter with a person on the
15:4515 minutes, 45 secondsrunway on landing, not on the takeoff roll. A little bit different, but they're cleared for landing at at Austin on 18 right southwest 1392. As they're
15:5515 minutes, 55 secondslanding, they strike a human being on the runway. It's not there's no explanation for why the person was there. They just were. The dialogue back
16:0416 minutes, 4 secondsand forth between them and the tower was the tower says, "Where was the person?" The airplane just says, "Well, behind us now." So, they're trying to put it together. they they weren't 100% sure
16:1316 minutes, 13 secondsthat they saw a person. This this they're 100% sure here in Denver that they saw cuz they say it right away, right from the beginning um that that's
16:2116 minutes, 21 secondswhat happened. So it has happened before. It's very tragic every time it happens and very sad. The next question is this. Could it have been prevented?
16:3016 minutes, 30 secondsAnd I I think the answer to that is doubtful. It's very doubtful that this could have been prevented. Denver airport is the largest in the country by
16:3816 minutes, 38 secondsgeography. It's 53 square miles. That's a huge airport. Just to put fencing around that airport is incredibly
16:4716 minutes, 47 secondsintense. And so there are points if you've ever been out to Denver and driven out of that airport or driven into that airport, it's extremely rural
16:5516 minutes, 55 secondsgoing into DIA. And most of the the roads around the perimeter fence are just gravel roads. So how did this
17:0317 minutes, 3 secondsperson get here? It we're going to find out in the days and weeks to come. Uh it may have been that they um they drove up, they might have walked up, they
17:1117 minutes, 11 secondsmight have biked up. It's going to be a long walk to get to where they were.
17:1417 minutes, 14 secondsThey were on the extreme eastern portion of the airport. It the early reports are telling us that they scaled the fence.
17:2117 minutes, 21 secondsThey could have clipped the fence. Uh now, aren't there sensors to to let people know that? Some airports there are. I'm not sure about Denver. It's
17:2817 minutes, 28 secondssuch a big airport that I don't know if they've got sensors everywhere to detect that stuff. And it doesn't make any difference. the the distance and I'll
17:3517 minutes, 35 secondsshow you a picture here. All right, here's a still photo runway 17 left.
17:4017 minutes, 40 secondsThere's the fence. Now, you say, "Well, isn't there barb wire on the top of the fence?" Yeah, there is barb wire right on the top of the fence. But look here between these two fence poles, there's
17:4817 minutes, 48 secondsno barb wire. And so, you could climb the fence and, you know, just kind of scoot in between over the barb wire. And
17:5517 minutes, 55 secondsthen the runway is, and I'll show you an airplane on that same runway. There's a still photo. You can see the fence in
18:0318 minutes, 3 secondsthe foreground, right? And it's just a short walk, maybe a minute, not even that long, to get up on runway 17 left.
18:1118 minutes, 11 secondsAnd so it'd be very easy to do. Even if you detected it on a sensor and you sent a car out there, it would take longer for the people to get out than the
18:1818 minutes, 18 secondsindividual to walk onto the runway. And if there was an airplane already cleared for takeoff, then the the story would be told. All
18:2518 minutes, 25 secondsright. How could something like this um happen? Well, um it's not likely that it could be
18:3318 minutes, 33 secondsprevented altogether. I think the first reaction we have is, you know, could there have been something we have done to prevent something like this? I I
18:4118 minutes, 41 secondsdon't think so. I I I think it's virtually impossible. Um it's also not likely that this was a
18:4818 minutes, 48 secondsmistake. um out for a stroll at almost midnight in the pitch black uh going through gravel roads. You'd have to
18:5718 minutes, 57 secondsscale a fence or cut a fence u to get through. Uh no evidence that they cut the fence. They probably scaled it. Uh
19:0419 minutes, 4 secondsand then walking onto an active airport and an active runway uh would imply, I think, on some level that that was intentional. Um you don't just stroll
19:1419 minutes, 14 secondsout onto an active runway. um you'd have to drive or bike or somehow get yourself there, then hop the fence. All of it is
19:2119 minutes, 21 secondsjust kind of a little bit uh too much to fathom. And then the airplane that's coming at you is lit up like a Christmas tree. I mean, there's every light you
19:2819 minutes, 28 secondscan imagine. It is really bright out in front of the airplane for the pilots to get as much visibility as they can. It's
19:3519 minutes, 35 secondsit it's not possible that you wouldn't see the airplane um coming. Now, anything is a possibility. Uh the the final question is why did this happen?
19:4519 minutes, 45 secondsAnd I I think it's very difficult for us to even come up with any answer to that.
19:5019 minutes, 50 secondsThe the big answer to that is it's just unknown. You know, uh was the person on drugs? Were they out of their mind? I I don't know. We don't know any of those
19:5819 minutes, 58 secondsthings. All we know is it didn't happen by mistake. There was too much that had to go into this um for this person to just be kind of in the wrong place at
20:0720 minutes, 7 secondsthe wrong time. I think we can count that one out. Uh, but at the same time, it's just simply unknown. There's just too many question marks. And as I said
20:1520 minutes, 15 secondsat the beginning of the video, uh, if you had any thoughts of self harm, there's an 800 number below. Please call that number. Uh, and if you were one of
20:2320 minutes, 23 secondsthe pilots or you know one of the pilots, uh, a lot of people watch these videos. And if you're one of that the crew, uh, the the uh, cabin crew, the
20:3120 minutes, 31 secondscockpit crew, or one of the passengers on that flight, and you'd like to talk to somebody who understands, who has training, uh, please email me. um down
20:4020 minutes, 40 secondsbelow and I will reach back out to you and we'll start that conversation. But I'm going to ask you now as we wrap this up to pray for the pilots. Um this must
20:4920 minutes, 49 secondsbe absolutely awful for them. They've got that flashbulb picture in their mind that they'll never get rid of. Uh and
20:5820 minutes, 58 secondspray for the the families uh of this man. The there's this man or I'm assuming it's a man. It might be a
21:0621 minutes, 6 secondswoman. this individual who's on the runway um no doubt has a family and they're getting the bad news now and also the people on the airplane, the
21:1421 minutes, 14 secondspassengers. So, there's a bunch of prayers to go around uh as we um look back over this tragedy in Denver this
21:2121 minutes, 21 secondsweekend and hope that it never happens again. Well, now you know. I'm Captain Steve"""


# Strip one leading timestamp blob (matches YouTube-style plain text: no timecodes).
# Export quirks: "0:099 secondslanguage" (= 0:09 + "9 seconds" + "language"),
# "1:031 minute, 3 secondsreality", "12:0012 minutesOver", "0:00Lately".
_STAMP_PATTERNS = (
    re.compile(r"^\d+:\d+\s+seconds"),  # "0:099 secondslanguage", "0:1717 secondsof"
    re.compile(
        r"^\d+:\d{2}\s*\d+\s+minutes?,\s*\d+\s+seconds?"
    ),  # "1:031 minute, 3 secondsreality"
    re.compile(r"^\d+:\d{2}\s*\d+\s+minutes?"),  # "12:0012 minutesOver", "48:0048 minutesthis"
    re.compile(r"^\d+:\d+(?=[A-Za-z\"'])"),  # "0:00Lately"
    re.compile(r"^\d+:\d+"),  # fallback: bare M:SS
)


def _strip_one_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if re.match(r"^Chapter \d+:", line):
        return ""
    if re.match(r"^\[.*\]$", line):
        return ""
    prev = None
    while prev != line:
        prev = line
        for pat in _STAMP_PATTERNS:
            m = pat.match(line)
            if m:
                line = line[m.end() :].lstrip()
                break
    return line.strip()


def clean_transcript(raw: str) -> str:
    parts = [_strip_one_line(L) for L in raw.split("\n")]
    return " ".join(p for p in parts if p)


def main():
    clean = clean_transcript(RAW_TRANSCRIPT)
    print(f"Clean transcript length: {len(clean)} chars")
    print(f"Preview: {clean[:200]}...")
    print()

    if re.search(r"\d:\d{2}", clean[:500]):
        print(
            "WARNING: first 500 chars still look like they contain time-like patterns; "
            "review output.",
            file=sys.stderr,
        )

    db = TranscriptDatabase(DB_PATH)
    db.add_video(VIDEO_ID, VIDEO_TITLE, VIDEO_CHANNEL, VIDEO_URL)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT id FROM transcripts WHERE video_id = ?", (VIDEO_ID,))
    row = cur.fetchone()

    if row:
        cur.execute(
            "UPDATE transcripts SET text = ?, language = 'en', confidence = 1.0 WHERE video_id = ?",
            (clean, VIDEO_ID),
        )
        print(f"Updated transcript for {VIDEO_ID}")
    else:
        cur.execute(
            """
            INSERT INTO transcripts (video_id, text, language, confidence, created_at)
            VALUES (?, ?, 'en', 1.0, ?)
            """,
            (VIDEO_ID, clean, ts),
        )
        print(f"Inserted transcript for {VIDEO_ID}")

    cur.execute(
        """
        UPDATE videos
        SET has_transcript = 1, transcript_fetched_at = ?
        WHERE video_id = ?
        """,
        (ts, VIDEO_ID),
    )

    conn.commit()

    for task in ("shorty", "synthetic_questions", "entities", "triples"):
        cur.execute(
            "SELECT id FROM processing_queue WHERE video_id = ? AND task = ?",
            (VIDEO_ID, task),
        )
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO processing_queue (video_id, task, status) VALUES (?, ?, 'pending')",
                (VIDEO_ID, task),
            )
            print(f"Queued task: {task}")
        else:
            cur.execute(
                "UPDATE processing_queue SET status = 'pending', error = NULL WHERE video_id = ? AND task = ?",
                (VIDEO_ID, task),
            )
            print(f"Reset task to pending: {task}")

    conn.commit()
    conn.close()
    print("\nDone. Run batch_processor to generate Shorty/entities/synq, then reindex.")


if __name__ == "__main__":
    main()
