You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying how tutors in K-12 math tutoring conversations decide when to push for rigor versus when to introduce scaffolds. Your task is to identify turn ranges where a scaffolding/rigor key moment occurs — a moment where the tutor's choice about how much support to provide meaningfully shapes the student's learning.

At each moment, the tutor decides how much support to provide. Too much support robs the student of productive struggle. Too little support leaves the student floundering. The right choice depends on reading the student's current state — their understanding, confidence, and engagement.

### Scaffolding

Scaffolding is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It is gradually removed as the student gains competence. The goal is to keep the student in their zone of proximal development — challenged but not overwhelmed.

Common scaffolding strategies:
...

Over-scaffolding occurs when the tutor provides more support than the student needs — doing the cognitive work for them, giving away answers, or funneling toward a solution without letting the student struggle productively.

### Rigor 

Rigor means maintaining or increasing the level of challenge for the student. It fosters independence and deeper understanding.

Common strategies that push for rigor:
...

### What to look for

...

### False positives to avoid

The mere presence of scaffolding does NOT make a key moment, as scaffolding is a tutor's primary tool. A key moment arises only when there is a key decision point for the tutor: the scaffolding is notably miscalibrated (too much, too little, wrong type) or when the student's signal creates a genuine fork in how to scaffold. If the tutor is simply teaching a procedure in a standard way and the student is following along, that is routine instruction, not a key moment.

Example scenarios that may lead to false positives: 
- Routine instruction are not key moments unless the tutor's approach is notably miscalibrated. Routine instruction includes when a tutor introduces or sets up a problem (e.g. reading the problem, defining terms), introduces a new concept or procedure for the first time, moves to the next problem in a natural sequence, or gives a standard explanation at a natural teaching point. 
- In games (Blooket, speed multiplication, Broken Calculator, Fish Feeder, matching games, etc.), isolated mistakes, post-game score reviews, quick tips after games, and replaying games are all routine. 
- If the tutor walks through an entire long division or word problem step by step, this is ONE moment (the decision to heavily scaffold), not five separate moments for each sub-step.
- If the student catches and fixes their own error or solves a problem correctly independently without struggle, then the tutor likely has no scaffolding decision to make.
- If the student performed well in a game and the tutor offers a quick tip or reviews one question, this is routine unless the tip reveals a significant pedagogical choice (e.g., teaching a strategy the student didn't need).
- If the student plays a non-math game (e.g., pattern matching, butterfly matching) and the tutor passively observes, the game itself is not a key moment unless the tutor's choice to allow it is the key moment. 

## How to Define Moment Boundaries and Length

Draw turn_start and turn_end around the **full scaffolding arc**: from the turn where the scaffolding/rigor decision point first arises (e.g., the problem is introduced, or the student's first signal appears) through to the turn where the tutor's approach has fully played out and the problem or topic shifts. Most key moments are focused decision arcs of 5-50 turns.

**Sizing guidance:**
...

Never create multiple overlapping or adjacent detections for what is essentially one continuous scaffolding/rigor interaction.

## Your Task

Identify **5-12 scaffolding moments** scaffolding/rigor moments per session. If you find fewer than 5, re-scan for missed rigor opportunities, session-level decisions, and tutor inaction. If you find more than 15, check whether you're fragmenting multi-step problems into sub-steps or flagging routine instruction. It is better to flag a moment that turns out to be minor than to miss one that matters. A separate analysis step will evaluate each moment in detail — focus on detection.

In addition to turn starts and ends, include a brief, one-sentence description to justify your choice of moment.

Respond with valid JSON only:

```json
{
  "detections": [
    {
      "turn_start": 45,
      "turn_end": 52,
      "annotation_type": "scaffolding",
      "situation": "..."
    }
  ]
}
```

## Transcript

{transcript}