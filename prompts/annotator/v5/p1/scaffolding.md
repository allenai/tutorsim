You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying how tutors in K-12 math tutoring conversations decide to push for rigor in a session by asking students to do harder tasks or tasks that involve more metacognition, versus when they choose to introduce scaffolds that make problems more accessible for students.

**Scaffolding** is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It includes breaking problems into steps, providing hints, modeling a process, using visual aids, or simplifying the task. The goal is to gradually remove support as the student gains competence.

**Rigor** (pushing for depth) means maintaining or increasing the cognitive demand on the student. It includes asking for explanations, requiring metacognition ("How did you know that?"), removing support to foster independence, extending problems, or challenging students who are coasting.

The core tradeoff: when a student struggles, the tutor must decide how much support to provide. When a student succeeds, the tutor must decide whether to push further or move on. Your task is to identify turn ranges where the tutor faces this tradeoff and makes a notable choice.

## What to Look For

Scan the transcript for moments where the tutor's scaffolding or rigor decisions are notable:

- Student makes an error or shows a misconception
- Student answers correctly and could be pushed for deeper thinking
- Student shows insight, reasoning, or connections that could be extended
- Tutor provides a hint, breaks a problem into steps, or models a process
- Tutor explains when they could have questioned instead
- Tutor gives away the answer or does the cognitive work for the student
- Student is frustrated, stuck, or says "I don't know"
- Student demonstrates independence or transfers a concept to a new problem
- Tutor repeats the same strategy that isn't working
- Transitions between problem types or activities
- Tutor shifts approach mid-interaction (from questioning to telling, or vice versa)

## Important: Cast a Wide Net

Identify **8-15 moments** per session. It is better to flag a moment that turns out to be minor than to miss one. Include brief moments (even 2-3 turns) as well as longer interactions.

When in doubt, include it. A separate analysis step will evaluate each moment in detail.

## How to Define Moment Boundaries

Draw turn_start and turn_end around the **core interaction**: from the turn where the scaffolding/rigor decision arises to the turn where the tutor's approach plays out or the activity shifts. Include the student's triggering turn (error, success, "I don't know") and the tutor's response sequence. Brief moments (2-3 turns) are fine — stop once the interaction resolves.

## How to Choose a Cut Point

Each detection needs a `suggested_cut_turn`. Here is why: the transcript will be sliced at this turn, and a **different AI tutor** will take over from that point. They see everything up to and including the cut turn, and nothing after. Their first response is what gets evaluated. The cut point is designing a test — the transcript-so-far is the prompt, the tutor's continuation is the answer being graded.

A good cut point satisfies three criteria:

1. **Enough context to understand the situation.** The math topic, current problem, and the student's cognitive state should be clear from preceding turns. Relevant patterns (e.g., the student has been struggling for several turns, or coasting with bare answers) should be visible.

2. **A genuine decision to make.** The moment presents a fork where multiple reasonable scaffolding or rigor approaches exist, and the choice reveals something about tutor skill. If there is only one obvious response, the cut point is less useful.

3. **No preview of the original tutor's approach.** The cut lands *before* the original tutor responds to the scaffolding-relevant moment, so the incoming tutor must make their own choice without being anchored by what they saw.

The cut turn must be a **STUDENT turn** (or non-tutor turn) so the tutor speaks next. Place the cut at the student turn that sets up the decision — the thing the incoming tutor must respond to.

<examples>

<example>
<title>Good cut — student makes a revealing error</title>
<scenario>At turn 55, the student writes "3/4 + 1/2 = 4/6."</scenario>
<cut_turn>55</cut_turn>
<reasoning>The misconception (adding numerators and denominators separately) is visible. The incoming tutor must decide: point out the error directly? Ask the student to check their work? Provide a visual model? Model the correct procedure? Each approach reflects a different scaffolding philosophy.</reasoning>
</example>

<example>
<title>Good cut — student has been coasting</title>
<scenario>Over turns 30-40, the student has answered several problems correctly but with no explanation — just bare answers. At turn 41, another bare answer: "8."</scenario>
<cut_turn>41</cut_turn>
<reasoning>The incoming tutor can see the pattern of correct-but-shallow responses and must decide: accept and move on, or push for deeper understanding ("How did you get that?"). This tests whether the tutor prioritizes pace or rigor.</reasoning>
</example>

<example>
<title>Bad cut — tutor already scaffolded</title>
<scenario>Student is stuck at turn 44. The tutor breaks the problem into steps at turns 45-46.</scenario>
<cut_turn>Should be 44, not 46</cut_turn>
<reasoning>Cutting at 46 means the incoming tutor sees the original tutor's scaffolding strategy and is anchored by it. Always cut before the tutor acts on the moment.</reasoning>
</example>

</examples>

## Output Format

Respond with valid JSON only:

```json
{
  "detections": [
    {
      "turn_start": 45,
      "turn_end": 52,
      "annotation_type": "scaffolding",
      "brief_description": "Student struggles with fraction division, tutor decides whether to hint or model",
      "suggested_cut_turn": 46
    }
  ]
}
```

## Transcript

{transcript}