You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying rapport building behaviors in K-12 math tutoring conversations. Rapport is the interpersonal connection between tutor and student that creates a safe, productive learning environment. It encompasses how the tutor relates to the student as a person -- showing genuine interest, managing emotions, building trust, and navigating the social dynamics of the session.

Your task is to identify turn ranges where the tutor makes an attempt at building rapport with the student, or where a rapport-building opportunity arises.

## What to Look For

Scan the ENTIRE transcript for moments involving social-psychological dynamics:

- Session opening / greeting / check-in
- Session closing / goodbye / wrap-up
- Transitions between activities
- Student expresses emotion (frustration, excitement, tiredness, boredom, anxiety)
- Tutor acknowledges or ignores student emotional state
- Small talk, personal sharing, off-topic conversation
- Praise (generic or specific)
- Humor, playfulness, jokes, or tension
- Student resistance, refusal, or disengagement
- Tutor builds on student interests or remembers personal details
- Power dynamics (tutor flexibility or rigidity about rules, posture, time)
- Student shares personal information unprompted
- Moments where the tutor's rapport attempt falls flat or creates awkwardness
- Academic content used as rapport (making problems personal, using student interests)

## Important: Cast a Wide Net

Identify **5-12 moments** per session. It is better to flag a moment that turns out to be minor than to miss one. Look throughout the ENTIRE transcript, not just at the beginning and end.

When in doubt, include it. A separate analysis step will evaluate each moment in detail.

## How to Define Moment Boundaries

Draw turn_start and turn_end around the **core interaction**: from the turn that initiates the rapport-relevant moment to the turn where it resolves or the tutor moves on. Include the triggering student behavior and the tutor's response, but stop once the moment is over. If the same dynamic persists across many turns, that is one moment with a wider range, not multiple overlapping detections.

## How to Choose a Cut Point

Each detection needs a `suggested_cut_turn`. Here is why: the transcript will be sliced at this turn, and a **different AI tutor** will take over from that point. They see everything up to and including the cut turn, and nothing after. Their first response is what gets evaluated. The cut point is designing a test — the transcript-so-far is the prompt, the tutor's continuation is the answer being graded.

A good cut point satisfies three criteria:

1. **Enough context to understand the situation.** The student's emotional and behavioral state should be readable from preceding turns. Relevant patterns (e.g., disengagement building over several turns, or a sudden emotional shift) should be visible in the transcript before the cut.

2. **A genuine decision to make.** The moment presents a fork where multiple reasonable tutor responses exist, and the choice reveals something about tutor skill. If there is only one obvious response, the cut point is less useful.

3. **No preview of the original tutor's approach.** The cut lands *before* the original tutor responds to the rapport-relevant moment, so the incoming tutor must make their own choice without being anchored by what they saw.

The cut turn must be a **STUDENT turn** (or non-tutor turn) so the tutor speaks next. Place the cut at the student turn that sets up the decision — the thing the incoming tutor must respond to.

<examples>

<example>
<title>Good cut — student expresses frustration</title>
<scenario>The student says "this is so stupid I hate fractions" at turn 47.</scenario>
<cut_turn>47</cut_turn>
<reasoning>The incoming tutor sees clear frustration and must decide: address the emotion directly? Redirect to the math? Validate and then re-engage? The student's state is clear from context, but the response strategy is genuinely open.</reasoning>
</example>

<example>
<title>Good cut — disengagement pattern building over time</title>
<scenario>Over turns 60-70, the student's responses have gotten shorter and more listless ("ok," "sure," "idk"). At turn 71, the student gives another one-word answer.</scenario>
<cut_turn>71</cut_turn>
<reasoning>The incoming tutor can see the pattern of disengagement across multiple turns and must decide how to re-engage. The context is rich (the pattern is visible), but the strategy is open.</reasoning>
</example>

<example>
<title>Bad cut — too late, tutor already responded</title>
<scenario>Student expresses anxiety at turn 30. The tutor responds with reassurance at turn 31.</scenario>
<cut_turn>Should be 30, not 31</cut_turn>
<reasoning>Cutting at 31 means the incoming tutor sees the original tutor's approach and is anchored by it. Always cut before the tutor acts on the rapport moment.</reasoning>
</example>

</examples>

## Output Format

Respond with valid JSON only:

```json
{
  "detections": [
    {
      "turn_start": 1,
      "turn_end": 25,
      "annotation_type": "rapport",
      "brief_description": "Session opening with check-in about student's weekend",
      "suggested_cut_turn": 3
    }
  ]
}
```

## Transcript

{transcript}