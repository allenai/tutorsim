You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying rapport building behaviors in K-12 math tutoring conversations. Your task is to identify turn ranges where a rapport-relevant key moment occurs — a moment where the tutor's rapport choice meaningfully shapes the student relationship or learning environment.

Rapport is the interpersonal connection between tutor and student that creates a safe, productive learning environment. In tutoring, rapport serves learning — it is the foundation that makes academic work possible, not an end in itself.

### Rapport Strategies

The challenge of rapport in tutoring is calibrating it to the student and the moment. 

Common rapport-building approaches tutors use:
...

Look for: 
...

### Missed Opportunities are Key Moments

If the session context or student behavior creates an opportunity for rapport and the tutor ignores it, skips it, or handles it superficially, that is also a key moment. 

When detecting a missed opportunity for rapport, include the turns where the opportunity arose and the tutor's insufficient response (or lack of response). The key moment is the gap between what happened and what could have happened.

### False positives to avoid

- Brief generic praise after a single correct answer or a routine problem with no preceding struggle or emotional context. 
- Routine transitions between individual problems where the tutor is simply moving to the next question without any rapport-relevant exchange. 
- A single polite or warm comment in passing that does not constitute a sustained interaction or respond to a student signal. 
- A brief off-topic student comment that the tutor acknowledges in one turn and moves on — no sustained exchange develops. 
- A student celebration ('Yay!', 'My math is now mathing!') that the tutor briefly acknowledges ('Nice', 'All right') without either sustained celebration or conspicuous dismissal — the moment is pleasant but unremarkable. 
- A student sharing a personal detail in passing (nickname origin, favorite food, brief curiosity about a topic) where the tutor acknowledges it in 1-2 turns and neither party develops the thread further. 
- A brief playful or cultural exchange (e.g., student uses a foreign word, tutor responds) that lasts a couple turns but doesn't connect to the student's identity, interests, or emotional state in a meaningful way.

## How to Define Moment Boundaries and Length

Draw turn_start and turn_end around the **full rapport interaction**: from the turn that initiates the rapport-relevant moment to the turn where the conversation clearly shifts to a different mode (e.g., social conversation ends and academic work begins, or vice versa). Do NOT clip to just the triggering moment — include the entire exchange that constitutes the rapport event.

**Sizing guidance:**
...

Never create multiple overlapping or adjacent detections for what is essentially one continuous rapport interaction.

## Your Task

Identify **3-8 rapport moments** rapport moments per session. Sessions with rich social conversation or multiple emotional signals may have more. A separate analysis step will evaluate each moment in detail — focus on detection. 

In addition to turn starts and ends, include a brief, one-sentence description to justify your choice of moment.

Respond with valid JSON only:

```json
{
  "detections": [
    {
      "turn_start": 1,
      "turn_end": 25,
      "annotation_type": "rapport",
      "brief_description": "..."
    }
  ]
}
```

## Transcript

{transcript}