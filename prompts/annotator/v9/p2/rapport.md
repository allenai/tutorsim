You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

Rapport is the interpersonal connection between tutor and student that creates a safe, productive learning environment. Your task is to identify turn ranges where a rapport-relevant key moment occurs. 

The challenge of rapport in tutoring is calibrating it to the student and the moment. Good rapport builds trust and relational continuity, and it requires emotional attunement and genuine care. Rapport can be excessive if it is done at the expense of learning; such imbalances are themselves key moments. Key rapport moments can occur before, after, and during math instruction. If an interaction involves a notable rapport-related decision for the tutor, it is a key moment. 

### Rapport Strategies

Common rapport-building approaches tutors use:
- **Check-ins**: Asking about the student's current emotional or cognitive state, or their life or interests outside tutoring. 
- **Active listening**: Asking follow-up questions to student sharing, with genuine curiosity. 
- **Collaborative activities**: Examples include drawing together, warm-up or icebreaker games, or building ongoing creative projects. 
- **Structured choices**: Giving the student agency by letting them choose activities, topics, games, or session order. 
- **Personal sharing**: The tutor shares their own experiences or interests, models vulnerability, or relates to the student's experiences.
- **Emotional validation**: Acknowledging frustration or difficulty, normalizing confusion and mistakes, or reframing setbacks as part of learning.
- **Humor and playfulness**: Jokes, playful banter, celebratory moments, lighthearted commentary.
- **Specific praise**: Acknowledging particular effort, thinking, strategy, process, or growth rather than generic praise.
- **Relational continuity**: Remembering details from prior sessions, maintaining shared references, following up on promises. 
- **Goal-setting and reflection**: Collaboratively setting goals, reflecting on progress, discussing accomplishments.
- **Patience through struggle**: Navigating student struggle, resistance, or frustration with warmth, calm, flexibility, support, and respect, including redirection. 
- **Interest integration**: Using the student's interests to frame problems or examples.
- **Handling disruptions**: Patiently and flexibly responding to or resolving technical problems and environmental distractions. 

## Considerations

When assessing effectiveness, consider the following. 

**Student Behavior**
...

**Timing & Duration**
...

**What is often perceived as effective or ineffective?**
...

Examples of moments that are perceived as effective: 
...

Examples of moments that are perceived as ineffective: 
...

### Calibrating Mixed Assessments

...

## Your Task

For the flagged moment, provide:
- **Situation**: First, *evaluate whether this was an appropriate moment for rapport building*. Then, describe where you are in the session flow (opening, mid-task, transition, closing) and what the student is currently doing. Describe the student's emotional or social state, and what makes this moment notable for rapport.
- **Action**: Describe the strategy the the tutor used to build rapport. Name the strategy type (specific praise, structured choice, etc.) where applicable. If the tutor missed a rapport opportunity, describe what happened instead.
- **Result**: How effective is the tutor's action for building a productive learning relationship? Describe potential reasons why the tutor's strategy may be effective and/or why it may not be. 

## Examples

### Example 1: Tutor recalls a personal detail

```json
{
  "annotation_type": "rapport",
  "turn_start": 5,
  "turn_end": 8,
  "situation": "The opening of a tutoring session. This is a natural place for rapport building before transitioning to academic work.",
  "action": "The tutor remembered the student's baseball award from a previous session. The tutor used relational continuity and a specific follow-up: 'So tell me about that baseball award -- what was it for?'",
  "result": "The approach was effective. The tutor remembered a specific detail and asked an open-ended question, which is genuine rather than formulaic. The exchange lasted 3 turns and transitioned naturally into the session's math work without displacing instructional time. The student shared unprompted detail about their values, but the approach would have been effective even with shorter responses -- the tutor demonstrated that they care about the student as a person and built relational continuity."
}
```

### Example 2: Enthusiastic but pedagogically harmful praise

```json
{
  "annotation_type": "rapport",
  "turn_start": 50,
  "turn_end": 55,
  "situation": "The student just answered a straightforward math problem correctly. The tutor wants to reinforce the student's confidence.",
  "action": "The tutor used enthusiastic person-praise: 'Oh my gosh, you are SO smart!' The student responded happily and the exchange was warm and playful.",
  "result": "The approach was ineffective despite the student's positive response. Praising innate ability ('you are smart') rather than effort or strategy undermines growth mindset -- it teaches the student that success comes from being smart rather than from working hard, which makes them less resilient when they encounter difficulty. The student enjoyed the praise in the moment, but enjoyment does not make the approach pedagogically sound. An effective alternative would be specific process-praise: 'You got that fast -- did you use the shortcut we practiced?'"
}
```

### Example 3: One-sided check-in

```json
{
  "annotation_type": "rapport",
  "turn_start": 85,
  "turn_end": 89,
  "situation": "Transition between activities -- a natural place for a check-in. The student had been working quietly for several minutes.",
  "action": "The tutor used a generic check-in: 'How's your day going?' When the student gave a one-word answer ('fine'), the tutor responded warmly ('That's good!') but moved on without probing.",
  "result": "The approach was ineffective. The check-in was formulaic -- a generic question followed by no adaptation when the student gave a minimal response. The tutor did not follow up with a more specific question, share something about their own day, or try a different angle. The interaction stayed surface-level and did not build connection beyond the routine."
}
```

### Example 4: Missed emotional cue

```json
{
  "annotation_type": "rapport",
  "turn_start": 145,
  "turn_end": 158,
  "situation": "The student was resisting the session duration, repeatedly asking when they could stop. This signals frustration or fatigue -- a moment where emotional attunement is needed.",
  "action": "The tutor engaged in a 14-turn negotiation about session time ('We still have 10 minutes') rather than acknowledging the student's emotional state.",
  "result": "The approach was ineffective. The tutor responded to the logistics of time rather than the underlying frustration. The student's resistance continued throughout the exchange and 14 turns were consumed without resolution. An alternative would have been to acknowledge the frustration ('I hear you, this is tough') and offer a structured choice ('Want to try something different, or power through the last problem?')."
}
```

### Example 5: Incidental positives within an ineffective approach

```json
{
  "annotation_type": "rapport",
  "turn_start": 70,
  "turn_end": 85,
  "situation": "The student mentioned an upcoming medical appointment, providing an opportunity for the tutor to build rapport. The tutor already knew about the appointment from a previous session.",
  "action": "The tutor acknowledged the appointment and confirmed the date, showing they remembered and demonstrating relational continuity. However, the tutor then spent 15 turns discussing scheduling logistics -- which days to reschedule tutoring, whether Monday or Tuesday would work -- rather than asking how the student felt about the appointment.",
  "result": "The approach was ineffective. While the tutor demonstrated relational continuity by remembering the appointment, this incidental positive does not redeem the core action: spending the moment on adult logistics rather than addressing the student's emotional experience. The student needed to feel that the tutor cared about them, not about the tutoring schedule. The relational continuity was a foundation the tutor failed to build on -- knowing about the appointment is not the same as using that knowledge to connect with the student."
}
```

### Example 6: Poorly timed rapport -- student actively on task

```json
{
  "annotation_type": "rapport",
  "turn_start": 502,
  "turn_end": 507,
  "situation": "The student is in the middle of working through a math solution, focused and on task. They are not showing signs of distraction, boredom, or frustration. This is not an appropriate time for rapport building.",
  "action": "The tutor, inspired by the topic of the math problem, asked the student about their personal likes and dislikes, interrupting the student's problem-solving flow.",
  "result": "The approach was ineffective. The tutor interrupted a focused student to initiate a social tangent that was neither needed nor wanted in that moment. The student answered briefly and redirected back to the solution on their own -- a clear signal that the rapport move was unwelcome. Timing is not incidental to rapport; it is fundamental. A well-intentioned question at the wrong moment teaches the student that the tutor does not respect their focus. This question belonged at a transition point, not mid-problem."
}
```

## Detected Moment

The following excerpt contains the flagged turn range, marked with >>> DETECTED MOMENT START <<< and >>> DETECTED MOMENT END <<<. The surrounding turns provide context.

{excerpt}

## Output Format

Respond with valid JSON only:

```json
{
  "annotation_type": "rapport",
  "turn_start": {turn_start},
  "turn_end": {turn_end},
  "situation": "...",
  "action": "...",
  "result": "..."
}
```