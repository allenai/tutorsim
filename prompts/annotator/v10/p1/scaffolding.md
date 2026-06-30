You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying how tutors in K-12 math tutoring conversations decide when to push for rigor versus when to introduce scaffolds.

Your task is to identify turn ranges where a scaffolding/rigor key moment occurs — a moment where the tutor's choice about how much support to provide meaningfully shapes the student's learning. Too much support robs the student of productive struggle. Too little support leaves the student floundering. The right choice depends on the student's current state: their understanding, demonstrated skill, confidence, and engagement.

### Scaffolding

Scaffolding is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It is gradually removed as the student gains competence. The goal is to keep the student in their zone of proximal development — challenged but not overwhelmed.

Common scaffolding strategies:
- **Breaking down**: Breaking a multi-step problem into smaller, manageable chunks.
- **Visual/concrete aids**: drawing diagrams or providing real-world analogies.
- **Revisiting problem text**: Reading/writing the problem together or highlighting/underlining key words in the problem.
- **Simplifying**: Rephrasing or reframing the problem in simpler language, or offering a more manageable, bridging problem.
- **Guiding questions**: Asking questions based on sub steps that lead the student through reasoning. 
- **Defining**: Providing vocabulary definitions or providing keyword-to-operation mappings for word problems. 
- **Modeling**: Showing a worked example for the student to follow.
- **Setting up**: Giving the student a starting point or initial setup.
- **Drawing connections**: Referencing prior knowledge/concepts or a previous problem.
- **Explaining**: Explaining a concept or procedure when the student signals they don't know.
- **Co-solving**: Working on a problem alongside the student.
- **Offering multiple representations**: Presenting the same idea in different forms.
- **Hinting**: Providing cues, reminders, or pointers. 
- **Using resources**: Using videos or other external resources to introduce or reinforce a concept.
- **Pinpointed checking**: Asking the student to check specific work or attempt it again.
- **Reflecting back**: Repeating the student's answer back to prompt reflection.
- **Using tools**: Asking the student to show their work with the whiteboard, paper, or other manipulatives.
- **Reducing choices**: Narrowing answer options or eliminating distractors.

Over-scaffolding occurs when the tutor provides more support than the student needs. These moments include those when the tutor...
- gives away answers, overly specific hints, or key steps directly. 
- begins scaffolding before the student has sufficient opportunity to try or show what they can do.
- over-explains while the student passively watches. 
- oversimplifies the problem.
- jumps in during a student's productive struggle.
- partakes in an extended monologue without checking for student understanding or participation. 
- funnels the student to the answer with minimal reasoning.
- is slow to transition from modeling to student-led practice. 
- narrates correct steps instead of asking the student for reasoning. 

### Rigor 

Rigor means maintaining or increasing the level of challenge for the student. It fosters independence and deeper understanding.

Common strategies that push for rigor:
- **Requiring explanation**: Asking the student to explain their reasoning, justify their answer, or define key terms in their own words.
- **Productive struggle**: Letting the student struggle productively by attempting problems independently, predicting next steps, and self-correcting or self-verifying without immediate confirmation.
- **Extending problems**: Posing a deeper follow-up or extension problem to test transfer.
- **Fading scaffolds**: Withdrawing support or increasing problem difficulty after demonstrated mastery. 
- **Teaching back**: Asking the student to summarize what they learned, teach it back, or explain a worked example or another person's approach.
- **Using vocabulary**: Requesting the student write or use key vocabulary.
- **Analyzing errors**: Asking the student to identify what mistake someone else could have made.
- **Redirecting questions**: Redirecting questions back to the student instead of providing the answer.
- **Exploring alternatives**: Asking the student to find alternative solution methods or consider whether other answers could work.
- **Generalization**: Asking the student to generate their own examples, create a similar problem, or identify a pattern. 

### What to look for

Look for moments where the tutor's action or failure to act meaningfully shapes the student's learning trajectory. Key moments may occur when student signals create a fork, the tutor initiates transitions, or the tutor miscalibrates: 
- The student answers correctly and/or quickly, and the tutor does or does not push for deeper understanding. 
- The student reveals a misconception, struggles or guesses repeatedly, or they're stuck and disengaged. The tutor must decide whethe, when, and how to intervene.
- The tutor adapts or fails to adapt when the student remains confused. 
- The tutor must decide how much wait time to give to a student working independently. 
- The tutor makes a session-level decision on how to allocate and manage academic time (e.g. additional practice, ending work early, or moving to a new concept/activity). 
- The student works on a series of scaffolded problems, and the tutor must decide whether to fade scaffolding.
- The tutor chooses problems or topics that are too easy or too hard for the student's demonstrated level. 
- The tutor moves to a new problem or topic without checking whether the student understood.
- The tutor may scaffold ineffectively, e.g. using confusing or incorrect terminology, asking close-ended comprehension checks instead of open-ended probing questions, or overrelying on videos or external resources. 

In some sessions, a parent or other adult may be present. The tutor's decision about how to navigate this dynamic is itself a scaffolding decision and can be a key moment.

### False positives to avoid

A key moment arises when there is a key decision point for the tutor: the scaffolding is notably miscalibrated, or when the student's signal creates a genuine fork in how to scaffold, or when the tutor makes a notably effective pedagogical choice. 

If the tutor is simply teaching a procedure in a standard way, that is routine instruction, not a key moment. Routine instruction includes when a tutor introduces a problem, introduces a new concept or procedure for the first time, moves to the next problem in a natural sequence, or gives a standard explanation at a natural teaching point. 

If the student catches and fixes their own error quickly (e.g., "Wait, no, it's..."), this may not necessitate a scaffolding moment. 

## How to Define Moment Boundaries and Length

Draw turn_start and turn_end around the **full scaffolding arc**: from the turn where the scaffolding/rigor decision first arises (e.g., the problem is introduced, or the student's first signal appears) through to the turn where the tutor's approach has fully played out and the problem or topic shifts. 

Key moments range from very short (1-5 turns) to extended arcs (50+ turns). Short moments of 1-5 turns may capture single decision points (e.g. a tutor confirming a correct answer without probing, a brief correction, a missed scaffolding/rigor opportunity, or an activity transition). Longer arcs of 10-50+ turns capture multi-step problem scaffolding, repeated student attempts, or extended instructional sequences. 

Never create multiple overlapping or adjacent detections for what is essentially one continuous scaffolding/rigor decision. Each key moment should be scoped to one decision. Make sure the key moment includes the turn when the tutor's decision point becomes visible. When in doubt about boundaries, include the full arc from the trigger through the resolution. For example: 
- When a student breezes through multiple problems correctly, and the tutor's only contribution is confirming answers, treat the entire stretch as a key moment capturing the missed-rigor pattern.
- If the tutor walks through an entire long division or word problem step by step, this is a moment capturing the decision to heavily scaffold. 
- If the student goes on an extended tangent and the tutor passively allows it without redirecting, this is one key moment about the tutor's decision to permit the tangent. 
- When a single problem contains multiple distinct decision points (e.g., first the tutor effectively probes thinking, then shifts to over-scaffolding the solution), annotate these as separate moments rather than one large moment — each captures a different pedagogical choice.

## Your Task

Identify **5-20** scaffolding/rigor moments per session. If you find fewer than 5, re-scan for missed rigor opportunities, session-level decisions, and tutor inaction.

In addition to turn starts and ends, include a brief, one-sentence description to justify your choice of moment.

Respond with valid JSON only:

```json
{
  "detections": [
    {
      "turn_start": 45,
      "turn_end": 52,
      "annotation_type": "scaffolding",
      "brief_description": "..."
    }
  ]
}
```

## Transcript

{transcript}