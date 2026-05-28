You are an expert in pedagogy and an experienced teaching coach. A previous review of this tutoring transcript flagged the turn range below as a scaffolding-related moment. Your task is to analyze it in detail.

## Research Context

We are studying how tutors in K-12 math tutoring conversations decide to push for rigor in a session by asking students to do harder tasks or tasks that involve more metacognition, versus when they choose to introduce scaffolds that make problems more accessible for students. For each flagged moment, you will analyze the provided situation, and then describe the tutor's strategy and how well it worked. 

### Scaffolding

Scaffolding is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It is gradually removed as the student gains competence. The goal is to keep the student in their zone of proximal development — challenged but not overwhelmed.

Common scaffolding strategies:
- **Breaking down**: Decomposing a complex problem into smaller, manageable steps
- **Hinting**: Providing a cue that points the student toward the answer without giving it away
- **Modeling**: Demonstrating a process or worked example for the student to follow
- **Simplifying**: Reducing the complexity of the task while preserving the core concept
- **Visual/concrete aids**: Using diagrams, manipulatives, or real-world analogies

Over-scaffolding occurs when the tutor provides more support than the student needs — doing the cognitive work for them, giving away answers, or funneling toward a solution without letting the student struggle productively.

### Rigor (Pushing for Depth)

Rigor means maintaining or increasing the cognitive demand on the student. It fosters independence and deeper understanding.

Common rigor strategies:
- **Requiring explanation**: "How did you know that?" or "Why does that work?"
- **Extending problems**: Adding complexity, edge cases, or transfer to new contexts
- **Fading support**: Deliberately removing scaffolds to see if the student can work independently
- **Challenging correct answers**: Asking students to generalize, prove, or find alternative approaches
- **Metacognition prompts**: Asking the student to reflect on their own thinking process

### The Core Tradeoff

At each moment, the tutor decides how much support to provide. Too much support robs the student of productive struggle. Too little support leaves the student floundering. The right choice depends on reading the student's current state — their understanding, confidence, and engagement.

## Your Task

For the flagged moment, provide:
- **Situation**: First, *evaluate why this is or isn't an appropriate time to push for rigor or scaffold*. Then, describe the student's state (e.g. stuck, confused, succeeding, coasting), and what pedagogical context makes this moment notable for scaffolding/rigor.  
- **Action**: Describe what strategy the tutor used to push for rigor or scaffold. Name the strategy type (breaking down, hinting, modeling, requiring explanation, etc.) where applicable.
- **Result**: How well did the tutor's strategy work? Did the student make learning progress? Was the level of support appropriate given the student's state and the topic's novelty? 

## Examples

### Output example 1: Tutor connects to prior knowledge

```json
{
  "annotation_type": "scaffolding",
  "turn_start": 45,
  "turn_end": 48,
  "situation": "The student was intimidated by large numbers in division (5600 / 70) and said 'I don't know how to do this.' This is an appropriate time to scaffold -- the student's block is intimidation, not lack of knowledge.",
  "action": "The tutor used hinting -- connecting the problem to a known fact (56 / 7 = 8) and asking the student to apply the strategy of canceling zeros.",
  "result": "The scaffold was well-calibrated. The tutor correctly identified that the student already knew the underlying fact and just needed a bridge to see it. The student recognized 56 / 7 from memory and articulated the simplified problem on their own. The tutor provided minimal support and the student did the reasoning."
}
```

### Output example 2: Tutor explains without checking understanding

```json
{
  "annotation_type": "scaffolding",
  "turn_start": 76,
  "turn_end": 86,
  "situation": "The student added denominators incorrectly (1/4 + 1/4 = 2/8), revealing a misconception about fraction addition. This is an appropriate time to scaffold the specific misconception.",
  "action": "The tutor explained the rule ('the bottom number stays the same') and used a hands analogy. When the student self-corrected to 2/4, the tutor said 'perfect' and moved on without a follow-up problem.",
  "result": "The analogy was well-chosen and addressed the misconception directly -- the student corrected the error. However, the tutor did not check whether the student understood why the denominator stays the same. There was no follow-up question or transfer problem, so it is unclear whether the student learned a rule or understood the concept."
}
```

### Output example 3: Tutor doesn't push when student is coasting

```json
{
  "annotation_type": "scaffolding",
  "turn_start": 55,
  "turn_end": 63,
  "situation": "The student solved three single-digit multiplication problems quickly and correctly with no hesitation. This signals readiness for more challenge -- an appropriate time to push for rigor.",
  "action": "The tutor praised each answer ('Good job!') and gave another problem at the same difficulty level. No attempt to extend, require explanation, or increase complexity.",
  "result": "The student was coasting and the tutor maintained the same level. No productive struggle or growth occurred in this segment. The tutor could have asked 'How did you figure that out so fast?' to push for metacognition, or introduced a harder problem to find where the student's understanding breaks down."
}
```

### Output example 4: Tutor tells the student the answer with incidental correct diagnosis

```json
{
  "annotation_type": "scaffolding",
  "turn_start": 32,
  "turn_end": 38,
  "situation": "The student subtracted 15 - 8 and got 13, suggesting a counting error. This is an appropriate time to scaffold.",
  "action": "The tutor said 'Not quite -- remember, 15 minus 8. Count back from 15: fourteen, thirteen, twelve, eleven, ten, nine, eight, seven. So it's 7.' The tutor counted aloud for the student.",
  "result": "Although the tutor correctly identified the error type (counting), the tutor did all the cognitive work -- counting back from 15 while the student listened passively. The student then wrote 7 and moved on, but there is no evidence they understood why their original answer was wrong or practiced the counting themselves. The correct diagnosis does not salvage the fact that the student did no thinking."
}
```

## Detected Moment

The following excerpt contains the flagged turn range, marked with >>> DETECTED MOMENT START <<< and >>> DETECTED MOMENT END <<<. The surrounding turns provide context.

{excerpt}

## Output Format

Respond with valid JSON only:

```json
{
  "situation": "...",
  "action": "...",
  "result": "..."
}
```