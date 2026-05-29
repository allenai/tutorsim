You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying how tutors in K-12 math tutoring conversations decide to push for rigor in a session by asking students to do harder tasks or tasks that involve more metacognition, versus when they choose to introduce scaffolds that make problems more accessible for students.

**Scaffolding** is temporary, calibrated support that helps a student accomplish a task they cannot yet do independently. It includes breaking problems into steps, providing hints, modeling a process, using visual aids, or simplifying the task. The goal is to gradually remove support as the student gains competence.

**Rigor** (pushing for depth) means maintaining or increasing the cognitive demand on the student. It includes asking for explanations, requiring metacognition ("How did you know that?"), removing support to foster independence, extending problems, or challenging students who are coasting.

The core tradeoff: when a student struggles, the tutor must decide how much support to provide. When a student succeeds, the tutor must decide whether to push further or move on. Your task is to identify turn ranges where the tutor faces this tradeoff and makes a notable choice.

## Important: Cast a Wide Net

Identify **8-15 moments** per session. It is better to flag a moment that turns out to be minor than to miss one. Include brief moments (even 2-3 turns) as well as longer interactions.

When in doubt, include it. A separate analysis step will evaluate each moment in detail.

## Output Format

Respond with only a list of JSON:
{
  "detections": [
    {
      "start": 45,
      "end": 52,
      "description": "..."
    }
  ]
}

## Transcript

{transcript}
