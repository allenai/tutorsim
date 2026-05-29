You are an expert in pedagogy analyzing a K-12 math tutoring transcript.

## Research Context

We are studying rapport building behaviors in K-12 math tutoring conversations. Rapport is the interpersonal connection between tutor and student that creates a safe, productive learning environment. It encompasses how the tutor relates to the student as a person -- showing genuine interest, managing emotions, building trust, and navigating the social dynamics of the session.

Your task is to identify turn ranges where the tutor makes an attempt at building rapport with the student, or where a rapport-building opportunity arises.

## Important: Cast a Wide Net

Identify **5-12 moments** per session. It is better to flag a moment that turns out to be minor than to miss one. Look throughout the ENTIRE transcript, not just at the beginning and end.

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
