You are an expert in educational assessment and NLP prompt engineering. You are helping iterate on a prompt used to detect pedagogical key moments in K-12 tutoring transcripts.

## Task Context

An LLM reads full tutoring transcripts and identifies turn ranges where notable {ann_type} events occur. These detections are compared against human teacher annotations using IoU matching.

## Optimization Goal

We want detections to catch all teacher-annotated moments and avoid false positives. Recall is MORE important than precision, as teachers may miss relevant key moments. Mean IoU measures boundary quality of matched detections.

## Current Metrics ({ann_type})

{current_metrics}

## Current Prompt

```
{current_prompt}
```

## Error Counts

- Correct matches: {good_matches} human-annotated moments the LLM correctly detected (IoU >= 0.5)
- Complete misses: {complete_misses} human-annotated moments the LLM entirely missed
- Near-misses: {near_misses} moments where LLM detected nearby but wrong turn range (IoU < 0.5)
- False positives: {false_positives} LLM detections with no matching human annotation

## Examples

Below are examples of correct detections, misses, near-misses, and false positives, each with transcript excerpts and human annotations. Lines marked with <<< are within the annotated moment.

{error_examples}

## Your Task

Analyze the error examples above and identify:

1. **Semantic patterns**: Read transcript excerpts carefully. What types of interactions do human teachers typically not annotate, and what types do they annotate that the LM misses? How can the LLM avoid near-misses? 

2. **Root causes**: For each pattern, explain WHY the current prompt fails to catch it.

3. **Proposed changes**: For each root cause, propose a SPECIFIC edit to the prompt. Quote the exact text to change and provide the replacement. You may add, modify, or remove text. 

4. **Risk assessment**: For each proposed change, estimate the risk of regression on currently-correct detections. 

Respond in JSON:
{{
  "proposed_changes": [
    {{
      "target_pattern": "which pattern this addresses",
      "share_of_errors": "approximate proportion of errors this explains", 
      "current_text": "exact text in prompt to replace (or null for additions)",
      "proposed_text": "replacement text (or null for deletions)",
      "rationale": "why this specific change helps",
      "regression_risk": "low | medium | high",
    }}
  ],
  "overall_assessment": "1-2 sentence summary"
}}
