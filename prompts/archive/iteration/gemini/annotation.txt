You are an expert in educational assessment and NLP prompt engineering. You are helping iterate on a prompt used to annotate pedagogical key moments in K-12 tutoring transcripts.

## Task Context

An LLM reads a tutoring transcript excerpt around a detected key moment and produces a situation/action/result analysis. A separate labeler then classifies the result text as effective/partial/ineffective. We compare this result against mean-aggregated human teacher annotations. 

## Optimization Goal

The LLM annotator aims to capture different teachers' perspectives, yet its output should nudge the labeller to produce an overall judgement. If teachers are likely to convey that a moment as effective, the LLMs' output should also lean effective, and similarly, if teachers are likely to deem a moment as ineffective, the LLM's output should also lean ineffective. If teachers provide mixed responses for a contentious moment, then the LLM's output should convey partial effectiveness, and provide evidence for and against effectiveness. 

Prompt guidelines for the LLM should reflect the types of considerations teachers make, and cover the range of situations and actions that teachers assess.

## Current Metrics ({ann_type})

{current_metrics}

## Current Prompt

```
{current_prompt}
```

## Error Statistics

- Total matched pairs: {total_pairs}
- Agreements: {agreements}
- Disagreement breakdown: {confusion_summary}

## Examples

Below are examples of agreements and disagreements between teachers and the LM. 

{error_examples}

## Your Task

Analyze the disagreement examples above and identify:

1. **Semantic patterns**: Does the LM consistently rate too harshly or too generously? Read the action and result descriptions carefully -- focus on what the LLM writes and argues differently, and not just the label.

2. **Root causes**: What factors, criteria, or context do teachers weigh that the LM is overlooking? What types of evidence suggest a tutor's strategy to be effective or ineffective? What leads to a mixed and/or partial outcome among teachers? 

3. **Proposed changes**: For each root cause, propose a SPECIFIC edit to the prompt. Fit proposed edits under the following prompt headings: "Your Task", "Ineffectiveness Considerations", and "Effectiveness Considerations". If changing text, quote the exact text to change and provide the replacement. 

Respond in JSON:
{{
  "patterns": [
    {{
      "name": "pattern name",
      "direction": "too_harsh | too_generous",
      "share_of_errors": "approximate % of disagreements this explains",
      "description": "what the LLM gets wrong and why",
      "examples_referenced": ["Disagreement 1", "Disagreement 3"]
    }}
  ],
  "proposed_changes": [
    {{
      "target_pattern": "which pattern this addresses",
      "change_type": "short description of the type of edit (e.g. modify section, add calibration note, remove text, restructure, add counterfactual)",
      "current_text": "exact text in prompt to replace (or null for additions)",
      "proposed_text": "replacement text (or null for deletions)",
      "rationale": "why this specific change helps",
      "directional_effect": "makes LLM harsher | makes LLM more generous | neutral",
      "expected_impact": "which annotation type benefits and which might regress"
    }}
  ],
  "overall_assessment": "1-2 sentence summary of the main bias and whether prompt changes can fix it"
}}
