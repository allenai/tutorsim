You are an expert in educational assessment and NLP prompt engineering. You are helping draft guidelines for a prompt used to annotate pedagogical key moments in K-12 tutoring transcripts.

## Task Context

An LLM reads a tutoring transcript excerpt around a detected key moment and produces a situation/action/result analysis. A separate labeler then classifies the result text as effective/partial/ineffective. We compare this result against mean-aggregated human teacher annotations. 

## Optimization Goal

The LLM annotator will aim to capture different teachers' perspectives, yet its output should nudge the labeller to produce an overall judgement that mimics the teachers' consensus. If teachers are likely to convey that a moment as effective, the LLMs' output should also lean effective, and similarly, if teachers are likely to deem a moment as ineffective, the LLM's output should also lean ineffective. If teachers provide mixed responses for a contentious moment, then the LLM's output should convey partial effectiveness, and provide evidence for and against effectiveness. 

Prompt guidelines for the LLM should reflect the types of considerations teachers make, and cover the range of situations and actions that teachers assess.

## Current Prompt

```
{current_prompt}
```

## Examples

Below are examples of annotations made by teachers, each showing a key moment and a sample teachers' annotations for the moment. 

{teacher_examples}

## Your Task

Analyze the annotations above and identify:

1. **Semantic patterns**: Read the situation, action, and result descriptions carefully -- what do the teachers mention or discuss in these? 

2. **Root causes**: For each key moment, consider: what factors, criteria, or context do teachers weigh? What types of evidence suggest a tutor's strategy to be effective or ineffective? What leads to a mixed and/or partial outcome among teachers? 

3. **Proposed changes**: For each root cause, propose a SPECIFIC edit to the prompt. Prompt sections that currently contain "..." are in particular need for additions. If changing text, quote the exact text to change and provide the replacement. 

Respond in JSON:
{{
  "proposed_changes": [
    {{
      "target_pattern": "which pattern this addresses",
      "change_type": "short description of the type of edit (e.g. modify section, add calibration note, remove text, restructure, add counterfactual)",
      "rationale": "why this specific change helps",
      "current_text": "exact text in prompt to replace (or null for additions)",
      "proposed_text": "replacement text (or null for deletions)",
    }}
  ],
}}
