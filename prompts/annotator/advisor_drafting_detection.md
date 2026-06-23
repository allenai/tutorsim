You are an expert in educational assessment and NLP prompt engineering. You are helping draft guidelines for a prompt used to detect pedagogical key moments in K-12 tutoring transcripts.

## Task Context

An LLM reads full tutoring transcripts and identifies turn ranges where notable {ann_type} events occur. These detections are compared against human expert annotations. 

## Optimization Goal

We want to maximize BOTH cluster recall (catching all human-annotated moments) and moment precision (not flagging things humans didn't annotate), with recall weighted slightly higher. 

Prompt guidelines for the LLM should reflect the types of considerations teachers make when selecting a moment span, and cover the range of situations and actions that teachers identify. 

## Current Prompt

```
{current_prompt}
```

## Examples

Below are examples of annotations made by teachers, each showing a key moment and a teacher's annotation for that moment. Ignore descriptions that state that a key moment isn't actually a {ann_type} key moment; treat those as noise. 

{teacher_examples}

## Your Task

Analyze the key moments above and identify:

1. **Semantic patterns**: What types of moments do the teachers identify? Why might other transcript interactions not count as a key moment?

2. **Root causes**: What patterns describe the inclusion or exclusion of certain tutoring interactions as key moments? 

3. **Proposed changes**: For each root cause, propose a SPECIFIC edit to the prompt. There are several areas in the current prompt that are missing useful information; focus on adding to those. If changing text, quote the exact text to change and provide the replacement. 

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
