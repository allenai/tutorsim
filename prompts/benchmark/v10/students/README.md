# Student prompts (benchmark v6)

Mode templates loaded by `benchmark/core/students.py:build_student_system_prompt`:

| File | Mode | Placeholders |
|---|---|---|
| `simple.txt` | `simple` | `[[NEXT_CONVERSATION_INFORMATION_HERE]]`, `{shared_generation_instructions}` |
| `expert.txt` | `expert` | same |
| `imitate_example.txt` | `imitate_example` | + `[[EXAMPLE_CONVERSATION_HERE]]` |
| `paraphrase_with_example.txt` | `paraphrase_with_example` | + `[[EXAMPLE_CONVERSATION_HERE]]` |
| `trait.txt` | `trait`, `<dim>-<n>`, `joined-<n>` | + `[[PERSONA_DESCRIPTION_HERE]]` |
| `trait_with_example.txt` | `trait_with_example` | + `[[EXAMPLE_CONVERSATION_HERE]]`, `[[STUDENT_DESCRIPTION_HERE]]` |

## Sub-directories

- `trait_generator/system.txt` + `trait_generator/user.txt` — prompts the trait-mode pipeline sends to the LLM to *generate* a persona from a transcript prefix (oracle-safe). Loaded by `benchmark/synth_students/traits.py:TraitGenerator.get_trait_system_prompt`.
- `dimensions/*.txt` — descriptions of the five behavior dimensions (`distractedness`, `active_vs_passive_learning`, `affect`, `misconceptions`, `learning_efficiency`). Loaded by `benchmark/synth_students/dimension.py:get_dimension_description`. The `joined` and `joined_*` modes concatenate multiple of these and orchestration stays in code.

## What's still in code (and why)

- `get_shared_generation_instructions` in `benchmark/synth_students/prompts.py` — small parametrized helper that selects one of four short instruction variants based on `(convo_length, include_example)`. The four variants are tiny and conditional logic dominates. Substituted into the `{shared_generation_instructions}` placeholder.
- `joined` / `joined_misconceptions_distractedness` / `joined_misconceptions_affect` orchestration in `traits.py` — generates per-dimension personas in parallel and concatenates them. Pure orchestration; no new prompt text.
- Sentence-count suffix (`The description should be N sentences long.`) — appended to the trait-generator system prompt via the `{sentence_count_suffix}` placeholder.
