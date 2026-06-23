"""Student prompt classes -- thin shims over prompt files on disk.

The actual prompt text lives in `prompts/benchmark/v6/students/*.txt`. This
module is a verbatim port of synth-students' class structure so calling code
can keep saying `SimpleMultiTurnStudentPrompt(num_turns).get_system_prompt()`.
The class bodies are now one-liners that load the corresponding .txt and
substitute `{shared_generation_instructions}` with the parametrized helper
below. All other placeholders (`[[NEXT_CONVERSATION_INFORMATION_HERE]]`,
`[[EXAMPLE_CONVERSATION_HERE]]`, `[[PERSONA_DESCRIPTION_HERE]]`,
`[[STUDENT_DESCRIPTION_HERE]]`) are filled by `benchmark/core/students.py`
after this class returns its template.

`get_shared_generation_instructions` stays here because it's a small
parametrized helper (4 conditional variants based on convo_length + example
inclusion) -- not a prompt body. It's substituted into the file-loaded
template via the `{shared_generation_instructions}` placeholder.
"""
from pathlib import Path
from typing import Optional

# TUTOR_NAME is the single dep from src/global_vars.py; inlined here to avoid
# pulling synth-students' globals module. Value preserved verbatim.
TUTOR_NAME = "TUTOR"

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "benchmark"
_PROMPT_VERSION = "v8"  # bump if a new student prompt set ships


def _load_student_template(mode: str) -> str:
    """Load prompts/benchmark/{version}/students/{mode}.txt."""
    path = _PROMPTS_DIR / _PROMPT_VERSION / "students" / f"{mode}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def get_shared_generation_instructions(
    convo_length: Optional[int] = None, include_example: bool = True
):
    # convo_length and include_example are accepted for synth-students API
    # compatibility but no longer used: our pipeline caps turns via max_turns
    # and doesn't take an 'end' key from the student, so the original
    # length/end-key sentences were incoherent in our context and have been
    # stripped (see chat 2026-06-15).
    return "You may generate multiple student turns in a row as needed. Do *not* generate any turns as the tutor. You should only generate turns that involve student utterances or actions. Remember to wait for the tutor to respond before generating your next turns."


def _render(mode: str, num_turns: Optional[int], include_example: bool) -> str:
    template = _load_student_template(mode)
    shared = get_shared_generation_instructions(num_turns, include_example=include_example)
    return template.replace("{shared_generation_instructions}", shared)


class ExpertMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("expert", num_turns, include_example=False)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class SimpleMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("simple", num_turns, include_example=False)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class ImitateExampleMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("imitate_example", num_turns, include_example=True)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class TraitWithExampleMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("trait_with_example", num_turns, include_example=True)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class TraitMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("trait", num_turns, include_example=False)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class ParaphraseWithExampleMultiTurnStudentPrompt:
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("paraphrase_with_example", num_turns, include_example=True)

    def get_system_prompt(self) -> str:
        return self.system_prompt


class OracleMomentStudentPrompt:
    """Oracle student: imitate the real student using the full post-cut
    transcript as an example conversation, plus a generated trait persona.
    Modeled on ImitateExampleMultiTurnStudentPrompt (synth-students) with
    an added persona section.
    """
    def __init__(self, num_turns: Optional[int] = None):
        self.num_turns = num_turns
        self.system_prompt = _render("oracle", num_turns, include_example=True)

    def get_system_prompt(self) -> str:
        return self.system_prompt
