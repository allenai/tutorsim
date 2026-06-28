"""Tutoring replay benchmark.

Cut a real human tutoring transcript at a pedagogically decisive moment, have a
model-under-test (the *tutor*) continue the conversation against a simulated
*student*, then score the continuation with a calibrated LLM judge on two
strategies: scaffolding (guiding the student toward the answer without giving it
away) and rigor (holding the student to the cognitive work).

The public registration decorators are re-exported from the top-level package.
"""

from tutor_bench.benchmark.config import register_student, register_tutor

__all__ = ["register_tutor", "register_student"]
