"""tutor-bench: A benchmarking and evaluation framework for AI tutoring systems."""

from tutor_bench.benchmark import register_student, register_tutor
from tutor_bench.version import __version__

__all__ = [
    "register_tutor",
    "register_student",
    "__version__",
]
