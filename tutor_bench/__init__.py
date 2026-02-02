"""tutor-bench: A benchmarking and evaluation framework for AI tutoring systems."""

from tutor_bench.annotator import Annotator
from tutor_bench.evaluator import Evaluator
from tutor_bench.version import __version__

__all__ = [
    "Annotator",
    "Evaluator",
    "__version__",
]
