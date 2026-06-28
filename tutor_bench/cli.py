"""Command-line entry point for tutor-bench.

Thin re-export of the benchmark CLI so the ``tutor-bench`` console script
dispatches the benchmark subcommands (``run``, ``report``, ``view``, ...).
"""

from tutor_bench.benchmark.cli import main

if __name__ == "__main__":
    main()
