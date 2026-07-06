"""Read packaged resource files (prompts) from the tutorsim_build package.

Mirrors tutorsim.resources but anchored at files("tutorsim_build") — the
runtime helper is hard-wired to the tutorsim package and cannot read
build-only prompts.
"""

from importlib.resources import files


def resource_text(relative_path: str) -> str:
    """Return the text of a resource file inside the tutorsim_build package."""
    return (files("tutorsim_build") / relative_path).read_text(encoding="utf-8")
