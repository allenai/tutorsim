"""Access packaged tutorsim runtime assets."""

from importlib.resources import files


def resource_text(relative_path: str) -> str:
    """Read a UTF-8 text resource from the tutorsim package."""
    return (files("tutorsim") / relative_path).read_text(encoding="utf-8")
