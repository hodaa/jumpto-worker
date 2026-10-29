"""Normalize provider LaTeX into the subset matplotlib mathtext accepts."""

from __future__ import annotations

import re

_LE_ALIAS = re.compile(r"\\le(?![a-zA-Z])")
_GE_ALIAS = re.compile(r"\\ge(?![a-zA-Z])")
_CONTROL_ESCAPES = {
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
}
_ENVIRONMENT = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}")
_ROW_SEPARATOR = re.compile(r"\\\\")
_ALIGNMENT = re.compile(r"&")
_SIZE_COMMAND = re.compile(r"\\(?:Bigg|bigg|Big|big)[lrm]?(?![a-zA-Z])")
_STYLE_COMMAND = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle)\s*")
_TEXT_COMMAND = re.compile(r"\\text\{([^{}]*)\}")
_EMPTY_FRACTION = re.compile(r"\\frac\s*\{[^{}]*\}\s*\{\s*\}|\\frac\s*\{\s*\}\s*\{[^{}]*\}")
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_FRAC_TOKEN = r"(\{[^{}]*\}|\\[a-zA-Z]+|[^{}\s])"
_FRACTION = re.compile(rf"\\frac\s*{_FRAC_TOKEN}\s*{_FRAC_TOKEN}")


def _brace_fraction(match: re.Match[str]) -> str:
    """Brace both arguments of an unbraced \\frac shorthand."""
    numerator = match.group(1).strip("{}")
    denominator = match.group(2).strip("{}")
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def restore_latex_escapes(latex: str) -> str:
    """Undo JSON control escapes a model used for single-backslash LaTeX.

    A model that writes ``\\frac`` with one backslash in JSON produces a form
    feed once parsed (``\x0crac``), and likewise for ``\\times``, ``\\neq``,
    ``\\begin``, and friends. Restoring the control characters to their
    backslash forms recovers the command the model meant.
    """
    for control, escaped in _CONTROL_ESCAPES.items():
        latex = latex.replace(control, escaped)
    return latex


def _drop_non_ascii_text(match: re.Match[str]) -> str:
    """Keep ASCII ``\\text`` labels; mathtext cannot draw Arabic ones."""
    content = match.group(1)
    return match.group(0) if content.isascii() else ""


def _flatten_environment(latex: str) -> str:
    """Turn a LaTeX environment such as ``cases`` into a single mathtext line."""
    latex = _ENVIRONMENT.sub("", latex)
    latex = _ROW_SEPARATOR.sub(r",\\quad ", latex)
    return _ALIGNMENT.sub("", latex)


def normalize_math(latex: str) -> str:
    """Rewrite constructs that matplotlib mathtext does not accept."""
    latex = restore_latex_escapes(latex)
    latex = _flatten_environment(latex)
    latex = _SIZE_COMMAND.sub("", latex)
    latex = _STYLE_COMMAND.sub("", latex)
    latex = _TEXT_COMMAND.sub(_drop_non_ascii_text, latex)
    latex = _EMPTY_FRACTION.sub("", latex)
    latex = _ARABIC.sub("", latex)
    latex = _FRACTION.sub(_brace_fraction, latex)
    latex = _LE_ALIAS.sub(r"\\leq", latex)
    return _GE_ALIAS.sub(r"\\geq", latex)
