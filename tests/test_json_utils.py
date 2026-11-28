"""LLM JSON parsing must survive LaTeX backslashes, code fences and surrounding prose."""

import pytest

from src.llm.json_utils import parse_llm_json


def test_latex_backslashes_are_kept_literally():
    raw = '```json\n{"items": [{"text": "S_A^{(2)} = -\\log \\mathrm{Tr} \\rho_A^2", "evidence": "\\alpha"}]}\n```'
    data = parse_llm_json(raw)
    assert data["items"][0]["text"] == "S_A^{(2)} = -\\log \\mathrm{Tr} \\rho_A^2"
    assert data["items"][0]["evidence"] == "\\alpha"


def test_latex_that_starts_with_a_valid_escape_letter_is_not_a_control_character():
    data = parse_llm_json('{"t": "\\theta + \\tau + \\nu + \\beta + \\frac{1}{2} + \\upsilon"}')
    assert data["t"] == "\\theta + \\tau + \\nu + \\beta + \\frac{1}{2} + \\upsilon"
    assert "\r" not in data["t"] and "\t" not in data["t"]


def test_already_escaped_backslashes_are_untouched():
    assert parse_llm_json('{"t": "a \\\\rho b"}')["t"] == "a \\rho b"


def test_valid_escapes_still_decode_and_prose_is_ignored():
    data = parse_llm_json('Here you go: {"a": "line\\nbreak \\u00e9 \\"q\\""} thanks')
    assert data == {"a": 'line\nbreak \u00e9 "q"'}


def test_no_json_raises():
    with pytest.raises(ValueError):
        parse_llm_json("I could not find anything.")
