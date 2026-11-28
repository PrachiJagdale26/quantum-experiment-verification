"""Robust parsing of JSON returned by LLMs (code fences, LaTeX backslashes, text around the object)."""

import json
import re
from typing import Any

# Backslashes to keep literally:
#  - ones that do not start a valid JSON escape (LaTeX "\alpha", "\log", "\mathrm");
#  - known LaTeX commands that happen to start with a valid escape letter ("\rho", "\tau", "\theta", "\nu",
#    "\beta", "\frac", "\rangle"); otherwise "\rho" would silently decode to a carriage return + "ho".
#    Only listed command names count, so a real newline before a word ("line\nbreak") still decodes;
#  - "\u" not followed by four hex digits ("\upsilon").
# A raw backslash already doubled ("\\") is left alone.
_LATEX_COMMANDS = ("beta|bar|bra|big|bf|boldsymbol|binom|bullet|frac|forall|flat|nu|nabla|neq|not|nonumber|ni|"
                   "rho|rangle|right|rm|rvert|rceil|rfloor|tau|theta|text|times|tilde|top|to|tr|triangle|tfrac")
_LITERAL_BACKSLASH = re.compile(
    r'(?<!\\)\\(?:(?!["\\/bfnrtu])|(?=(?:' + _LATEX_COMMANDS + r')(?![a-z]*[A-Z]))|(?=u(?![0-9a-fA-F]{4})))')


def parse_llm_json(raw: str) -> Any:
    """
    Parse a model's JSON answer. Handles markdown fences, text around the JSON object, and invalid
    escapes such as LaTeX in physics text, which are kept as literal backslashes.
    Raises ValueError if no JSON object can be recovered.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    candidate = _LITERAL_BACKSLASH.sub(r"\\\\", text[start:end + 1])
    return json.loads(candidate)
