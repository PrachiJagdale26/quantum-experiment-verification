"""Test isolation: tests never write to the real research database or the real Chroma store."""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="research_test_")
os.environ["RESEARCH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'test_research.db')}"
os.environ["CHROMA_PERSIST_DIRECTORY"] = os.path.join(_TMP, "chroma")
