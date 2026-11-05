"""T-RET-06 -- the retriever never calls an LLM (INV-RET-05).

Static check: nothing under ``src/retriever`` outside ``eval/`` may import
``rag_common.llm`` or ``rag_common.rewrite``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "retriever"
FORBIDDEN = {"rag_common.llm", "rag_common.rewrite"}


def _iter_py_files(root: Path):
    for path in root.rglob("*.py"):
        if "eval" in path.parts:
            continue
        yield path


def test_no_llm_imports_in_retriever_src():
    offenders = []
    for path in _iter_py_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN:
                        offenders.append((path, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module in FORBIDDEN:
                    offenders.append((path, node.module))
    assert not offenders, f"forbidden imports: {offenders}"


def test_eval_may_import_rewrite():
    # the eval variant module is allowed to import rag_common.rewrite lazily
    variants = SRC / "eval" / "variants.py"
    text = variants.read_text(encoding="utf-8")
    assert "rewrite" in text
