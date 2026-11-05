"""T-RET-20 -- dictionary expansion on word boundaries, ambiguity via context."""

from __future__ import annotations

from pathlib import Path

import pytest

from retriever.query.dictionary import Dictionary

DICT_PATH = Path(__file__).resolve().parents[1] / "src" / "retriever" / "dictionaries" / "rudn_synonyms.yaml"


@pytest.fixture
def dictionary() -> Dictionary:
    return Dictionary.from_yaml(DICT_PATH)


def test_slang_expanded(dictionary):
    assert "общежитие" in dictionary.expand("где общага")


def test_word_boundary_no_false_match(dictionary):
    # "общагами" contains "общага" as a prefix but is a different word; the
    # word-boundary rule must not fire the "общага" entry.
    assert "общежитие" not in dictionary.expand("общагами довольны все")


def test_ambiguous_needs_context(dictionary):
    # "ПК" about admissions -> приёмная комиссия
    out = dictionary.expand("сроки ПК для поступления")
    assert "приёмная комиссия" in out
    # "ПК" about courses -> повышение квалификации
    out2 = dictionary.expand("курсы ПК для преподавателей")
    assert "повышение квалификации" in out2


def test_ambiguous_without_context_not_expanded(dictionary):
    # Neither context word present -> no expansion of the ambiguous entry.
    out = dictionary.expand("расскажи про ПК")
    assert "приёмная комиссия" not in out
    assert "повышение квалификации" not in out


def test_dictionary_validates_duplicates(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nentries:\n"
        "  - {term: x, expand: [y], type: slang}\n"
        "  - {term: X, expand: [z], type: slang}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        Dictionary.from_yaml(bad)


def test_ambiguous_requires_context(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nentries:\n  - {term: ПК, expand: [x], type: abbr, ambiguous: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        Dictionary.from_yaml(bad)

