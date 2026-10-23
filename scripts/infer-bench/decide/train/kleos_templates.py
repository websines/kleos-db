"""The decider-nano prompts exactly as kleos renders them (kleos-core/src/decide.rs,
`templates::spec`, Family::Nano). Training uses these strings so the Rust path needs no
prompt change. check_against_rust() greps decide.rs to catch drift."""
import re
from pathlib import Path

NLI_OPTIONS = [
    "entailment: The hypothesis follows from the premise",
    "neutral: The hypothesis might be true or false given the premise",
    "contradiction: The hypothesis contradicts the premise",
]
NLI_Q = "What is the relationship between the premise and the hypothesis?"
CNLI_OPTIONS = [
    "Entailment: The old note implies the new statement is true",
    "Contradiction: The old note and the new statement cannot both be true",
    "NotMentioned: The old note neither implies nor contradicts the new statement",
]
ENTITY_OPTIONS = [
    "same: both mentions refer to one and the same entity",
    "different: they are different entities that may share a name",
    "unclear: there is not enough information to tell",
]
ENTITY_Q = "Are mention A and mention B the same entity?"

ENT, NEU, CON = 0, 1, 2          # NLI option indices
C_ENT, C_CON, C_NM = 0, 1, 2     # CNLI option indices
SAME, DIFF, UNCLEAR = 0, 1, 2


def nli(a, b):
    """supersedes / contradicts / duplicate (Family::Nano)."""
    return f"Premise: {a}\nHypothesis: {b}", NLI_Q, NLI_OPTIONS


def cnli(old, new):
    """relation (both families)."""
    return old, f"Classify the relationship between the old note and this new statement:\n{new}", CNLI_OPTIONS


def entity(a, b):
    """same_entity@v2."""
    return f"Mention A: {a}\nMention B: {b}", ENTITY_Q, ENTITY_OPTIONS


# allowed NLI / CNLI options per memory relation class
NLI_TARGET = {"update": [CON], "contradiction": [CON], "compatible": [NEU], "duplicate": [ENT]}
CNLI_TARGET = {"update": [C_CON], "contradiction": [C_CON], "compatible": [C_NM], "duplicate": [C_ENT]}


def check_against_rust(decide_rs):
    src = Path(decide_rs).read_text()
    for s in NLI_OPTIONS + CNLI_OPTIONS + ENTITY_OPTIONS + [NLI_Q, ENTITY_Q]:
        assert f'"{s}"' in src, f"template drift: {s!r} not in decide.rs"
    assert "Premise: {a}\\nHypothesis: {b}" in src
    assert "Classify the relationship between the old note and this new statement:\\n{new}" in src
    assert "Mention A: {a}\\nMention B: {b}" in src
    return True
