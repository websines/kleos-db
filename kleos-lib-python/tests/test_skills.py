"""The skill library through the Python binding (docs/specs/skills.md)."""
import os

from kleos import CozoDbPy

SKILLS = os.path.join(os.path.dirname(__file__), "..", "..", "skills")


def test_import_find_outcome_and_rollback():
    db = CozoDbPy("mem", "", "{}")
    r = db.skill_call("import", {"path": SKILLS, "activate": True})
    assert len(r["imported"]) >= 4
    f = db.skill_call("find", {"task": "fork memory before a risky experiment", "k": 2})
    assert f["skills"][0]["name"] == "kleos-branches"
    o = db.skill_call("record_outcome", {"name": "kleos-branches", "success": True, "route": f["route"]})
    assert o["success"] is True
    md = "---\nname: kleos-branches\ndescription: Branches.\n---\nv2\n"
    db.skill_call("put", {"skill_md": md, "activate": True})
    assert db.skill_call("rollback", {"name": "kleos-branches"})["status"] == "active"
    e = db.skill_call("entry", {"uri": "skill://kleos-branches/SKILL.md"})
    assert e["resources"][0]["digest"].startswith("sha256:")
