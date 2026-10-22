"""CPU checks for the public source tree; does not import Isaac Sim."""
import ast
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
errors = []
count = 0
for base in (root / "source", root / "scripts", root / "deploy"):
    for path in base.rglob("*.py"):
        count += 1
        text = path.read_text()
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
        if re.search(r"/(?:home|Users)/[A-Za-z0-9_.-]+/", text):
            errors.append(f"Machine-specific path in {path.relative_to(root)}")
for path in (root / "results").rglob("*.json"):
    json.loads(path.read_text())
for path in (root / "README.md", *sorted((root / "docs").glob("*.md"))):
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        if not (path.parent / target.split("#")[0]).exists():
            errors.append(f"Broken relative link in {path.name}: {target}")
registered = []
for path in (root / "source").rglob("*.py"):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            registered.extend(kw.value.value for kw in node.keywords if kw.arg == "id" and isinstance(kw.value, ast.Constant))
assert registered == ["Unitree-G1-23dof-Duet-Velocity"], registered
recipe = json.loads((root / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/duet_velocity/upstream_config.json").read_text())
assert len(recipe["rewards"]) == 22
evidence = json.loads((root / "results/duet_22000/evaluation.json").read_text())
assert len(evidence["results"]) == 19
assert sum(row["total"] for row in evidence["results"].values()) == 304
assert json.loads((root / "results/training_25001.json").read_text())["completed_updates"] == 25001
if errors:
    raise SystemExit("\n".join(errors))
print(f"RELEASE_STATIC_PASS: {count} Python files, links and evidence schema checked")
