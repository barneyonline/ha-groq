"""Install only requirements declared by Groq's pinned Core dependency closure."""

import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

import homeassistant
from packaging.requirements import Requirement

component_root = Path(homeassistant.__file__).parent / "components"
manifest = json.loads(Path("/opt/groq-test/manifest.json").read_text())
pending = [*manifest["dependencies"], *manifest.get("after_dependencies", [])]
components = {}
while pending:
    domain = pending.pop(0)
    if domain in components:
        continue
    data = json.loads((component_root / domain / "manifest.json").read_text())
    components[domain] = {
        "dependencies": data.get("dependencies", []),
        "requirements": data.get("requirements", []),
    }
    pending.extend(components[domain]["dependencies"])
requirements = sorted(
    {value for item in components.values() for value in item["requirements"]}
)
assert (
    len(components) <= 20 and len(requirements) <= 12
), "Dependency closure exceeded prototype bound"
Path("/opt/groq-test/component-requirements.json").write_text(
    json.dumps(components, indent=2) + "\n"
)
print(
    json.dumps(
        {"components": list(components), "requirements": requirements}, indent=2
    ),
    flush=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-compile",
        "--constraint",
        "/opt/groq-test/core-constraint.txt",
        *requirements,
    ],
    check=True,
)

# The integration's own optional requirement is distinct from Core's closure.
for value in manifest["requirements"]:
    requirement = Requirement(value)
    actual = version(requirement.name)
    if actual not in requirement.specifier:
        raise RuntimeError(
            f"{requirement.name}=={actual} violates {requirement.specifier}"
        )
