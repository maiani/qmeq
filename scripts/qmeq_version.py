"""Print QmeQ's source version without importing the package."""

from __future__ import annotations

import ast
from pathlib import Path


def source_version() -> str:
    init_path = Path(__file__).resolve().parents[1] / "qmeq" / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        if not isinstance(value, str):
            break
        return value
    raise RuntimeError(f"Could not find a literal __version__ in {init_path}")


if __name__ == "__main__":
    print(source_version())
