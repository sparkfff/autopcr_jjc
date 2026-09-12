import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES_INIT = ROOT / "autopcr" / "module" / "modules" / "__init__.py"


def module_list_members(name: str):
    tree = ast.parse(MODULES_INIT.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return {
        item.id
        for item in assignment.value.args[2].elts
        if isinstance(item, ast.Name)
    }


class LabyrinthIntegrationTests(unittest.TestCase):
    def test_labyrinth_run_is_registered_only_as_dangerous(self):
        self.assertIn("labyrinth_run", module_list_members("danger_modules"))
        self.assertNotIn("labyrinth_run", module_list_members("tool_modules"))

    def test_existing_labyrinth_module_is_not_used_as_a_runner_dependency(self):
        source = (
            ROOT / "autopcr" / "module" / "modules" / "labyrinth_run.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from .labyrinth import", source)


if __name__ == "__main__":
    unittest.main()
