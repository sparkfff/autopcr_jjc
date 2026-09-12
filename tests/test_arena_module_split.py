import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "autopcr" / "module" / "modules"


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class ArenaModuleSplitTests(unittest.TestCase):
    expected = {
        "jjc_daily",
        "pjjc_daily",
        "jjc_back",
        "pjjc_back",
        "jjc_info",
        "pjjc_info",
        "pjjc_def_shuffle_team",
        "pjjc_atk_shuffle_team",
    }

    def test_arena_module_contains_all_custom_features(self):
        classes = {
            node.name
            for node in parse(MODULES / "arena.py").body
            if isinstance(node, ast.ClassDef)
        }
        self.assertTrue(self.expected.issubset(classes))

    def test_custom_features_are_not_defined_in_daily_or_tools(self):
        for filename in ("daily.py", "tools.py"):
            classes = {
                node.name
                for node in parse(MODULES / filename).body
                if isinstance(node, ast.ClassDef)
            }
            self.assertTrue(self.expected.isdisjoint(classes), filename)

    def test_arena_tab_is_registered_after_danger(self):
        init_tree = parse(MODULES / "__init__.py")
        arena_assignment = next(
            node
            for node in init_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "arena_modules"
                for target in node.targets
            )
        )
        call = arena_assignment.value
        self.assertEqual(ast.literal_eval(call.args[0]), "竞技场")
        self.assertEqual(ast.literal_eval(call.args[1]), "arena")
        self.assertEqual(
            {item.id for item in call.args[2].elts if isinstance(item, ast.Name)},
            self.expected,
        )

        manager_tree = parse(ROOT / "autopcr" / "module" / "modulelistmgr.py")
        manager = next(
            node
            for node in manager_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModuleListManager"
        )
        modules_assignment = next(
            node
            for node in manager.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "modules"
        )
        keys = [
            key.value.id
            for key in modules_assignment.value.keys
            if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
        ]
        self.assertEqual(keys.index("arena_modules"), keys.index("danger_modules") + 1)


if __name__ == "__main__":
    unittest.main()
