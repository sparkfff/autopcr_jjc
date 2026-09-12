import unittest
from types import SimpleNamespace

from autopcr.model.common import LabyrinthMapInfo, LabyrinthStatus
from autopcr.model.enums import eLabyrinthBlockType as Block, eLabyrinthStatusType as Status
from autopcr.model.responses import LabyrinthResumeResponse
from autopcr.module.modules.labyrinth_run import labyrinth_run
from autopcr.util.labyrinth import parse_run_command_args

GUILD_NAMES = {1: "美食殿堂", 2: "破晓之星", 5: "拉比林斯"}


def find_test_guild(token):
    if token.isdigit():
        guild_id = int(token)
        return guild_id if guild_id in GUILD_NAMES else None
    for guild_id, name in GUILD_NAMES.items():
        if token in name:
            return guild_id
    return None


class LabyrinthCommandTests(unittest.TestCase):
    def test_no_arguments_keep_web_settings(self):
        args = []
        self.assertEqual(parse_run_command_args(args), {})
        self.assertEqual(args, [])

    def test_route_and_bare_target_override_only_requested_settings(self):
        args = ["完美路线", "45000"]
        self.assertEqual(
            parse_run_command_args(args),
            {
                "labyrinth_run_route": "完美路线",
                "labyrinth_run_target_score": 45000,
            },
        )
        self.assertEqual(args, [])

    def test_guild_before_route(self):
        args = ["美食殿堂", "完美路线", "45000"]
        self.assertEqual(
            parse_run_command_args(args, find_test_guild),
            {
                "labyrinth_run_guild_id": 1,
                "labyrinth_run_route": "完美路线",
                "labyrinth_run_target_score": 45000,
            },
        )
        self.assertEqual(args, [])

    def test_guild_id_is_not_treated_as_target_score(self):
        args = ["5", "性价比"]
        self.assertEqual(
            parse_run_command_args(args, find_test_guild),
            {
                "labyrinth_run_guild_id": 5,
                "labyrinth_run_route": "性价比",
            },
        )
        self.assertEqual(args, [])

    def test_guild_name_substring(self):
        args = ["破晓"]
        self.assertEqual(
            parse_run_command_args(args, find_test_guild),
            {"labyrinth_run_guild_id": 2},
        )
        self.assertEqual(args, [])

    def test_rejects_conflicting_guilds(self):
        with self.assertRaisesRegex(ValueError, "只能选择一个公会"):
            parse_run_command_args(["1", "破晓之星"], find_test_guild)

    def test_guild_is_required_when_lookup_is_provided(self):
        for args in ([], ["完美路线"], ["完美路线", "45000"]):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "请指定公会"):
                    parse_run_command_args(list(args), find_test_guild)

    def test_no_requirement_and_labeled_target(self):
        for args in (["不要求", "目标分数", "46800"], ["随机开局", "目标:46800"]):
            with self.subTest(args=args):
                tokens = list(args)
                self.assertEqual(
                    parse_run_command_args(tokens),
                    {
                        "labyrinth_run_route": "不要求",
                        "labyrinth_run_target_score": 46800,
                    },
                )
                self.assertEqual(tokens, [])

    def test_unknown_tokens_are_left_for_the_common_error_handler(self):
        args = ["不要求", "未知参数"]
        self.assertEqual(
            parse_run_command_args(args),
            {"labyrinth_run_route": "不要求"},
        )
        self.assertEqual(args, ["未知参数"])

    def test_rejects_conflicting_or_invalid_values(self):
        for args, message in (
            (["完美路线", "性价比"], "只能选择一个"),
            (["不要求", "完美"], "只能选择一个"),
            (["目标分数"], "缺少数字"),
            (["45123"], "100倍数"),
            (["100100"], "0～100000"),
            (["45000", "目标45000"], "只能设置一次"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, message):
                    parse_run_command_args(list(args))

    def test_perfect_alias_and_value_route(self):
        for args, route in ((["完美"], "完美路线"), (["性价比"], "性价比")):
            with self.subTest(args=args):
                tokens = list(args)
                self.assertEqual(
                    parse_run_command_args(tokens),
                    {"labyrinth_run_route": route},
                )
                self.assertEqual(tokens, [])


class LabyrinthBotOverrideTests(unittest.TestCase):
    def make_module(self):
        config = {
            "labyrinth_run_route": "完美路线",
            "labyrinth_run_preference": "遗物优先",
        }
        module = labyrinth_run(
            SimpleNamespace(id="command-test", get_config=lambda key, default=None: config.get(key, default))
        )
        module.get_config = config.__getitem__
        module.strategy = "完美路线"
        module.expected = module._build_expected_block_types("任意")
        return module

    def make_graph(self):
        graph = []
        for area, columns in labyrinth_run.AREA_REQUIREMENTS.items():
            for column, kinds in columns.items():
                kind = Block.RELIC if Block.RELIC in kinds else next(iter(kinds))
                graph.append(
                    LabyrinthMapInfo(
                        block_id=area * 10000 + column * 100 + 1,
                        area=area,
                        column=column,
                        row=1,
                        block_type=kind,
                        is_visited=0,
                        next_block_id_list=[],
                    )
                )
        for current, following in zip(graph, graph[1:]):
            current.next_block_id_list = [following.block_id]
        return graph

    def test_no_reroll_accepts_imperfect_map_as_score_route(self):
        module = self.make_module()
        graph = self.make_graph()
        next(block for block in graph if block.area == 1 and block.column == 3).block_type = Block.RELIC
        snapshot = LabyrinthResumeResponse(
            block_id=graph[0].block_id,
            map_list=graph,
            status=LabyrinthStatus(type=Status.NONE),
        )

        module.reroll_opening = True
        self.assertIsNone(module._opening_route(snapshot, 5))
        module.reroll_opening = False
        self.assertIsNotNone(module._opening_route(snapshot, 5))
        self.assertFalse(module.pin_opening_route)


if __name__ == "__main__":
    unittest.main()
