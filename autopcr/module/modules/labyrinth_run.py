"""State-driven Labyrinth runner with score-oriented choices and quick settlement."""

from random import choice, randint
import json
from typing import Dict, List, Set
from ..modulebase import Module, description, name
from ..config import LabyrinthGuildConfig, singlechoice, inttype, booltype
from ...db.database import db
from ...model.error import AbortError
from ...model.enums import (
    eLabyrinthStatusType as Status,
    eInventoryType as Inventory,
    eLabyrinthBlockType as Block,
)
from ...model.common import (
    UnitHpInfo,
    TowerWaveResultInfo,
    TowerWaveResultUnitInfo,
    DungeonQueryUnit,
)
from ...util.labyrinth import (
    DIFFICULTY_SCORE_RATE,
    VALUE_ROUTE_DIFFICULTIES,
    battle_decks,
    best_route,
    event_choice_score,
    reward_score,
    route_value,
    score_shop_plan,
    shop_plan,
    shop_reset_cost,
    unit_rank,
)
from ...util.questutils import create_battle_start_token
from ...util.labyrinth_state import LabyrinthState


LABYRINTH_BLOCK_TYPE_NAME = {
    Block.NONE: "起点",
    Block.NORMAL_QUEST: "普通怪物",
    Block.HARD_QUEST: "EX怪物",
    Block.TICKET: "角色",
    Block.EVENT: "事件",
    Block.RELIC: "遗物",
    Block.SHOP: "商店",
    Block.BOSS_QUEST: "Boss",
}


class LabyrinthRouteMixin:
    """通关模块自用的地图和路线基础方法，不改动已有刷开局模块。"""

    AREA_REQUIREMENTS: Dict[int, Dict[int, Set[Block]]] = {
        1: {
            1: {Block.NONE},
            2: {Block.NORMAL_QUEST},
            3: {Block.TICKET},
            4: {Block.NORMAL_QUEST},
            5: {Block.TICKET},
            6: {Block.RELIC},
        },
        2: {
            1: {Block.NONE},
            2: {Block.TICKET},
            3: {Block.NORMAL_QUEST},
            4: {Block.RELIC},
            5: {Block.HARD_QUEST},
            6: {Block.TICKET},
            7: {Block.RELIC},
        },
        3: {
            1: {Block.NONE},
            2: {Block.NORMAL_QUEST},
            3: {Block.EVENT, Block.RELIC},
            4: {Block.TICKET},
            5: {Block.HARD_QUEST},
            6: {Block.SHOP},
            7: {Block.BOSS_QUEST},
        },
        4: {
            1: {Block.NONE},
            2: {Block.TICKET},
            3: {Block.HARD_QUEST},
            4: {Block.EVENT},
            5: {Block.HARD_QUEST},
            6: {Block.NORMAL_QUEST},
            7: {Block.TICKET},
            8: {Block.SHOP},
        },
        5: {
            1: {Block.NONE},
            2: {Block.NORMAL_QUEST},
            3: {Block.EVENT, Block.RELIC},
            4: {Block.HARD_QUEST},
            5: {Block.RELIC},
            6: {Block.HARD_QUEST},
            7: {Block.SHOP},
            8: {Block.BOSS_QUEST},
        },
    }

    def _block_type(self, block) -> int:
        return int(block.ChangeBlockType or block.block_type or 0)

    def _target_areas(self, difficulty: int) -> List[int]:
        if difficulty == 1:
            return [1, 2, 3]
        return sorted(self.AREA_REQUIREMENTS)

    def _max_unlocked_difficulty(self, top) -> int:
        cleared = [
            info.difficulty
            for info in (top.guild_cleared_difficulty_list or [])
            if info.difficulty is not None
        ]
        if not cleared:
            return 1
        return min(max(cleared) + 1, 5)

    def _build_expected_block_types(
        self, third_block_type: str
    ) -> Dict[int, Dict[int, Set[Block]]]:
        expected = {
            area: {column: set(types) for column, types in requirements.items()}
            for area, requirements in self.AREA_REQUIREMENTS.items()
        }
        third = {
            "遗物": {Block.RELIC},
            "事件": {Block.EVENT},
            "任意": {Block.EVENT, Block.RELIC},
        }.get(third_block_type, {Block.EVENT, Block.RELIC})
        expected[3][3] = set(third)
        expected[5][3] = set(third)
        return expected

    def _boss_unit_ids(self, block) -> Set[int]:
        quest_id = getattr(block, "quest_id", None)
        quest = db.labyrinth_quest_data.get(quest_id) if quest_id else None
        wave_group = db.labyrinth_wave_group_data.get(quest.wave_group_id) if quest else None
        if not wave_group:
            return set()
        return {
            enemy.unit_id
            for enemy in map(db.labyrinth_enemy_parameter.get, wave_group.get_enemy_ids())
            if enemy
        }

    def _boss_names(self, block) -> str:
        units = self._boss_unit_ids(block)
        boss_info = db.labyrinth_boss_info.get(block.area, {})
        names = [boss_info[unit_id] for unit_id in sorted(units) if unit_id in boss_info]
        if names:
            return "/".join(names)
        if units:
            return "/".join(str(unit_id) for unit_id in sorted(units))
        quest_id = getattr(block, "quest_id", None)
        return str(quest_id) if quest_id else "未知Boss"

    def _position_name(self, block, area_columns: Dict[int, List]) -> str:
        rows = [item.row for item in area_columns.get(block.column, [])]
        max_row = max(rows) if rows else block.row
        if max_row <= 1:
            return "合流"
        if max_row == 2:
            return {1: "下", 2: "上"}.get(block.row, str(block.row))
        if max_row == 3:
            return {1: "下", 2: "中", 3: "上"}.get(block.row, str(block.row))
        return str(block.row)

    def _format_route(self, area: int, route: List, map_list: List) -> str:
        area_columns: Dict[int, List] = {}
        for block in map_list:
            if block.area == area:
                area_columns.setdefault(block.column, []).append(block)
        parts = []
        for block in route:
            block_type = self._block_type(block)
            block_name = LABYRINTH_BLOCK_TYPE_NAME.get(block_type, str(block_type))
            extra = (
                f"({self._boss_names(block)})"
                if block_type == Block.BOSS_QUEST
                else ""
            )
            parts.append(
                f"{block.column}{self._position_name(block, area_columns)}"
                f"【{block_name}{extra}】"
            )
        return f"区域{area}：" + "-".join(parts)


@description(
    "!!!!!!!!!!!!!危险，高概率封号!!!!!!!!!!!!!!!"
    "\n进入最高可挑战难度，完成指定次数或单局达到目标分数后停止。支持续跑已有开局，不检查本地票数。"
    "\n完美路线、性价比先刷到符合要求的开局再推图；不要求则直接开始。完美路线按规划走，其他按剩余路线积分估值选路。"
    "\n性价比沿用刷开局规则：难度4/5要求区域4同路双EX、区域2遗物直连EX；低难度不加此限制。"
    "\n自动结算按当前队伍和敌人数据构造胜利战报，战报耗时随机13～20秒，不实际等待。"
)
@name("黎明界通关")
@booltype("labyrinth_run_auto_clear", "自动结算（构造胜利战报）", True)
@singlechoice(
    "labyrinth_run_event",
    "事件选项",
    "收益优先",
    ["收益优先", "首个免费选项", "随机免费选项"],
)
@inttype("labyrinth_run_shop_reset_limit", "区域3/5商店刷新上限", 2, list(range(3)))
@inttype(
    "labyrinth_run_shop_reserve", "途中商店保留金币", 2700, list(range(0, 20001, 50))
)
@singlechoice(
    "labyrinth_run_third_block_type",
    "完美路线：区域3/5第3格",
    "任意",
    ["遗物", "事件", "任意"],
)
@singlechoice(
    "labyrinth_run_preference",
    "同分路线偏好",
    "遗物优先",
    ["遗物优先", "事件优先", "无偏好"],
)
@inttype(
    "labyrinth_run_reroll_limit",
    "每局刷开局上限（未命中则停止）",
    300,
    [100, 300],
)
@singlechoice(
    "labyrinth_run_route", "开局路线", "性价比", ["完美路线", "性价比", "不要求"]
)
@inttype(
    "labyrinth_run_target_score",
    "目标分数（0不限，100递增）",
    0,
    list(range(0, 100001, 100)),
)
@inttype("labyrinth_run_count", "次数上限", 1, list(range(1, 101)))
@LabyrinthGuildConfig("labyrinth_run_guild_id", "公会", 5)
class labyrinth_run(LabyrinthRouteMixin, Module):
    MAX_STEPS = 500

    def _checkpoint(self):
        self.save_cache("statistics", self.statistics)

    def _begin_run(self, enter_id, difficulty):
        current = self.statistics.get("current_run", {})
        if current.get("enter_id") != enter_id:
            self.statistics["current_run"] = {
                "enter_id": enter_id,
                "difficulty": difficulty,
                "seen": [],
            }
        self.enter_id = enter_id
        self._checkpoint()

    def _remaining_route(self, snapshot):
        blocks = snapshot.map_list or []
        pinned = getattr(self, "perfect_route", None)
        if pinned:
            by_id = {b.block_id: b for b in blocks}
            if snapshot.block_id not in pinned:
                return None
            ids = pinned[pinned.index(snapshot.block_id) :]
            route = [by_id[id] for id in ids if id in by_id]
            if len(route) != len(ids) or any(
                b.block_id not in a.next_block_id_list for a, b in zip(route, route[1:])
            ):
                return None
            if any(
                self._block_type(b)
                not in self.expected.get(b.area, {}).get(b.column, set())
                for b in route[1:]
            ):
                return None
            return route
        return best_route(
            blocks,
            snapshot.block_id,
            preference=self.get_config("labyrinth_run_preference"),
            battle_seconds=16.5,
        )

    def _opening_route(self, snapshot, difficulty):
        blocks = snapshot.map_list or []
        if not blocks:
            raise AbortError("地图为空，无法筛选开局")
        self.pin_opening_route = False
        if (
            self.strategy == "不要求"
            or not self.reroll_opening
            and self.strategy != "完美路线"
        ):
            return self._remaining_route(snapshot)
        # A resumed perfect run must agree with its already visited branches.
        visited = {(b.area, b.column): b.block_id for b in blocks if b.is_visited}
        visited.update(
            {
                (b.area, b.column): b.block_id
                for b in blocks
                if b.block_id == snapshot.block_id
            }
        )
        perfect = self.strategy == "完美路线"
        compatible = [
            b
            for b in blocks
            if not perfect or visited.get((b.area, b.column), b.block_id) == b.block_id
        ]
        start = min(blocks, key=lambda b: (b.area, b.column, b.row))
        required_areas = set(self._target_areas(difficulty))
        if not required_areas.issubset({b.area for b in blocks}):
            raise AbortError("地图区域数据不全，不能判断开局")
        route = best_route(
            compatible,
            start.block_id,
            expected=self.expected if perfect else None,
            value_requirement=not perfect and difficulty in VALUE_ROUTE_DIFFICULTIES,
            preference=self.get_config("labyrinth_run_preference"),
            battle_seconds=16.5,
        )
        if route and perfect:
            positions = [
                (area, col)
                for area in sorted(required_areas)
                for col in sorted(self.expected[area])
            ]
            if [(b.area, b.column) for b in route] != positions:
                return None
            saved = (
                getattr(self, "statistics", {})
                .get("current_run", {})
                .get("perfect_route")
                or []
            )
            by_id = {b.block_id: b for b in compatible}
            if saved and all(id in by_id for id in saved):
                cached = [by_id[id] for id in saved]
                if (
                    [(b.area, b.column) for b in cached] == positions
                    and all(
                        self._block_type(b) in self.expected[b.area][b.column]
                        for b in cached
                    )
                    and all(
                        b.block_id in (a.next_block_id_list or [])
                        for a, b in zip(cached, cached[1:])
                    )
                ):
                    route = cached
        if route and perfect:
            self.pin_opening_route = True
        if not route and not self.reroll_opening:
            return best_route(
                blocks,
                snapshot.block_id,
                preference=self.get_config("labyrinth_run_preference"),
                battle_seconds=16.5,
            )
        return route

    def _accept(self, operation, response, **context):
        before = self.state.snapshot.alpha_rupee
        self.state.apply(operation, response, **context)
        self.statistics["current_run"]["pending"] = self.state.dump_pending()
        self._checkpoint()
        if self.state.snapshot.alpha_rupee != before:
            pass
            # self._log(f"金币更新：{self.state.snapshot.alpha_rupee}（上一步{before}）")

    def _next_event_effect(self, status):
        result = db.labyrinth_event_result.get(status.event_result_id)
        if result is None:
            raise AbortError(
                f"事件结果{status.event_result_id}缺少效果表，请更新数据库"
            )
        return next(
            (
                n
                for n in range(status.effect_num + 1, 6)
                if getattr(result, f"effect_type_{n}", 0)
            ),
            None,
        )

    def _record(self, category, key):
        counter = self.statistics.setdefault(category, {})
        key = str(key)
        counter[key] = counter.get(key, 0) + 1

    def _observe(self, snapshot):
        """Count exposures once, including across interrupted task executions."""
        seen = self.statistics["current_run"]["seen"]
        for block in snapshot.map_list or []:
            key = f"map:{block.block_id}:{block.quest_id}"
            if block.quest_id and key not in seen:
                self._record("quests", block.quest_id)
                seen.append(key)
        status = snapshot.status
        if status is None:
            raise AbortError("响应缺少 status，已保留当前开局")
        # Most ordinary Quest IDs are hidden in the initial map and only appear
        # in status after entering a battle block.
        quest_key = f"map:{snapshot.block_id}:{status.quest_id}"
        if status.quest_id and quest_key not in seen:
            self._record("quests", status.quest_id)
            seen.append(quest_key)
        current_run = self.statistics["current_run"]
        fingerprint = json.dumps(self._fingerprint(snapshot), ensure_ascii=False)
        if current_run.get("observed_fingerprint") != fingerprint:
            current_run["exposure"] = current_run.get("exposure", 0) + 1
            current_run["observed_fingerprint"] = fingerprint
        source = f'{snapshot.block_id}:{status.type}:{current_run["exposure"]}'
        observations = []
        if status.event_id:
            observations.append(("events", source, status.event_id))
        for candidate in status.candidate_unit_list or []:
            observations.append(("unit_candidates", source, candidate.unit_id))
        for item in status.choices or []:
            if item.reward_type == Inventory.LabyrinthRelic:
                observations.append(
                    (
                        "relic_candidates",
                        f"{source}:{status.current_choice_count}:{item.choice_num}",
                        item.reward_id,
                    )
                )
        for item in status.shop_lineup_list or []:
            if item.reward_type == Inventory.LabyrinthRelic:
                observations.append(
                    (
                        "relic_candidates",
                        f"shop:{snapshot.block_id}:{status.shop_reset_count}:{item.lineup_id}",
                        item.reward_id,
                    )
                )
        for category, origin, value in observations:
            key = f"{category}:{origin}:{value}"
            if value and key not in seen:
                self._record(category, value)
                seen.append(key)
        self.statistics["current_run"]["block_id"] = snapshot.block_id
        self.statistics["current_run"]["status"] = (
            int(status.type) if status.type is not None else None
        )
        self._checkpoint()

    @staticmethod
    def _fingerprint(snapshot):
        # Responses update this snapshot, including authoritative currency stocks.
        return (
            snapshot.block_id,
            snapshot.alpha_rupee,
            snapshot.status.json(sort_keys=True) if snapshot.status else None,
            tuple((u.unit_id, u.unit_type) for u in snapshot.unit_list or []),
            tuple(
                (r.relic_id, r.effected_count, r.enable)
                for r in snapshot.relic_list or []
            ),
            tuple(
                (b.block_id, b.is_visited, b.ChangeBlockType)
                for b in snapshot.map_list or []
            ),
        )

    def _score(self, item):
        return reward_score(
            item.reward_type,
            item.reward_id,
            item.reward_count,
            db.labyrinth_summon_unit,
        )

    async def _summon(self, client, snapshot):
        status = snapshot.status
        if not status.item_id:
            raise AbortError("召唤状态缺少 item_id，需要抓取 roll_unit / resume")
        ticket = db.labyrinth_summon_unit.get(status.item_id)
        if not ticket or not ticket.summon_num:
            raise AbortError(
                f"未知角色券{status.item_id}，需更新数据库或抓取 summon_unit"
            )
        # summon_type=3 is server-random: the captured client sends [] directly,
        # once per ticket, and the server may return another SUMMON_UNIT status.
        if ticket.summon_type == 3:
            result = await client.labyrinth_summon_unit(
                self.enter_id, status.item_id, []
            )
            self._log_summoned(result)
            return result
        if ticket.summon_type not in (1, 2):
            raise AbortError(
                f"角色券{status.item_id}的召唤类型{ticket.summon_type}未知，需要抓包"
            )
        candidates = {
            u.unit_id: u for u in status.candidate_unit_list or [] if u.unit_id
        }
        if not candidates:
            # Captures confirm type 1/2 include candidates. Do not spend or reroll
            # when an unexpected response omits them.
            raise AbortError(
                f"选择角色券{status.item_id}缺少候选，需补充 resume / roll_unit 抓包"
            )
        owned = {unit.unit_id for unit in snapshot.unit_list or []}
        available = [unit_id for unit_id in candidates if unit_id not in owned]
        if len(available) < ticket.summon_num:
            raise AbortError(
                f"角色券{status.item_id}候选不足，需确认重复角色/不足人数的 summon_unit 协议"
            )
        # Jun (104701) can be granted by an event. Keep that future +100 score
        # available when another equally scoring recruit can be chosen.
        selected = sorted(
            available,
            key=lambda id: (id != 104701, unit_rank(candidates[id], client.data.unit)),
            reverse=True,
        )[: ticket.summon_num]
        result = await client.labyrinth_summon_unit(
            self.enter_id, status.item_id, selected
        )
        self._log_summoned(result)
        return result

    def _log_summoned(self, result):
        for unit in result.unit_info_list or []:
            self._log(f"召唤角色：{db.get_unit_name(unit.unit_id)}({unit.unit_id})")

    async def _choice_reward(self, client, snapshot):
        status = snapshot.status
        choices = [c for c in status.choices or [] if c.choice_num is not None]
        if not choices or status.current_choice_count is None:
            raise AbortError(
                "奖励选择缺少 choices / current_choice_count，需要 choice_reward 抓包"
            )
        owned_relics = {r.relic_id for r in snapshot.relic_list or []}
        owned_units = {u.unit_id for u in snapshot.unit_list or []}
        ticket_scores = {}
        for item in choices:
            if item.reward_type != Inventory.LabyrinthTicket:
                continue
            pool = await client.labyrinth_get_candidate_unit(item.reward_id)
            if pool.unit_id_list is None:
                raise AbortError(
                    f"角色券{item.reward_id}候选预览缺少 unit_id_list，已保留选择"
                )
            available = set(pool.unit_id_list) - owned_units
            ticket = db.labyrinth_summon_unit.get(item.reward_id)
            count = min(
                len(available),
                (item.reward_count or 0) * (ticket.summon_num if ticket else 1),
            )
            # Avoid using up the free Jun event reward when equivalent pools exist.
            jun_risk = count / len(available) if 104701 in available else 0
            ticket_scores[item.choice_num] = (count * 100, -jun_risk, len(available))

        def score(item):
            if item.choice_num in ticket_scores:
                return ticket_scores[item.choice_num] + (-item.choice_num,)
            value = self._score(item)
            if (
                item.reward_type == Inventory.LabyrinthRelic
                and item.reward_id in owned_relics
            ):
                value = 0
            relic = (
                db.labyrinth_relic.get(item.reward_id)
                if item.reward_type == Inventory.LabyrinthRelic
                else None
            )
            # Free choices favor high rarity to leave cheap relics for shops.
            return (
                value,
                getattr(relic, "rarity", 0) or 0,
                getattr(relic, "relic_mark_count", 0) or 0,
                -item.choice_num,
            )

        selected = max(choices, key=score)
        result = await client.labyrinth_choice_reward(
            self.enter_id, status.current_choice_count, selected.choice_num
        )
        kind = "遗物" if selected.reward_type == Inventory.LabyrinthRelic else "角色券"
        # self._log(f"选择【{kind}】奖励：第{selected.choice_num}项，{selected.reward_id}，预计{score(selected)[0]}分")

        return result

    async def _event(self, client, snapshot):
        status = snapshot.status
        event = db.labyrinth_event.get(status.event_id)
        if not event:
            raise AbortError(
                f"事件{status.event_id}缺少数据库配置，需要 choice_event 抓包"
            )
        available = []
        strategy = self.get_config("labyrinth_run_event")
        for num in range(1, 4):
            choice_id = getattr(event, f"choice_id_{num}")
            option = db.labyrinth_event_choice.get(choice_id)
            if not option:
                continue
            cost = (
                0
                if option.condition_type == 0
                else getattr(option, "condition_value", None)
            )
            if option.condition_type not in (0, 1) or cost is None or cost < 0:
                continue
            if cost and (
                snapshot.alpha_rupee is None
                or cost > snapshot.alpha_rupee
                or strategy != "收益优先"
            ):
                continue
            available.append((num, option, cost))
        if not available:
            raise AbortError(f"事件{status.event_id}无可用选项，需要核对条件分支")

        def value(entry):
            num, option, cost = entry
            score = event_choice_score(option.description) - cost * 200 / 1350
            if getattr(option, "unit_id", None) in {
                u.unit_id for u in snapshot.unit_list or []
            } and any(word in option.description for word in ("成为伙伴", "成為夥伴")):
                score -= 100
            return (score, -cost, -num)

        num, option, cost = (
            choice(available)
            if strategy == "随机免费选项"
            else max(available, key=value) if strategy == "收益优先" else available[0]
        )
        result = await client.labyrinth_choice_event(
            self.enter_id, status.event_id, num
        )
        self._log(
            f"选择事件{status.event_id}第{num}项：{option.description}（{strategy}，花费{cost}金币）"
        )

        return result

    async def _shop(self, client, snapshot):
        status = snapshot.status
        if snapshot.alpha_rupee is None:
            raise AbortError("商店缺少 alpha_rupee，无法确认可用金币")
        blocks = snapshot.map_list or []
        current = next((b for b in blocks if b.block_id == snapshot.block_id), None)
        if current is None:
            raise AbortError("商店所在格不在地图中")
        remaining = self._remaining_route(snapshot)
        if not remaining:
            raise AbortError("商店之后没有可达终点，需确认跨区域 move 协议")
        later_shop = any(self._block_type(b) == Block.SHOP for b in remaining[1:])
        reserve = self.get_config("labyrinth_run_shop_reserve") if later_shop else 0
        lineup = status.shop_lineup_list or []
        resets = status.shop_reset_count
        limit = (
            self.get_config("labyrinth_run_shop_reset_limit")
            if current.area in (3, 5)
            else 0
        )
        refresh, minimum_price = False, 1350
        if current.area == 2:
            # Buy only cheap one-star relics; save currency for the later shops.
            cheap = [
                item
                for item in lineup
                if item.reward_type == Inventory.LabyrinthRelic
                and (
                    getattr(db.labyrinth_relic.get(item.reward_id), "rarity", 0) == 1
                    or (
                        item.reward_id not in db.labyrinth_relic
                        and item.price is not None
                        and item.price <= 1350
                    )
                )
            ]
            plan = shop_plan(cheap, snapshot.alpha_rupee, 0, db.labyrinth_summon_unit)
        elif current.area == 4:
            # Four-star relics cost far more for the same 200 score. Save a full
            # final-shop cheap-relic budget (3 stocks x 3 slots plus 300+600).
            reserve = max(reserve, 13050) if later_shop else 0
            plan = shop_plan(
                [
                    item
                    for item in lineup
                    if item.reward_type == Inventory.LabyrinthTicket
                ],
                snapshot.alpha_rupee,
                reserve,
                db.labyrinth_summon_unit,
            )
        else:
            plan, refresh, minimum_price = score_shop_plan(
                lineup,
                snapshot.alpha_rupee,
                reserve,
                db.labyrinth_summon_unit,
                db.labyrinth_relic,
                resets,
                limit,
                db.labyrinth_setting,
            )
        if plan:
            item = plan[0]
            result = await client.labyrinth_shop_buy(
                self.enter_id, item.lineup_id, snapshot.alpha_rupee
            )
            self._log(
                f"购买区域{current.area}【商店】：第{item.lineup_id}项，{item.reward_id}x{item.reward_count}，"
                f"预计{self._score(item)}分，花费{item.price}。"
            )
            return "shop_buy", result, {"lineup_id": item.lineup_id}
        if refresh:
            cost = shop_reset_cost(resets, db.labyrinth_setting)
            result = await client.labyrinth_shop_reset(self.enter_id, resets)
            self._log(
                f"刷新区域{current.area}【商店】：第{resets + 1}次，预计消耗{cost}，预留购买金币{minimum_price}"
            )
            return "shop_reset", result, {}
        result = await client.labyrinth_shop_close(self.enter_id)
        # self._log(f"离开区域{current.area}【商店】格：剩余金币{snapshot.alpha_rupee}")

        return "shop_close", result, {}

    async def _battle(self, client, snapshot):
        status = snapshot.status
        if not self.get_config("labyrinth_run_auto_clear"):
            raise AbortError(
                f"已停在战斗格{snapshot.block_id}（关卡{status.quest_id}）。"
                "自动结算未开启；可开启此选项，或手动打完后再次执行续跑。"
            )
        if not status.quest_id:
            raise AbortError("战斗状态缺少 quest_id")
        boss = status.type == Status.BOSS
        decks = battle_decks(snapshot, client.data.unit, boss)
        if any(not deck.unit_list for deck in decks):
            raise AbortError(
                "已召唤角色不足以编队，保留当前战斗；需检查 resume.unit_list"
            )
        quest = db.labyrinth_quest_data.get(status.quest_id)
        wave = db.labyrinth_wave_group_data.get(quest.wave_group_id) if quest else None
        enemy_ids = [id for id in wave.get_enemy_ids() if id] if wave else []
        enemies = [db.labyrinth_enemy_parameter.get(id) for id in enemy_ids]
        if not enemies or any(enemy is None or not enemy.hp for enemy in enemies):
            raise AbortError(
                f"关卡{status.quest_id}缺少敌人数据，请更新数据库；未发起战斗"
            )
        if not quest.limit_time or quest.limit_time <= 0:
            raise AbortError(f"关卡{status.quest_id}缺少战斗时限，未发起战斗")
        # Report simulated combat duration without a real sleep or a fixed loading offset.
        combat_ms = randint(13000, 20000)
        remain_time = max(0, quest.limit_time * 1000 - combat_ms)
        if not client.viewer_id:
            raise AbortError("缺少当前账号 viewer_id，未发起战斗")
        # These are synthetic victory reports, NOT output from a battle engine.
        # The captured auto_clear=3 is retained as a request field, not assumed
        # to be a server-side simulation switch.
        # Wire enemy unit_id is enemy_id (770...), not display unit_id (30...).
        await client.labyrinth_update_deck(self.enter_id, decks)
        args = (self.enter_id, snapshot.block_id, status.quest_id)
        token = create_battle_start_token()
        if boss:
            started = await client.labyrinth_boss_battle_start(*args, token)
            if not started.battle_log_id_list or not started.seed_list:
                raise AbortError(
                    "首领开战未返回 seed_list / battle_log_id_list，已停止结算"
                )
            total_hp = sum(enemy.hp for enemy in enemies)
            first_team = decks[0].unit_list
            damage, remainder = divmod(total_hp, len(first_team))
            waves = [
                TowerWaveResultInfo(
                    wave_num=1,
                    remain_time=remain_time,
                    unit_info_list=[
                        TowerWaveResultUnitInfo(
                            unit_id=id,
                            owner_viewer_id=client.viewer_id,
                            damage=damage + (i < remainder),
                            is_alive=1,
                        )
                        for i, id in enumerate(first_team)
                    ],
                )
            ]
            versus = [
                DungeonQueryUnit(
                    owner_viewer_id=0,
                    unit_id=id,
                    retired=1,
                    hp=0,
                    energy=0,
                    skill_limit_counter=[],
                    damage=0,
                    parts_list=[],
                )
                for id in enemy_ids
            ]
            result = await client.labyrinth_boss_battle_finish(*args, waves, versus, 3)
        else:
            started = await client.labyrinth_battle_start(*args, token)
            if started.battle_log_id is None or started.seed is None:
                raise AbortError("开战未返回 seed / battle_log_id，已停止结算")
            hp = [
                UnitHpInfo(viewer_id=client.viewer_id, unit_id=id, hp=1)
                for id in decks[0].unit_list
            ]
            hp += [UnitHpInfo(viewer_id=0, unit_id=id, hp=0) for id in enemy_ids]
            result = await client.labyrinth_battle_finish(*args, remain_time, hp, 3)
        # Both winning finish captures returned 2, followed by rewards/status.
        if result.result_type != 2:
            raise AbortError(
                f"关卡{status.quest_id}未返回胜利 result_type=2（实际{result.result_type}），已停止，需核对结算抓包"
            )
        current = next(
            (b for b in snapshot.map_list or [] if b.block_id == snapshot.block_id),
            None,
        )
        kind = (
            LABYRINTH_BLOCK_TYPE_NAME.get(self._block_type(current), "事件战斗")
            if current
            else "战斗"
        )
        self._log(f"【{kind}】格战斗完成：关卡{status.quest_id}")

        return result

    async def _run_once(self, client):
        previous = None
        last_route = None
        for _ in range(self.MAX_STEPS):
            snapshot = self.state.snapshot
            self._observe(snapshot)
            fingerprint = self._fingerprint(snapshot)
            if fingerprint == previous:
                raise AbortError(
                    f"格{snapshot.block_id}响应状态未推进，已停止重复请求；请保留该操作请求及响应"
                )
            previous = fingerprint
            status = snapshot.status
            if status.type == Status.NONE:
                route = self._remaining_route(snapshot)
                if not route:
                    raise AbortError(
                        f"从格{snapshot.block_id}找不到到终点的可达路线，已保留开局；需确认 map_list / 跨区域 move"
                    )
                if len(route) == 1:
                    if (
                        self._block_type(route[0]) != Block.BOSS_QUEST
                        or not route[0].is_visited
                    ):
                        raise AbortError(
                            "终点未确认是已访问的首领格，不能结算；需核对 resume / exit 抓包"
                        )
                    # Only count success after exit returns a real score.
                    result = await client.labyrinth_exit(self.enter_id)
                    if result.result_score is None:
                        raise AbortError(
                            "exit 未返回 result_score，不能确认通关统计，需要结算抓包"
                        )
                    self._record("completed_scores", result.result_score)
                    self.statistics["completed_runs"] = (
                        self.statistics.get("completed_runs", 0) + 1
                    )
                    self.statistics["last_result"] = {
                        key: getattr(result, key)
                        for key in (
                            "unit_score",
                            "relic_score",
                            "move_score",
                            "battle_normal_score",
                            "battle_hard_score",
                            "battle_boss_score",
                            "result_score",
                        )
                    }
                    self.statistics.pop("current_run", None)
                    self._checkpoint()
                    rate = DIFFICULTY_SCORE_RATE.get(self.difficulty, 100) / 100
                    self._log(
                        f"黎明界结算：{result.result_score}分（难度{self.difficulty}，倍率×{rate:g}；角色{result.unit_score}，遗物{result.relic_score}，"
                        f"移动{result.move_score}，普通{result.battle_normal_score}，EX{result.battle_hard_score}，首领{result.battle_boss_score}）"
                    )
                    rewards = list(result.exit_reward_list or [])
                    for box in (result.treasure_box_reward_list or []) + (
                        result.rare_treasure_box_reward_list or []
                    ):
                        rewards.extend(box.reward_list or [])
                    if rewards:
                        self._log(await client.serialize_reward_summary(rewards))
                    return result.result_score
                if route[0].area != last_route:
                    area = route[0].area
                    area_route = [b for b in route if b.area == area]
                    self._log(self._format_route(area, area_route, snapshot.map_list))
                    value = route_value(route[1:], battle_seconds=16.5)[0]
                    rate = DIFFICULTY_SCORE_RATE.get(self.difficulty, 100) / 100
                    self._log(
                        f"难度{self.difficulty}×{rate:g}后约{value * rate:.0f}"
                        "（含金币购买力估值，不含固定踏破分；目标分数以最终结算为准）"
                    )
                    last_route = area
                next_block = route[1]
                result = await client.labyrinth_move(self.enter_id, next_block.block_id)
                self._accept("move", result, block_id=next_block.block_id)
                self._log(
                    f'移动到【{LABYRINTH_BLOCK_TYPE_NAME.get(self._block_type(next_block), "未知")}】格：{next_block.block_id}'
                )
            elif status.type == Status.SUMMON_UNIT:
                self._accept("summon_unit", await self._summon(client, snapshot))
            elif status.type in (Status.SUMMON_TICKET_CHOICE, Status.RELIC_CHOICE):
                self._accept(
                    "choice_reward", await self._choice_reward(client, snapshot)
                )
            elif status.type == Status.EVENT_CHOICE:
                self._accept("choice_event", await self._event(client, snapshot))
            elif status.type == Status.EVENT_EFFECT:
                if not status.event_result_id or status.effect_num is None:
                    raise AbortError(
                        "事件效果缺少 event_result_id / effect_num，需要 event_effect 抓包"
                    )
                next_effect = self._next_event_effect(status)
                result = await client.labyrinth_event_effect(
                    self.enter_id, status.event_result_id, status.effect_num
                )
                self._accept("event_effect", result, next_effect=next_effect)
                # self._log(f"处理事件效果：{status.event_result_id}，第{status.effect_num}项")
            elif status.type == Status.USE_RELIC:
                if not status.relic_id:
                    raise AbortError("使用遗物状态缺少 relic_id，需要 use_relic 抓包")
                self._accept(
                    "use_relic",
                    await client.labyrinth_use_relic(self.enter_id, status.relic_id),
                )
                self._log(f"使用遗物：{status.relic_id}")
            elif status.type == Status.SHOP:
                operation, result, context = await self._shop(client, snapshot)
                self._accept(operation, result, **context)
            elif status.type in (Status.QUEST, Status.BOSS):
                self._accept("battle_finish", await self._battle(client, snapshot))
            else:
                raise AbortError(
                    f"未知黎明界状态{status.type}，已保留开局，需要 resume 抓包"
                )
        raise AbortError(f"单局超过{self.MAX_STEPS}步，已停止，请检查缓存中的最后状态")

    async def do_task(self, client):
        self.statistics = self.find_cache("statistics") or {"version": 1}
        self.strategy = self.get_config("labyrinth_run_route")
        if self.strategy == "随机开局":
            self.strategy = "不要求"
        bot_reroll = self._get_raw_config("labyrinth_run_bot_reroll_opening", None)
        self.reroll_opening = (
            self.strategy != "不要求" if bot_reroll is None else bool(bot_reroll)
        )
        if self.reroll_opening and self.strategy == "不要求":
            raise AbortError(
                "机器人参数要求刷开局，但路线为不要求；请指定完美路线或性价比"
            )
        self.expected = self._build_expected_block_types(
            self.get_config("labyrinth_run_third_block_type")
        )
        count = self.get_config("labyrinth_run_count")
        target = self.get_config("labyrinth_run_target_score")
        guild_id = self.get_config("labyrinth_run_guild_id")
        self._log(
            f'路线策略：{self.strategy}，最多{count}次，目标分数：{target or "不限"}'
        )
        completed = 0
        try:
            for attempt in range(1, count + 1):
                top = await client.labyrinth_top()
                self.perfect_route = None
                self.difficulty = (
                    top.difficulty
                    if top.enter_id
                    else self._max_unlocked_difficulty(top)
                )
                if top.enter_id:
                    self._begin_run(top.enter_id, self.difficulty)
                    snapshot = await client.labyrinth_resume(self.enter_id)
                    current = self.statistics["current_run"]
                    pending = (
                        current.get("pending")
                        if current.get("block_id") == snapshot.block_id
                        else None
                    )
                    self.state = LabyrinthState(snapshot, pending)
                    self._log(
                        f"读取已有开局：公会{top.guild_id}，难度{self.difficulty}"
                    )
                else:
                    self.state = None
                limit = self.get_config("labyrinth_run_reroll_limit")
                for opening_attempt in range(1, limit + 1):
                    if self.state is None:
                        entered = await client.labyrinth_enter(
                            guild_id, self.difficulty
                        )
                        if not entered.enter_id:
                            raise AbortError("enter 未返回 enter_id")
                        self._begin_run(entered.enter_id, self.difficulty)
                        self.state = LabyrinthState.entered(entered, guild_id)
                    snapshot = self.state.snapshot
                    self._observe(snapshot)
                    if route := self._opening_route(snapshot, self.difficulty):
                        if self.pin_opening_route:
                            self.perfect_route = [b.block_id for b in route]
                        self.statistics["current_run"][
                            "perfect_route"
                        ] = self.perfect_route
                        self.statistics["current_run"][
                            "pending"
                        ] = self.state.dump_pending()
                        self._checkpoint()
                        if self.reroll_opening:
                            self._log(
                                f"开局符合{self.strategy}要求，本次筛选{opening_attempt}次，开始推图"
                            )
                        else:
                            self._log("已按机器人参数跳过开局筛选，开始按积分推图")
                        break
                    # self._log(f"第{opening_attempt}次开局不符合{self.strategy}要求，撤退重开")
                    await client.labyrinth_retire(self.enter_id)
                    self.statistics.pop("current_run", None)
                    self.statistics["rerolled_runs"] = (
                        self.statistics.get("rerolled_runs", 0) + 1
                    )
                    self._checkpoint()
                    self.state = None
                    await client.labyrinth_top()
                else:
                    raise AbortError(
                        f"尝试{limit}次仍未刷到{self.strategy}，未开始推图"
                    )
                score = await self._run_once(client)
                completed += 1
                self._log(f"第{attempt}/{count}次黎明界通关完成")
                if target and score >= target:
                    self._log(f"已达到目标分数{target}，停止后续挑战")
                    break
        finally:
            self._checkpoint()
            self._log(f"黎明界通关完成：{completed}/{count}。")
