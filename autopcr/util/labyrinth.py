"""Offline route and shop decisions. Scores are estimates, not server results."""

from functools import lru_cache
from random import choice
import re
import unicodedata

from ..model.enums import eInventoryType as Inventory, eLabyrinthBlockType as Block
from ..model.enums import eLabyrinthShopItemStatus as ShopStatus
from ..model.common import DeckListData


# Confirmed against the June TW captures, local labyrinth_setting (221..225),
# and first-hand gameplay reports. See docs/LABYRINTH_CAPTURE_CHECKLIST.md.
# Include the usual recruitment/relic rewards, but omit common movement scores.
BLOCK_SCORE = {
    Block.NONE: 0,
    Block.NORMAL_QUEST: 200,
    Block.HARD_QUEST: 1300,
    Block.TICKET: 300,
    Block.EVENT: 250,
    Block.RELIC: 200,
    Block.SHOP: 400,
    Block.BOSS_QUEST: 1500,
}

DIFFICULTY_SCORE_RATE = {1: 100, 2: 120, 3: 140, 4: 170, 5: 200}

# 区域4同路双EX、区域2遗物直连EX 只在高难度才要求（性价比开局）。
VALUE_ROUTE_DIFFICULTIES = {4, 5}

# Boss 难度分级（区域 -> unit_id -> 难度）。名称以数据库 labyrinth_boss_info 为准。
BOSS_DIFFICULTY = {
    3: {
        312505: "简单",  # 厄勒克特拉夫人
        319604: "简单",  # 冰霜魔狼
        303306: "普通",  # 暗黑滴水嘴兽
        301206: "普通",  # 巨型魔像
        306604: "困难",  # 毒液沙鳗蛇
    },
    5: {
        310103: "简单",  # 愤怒巨龙
        301701: "普通",  # 炸脖龙
        319401: "普通",  # 究极守护者
        315004: "困难",  # 领主哥布林
        302501: "困难",  # 奇美拉
    },
}
BOSS_PRESETS = ("简单", "普通", "菜鸟")
# 通关指令与网页「开局路线」共用；随机开局为旧配置值。
RUN_ROUTE_ALIASES = {
    "完美路线": "完美路线",
    "完美": "完美路线",
    "性价比": "性价比",
    "不要求": "不要求",
    "随机开局": "不要求",
}


def boss_difficulty(area, unit_id):
    return BOSS_DIFFICULTY.get(area, {}).get(unit_id)


def boss_preset(mode):
    """Bosses accepted per area for a preset: 简单 / 普通(含简单) / 菜鸟(仅魔狼、巨龙)."""
    if mode == "菜鸟":
        return {3: {319604}, 5: {310103}}
    allowed = {"简单": {"简单"}, "普通": {"简单", "普通"}}.get(mode)
    if allowed is None:
        raise ValueError(f"未知开局模式：{mode}")
    return {
        area: {unit_id for unit_id, level in bosses.items() if level in allowed}
        for area, bosses in BOSS_DIFFICULTY.items()
    }


def parse_run_command_args(args, find_guild=None):
    """Consume bot-only Labyrinth overrides from a token list.

    Format: <公会id|公会名> [完美路线|性价比|不要求] [目标分数].
    Route and score keep the web config when omitted; guild is required
    when find_guild is provided.
    """
    config = {}
    guild = None
    route = None
    target = None
    index = 0
    while index < len(args):
        token = args[index]
        if token in RUN_ROUTE_ALIASES:
            selected = RUN_ROUTE_ALIASES[token]
            if route is not None and route != selected:
                raise ValueError("完美路线、性价比和不要求只能选择一个")
            route = selected
            del args[index]
            continue

        raw_score = None
        if token in ("目标分数", "目标"):
            if index + 1 >= len(args):
                raise ValueError("目标分数后缺少数字")
            raw_score = args[index + 1]
            del args[index : index + 2]
        else:
            match = re.fullmatch(r"(?:目标分数|目标)?[=:：]?(\d+)", token)
            labeled = bool(re.match(r"(?:目标分数|目标|[=:：])", token))
            if match and (
                labeled or find_guild is None or find_guild(token) is None
            ):
                raw_score = match.group(1)
                del args[index]
        if raw_score is not None:
            if target is not None:
                raise ValueError("目标分数只能设置一次")
            target = int(raw_score)
            if target < 0 or target > 100000 or target % 100:
                raise ValueError("目标分数须为0～100000之间的100倍数")
            continue

        if find_guild is not None:
            guild_id = find_guild(token)
            if guild_id is not None:
                if guild is not None and guild != guild_id:
                    raise ValueError("只能选择一个公会")
                guild = guild_id
                del args[index]
                continue
        index += 1

    if find_guild is not None and guild is None:
        raise ValueError("请指定公会")
    if guild is not None:
        config["labyrinth_run_guild_id"] = guild
    if route is not None:
        config["labyrinth_run_route"] = route
    if target is not None:
        config["labyrinth_run_target_score"] = target
    return config


def shop_reset_cost(current_count, settings):
    """Setting IDs 101..106 match the observed increasing shop cost schedule."""
    if current_count is None or current_count < 0:
        return None
    defaults = (300, 600, 1200, 2400, 4800, 9999)
    if current_count >= len(defaults):
        return None
    return settings.get(101 + current_count, defaults[current_count])


def block_score(block, strategy="积分优先"):
    kind = block.ChangeBlockType or block.block_type
    score = BLOCK_SCORE.get(kind, 0)
    if kind == Block.SHOP and block.area == 2:
        score = 150  # Early shop has no free rewards and a limited coin budget.
    if strategy == "角色优先" and kind == Block.TICKET:
        score += 10000
    if strategy == "遗物优先" and kind == Block.RELIC:
        score += 10000
    return score


def best_route(
    map_list,
    current_id,
    strategy="积分优先",
    expected=None,
    *,
    value_requirement=False,
    preference="无偏好",
    battle_seconds=None,
):
    """Maximize the whole reachable suffix; dead ends and cycles never win.

    The occupied block is exempt from perfect-route filtering when resuming.
    No edges are inferred from block IDs, rows, or adjacent areas.
    """
    by_id = {b.block_id: b for b in map_list}
    if current_id not in by_id:
        return None
    final_position = max((b.area, b.column) for b in map_list)
    visiting = set()

    @lru_cache(None)
    def solve(block_id, relic_hard=False, hard_count=0):
        block = by_id[block_id]
        if block_id in visiting:
            return None
        kind = block.ChangeBlockType or block.block_type
        hard_count = (
            min(2, hard_count + (block.area == 4 and kind == Block.HARD_QUEST))
            if value_requirement
            else 0
        )
        if expected and block_id != current_id:
            allowed = expected.get(block.area, {}).get(block.column)
            if allowed is None or kind not in allowed:
                return None
        if (block.area, block.column) == final_position:
            if value_requirement and not (relic_hard and hard_count == 2):
                return None
            return (block,)
        visiting.add(block_id)
        routes = []
        for next_id in block.next_block_id_list or []:
            nxt = by_id.get(next_id)
            # Labyrinth is forward-only. Reject malformed/backward graph edges.
            if nxt and (nxt.area, nxt.column) > (block.area, block.column):
                next_kind = nxt.ChangeBlockType or nxt.block_type
                pair = (
                    block.area == nxt.area == 2
                    and kind == Block.RELIC
                    and next_kind == Block.HARD_QUEST
                )
                suffix = solve(
                    next_id,
                    relic_hard or pair if value_requirement else False,
                    hard_count,
                )
                if suffix:
                    routes.append((block,) + suffix)
        visiting.remove(block_id)
        if not routes:
            return None
        if strategy == "随机路线":
            return choice(routes)
        return max(
            routes,
            key=lambda route: route_value(route, strategy, preference, battle_seconds),
        )

    result = solve(current_id)
    return list(result) if result else None


def value_route_status(map_list):
    """Diagnose the two value-route requirements independently.

    Returns (area 4 has a path with two EX fights, most EX fights on one
    area 4 path, area 2 has a relic block leading directly into an EX fight).
    best_route(value_requirement=True) additionally needs both on one route.
    """

    def kind(block):
        return block.ChangeBlockType or block.block_type

    area4 = {b.block_id: b for b in map_list if b.area == 4}
    max_hard = 0

    def walk(block_id, hard, seen):
        nonlocal max_hard
        block = area4[block_id]
        hard += kind(block) == Block.HARD_QUEST
        max_hard = max(max_hard, hard)
        for next_id in block.next_block_id_list or []:
            if next_id in area4 and next_id not in seen:
                walk(next_id, hard, seen | {next_id})

    if area4:
        first = min(b.column for b in area4.values())
        for block in area4.values():
            if block.column == first:
                walk(block.block_id, 0, {block.block_id})

    area2 = {b.block_id: b for b in map_list if b.area == 2}
    relic_hard = any(
        kind(block) == Block.RELIC and kind(area2[next_id]) == Block.HARD_QUEST
        for block in area2.values()
        for next_id in block.next_block_id_list or []
        if next_id in area2
    )
    return max_hard >= 2, max_hard, relic_hard


def route_value(route, strategy="积分优先", preference="无偏好", battle_seconds=None):
    score, preferred = 0, 0
    desired = {"事件优先": Block.EVENT, "遗物优先": Block.RELIC}.get(preference)
    for block in route:
        kind = block.ChangeBlockType or block.block_type
        value = block_score(block, strategy)
        # In a score run, ordinary fights also finance later relic purchases.
        # Use 90 seconds as the map estimate; battle reports use actual quest limits.
        if battle_seconds is not None and kind in (
            Block.NORMAL_QUEST,
            Block.HARD_QUEST,
        ):
            coins = max(0, 90 - battle_seconds) * 10 + (
                300 if kind == Block.HARD_QUEST else 0
            )
            value += coins * 200 / 1350
        # Unknown event rewards have a conservative one-relic expectation.
        if battle_seconds is not None and kind == Block.EVENT:
            value = 200
        score += value
        preferred += kind == desired
    # A preference settles equal estimates; it cannot override an extra EX.
    return score, preferred


def reward_score(kind, reward_id, count, summon_data):
    count = count or 0
    if kind == Inventory.LabyrinthRelic:
        return 200 * count
    if kind == Inventory.LabyrinthUnit:
        return 100 * count
    if kind == Inventory.LabyrinthTicket:
        ticket = summon_data.get(reward_id)
        return 100 * count * (ticket.summon_num if ticket else 1)
    return 0


def event_choice_score(description):
    """Heuristic from localized option text; never controls protocol fields.

    Currency uses the cheapest observed relic exchange (1350 coins / 200 points).
    Event fights give recruits/relics, but no map battle-clear points.
    """
    text = (
        unicodedata.normalize("NFKC", description or "")
        .replace("\\n", "")
        .replace("\n", "")
    )
    score = 0
    end = 0
    for match in re.finditer(
        r"(遗物|遺物|印记|印記|金币|金幣)\s*[×x]\s*(\d+)(?:[~～〜](\d+))?", text
    ):
        probabilities = re.findall(r"(\d+)\s*%", text[end : match.start()])
        probability = int(probabilities[-1]) / 100 if probabilities else 1
        kind, count, upper = match.groups()
        quantity = (int(count) + int(upper)) / 2 if upper else int(count)
        value = (
            200
            if kind in ("遗物", "遺物")
            else 100 if kind in ("印记", "印記") else 200 / 1350
        )
        score += probability * quantity * value
        end = match.end()
    if any(word in text for word in ("敌人", "敵人")):
        score += 300 if any(word in text for word in ("极难", "極難", "EX")) else 100
    if any(word in text for word in ("成为伙伴", "成為夥伴")):
        score += 100
    return score


def unit_rank(unit, account_units):
    # USER candidates omit unit_data; NPC candidates carry their own stats.
    data = account_units.get(unit.unit_id) if unit.unit_type == 1 else None
    data = data or unit.unit_data
    return tuple(
        getattr(data, attr, 0) or 0
        for attr in ("unit_level", "promotion_level", "unit_rarity")
    ) + (-unit.unit_id,)


def battle_decks(snapshot, account_units, boss=False):
    """Preserve valid saved teams and fill vacancies with recruited units only."""
    owned = {unit.unit_id: unit for unit in snapshot.unit_list or [] if unit.unit_id}
    numbers = (132, 133, 134) if boss else (131,)
    saved = {
        int(deck.deck_number): deck
        for deck in snapshot.deck_list or []
        if deck.deck_number is not None
    }
    used, teams = set(), []
    for number in numbers:
        deck = saved.get(number)
        ids = [getattr(deck, f"unit_id_{i}", 0) for i in range(1, 6)]
        ids = list(dict.fromkeys(id for id in ids if id in owned and id not in used))
        used.update(ids)
        teams.append(ids)
    available = sorted(
        (id for id in owned if id not in used),
        key=lambda id: unit_rank(owned[id], account_units),
        reverse=True,
    )
    # Give each boss team a member before filling to five when the roster is small.
    for team in teams:
        if not team and available:
            team.append(available.pop(0))
    for team in teams:
        while len(team) < 5 and available:
            team.append(available.pop(0))
    return [
        DeckListData(deck_number=number, unit_list=ids)
        for number, ids in zip(numbers, teams)
    ]


def score_shop_plan(
    lineup, coins, reserve, summon_data, relic_data, resets, limit, settings
):
    """Save for cheap relics after refresh before spending on pricier stock.

    Future stock is unknown. Reserve at most the configured refresh rounds,
    using the observed number of relic slots and current discounted prices.
    At the last affordable refresh, use an exact knapsack for current stock.
    """
    relic_slots = [
        item for item in lineup if item.reward_type == Inventory.LabyrinthRelic
    ]

    def cheap(item):
        relic = relic_data.get(item.reward_id)
        return item.reward_type == Inventory.LabyrinthRelic and (
            getattr(relic, "rarity", 0) == 1
            or (relic is None and item.price is not None and item.price <= 1350)
        )

    prices = [
        item.price
        for item in relic_slots
        if cheap(item) and item.price is not None and item.price > 0
    ]
    minimum = min(prices) if prices else 1350
    reset_cost = shop_reset_cost(resets, settings)
    can_refresh = (
        bool(relic_slots)
        and resets is not None
        and resets < limit
        and reset_cost is not None
        and coins >= reserve + reset_cost + minimum
    )
    if not can_refresh:
        return shop_plan(lineup, coins, reserve, summon_data), False, minimum
    plan = shop_plan(
        [item for item in lineup if cheap(item)], coins, reserve, summon_data
    )
    if plan:
        return plan, False, minimum
    future_costs = [shop_reset_cost(i, settings) for i in range(resets, limit)]
    future_costs = [cost for cost in future_costs if cost is not None]
    future_budget = sum(future_costs) + len(future_costs) * len(relic_slots) * minimum
    tickets = [item for item in lineup if item.reward_type == Inventory.LabyrinthTicket]
    plan = shop_plan(tickets, coins, reserve + future_budget, summon_data)
    if plan:
        return plan, False, minimum
    ticket_budget = sum(
        item.price for item in tickets if item.price is not None and item.price > 0
    )
    plan = shop_plan(
        lineup,
        coins,
        reserve + future_budget + len(future_costs) * ticket_budget,
        summon_data,
    )
    return plan, not bool(plan), minimum


def shop_plan(lineup, coins, reserve, summon_data):
    """Choose an affordable subset maximizing total score, then minimizing cost."""
    budget = max(0, coins - reserve)
    candidates = [
        item
        for item in lineup
        if item.is_sold == 0
        and item.availability == ShopStatus.PURCHASE
        and item.price is not None
        and 0 <= item.price <= budget
        and reward_score(
            item.reward_type, item.reward_id, item.reward_count, summon_data
        )
        > 0
    ]
    # Sparse knapsack keeps this bounded by the budget, even for unusual lineups.
    plans = {0: (0, ())}
    for item in candidates:
        value = reward_score(
            item.reward_type, item.reward_id, item.reward_count, summon_data
        )
        for cost, (score, selected) in list(plans.items()):
            new_cost = cost + item.price
            if new_cost <= budget and score + value > plans.get(new_cost, (-1, ()))[0]:
                plans[new_cost] = score + value, selected + (item,)
    cost, (_, selected) = max(plans.items(), key=lambda pair: (pair[1][0], -pair[0]))
    return sorted(
        selected,
        key=lambda item: (
            -reward_score(
                item.reward_type, item.reward_id, item.reward_count, summon_data
            ),
            item.price,
            item.lineup_id,
        ),
    )
