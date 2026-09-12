"""竞技场相关日常、回刺查询、排名透视与换防功能。

独立于 daily.py 和 tools.py，减少同步上游时的冲突。
"""

import asyncio
import itertools
import random
from datetime import datetime
from typing import List, Set, Union

from ...core.apiclient import apiclient
from ...core.pcrclient import pcrclient
from ...db.database import db
from ...model.common import (
    ChangeRarityUnit,
    DeckListData,
    GrandArenaHistoryDetailInfo,
    GrandArenaHistoryInfo,
    GrandArenaSearchOpponent,
    ProfileUserInfo,
    RankingSearchOpponent,
    VersusResult,
    VersusResultDetail,
)
from ...model.custom import ArenaQueryResult
from ...model.enums import ePartyType
from ...model.error import AbortError, SkipError
from ...util.arena import instance as ArenaQuery
from ...util.questutils import create_battle_start_token
from ..config import booltype, inttype
from ..modulebase import Module, default, description, name

@description('仅进攻，不结算，会消耗次数')
@name('完成每日jjc任务')
@default(False)
class jjc_daily(Module):
    async def do_task(self, client: pcrclient):
        if client.data.is_empty_deck(ePartyType.ARENA):
            raise AbortError("未设置进攻队伍，请设置") # AUTO SET TODO

        info = await client.get_arena_info()
        if info.arena_info.battle_number != info.arena_info.max_battle_number:
            raise SkipError("今日jjc任务已完成")

        for _ in range(3):
            if info.search_opponent: break
            await asyncio.sleep(2)
            info = await client.get_arena_info()

        if not info.search_opponent:
            raise AbortError("无法搜到可攻击对手，请稍后再试")
        opponent = info.search_opponent[0]
        await client.arena_apply(opponent.viewer_id, opponent.rank)
        token = create_battle_start_token()
        await client.arena_start(token, opponent.viewer_id, info.arena_info.battle_number, 1)
        await client.logout()
        await asyncio.sleep(2)
        self._log(f"当前排名{info.arena_info.rank}，进攻第{opponent.rank}名的【{opponent.user_name}】")

@description('仅进攻，不结算，会消耗次数')
@name('完成每日pjjc任务')
@default(False)
class pjjc_daily(Module):
    async def do_task(self, client: pcrclient):
        if client.data.is_empty_deck(ePartyType.GRAND_ARENA_1) or \
        client.data.is_empty_deck(ePartyType.GRAND_ARENA_2) or \
        client.data.is_empty_deck(ePartyType.GRAND_ARENA_3):
            raise AbortError("未设置进攻队伍，请设置") # AUTO SET TODO

        info = await client.get_grand_arena_info()
        if info.grand_arena_info.battle_number != info.grand_arena_info.max_battle_number:
            raise SkipError("今日pjjc任务已完成")

        for _ in range(3):
            if info.search_opponent: break
            await asyncio.sleep(2)
            info = await client.get_grand_arena_info()

        if not info.search_opponent:
            raise AbortError("无法搜到可攻击对手，请稍后再试")
        opponent = info.search_opponent[0]
        await client.grand_arena_apply(opponent.viewer_id, opponent.rank)
        token = create_battle_start_token()
        await client.grand_arena_start(token, opponent.viewer_id, info.grand_arena_info.battle_number, 1)
        await client.logout()
        await asyncio.sleep(2)
        self._log(f"当前排名{info.grand_arena_info.rank}，进攻第{opponent.rank}名的【{opponent.user_name}】")

class Arena(Module):

    def target_rank(self) -> int: ...

    def present_defend(self, defen: Union[List[List[int]], List[int]]) -> str: ...

    def present_attack(self, attack: Union[List[List[ArenaQueryResult]], List[ArenaQueryResult]]) -> str: ...

    def get_rank_from_user_info(self, user_info: ProfileUserInfo) -> int: ...

    async def self_rank(self, client: pcrclient) -> int: ...

    async def choose_best_team(self, team: Union[List[ArenaQueryResult], List[List[ArenaQueryResult]]], rank_id: List[int], client: pcrclient) -> int: ...

    async def update_deck(self, units: Union[List[ArenaQueryResult], ArenaQueryResult], client: pcrclient): ...

    async def get_rank_info(self, client: pcrclient, rank: int) -> Union[RankingSearchOpponent, GrandArenaSearchOpponent]: ...

    async def get_opponent_info(self, client: pcrclient, viewer_id: int) -> Union[RankingSearchOpponent, GrandArenaSearchOpponent]: ...

    async def get_arena_history(self, client: pcrclient) -> Union[List[VersusResult], List[GrandArenaHistoryInfo]]: ...

    async def get_history_detail(self, log_id: int, client: pcrclient) -> Union[VersusResultDetail, GrandArenaHistoryDetailInfo]: ...

    async def get_defend_from_info(self, info: Union[RankingSearchOpponent, GrandArenaSearchOpponent]) -> Union[List[List[int]], List[int]]: ...

    async def get_defend_from_histroy_detail(self, history_detail: Union[VersusResultDetail, GrandArenaHistoryDetailInfo]) -> Union[List[List[int]], List[int]]: ...


    async def get_attack_team(self, defen: Union[List[List[int]], List[int]]) -> Union[List[List[ArenaQueryResult]], List[ArenaQueryResult]]: ...

    async def get_defend(self, client: pcrclient) -> Union[List[List[int]], List[int]]:
        target_rank: int = self.target_rank()
        self_rank = await self.self_rank(client)

        if target_rank > 0:
            target = await self.get_rank_info(client, target_rank)
            target_info = (await client.get_profile(target.viewer_id)).user_info
            self._log(f"{target_info.user_name}({target.viewer_id})")
            self._log(f"{self_rank} -> {target_rank}({target_info.user_name})")
            defend = await self.get_defend_from_info(target)
        else:
            historys = await self.get_arena_history(client)
            if not historys:
                raise AbortError("没有被刺记录")
            id = -target_rank
            if id == 0:
                for i, h in enumerate(historys):
                    h_detail = await self.get_history_detail(h.log_id, client)
                    if h_detail.is_challenge:
                        self._log(f"查找第{i + 1}条记录")
                        history = h
                        history_detail = h_detail
                        break
                else:
                    raise AbortError("没有刺人记录")
            else:
                self._log(f"查找第{id}条记录")
                if len(historys) < id:
                    raise AbortError(f"只有{len(historys)}条被刺记录")
                history = historys[id - 1]
                history_detail = await self.get_history_detail(history.log_id, client)

            target = history.opponent_user

            target_info = (await client.get_profile(target.viewer_id)).user_info
            target_rank = self.get_rank_from_user_info(target_info)

            self._log(f"{target.user_name}({target.viewer_id})\n{datetime.fromtimestamp(history.versus_time)} {'刺' if history_detail.is_challenge else '被刺'}")
            self._log(f"{self_rank} -> {target_rank}({target_info.user_name})")

            if history_detail.is_challenge:
                defend = await self.get_defend_from_histroy_detail(history_detail)
            else:
                target = await self.get_opponent_info(client, target.viewer_id)
                defend = await self.get_defend_from_info(target)


        if isinstance(defend[0], list):
            defend = [d[-5:] for d in defend]
        else:
            defend = defend[-5:]

        return defend


    async def do_task(self, client: pcrclient):
        self.available_unit: Set[int] = set(unit_id for unit_id in client.data.unit if client.data.unit[unit_id].promotion_level >= 7)

        defend = await self.get_defend(client)
        attack = await self.get_attack_team(defend)

        defend_str = self.present_defend(defend)

        if attack == []:
            raise AbortError(f'{defend_str}\n抱歉没有查询到解法\n※没有作业说明随便拆 发挥你的想象力～★\n')

        rank_id = list(range(len(attack)))
        best_team_id = await self.choose_best_team(attack, rank_id, client)
        if best_team_id >= 0 and best_team_id < len(attack):
            self._log(f"选择第{best_team_id + 1}支队伍作为进攻方队伍")
            await self.update_deck(attack[best_team_id], client)
        else:
            self._warn(f"队伍只有{len(attack)}支，无法选择第{best_team_id + 1}支队伍作为进攻方队伍")

        attack_str = self.present_attack(attack[:max(8, best_team_id + 1)])
        msg = [defend_str, "-------", attack_str]
        self._log('\n'.join(msg))

@description('查询jjc回刺阵容，并自动设置进攻队伍，对手排名=0则查找对战纪录第一条刺人的，<0则查找对战纪录，-1表示第一条，-2表示第二条，以此类推')
@name('jjc回刺查询')
@default(True)
@inttype("opponent_jjc_attack_team_id", "选择阵容", 1, [i for i in range(1, 10)])
@inttype("opponent_jjc_rank", "对手排名", -1, [i for i in range(-20, 101)])
class jjc_back(Arena):

    def target_rank(self) -> int:
        return self.get_config("opponent_jjc_rank")

    async def self_rank(self, client: pcrclient) -> int:
        return (await client.get_arena_info()).arena_info.rank

    def get_rank_from_user_info(self, user_info: ProfileUserInfo) -> int:
        return user_info.arena_rank

    def present_defend(self, defen: List[int]) -> str:
        msg = [db.get_unit_name(x) for x in defen]
        msg = f"防守方【{' '.join(msg)}】"
        return msg

    def present_attack(self, attack: List[ArenaQueryResult]) -> str:
        msg = ArenaQuery.str_result(attack)
        return msg

    async def choose_best_team(self, team: List[ArenaQueryResult], rank_id: List[int], client: pcrclient) -> int:
        id = int(self.get_config("opponent_jjc_attack_team_id")) - 1
        return id

    async def update_deck(self, units: ArenaQueryResult, client: pcrclient):
        units_id = [unit.id for unit in units.atk]
        star_change_unit = [unit_id for unit_id in units_id if client.data.unit[unit_id].unit_rarity == 5 and client.data.unit[unit_id].battle_rarity != 0]
        if star_change_unit:
            res = [ChangeRarityUnit(unit_id=unit_id, battle_rarity=5) for unit_id in star_change_unit]
            self._log(f"将{'|'.join([db.get_unit_name(unit_id) for unit_id in star_change_unit])}调至5星")
            await client.unit_change_rarity(res)

        under_rank_bonus_unit = [unit for unit in units_id if client.data.unit[unit].promotion_level < db.equip_max_rank - 1]
        if under_rank_bonus_unit:
            self._warn(f"无品级加成：{'，'.join([db.get_unit_name(unit_id) for unit_id in under_rank_bonus_unit])}")

        await client.deck_update(ePartyType.ARENA, units_id)

    async def get_rank_info(self, client: pcrclient, rank: int) -> RankingSearchOpponent:
        for page in range(1, 6):
            ranking = {info.rank: info for info in (await client.arena_rank(20, page)).ranking}
            if rank in ranking:
                return ranking[rank]
        raise AbortError("对手不在前100名，无法查询")

    async def get_opponent_info(self, client: pcrclient, viewer_id: int) -> RankingSearchOpponent:
        for page in range(1, 6):
            ranking = {info.viewer_id: info for info in (await client.arena_rank(20, page)).ranking}
            if viewer_id in ranking:
                return ranking[viewer_id]
        raise AbortError("对手不在前100名，无法查询")

    async def get_arena_history(self, client: pcrclient) -> List[VersusResult]:
        return (await client.get_arena_history()).versus_result_list

    async def get_history_detail(self, log_id: int, client: pcrclient) -> VersusResultDetail:
        return (await client.get_arena_history_detail(log_id)).versus_result_detail

    async def get_defend_from_info(self, info: RankingSearchOpponent) -> List[int]:
        return [unit.id for unit in info.arena_deck]

    async def get_defend_from_histroy_detail(self, history_detail: VersusResultDetail) -> List[int]:
        return [unit.id for unit in history_detail.vs_user_arena_deck]

    async def get_attack_team(self, defen: List[int]) -> List[ArenaQueryResult]:
        return await ArenaQuery.get_attack(self.available_unit, defen)

@description('查询pjjc回刺阵容，并自动设置进攻队伍，对手排名=0则查找对战纪录第一条刺人的，<0则查找对战纪录，-1表示第一条，-2表示第二条，以此类推')
@name('pjjc回刺查询')
@default(True)
@inttype("opponent_pjjc_attack_team_id", "选择阵容", 1, [i for i in range(1, 10)])
@inttype("opponent_pjjc_rank", "对手排名", -1, [i for i in range(-20, 101)])
class pjjc_back(Arena):
    def target_rank(self) -> int:
        return self.get_config("opponent_pjjc_rank")

    def present_defend(self, defen: List[List[int]]) -> str:
        msg = [' '.join([db.get_unit_name(y) for y in x]) for x in defen]
        msg = '\n'.join(msg)
        msg = f"防守方\n{msg}"
        return msg

    def present_attack(self, attack: List[List[ArenaQueryResult]]) -> str:
        msg = [f"第{id + 1}对策\n{ArenaQuery.str_result(x)}" for id, x in enumerate(attack)]
        msg = '\n\n'.join(msg)
        return msg

    def get_rank_from_user_info(self, user_info: ProfileUserInfo) -> int:
        return user_info.grand_arena_rank

    async def self_rank(self, client: pcrclient) -> int:
        return (await client.get_grand_arena_info()).grand_arena_info.rank

    async def choose_best_team(self, team: List[List[ArenaQueryResult]], rank_id: List[int], client: pcrclient) -> int:
        id = int(self.get_config("opponent_pjjc_attack_team_id")) - 1
        return id

    async def update_deck(self, units: List[ArenaQueryResult], client: pcrclient):
        units_id = [[uni.id for uni in unit.atk] for unit in units]
        star_change_unit = [uni_id for unit_id in units_id for uni_id in unit_id if
                            client.data.unit[uni_id].unit_rarity == 5 and
                            client.data.unit[uni_id].battle_rarity != 0]
        if star_change_unit:
            res = [ChangeRarityUnit(unit_id=unit_id, battle_rarity=5) for unit_id in star_change_unit]
            self._log(f"将{'|'.join([db.get_unit_name(unit_id) for unit_id in star_change_unit])}调至5星")
            await client.unit_change_rarity(res)

        under_rank_bonus_unit = [uni_id for unit_id in units_id for uni_id in unit_id if
                                 client.data.unit[uni_id].promotion_level < db.equip_max_rank - 1]
        if under_rank_bonus_unit:
            self._warn(f"无品级加成：{'，'.join([db.get_unit_name(unit_id) for unit_id in under_rank_bonus_unit])}")

        deck_list = []
        for i, unit_id in enumerate(units_id):
            deck_number = getattr(ePartyType, f"GRAND_ARENA_{i + 1}")
            sorted_unit_id = db.deck_sort_unit(unit_id)

            deck = DeckListData()
            deck.deck_number = deck_number
            deck.unit_list = sorted_unit_id
            deck_list.append(deck)

        await client.deck_update_list(deck_list)

    async def get_rank_info(self, client: pcrclient, rank: int) -> GrandArenaSearchOpponent:
        for page in range(1, 6):
            ranking = {info.rank: info for info in (await client.grand_arena_rank(20, page)).ranking}
            if rank in ranking:
                return ranking[rank]
        raise AbortError("对手不在前100名，无法查询")

    async def get_opponent_info(self, client: pcrclient, viewer_id: int) -> GrandArenaSearchOpponent:
        for page in range(1, 6):
            ranking = {info.viewer_id: info for info in (await client.grand_arena_rank(20, page)).ranking}
            if viewer_id in ranking:
                return ranking[viewer_id]
        # raise AbortError("对手不在前100名，无法查询")
        ret = GrandArenaSearchOpponent(viewer_id=viewer_id)
        return ret

    async def get_arena_history(self, client: pcrclient) -> List[GrandArenaHistoryInfo]:
        return (await client.get_grand_arena_history()).grand_arena_history_list

    async def get_history_detail(self, log_id: int, client: pcrclient) -> GrandArenaHistoryDetailInfo:
        return (await client.get_grand_arena_history_detail(log_id)).grand_arena_history_detail

    async def get_defend_from_info(self, info: GrandArenaSearchOpponent) -> List[List[int]]:
        ret = []
        if info.grand_arena_deck:
            if info.grand_arena_deck.first and info.grand_arena_deck.first[0].id != 2:
                ret.append([unit.id for unit in info.grand_arena_deck.first])
            if info.grand_arena_deck.second and info.grand_arena_deck.second[0].id != 2:
                ret.append([unit.id for unit in info.grand_arena_deck.second])
            if info.grand_arena_deck.third and info.grand_arena_deck.third[0].id != 2:
                ret.append([unit.id for unit in info.grand_arena_deck.third])

        if len(ret) < 2:
            ret = self.find_cache(str(info.viewer_id))
            if ret is None:
                raise AbortError("未知的对手防守，请尝试进攻一次")
            print("读取缓存队伍阵容")
        return ret

    async def get_defend_from_histroy_detail(self, history_detail: GrandArenaHistoryDetailInfo) -> List[List[int]]:
        ret = []
        if history_detail.vs_user_grand_arena_deck.first[0].id != 2:
            ret.append([unit.id for unit in history_detail.vs_user_grand_arena_deck.first])
        if history_detail.vs_user_grand_arena_deck.second[0].id != 2:
            ret.append([unit.id for unit in history_detail.vs_user_grand_arena_deck.second])
        if history_detail.vs_user_grand_arena_deck.third[0].id != 2:
            ret.append([unit.id for unit in history_detail.vs_user_grand_arena_deck.third])
        self.save_cache(str(history_detail.vs_user_viewer_id), ret)
        return ret

    async def get_attack_team(self, defen: List[List[int]]) -> List[List[ArenaQueryResult]]:
        return await ArenaQuery.get_multi_attack(self.available_unit, defen)

class ArenaInfo(Module):

    @property
    def use_cache(self) -> bool: ...

    async def get_rank_info(self, client: pcrclient, num: int, page: int) -> List[Union[GrandArenaSearchOpponent, RankingSearchOpponent]]: ...

    async def get_user_info(self, client: pcrclient, viewer_id: int) -> str:
        user_name = self.find_cache(str(viewer_id))
        if user_name is None or not self.use_cache:
            user_name = (await client.get_profile(viewer_id)).user_info.user_name
            self.save_cache(str(viewer_id), user_name)
        return user_name

    async def do_task(self, client: pcrclient):
        time = db.format_time(apiclient.datetime)
        self._log(f"时间：{time}")
        for page in range(1, 4):
            ranking = await self.get_rank_info(client, 20, page)
            for info in ranking:
                if info.rank > 51:
                    break
                user_name = await self.get_user_info(client, info.viewer_id)
                you = " <--- 你" if info.viewer_id == client.data.uid else ""
                self._log(f"{info.rank:02}: ({info.viewer_id}){user_name}{you}")

@booltype("jjc_info_cache", "使用缓存信息", True)
@description('jjc透视前51名玩家的名字')
@name('jjc透视')
@default(True)
class jjc_info(ArenaInfo):
    @property
    def use_cache(self) -> bool: return self.get_config("jjc_info_cache")

    async def get_rank_info(self, client: pcrclient, num: int, page: int) -> List[RankingSearchOpponent]:
        return (await client.arena_rank(num, page)).ranking

@booltype("pjjc_info_cache", "使用缓存信息", True)
@description('pjjc透视前51名玩家的名字')
@name('pjjc透视')
@default(True)
class pjjc_info(ArenaInfo):
    @property
    def use_cache(self) -> bool: return self.get_config("pjjc_info_cache")

    async def get_rank_info(self, client: pcrclient, num: int, page: int) -> List[GrandArenaSearchOpponent]:
        return (await client.grand_arena_rank(num, page)).ranking

class ShuffleTeam(Module):
    def team_cnt(self) -> int: ...
    def deck_num(self, num: int) -> ePartyType: ...
    async def check_limit(self, client: pcrclient):
        pass

    def shuffle_candidate(self) -> List[List[int]]:
        teams = [list(x) for x in itertools.permutations(range(self.team_cnt()))]
        teams = [x for x in teams if all(x[i] != i for i in range(self.team_cnt()))]
        return teams

    async def do_task(self, client: pcrclient):
        ids = random.choice(self.shuffle_candidate())
        deck_list: List[DeckListData] = []
        for i in range(self.team_cnt()):
            deck_number = self.deck_num(i)
            units = client.data.deck_list[deck_number]
            units_id = [getattr(units, f"unit_id_{i + 1}") for i in range(5)]

            deck = DeckListData()
            deck_number = self.deck_num(ids[i])
            deck.deck_number = deck_number
            deck.unit_list = units_id
            deck_list.append(deck)

        await self.check_limit(client)
        deck_list.sort(key=lambda x: x.deck_number)
        self._log('\n'.join([f"{i} -> {j}" for i, j in enumerate(ids)]))
        await client.deck_update_list(deck_list)

class PJJCShuffleTeam(ShuffleTeam):
    def team_cnt(self) -> int: return 3

@description('将pjjc进攻阵容随机错排')
@name('pjjc换攻')
class pjjc_atk_shuffle_team(PJJCShuffleTeam):
    def deck_num(self, num: int) -> ePartyType: return getattr(ePartyType, f"GRAND_ARENA_{num + 1}")

@description('将pjjc防守阵容随机错排')
@name('pjjc换防')
class pjjc_def_shuffle_team(PJJCShuffleTeam):
    def deck_num(self, num: int) -> ePartyType: return getattr(ePartyType, f"GRAND_ARENA_DEF_{num + 1}")
    async def check_limit(self, client: pcrclient):
        info = await client.get_grand_arena_info()
        limit_info = info.update_deck_times_limit
        if limit_info.round_times == limit_info.round_max_limited_times:
            ok_time = db.format_time(db.parse_time(limit_info.round_end_time))
            raise AbortError(f"已达到换防次数上限{limit_info.round_max_limited_times}，请于{ok_time}后再试")
        if limit_info.daily_times == limit_info.daily_max_limited_times:
            raise AbortError(f"已达到换防次数上限{limit_info.daily_max_limited_times}，请于明日再试")
        msg = f"{db.format_time(db.parse_time(limit_info.round_end_time))}刷新" if limit_info.round_times else ""
        self._log(f'''本轮换防次数{limit_info.round_times}/{limit_info.round_max_limited_times}，{msg}
今日换防次数{limit_info.daily_times}/{limit_info.daily_max_limited_times}''')
