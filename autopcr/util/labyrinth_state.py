"""Apply Labyrinth response deltas without reloading the run after every action."""
from ..model.common import LabyrinthStatus, LabyrinthRelicInfo, LabyrinthUnitInfo
from ..model.enums import eInventoryType as Inventory, eLabyrinthStatusType as Status
from ..model.responses import LabyrinthResumeResponse
from ..model.error import AbortError


class LabyrinthState:
    def __init__(self, snapshot, pending=None):
        self.snapshot = snapshot.copy(deep=True)
        # Return contexts survive rewards that temporarily replace shop/event status.
        self.pending = [LabyrinthStatus.parse_obj(s) for s in pending or []]
        self._continue()

    @classmethod
    def entered(cls, response, guild_id):
        blocks = response.map_list or []
        if not blocks or response.status is None:
            raise AbortError('enter 缺少地图或状态')
        start = min(blocks, key=lambda b: (b.area, b.column, b.row))
        state = cls(LabyrinthResumeResponse(
            guild_id=guild_id, block_id=start.block_id, alpha_rupee=0,
            status=response.status, map_list=blocks, unit_list=[], relic_list=[], deck_list=[], treasure_box=0))
        state._rewards(response.reward_list)
        return state

    def dump_pending(self):
        return [s.dict(exclude_none=True) for s in self.pending]

    def _merge_units(self, units):
        current = {u.unit_id: u for u in self.snapshot.unit_list or []}
        for unit in units or []:
            old = current.get(unit.unit_id)
            current[unit.unit_id] = LabyrinthUnitInfo.parse_obj({**old.dict(), **unit.dict(exclude_none=True)}) if old else unit.copy(deep=True)
        self.snapshot.unit_list = list(current.values())

    def _rewards(self, rewards):
        snapshot = self.snapshot
        relics = {r.relic_id: r for r in snapshot.relic_list or []}
        for item in rewards or []:
            delta = item.received if item.received is not None else item.count or 0
            if item.type == Inventory.AlphaRupee:
                snapshot.alpha_rupee = item.stock if item.stock is not None else (snapshot.alpha_rupee or 0) + delta
            elif item.type == Inventory.LabyrinthTreasureBox:
                snapshot.treasure_box = item.stock if item.stock is not None else (snapshot.treasure_box or 0) + delta
            elif item.type == Inventory.LabyrinthRelic and (item.stock if item.stock is not None else delta) > 0:
                relics.setdefault(item.id, LabyrinthRelicInfo(relic_id=item.id, effected_count=0, enable=1))
            elif item.type == Inventory.LabyrinthUnit:
                # Direct event recruits may be returned in reward_list. Do not
                # invent USER/NPC type; summon_unit's unit_info_list fills it in.
                self._merge_units([LabyrinthUnitInfo(unit_id=item.id, unit_data=item.unit_data)])
        snapshot.relic_list = list(relics.values())

    def _continue(self):
        status = self.snapshot.status
        if status is None or status.type is None:
            raise AbortError('响应缺少黎明界 status.type，已保留当前开局')
        # An explicit parent returned by the server takes precedence over the
        # saved context. NONE means the child is complete, not necessarily the tile.
        if self.pending and status.type == self.pending[-1].type and (
                status.type != Status.EVENT_EFFECT or status.event_result_id == self.pending[-1].event_result_id):
            self.pending.pop()
        if status.type == Status.NONE and self.pending:
            self.snapshot.status = self.pending.pop()
        if self.snapshot.status.type == Status.SHOP:
            owned = {r.relic_id for r in self.snapshot.relic_list or []}
            for item in self.snapshot.status.shop_lineup_list or []:
                if item.reward_type == Inventory.LabyrinthRelic and item.reward_id in owned:
                    item.availability = 3

    def apply(self, operation, response, *, block_id=None, lineup_id=None, next_effect=None):
        snapshot = self.snapshot
        old = snapshot.status
        if operation == 'shop_buy' and response.after_currency_num is None:
            raise AbortError('shop_buy 缺少剩余金币，已停止继续购买')
        if operation == 'shop_reset':
            if response.shop_lineup_list is None or response.shop_reset_count is None or response.after_currency_num is None:
                raise AbortError('shop_reset 缺少货架、刷新计数或剩余金币')
            snapshot.status = old.copy(update={
                'shop_lineup_list': response.shop_lineup_list,
                'shop_reset_count': response.shop_reset_count}, deep=True)
        else:
            if response.status is None or response.status.type is None:
                raise AbortError(f'{operation} 响应缺少 status.type')
            if operation == 'shop_buy' and response.status.type != Status.SHOP:
                shop = old.copy(deep=True)
                for item in shop.shop_lineup_list or []:
                    if item.lineup_id == lineup_id:
                        item.is_sold = 1
                self.pending.append(shop)
            if operation == 'event_effect' and next_effect is not None:
                self.pending.append(LabyrinthStatus(type=Status.EVENT_EFFECT,
                    event_result_id=old.event_result_id, effect_num=next_effect))
            snapshot.status = response.status.copy(deep=True)
        if operation == 'move':
            snapshot.block_id = block_id
            for block in snapshot.map_list or []:
                if block.block_id == block_id:
                    block.is_visited = 1
        self._rewards(getattr(response, 'reward_list', None))
        self._rewards(getattr(response, 'purchase_list', None))
        coins = getattr(response, 'after_currency_num', None)
        if coins is not None:
            snapshot.alpha_rupee = coins  # Absolute balance, including zero.
        self._merge_units(getattr(response, 'unit_info_list', None))
        used = old.relic_id if operation == 'use_relic' else getattr(response, 'use_relic_id', None)
        if used:
            for relic in snapshot.relic_list or []:
                if relic.relic_id == used:
                    relic.effected_count = (relic.effected_count or 0) + 1
        self._continue()
