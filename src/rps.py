from enum import IntEnum

import numpy as np
import pandas as pd


class Move(IntEnum):
    rock = 1
    paper = 2
    scissors = 3

    @property
    def beats(self) -> "Move":
        return RULES[self]

    @property
    def counter(self) -> "Move":
        return self.beats.beats

    def outcome(self, other: "Move") -> str:
        # from perspective of agent
        if self == other:
            return "draw"
        if self.beats == other:
            return "win"
        return "lose"


RULES = {Move.rock: Move.scissors, Move.paper: Move.rock, Move.scissors: Move.paper}


class Dyad:
    def __init__(self, group: pd.DataFrame):
        self.self_move = group["self_resp"]
        self.partner_move = group["partner_resp"]
        self.self_rt = group["self_rt"]
        self.partner_rt = group["partner_rt"]
        self._missing = self.self_move.isna()

    def _mask_missing(self, result: pd.Series, other: pd.Series) -> pd.Series:
        # clean missing entries
        return result.mask(self._missing | other.isna())

    def outcome(self) -> pd.Series:
        beats = self.self_move.map(RULES)
        outcome = pd.Series(
            np.select([self.self_move == self.partner_move, beats == self.partner_move], ["draw", "win"], default="lose"),
            index=self.self_move.index,
        )
        return self._mask_missing(outcome, self.partner_move)

    def stay(self) -> pd.Series:
        prev = self.self_move.shift(1)
        return self._mask_missing(self.self_move == prev, prev).astype("boolean")

    def shift(self) -> pd.Series:
        partner_prev = self.partner_move.shift(1)
        counter = partner_prev.map(RULES).map(RULES)
        direction = pd.Series(
            np.select([self.self_move == partner_prev, self.self_move == counter], ["imitate", "counter"], default="other"),
            index=self.self_move.index,
        )
        return self._mask_missing(direction, partner_prev)

    @property
    def loss_shift_rates(self) -> pd.Series:
        # % of shifting on subsequent trial following a loss
        after_loss_shift = (self.outcome().shift(1) == "lose") & ~self.stay()
        counts = self.shift()[after_loss_shift].value_counts()
        total = counts.sum()
        return counts / total if total else counts

    @property
    def imitate_rate(self) -> float:
        rates = self.loss_shift_rates
        return rates.get("imitate", 0.0) if not rates.empty else np.nan

    @classmethod
    def log(cls, frame: pd.DataFrame) -> pd.DataFrame:
        dyad = cls(frame)
        return pd.DataFrame({"outcome": dyad.outcome(), "decision": dyad.stay().map({True: "stay", False: "shift"})})
