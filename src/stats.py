from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control

from actr import actr_simulate
from rps import Dyad

REPEATS = 5


class Significance:
    def __init__(self, p_values: pd.Series, alpha: float = 0.05):
        self.q_values = pd.Series(np.nan, index=p_values.index)
        valid = p_values.notna()
        self.q_values[valid] = false_discovery_control(p_values[valid], method="bh")
        self.alpha = alpha

    @property
    def flag(self) -> pd.Series:
        return (self.q_values < self.alpha).mask(self.q_values.isna()).astype("boolean")

    def mask(self, series: pd.Series) -> pd.Series:
        return series.mask(~self.flag.fillna(False))


class Contingency:
    def __init__(self, tbl: pd.DataFrame):
        self.tbl = tbl
        if tbl.shape[0] < 2 or tbl.shape[1] < 2:
            self.chi2, self.p_value = np.nan, np.nan
        else:
            self.chi2, self.p_value, *_ = chi2_contingency(tbl, correction=False)

    @property
    def cramers_v(self) -> float:
        # V ≤ 0.10 is weak; V ≥ 0.25 is a strong effect size
        return (self.chi2 / self.tbl.to_numpy().sum()) ** 0.5 if not np.isnan(self.chi2) else np.nan

    @cached_property
    def _stay_rate(self) -> pd.Series:
        if np.isnan(self.chi2):
            return pd.Series(dtype=float)
        return self.tbl[True] / self.tbl.sum(axis=1)

    @property
    def win_stay_rate(self) -> float:
        return self._stay_rate.get("win", np.nan)

    @property
    def lose_stay_rate(self) -> float:
        return self._stay_rate.get("lose", np.nan)


@dataclass
class Strategy:
    win_stay_rate: float
    lose_stay_rate: float
    counter_rate: float

    @property
    def cluster(self) -> str:
        if pd.isna(self.win_stay_rate) or pd.isna(self.lose_stay_rate):
            return "none"
        if self.win_stay_rate < self.lose_stay_rate:
            return "WCLS"
        return "WSLC"

    @classmethod
    def classify(cls, win_stay_rate: pd.Series, lose_stay_rate: pd.Series, counter_rate: pd.Series) -> pd.Series:
        return pd.Series(
            [cls(w, l, c).cluster for w, l, c in zip(win_stay_rate, lose_stay_rate, counter_rate)],
            index=win_stay_rate.index,
        )


@dataclass
class Profile:
    chi2: float
    p_value: float
    cramers_v: float
    win_stay_rate: float
    lose_stay_rate: float
    counter_rate: float

    @classmethod
    def from_frame(cls, group: pd.DataFrame) -> "Profile":
        dyad = Dyad(group)
        outcome, stayed = dyad.outcome(), dyad.stay()

        test = Contingency(pd.crosstab(outcome.shift(1), stayed))
        return cls(
            chi2=test.chi2, # magnitude of association between prior outcome & staying
            p_value=test.p_value, # significance of chi2 relationship
            cramers_v=test.cramers_v, # normalized effect size of chi2
            win_stay_rate=test.win_stay_rate, # % stayed after a win
            lose_stay_rate=test.lose_stay_rate, # % stayed after a loss
            counter_rate=dyad.counter_rate, # % of post-loss shifts that beat opponent's last move
        )

    @classmethod
    def sample(cls, model_cls, frame: pd.DataFrame, repeats: int) -> list["Profile"]:
        profiles = []
        for _ in range(repeats):
            predicted = actr_simulate(model_cls(), frame)
            sim_frame = pd.DataFrame({
                "self_resp": predicted,
                "partner_resp": frame["partner_resp"],
                "self_rt": 0,
                "partner_rt": 0,
            })
            profiles.append(cls.from_frame(sim_frame))
        return profiles

    @classmethod
    def simulate(cls, model_cls, frame: pd.DataFrame, repeats: int = REPEATS) -> "Profile":
        profiles = cls.sample(model_cls, frame, repeats)
        return cls(**{
            field: np.nanmean([getattr(p, field) for p in profiles])
            for field in cls.__dataclass_fields__
        })
