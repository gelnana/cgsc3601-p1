import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control


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
        return (self.chi2 / self.tbl.to_numpy().sum()) ** 0.5 if not np.isnan(self.chi2) else np.nan

    @property
    def continuum(self) -> float:
        if np.isnan(self.chi2):
            return np.nan
        # WSLC (+1) <-> WCLS (-1)
        stay_rate = self.tbl[True] / self.tbl.sum(axis=1)
        return stay_rate.get("win", np.nan) - stay_rate.get("lose", np.nan)


class Strategy:
    def __init__(self, continuum: float, imitate_rate: float):
        self.continuum = continuum
        self.imitate_rate = imitate_rate

    @property
    def cluster(self) -> str:
        if pd.isna(self.continuum):
            return "none"
        if self.continuum < 0:
            return "WCLS"
        if pd.isna(self.imitate_rate):
            return "WSLC"
        return "WSLC-I" if self.imitate_rate > 0.5 else "WSLC-C"

    @classmethod
    def classify(cls, continuum: pd.Series, imitate_rate: pd.Series) -> pd.Series:
        return pd.Series([cls(c, i).cluster for c, i in zip(continuum, imitate_rate)], index=continuum.index)
