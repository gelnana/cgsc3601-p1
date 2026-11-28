from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import pingouin as pg


@dataclass
class Trials:
    frame: pd.DataFrame
    group_keys: Iterable[str]
    time_key: str

    def __post_init__(self):
        self.group_keys = list(self.group_keys)

    @classmethod
    def from_excel(cls, path: str, group_keys: Iterable[str], time_key: str) -> "Trials":
        group_keys = list(group_keys)
        frame = pd.read_excel(path, index_col=list(range(len(group_keys) + 1)))
        return cls(frame, group_keys, time_key)

    def groups(self) -> Iterator[tuple[tuple, "Trials"]]:
        # index using one (id, trials) pair per-participant
        for key, group in self.frame.groupby(level=self.group_keys, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            yield key, Trials(group.droplevel(self.group_keys), [], self.time_key)

    def partition(self, start=None, stop=None) -> "Trials":
        # retrieve slice of trial series
        values = self.frame.index.get_level_values(self.time_key)
        keep = np.ones(len(values), dtype=bool)
        if start is not None:
            keep &= values >= start
        if stop is not None:
            keep &= values < stop
        return Trials(self.frame[keep], self.group_keys, self.time_key)


@dataclass
class Analysis:
    profiler: Callable[[pd.DataFrame], dict]

    def run(self, trials: Trials) -> pd.DataFrame:
        # summarize each group into one row, organized by key
        rows = {key: self.profiler(group.frame) for key, group in trials.groups()}
        rows = {key: asdict(value) if is_dataclass(value) else value for key, value in rows.items()}
        summary = pd.DataFrame.from_dict(rows, orient="index")
        summary.index = pd.MultiIndex.from_tuples(summary.index, names=trials.group_keys)
        return summary


@dataclass
class Transform:
    builder: Callable[[pd.DataFrame], pd.DataFrame]

    def run(self, trials: Trials) -> pd.DataFrame:
        # extend existing trial column per group then return set
        keys, frames = zip(*((key, self.builder(group.frame)) for key, group in trials.groups()))
        return pd.concat(frames, keys=keys, names=trials.group_keys)


@dataclass
class Run:
    simulate: Callable[[type, pd.DataFrame], object]

    def begin(self, policies: type, trials: Trials) -> dict[tuple, object]:
        return {key: self.simulate(policies, group.frame) for key, group in trials.groups()}

@dataclass
class Evaluate:
    stats: list[str]

    def compare(
        self,
        predicted: dict[tuple, object],
        observed: dict[tuple, object],
        cohorts: pd.Series | None = None,
        group_keys: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for key, pred in predicted.items():
            obs = observed[key]
            row = dict(zip(group_keys, key)) if group_keys is not None else {"group": key}
            for stat in self.stats:
                row[f"{stat}_real"] = getattr(obs, stat)
                row[f"{stat}_model"] = getattr(pred, stat)
                row[f"{stat}_error"] = abs(getattr(pred, stat) - getattr(obs, stat))
            if cohorts is not None:
                row["cohort"] = cohorts.loc[key]
            rows.append(row)
        return pd.DataFrame(rows)

@dataclass
class Agreement:
    stats: list[str]

    def test(self, matched: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for cohort, group in matched.groupby("cohort"):
            for stat in self.stats:
                real = group[f"{stat}_real"].to_numpy(dtype=float)
                model = group[f"{stat}_model"].to_numpy(dtype=float)
                valid = ~np.isnan(real) & ~np.isnan(model)
                real, model = real[valid], model[valid]
                diff = model - real

                bias = diff.mean() if len(diff) else np.nan
                spread = diff.std(ddof=1) if len(diff) > 1 else np.nan
                margin = 1.96 * spread
                limit_lower, limit_upper = bias - margin, bias + margin

                if len(diff) > 1 and np.any(diff != 0):
                    wilcoxon_p = pg.wilcoxon(diff)["p_val"].iloc[0]
                else:
                    wilcoxon_p = np.nan

                rows.append({
                    "cohort": cohort, # strategy cluster
                    "stat": stat, # statistic being measured (cramers v, and win-stay/lose-stay/counter rate) 
                    "n": len(real), # valid dyads'
                    "bias": bias, # averages of Model - Participant error, positive value overpredicts
                    "limit_lower": limit_lower, # lower ~95% agreement bound for individual differences.
                    "limit_upper": limit_upper, # upper ~95% agreement bound for individual differences.
                    "wilcoxon_p": wilcoxon_p, # significance of the bias being non-zero
                })
        return pd.DataFrame(rows)


@dataclass
class Export:
    path: str

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        # excel is so picky :(
        bool_cols = frame.select_dtypes(include="boolean").columns
        return frame.assign(**{col: frame[col].map({True: "yes", False: "no"}) for col in bool_cols})

    def write(self, sheets: dict[str, pd.DataFrame], **sheet_kwargs: dict) -> None:
        with pd.ExcelWriter(self.path) as writer:
            for name, frame in sheets.items():
                self._clean(frame).to_excel(writer, sheet_name=name, **sheet_kwargs.get(name, {}))