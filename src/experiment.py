from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd


class Trials:
    def __init__(self, frame: pd.DataFrame, group_keys: Iterable[str], time_key: str):
        self.frame = frame
        self.group_keys = list(group_keys)
        self.time_key = time_key

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


class Analysis:
    def __init__(self, profiler: Callable[[pd.DataFrame], dict]):
        self.profiler = profiler

    def run(self, trials: Trials) -> pd.DataFrame:
        # summarize each group into one row, organized by key
        rows = {key: self.profiler(group.frame) for key, group in trials.groups()}
        summary = pd.DataFrame.from_dict(rows, orient="index")
        summary.index = pd.MultiIndex.from_tuples(summary.index, names=trials.group_keys)
        return summary


class Transform:
    def __init__(self, builder: Callable[[pd.DataFrame], pd.DataFrame]):
        self.builder = builder

    def run(self, trials: Trials) -> pd.DataFrame:
        # extend existing trial column per group then return set
        keys, frames = zip(*((key, self.builder(group.frame)) for key, group in trials.groups()))
        return pd.concat(frames, keys=keys, names=trials.group_keys)


class Run:
    def __init__(self, simulate: Callable[[object, pd.DataFrame], pd.Series]):
        self.simulate = simulate

    def begin(self, policies: Callable[[], object], trials: Trials) -> dict[tuple, pd.Series]:
        return {key: self.simulate(policies(), group.frame) for key, group in trials.groups()}

class Evaluate:
    def __init__(self, metrics: dict[str, Callable[[pd.Series, pd.Series], float]]):
        self.metrics = metrics

    @staticmethod
    def agreement(pred: pd.Series, obs: pd.Series) -> float:
        # % of trials where predicted == observed data
        return (pred == obs).mean()

    def compare(
        self,
        predicted: dict[tuple, pd.Series],
        observed: dict[tuple, pd.Series],
        cohorts: pd.Series | None = None,
        group_keys: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for key, pred in predicted.items():
            obs = observed[key]
            row = dict(zip(group_keys, key)) if group_keys is not None else {"group": key}
            for name, metric in self.metrics.items():
                row[name] = metric(pred, obs)
            if cohorts is not None:
                row["cohort"] = cohorts.loc[key]
            rows.append(row)
        return pd.DataFrame(rows)

class Export:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def _clean(frame: pd.DataFrame) -> pd.DataFrame:
        # excel is so picky :(
        bool_cols = frame.select_dtypes(include="boolean").columns
        return frame.assign(**{col: frame[col].map({True: "yes", False: "no"}) for col in bool_cols})

    def write(self, sheets: dict[str, pd.DataFrame], **sheet_kwargs: dict) -> None:
        with pd.ExcelWriter(self.path) as writer:
            for name, frame in sheets.items():
                self._clean(frame).to_excel(writer, sheet_name=name, **sheet_kwargs.get(name, {}))