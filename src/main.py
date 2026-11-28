import pandas as pd

from actr import Fit
from experiment import Agreement, Analysis, Evaluate, Export, Run, Trials, Transform
from models import Baseline, WSLC, WSLC_I, WSLC_C, WCLS
from rps import Dyad
from stats import Profile, Significance, Strategy

from concurrent.futures import ProcessPoolExecutor

STATS = ["win_stay_rate", "lose_stay_rate", "counter_rate", "cramers_v"]
COHORT_MODELS = {"none": (Baseline,), "WCLS": (WCLS,), "WSLC": (WSLC_I, WSLC_C)}


def _thread(key, candidates, target, calib_frame, holdout_frame):
    fit = Fit.select(candidates, calib_frame, target)
    predicted = Profile.simulate(fit, holdout_frame)
    return key, predicted, fit.model_cls.__name__


def _parallel(tasks):
    # delegate each model/participant fit to cores, so this doesn't take >1hr
    with ProcessPoolExecutor() as pool:
        return pool.map(_thread, *zip(*tasks))


def run(trials, output_path="data/clusters.xlsx", calib_bounds=(1, 241), holdout_bounds=(241, 481)):
    # one half fits strategies, the other evaluates model
    calibration = trials.partition(*calib_bounds)
    holdout = trials.partition(*holdout_bounds)

    # summarize participant behaviour on calibration trials
    summary = Analysis(Profile.from_frame).run(calibration)
    # adjust for false positives + flag outcome-driven patterns
    significance = Significance(summary["p_value"], alpha=0.05)
    summary["q_value"] = significance.q_values
    summary["outcome_dependent"] = significance.flag
    # cluster only on significant outcome-dependence
    masked_win_stay = significance.mask(summary["win_stay_rate"]) # % playing same move after win
    masked_lose_stay = significance.mask(summary["lose_stay_rate"]) # % playing same move after loss
    masked_counter = significance.mask(summary["counter_rate"]) # % post-loss shifts that beat opponent's last move
    summary["cluster"] = Strategy.classify(masked_win_stay, masked_lose_stay, masked_counter)
    summary["max_stay_rate"] = summary[["win_stay_rate", "lose_stay_rate"]].max(axis=1) # how `sticky` this player is, or tendency to play same move overall

    # compare each player's calibrated behavior against holdout trials
    observed = {key: Profile.from_frame(group.frame) for key, group in holdout.groups()}

    # map Model × Cohort
    models = {cls.__name__: cls for cls in (Baseline, WSLC, WSLC_I, WSLC_C, WCLS)}

    comparisons = []
    # run each ACT-R model against participant data and evaluate it
    for name, model in models.items():
        predicted = Run(Profile.simulate).begin(model, holdout)
        result = Evaluate(STATS).compare(
            predicted, observed, cohorts=summary["cluster"], group_keys=trials.group_keys
        )
        result["model"] = name
        comparisons.append(result)

    # combine all model eval rows into a single table.
    comparison = pd.concat(comparisons, ignore_index=True)

    # keep only each cohort's candidate model(s) at stock parameters, for a no-tuning baseline
    model_cohort = {cls.__name__: cohort for cohort, classes in COHORT_MODELS.items() for cls in classes}
    matched = comparison[comparison["model"].map(model_cohort) == comparison["cohort"]]
    agreement = Agreement(STATS).test(matched)

    # fit only cohorts with strategem, Baseline has no free parameters to tune
    fit_models = {cohort: classes for cohort, classes in COHORT_MODELS.items() if cohort != "none"}
    calib_frames = {key: group.frame for key, group in calibration.groups()}
    tasks = []
    for key, group in holdout.groups():
        candidates = fit_models.get(summary["cluster"].loc[key])
        if candidates is None:
            continue
        # this participants's calibration behavior, to fit against
        target = Profile(**{field: summary.loc[key, field] for field in Profile.__dataclass_fields__})
        tasks.append((key, candidates, target, calib_frames[key], group.frame))

    fitted_predicted, fitted_model = {}, {}
    for key, predicted, model_name in _parallel(tasks):
        fitted_predicted[key] = predicted
        fitted_model[key] = model_name # record which candidate won

    fitted_comparison = Evaluate(STATS).compare(
        fitted_predicted, observed, cohorts=summary["cluster"], group_keys=trials.group_keys
    )
    # record which candidate won per player
    fitted_comparison["fitted_model"] = fitted_comparison.set_index(trials.group_keys).index.map(fitted_model).values
    fitted_agreement = Agreement(STATS).test(fitted_comparison)

    # same agreement test, but grouped by which WSLC sub-strategy won instead of cohort
    submodel_comparison = fitted_comparison[fitted_comparison["cohort"] == "WSLC"].assign(cohort=lambda d: d["fitted_model"])
    submodel_agreement = Agreement(STATS).test(submodel_comparison)

    # mean agreement per cohort/model
    breakdown = comparison.pivot_table(
        index="cohort",
        columns="model",
        values=["win_stay_rate_error", "lose_stay_rate_error", "counter_rate_error", "cramers_v_error"],
        aggfunc="mean",
    )
    breakdown["n"] = comparison.drop_duplicates(subset=trials.group_keys).groupby("cohort").size()

    # aggregate & export trial logs
    trials_sheet = (
        trials.frame
        .join(summary[["outcome_dependent"]], on=trials.group_keys)
        .join(Transform(Dyad.log).run(trials))
    )

    Export(output_path).write(
        {
            "participants": summary,
            "trials": trials_sheet,
            "comparison": comparison,
            "breakdown": breakdown,
            "agreement": agreement,
            "fitted_comparison": fitted_comparison,
            "fitted_agreement": fitted_agreement,
            "submodel_agreement": submodel_agreement,
        },
        participants={"merge_cells": False},
        trials={"merge_cells": False},
        comparison={"index": False},
        agreement={"index": False},
        fitted_comparison={"index": False},
        fitted_agreement={"index": False},
        submodel_agreement={"index": False},
    )


def main():
    trials = Trials.from_excel("data/output.xlsx", group_keys=["player", "session"], time_key="trial_num")
    run(trials)  # fold 1 
    run(trials, "data/clusters_fold2.xlsx", calib_bounds=(241, 481), holdout_bounds=(1, 241))  # fold 2 - flipped


if __name__ == "__main__":
    main()
