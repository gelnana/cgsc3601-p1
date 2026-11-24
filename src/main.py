import pandas as pd

from actr import actr_simulate
from experiment import Analysis, Evaluate, Export, Run, Trials, Transform
from models import Baseline, WSLC, WSLC_I, WSLC_C, WCLS
from rps import Dyad
from stats import Contingency, Significance, Strategy


def profiler(group: pd.DataFrame) -> dict:
    dyad = Dyad(group)
    outcome, stayed = dyad.outcome(), dyad.stay()

    test = Contingency(pd.crosstab(outcome.shift(1), stayed))
    return {
        "chi2": test.chi2, # magnitude of association between prior outcome & staying
        "p_value": test.p_value, # significance of chi2 relationship
        "cramers_v": test.cramers_v, # normalized effect size of chi2
        "continuum": test.continuum, # gradient between WCLS & WSLC behaviour
        "imitate_rate": dyad.imitate_rate, # % of post-loss shifts imitating opp last move
    }


def main():
    trials = Trials.from_excel("data/output.xlsx", group_keys=["player", "session"], time_key="trial_num")
    
    # maybe adjust split here?
    # first half fits strategies, second half evaluates model
    calibration = trials.partition(1, 241)
    holdout = trials.partition(241, 481)

    summary = Analysis(profiler).run(calibration)
    significance = Significance(summary["p_value"], alpha=0.05)
    summary["q_value"] = significance.q_values
    summary["outcome_dependent"] = significance.flag
    # zero out strategy metrics for non-significant players
    summary["continuum"] = significance.mask(summary["continuum"])
    summary["imitate_rate"] = significance.mask(summary["imitate_rate"]) # % player imitates opponents last move
    summary["cluster"] = Strategy.classify(summary["continuum"], summary["imitate_rate"])

    # run each ACT-R model against participant data and evaluate it
    models = {"Baseline": Baseline, "WSLC": WSLC, "WSLC_I": WSLC_I, "WSLC_C": WSLC_C, "WCLS": WCLS}
    observed = {key: group.frame["self_resp"] for key, group in holdout.groups()}

    comparisons = []
    for name, model in models.items():
        predicted = Run(actr_simulate).begin(model, holdout)
        result = Evaluate({"agreement": Evaluate.agreement}).compare(
            predicted, observed, cohorts=summary["cluster"], group_keys=trials.group_keys
        )
        result["model"] = name
        comparisons.append(result)

    comparison = pd.concat(comparisons, ignore_index=True)

    # aggregate & export trial logs
    trials_sheet = (
        trials.frame
        .join(summary[["outcome_dependent"]], on=trials.group_keys)
        .join(Transform(Dyad.log).run(trials))
    )

    Export("data/clusters.xlsx").write(
        {"participants": summary, "trials": trials_sheet, "comparison": comparison},
        participants={"merge_cells": False},
        trials={"merge_cells": False},
        comparison={"index": False},
    )


if __name__ == "__main__":
    main()
