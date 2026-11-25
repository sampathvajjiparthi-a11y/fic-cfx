"""Step 8: statistics for RQ2/H2 (method vs method) and RQ3/H3 (model vs model).

Reads every CSV in outputs/counterfactuals/<dataset>/ and runs:
  * RQ2: for each model, Wachter vs prototype on the SAME images
         McNemar on validity, Wilcoxon on L1, L2, sparsity, plausibility
  * RQ3: for each method, model vs model on the SAME images
         Wilcoxon on L1 and sparsity (and McNemar on validity)
All tests are Holm-Bonferroni corrected together; effect sizes get bootstrap CIs.

    python scripts/08_analyse_counterfactuals.py --dataset small
"""

import _bootstrap  # noqa: F401
from itertools import combinations

import pandas as pd

from fic.cli import base_parser, setup
from fic.config import output_path
from fic.data import load_dataset
from fic.stats import bootstrap_ci, holm_correction, mcnemar_test, wilcoxon_test

METRICS = ["l1", "l2", "sparsity", "plausibility_ae"]


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.merge(b, on="test_index", suffixes=("_a", "_b"))


def main():
    args = base_parser(__doc__).parse_args()
    cfg, _, _, log = setup(args, "08_analyse_counterfactuals")
    name = load_dataset(args.dataset, cfg).name if args.dataset in ("small", "big") else args.dataset
    folder = output_path(cfg, "counterfactuals", name, "x").parent
    files = [f for f in folder.glob("*.csv")]
    if not files:
        raise SystemExit(f"No counterfactual CSVs in {folder}. Run step 7 first.")
    data = {(d.model[0], d.method[0]): d for d in (pd.read_csv(f) for f in files)}
    log.info(f"found: {sorted(data)}")

    # summary table
    summary = pd.concat(data.values()).groupby(["model", "method"]).agg(
        n=("valid", "size"), validity=("valid", "mean"),
        l1_median=("l1", "median"), l2_median=("l2", "median"),
        sparsity_median=("sparsity", "median"), plausibility_median=("plausibility_ae", "median"),
        seconds_per_image=("seconds_per_image", "mean")).reset_index()
    summary.to_csv(output_path(cfg, "metrics", name, "cf_summary.csv"), index=False)
    log.info("\n" + summary.to_string(index=False))

    tests = []
    B = cfg.evaluation.bootstrap
    # RQ2: method comparison within each model
    models = sorted({m for m, _ in data})
    for m in models:
        methods = sorted(k for mm, k in data if mm == m)
        for ma, mb in combinations(methods, 2):
            pr = paired(data[(m, ma)], data[(m, mb)])
            tests.append({"rq": "RQ2", "group": m, "comparison": f"{ma} vs {mb}", "metric": "valid",
                          **mcnemar_test(pr.valid_a, pr.valid_b)})
            both = pr[(pr.valid_a == 1) & (pr.valid_b == 1)]   # compare quality on valid pairs only
            for met in METRICS:
                if len(both) < 10:
                    continue
                r = wilcoxon_test(both[f"{met}_a"], both[f"{met}_b"])
                lo, hi = bootstrap_ci(both[f"{met}_a"].values, both[f"{met}_b"].values, n=B)
                tests.append({"rq": "RQ2", "group": m, "comparison": f"{ma} vs {mb}", "metric": met,
                              "n_pairs": len(both), "ci_low": lo, "ci_high": hi, **r})
    # RQ3: model comparison within each method
    methods = sorted({k for _, k in data})
    for k in methods:
        ms = sorted(m for m, kk in data if kk == k)
        for ma, mb in combinations(ms, 2):
            pr = paired(data[(ma, k)], data[(mb, k)])
            both = pr[(pr.valid_a == 1) & (pr.valid_b == 1)]
            for met in ["l1", "sparsity"]:
                if len(both) < 10:
                    continue
                r = wilcoxon_test(both[f"{met}_a"], both[f"{met}_b"])
                lo, hi = bootstrap_ci(both[f"{met}_a"].values, both[f"{met}_b"].values, n=B)
                tests.append({"rq": "RQ3", "group": k, "comparison": f"{ma} vs {mb}", "metric": met,
                              "n_pairs": len(both), "ci_low": lo, "ci_high": hi, **r})
    if not tests:
        log.warning("not enough paired results for tests yet")
        return
    res = holm_correction(pd.DataFrame(tests), cfg.evaluation.alpha)
    res["practically_meaningful"] = (res.effect_name == "rank_biserial_r") & (res.effect.abs() >= cfg.evaluation.practical_r)
    res.to_csv(output_path(cfg, "metrics", name, "cf_statistics.csv"), index=False)
    log.info("\n" + res[["rq", "group", "comparison", "metric", "p_holm", "effect", "significant"]]
             .to_string(index=False))


if __name__ == "__main__":
    main()
