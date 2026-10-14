"""
Final round (2026-10-14) — shared helpers. All analyses are exploratory / post hoc.
Pre-registration: results/statistical_tests/prereg_final_2026-10-14.md.

Reuses the EARLY sample builder (leakbreadth_common.build_early -> wdearly_common.build_sample)
and the model / McNemar / bootstrap / BH code of robustness_common and ci_common, imported unchanged.
"""
import os
import pandas as pd

from utils import RESULTS_STATS
from ci_common import guard_absent

EXPLORATORY = "EXPLORATORY / POST HOC (prereg_final_2026-10-14.md); hypothesis-generating only."


def write_csv(df: pd.DataFrame, name: str, header_lines=()):
    assert name.startswith("final_"), f"output name must start with final_: {name}"
    path = guard_absent(os.path.join(RESULTS_STATS, name))
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in [EXPLORATORY, *header_lines]:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)", flush=True)


def flag_units(c: pd.DataFrame, keys) -> pd.DataFrame:
    """Pre-registered flag: BH p < 0.05 AND bootstrap acc CI excludes 0, same direction, for BOTH LR and GBM."""
    c = c.copy()
    lo, hi = c["boot_acc_diff_ci_low"], c["boot_acc_diff_ci_high"]
    c["ci_dir"] = (lo > 0).astype(int) - (hi < 0).astype(int)
    c["family_ok"] = (c["p_value_bh"] < 0.05) & (c["ci_dir"] != 0)
    flag = {}
    for k, g in c.groupby(list(keys), sort=False):
        g = g.set_index("family")
        flag[k] = bool(g.loc["LR", "family_ok"] and g.loc["GBM", "family_ok"]
                       and g.loc["LR", "ci_dir"] == g.loc["GBM", "ci_dir"])
    c["flagged"] = [flag[tuple(r)] for r in c[list(keys)].itertuples(index=False)]
    return c
