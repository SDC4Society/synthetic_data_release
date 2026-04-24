"""Post-run consistency check: verify that model keys agree across the three report types."""

import glob
import json
import os
import sys


def check_report_key_consistency(outdir):
    """Warn if model keys in ResultsUtilAgg, ResultsMIA, and ResultsMLEAI diverge.

    Divergence means a model ran in the privacy games but not in utility (or vice
    versa), making Accuracy × PrivacyGain scatter plots impossible to join.
    """
    util_files  = glob.glob(os.path.join(outdir, "ResultsUtilAgg_*.json"))
    mia_files   = glob.glob(os.path.join(outdir, "ResultsMIA_*.json"))
    mleai_files = glob.glob(os.path.join(outdir, "ResultsMLEAI_*.json"))

    if not (util_files and mia_files and mleai_files):
        # Not all three reports exist yet — skip silently
        return

    util_path  = util_files[0]
    mia_path   = mia_files[0]
    mleai_path = mleai_files[0]

    util_data  = json.load(open(util_path))
    mia_data   = json.load(open(mia_path))
    mleai_data = json.load(open(mleai_path))

    # Extract model names from each report
    util_models = set()
    for ut_name, by_model in util_data.items():
        util_models |= set(by_model.keys()) - {"Raw"}

    mia_models = set()
    for tid_val in mia_data.values():
        mia_models |= set(tid_val.keys())

    mleai_models = set()
    for tid_val in mleai_data.values():
        for sa_val in tid_val.values():
            mleai_models |= set(sa_val.keys())

    priv_models = mia_models | mleai_models

    missing_in_util = priv_models - util_models
    missing_in_priv = util_models - priv_models

    if missing_in_util or missing_in_priv:
        sys.stderr.write(
            "[WARN] Report key mismatch — Utility × Privacy join will be incomplete.\n"
            f"  Models in privacy reports but missing from UtilAgg : {missing_in_util or 'none'}\n"
            f"  Models in UtilAgg but missing from privacy reports : {missing_in_priv or 'none'}\n"
        )
    else:
        print(f"[OK] Report key consistency check passed ({len(util_models)} models).")
