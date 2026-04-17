"""Privacy–utility tradeoff plotting utilities.

Extracted from `notebooks/Privacy-Utility tradeoff.ipynb` so the aggregation
logic (which governs how results are filtered before plotting) can be unit
tested.
"""
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pandas import DataFrame


# (regex, method name, parameter label)
_MODEL_PATTERNS = [
    (r'^BayesianNetBins(\d+\.?\d*)$', 'BayesianNet', 'bins'),
    (r'^PrivBayesEps(\d+\.?\d*)$', 'PrivBayes', 'eps'),
    (r'^PateGanEps(\d+\.?\d*)$', 'PATEGAN', 'eps'),
    (r'^SanitiserNHSk(\d+\.?\d*)$', 'SanitiserNHS', 'k'),
    (r'^SanitiserMondrianK(\d+\.?\d*)$', 'SanitiserMondrian', 'k'),
    (r'^SanitiserNHSMondrianK(\d+\.?\d*)$', 'SanitiserNHSMondrian', 'k'),
]

_PARAM_SYMBOLS = {
    'BayesianNet': 'bins',
    'PrivBayes': r'$\varepsilon$',
    'PATEGAN': r'$\varepsilon$',
    'SanitiserNHS': r'$k$',
    'SanitiserMondrian': r'$k$',
    'SanitiserNHSMondrian': r'$k$',
}


def parse_target_model(name):
    """Return (method, param, param_label) parsed from a TargetModel string."""
    for pattern, method, param_label in _MODEL_PATTERNS:
        m = re.fullmatch(pattern, name)
        if m:
            return method, float(m.group(1)), param_label
    return name, np.nan, None


def normalize_target_model(val):
    """Coerce tuple/list/str TargetModel entries to a trimmed string."""
    if isinstance(val, (tuple, list)):
        val = val[0]
    if not isinstance(val, str):
        return None
    return val.strip()


def add_method_param(df, model_col='TargetModel'):
    """Add Method / Param / ParamLabel columns parsed from ``model_col``."""
    df = df.copy()
    df[model_col] = df[model_col].map(normalize_target_model)
    parsed = df[model_col].map(parse_target_model)
    df['Method'] = parsed.map(lambda x: x[0])
    df['Param'] = parsed.map(lambda x: x[1])
    df['ParamLabel'] = parsed.map(lambda x: x[2])
    return df


def aggregate_tradeoff(
    res_linkage,
    res_inference,
    res_utility_agg,
    featureset_filter='Correlations',
    sensitive_attr_filter=None,
    pred_model_filter=None,
    label_var_filter=None,
):
    """Filter and aggregate the three result DataFrames into a single merged table.

    Returns the merged ``DataFrame`` with columns:
    ``Method, Param, PG_link, PG_link_std, PG_inf, PG_inf_std, Acc, Acc_std``.

    Split out from :func:`plot_privacy_utility_tradeoff` so filter/aggregation
    behaviour can be unit tested without invoking matplotlib.
    """
    rl = add_method_param(res_linkage)
    if featureset_filter is not None:
        rl = rl[rl['FeatureSet'] == featureset_filter]

    ri = add_method_param(res_inference)
    if sensitive_attr_filter is not None:
        ri = ri[ri['SensitiveAttribute'] == sensitive_attr_filter]

    ru = add_method_param(res_utility_agg)

    pg_link = (
        rl.groupby(['Method', 'Param'])['PrivacyGain']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'PG_link', 'std': 'PG_link_std'})
    )
    pg_inf = (
        ri.groupby(['Method', 'Param'])['PrivacyGain']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'PG_inf', 'std': 'PG_inf_std'})
    )

    ru_sub = ru.copy()
    if pred_model_filter is not None:
        ru_sub = ru_sub[ru_sub['PredictionModel'] == pred_model_filter]
    if label_var_filter is not None:
        ru_sub = ru_sub[ru_sub['LabelVar'] == label_var_filter]
    if 'Split' in ru_sub.columns:
        ru_sub = ru_sub[ru_sub['Split'] == 'OUT']

    acc = (
        ru_sub.groupby(['Method', 'Param'])['Accuracy']
        .agg(['mean', 'std']).reset_index()
        .rename(columns={'mean': 'Acc', 'std': 'Acc_std'})
    )

    merged = (
        pg_link.merge(pg_inf, on=['Method', 'Param'], how='outer')
        .merge(acc, on=['Method', 'Param'], how='inner')
    )
    return merged.sort_values(['Method', 'Param']).reset_index(drop=True)


def plot_privacy_utility_tradeoff(
    res_linkage: DataFrame,
    res_inference: DataFrame,
    res_utility_agg: DataFrame,
    model_col: str = 'TargetModel',
    featureset_filter: str = 'Correlations',
    sensitive_attr_filter: str = None,
    pred_model_filter: str = None,
    label_var_filter: str = None,
    figsize: tuple = (14, 6),
):
    """Plot privacy (linkage / inference PG) against utility (accuracy).

    Parameters
    ----------
    res_linkage, res_inference, res_utility_agg
        Outputs of ``load_results_linkage``, ``load_results_inference``,
        ``load_results_utility`` respectively.
    featureset_filter
        MIA FeatureSet to keep (e.g. ``'Correlations'``). ``None`` keeps all.
    sensitive_attr_filter
        AIA SensitiveAttribute to keep (CamelCase, matching
        ``load_results_inference`` normalisation — e.g. ``'LengthOfStay'``).
        ``None`` averages across all attributes.
    pred_model_filter, label_var_filter
        Optional utility filters.
    """
    merged = aggregate_tradeoff(
        res_linkage,
        res_inference,
        res_utility_agg,
        featureset_filter=featureset_filter,
        sensitive_attr_filter=sensitive_attr_filter,
        pred_model_filter=pred_model_filter,
        label_var_filter=label_var_filter,
    )

    methods = sorted(merged['Method'].unique())
    cmap = cm.get_cmap('tab10', max(len(methods), 1))
    color_map = {m: cmap(i) for i, m in enumerate(methods)}
    marker_list = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', '+']
    marker_map = {m: marker_list[i % len(marker_list)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    inf_title = 'Privacy Gain (Inference)'
    if sensitive_attr_filter is not None:
        inf_title += f' — {sensitive_attr_filter}'
    titles = ['Privacy Gain (Linkage)', inf_title]
    pg_cols = ['PG_link', 'PG_inf']

    for ax, title, pg_col in zip(axes, titles, pg_cols):
        for method, grp in merged.groupby('Method'):
            grp = grp.dropna(subset=[pg_col, 'Acc']).sort_values('Param')
            if grp.empty:
                continue

            color = color_map[method]
            marker = marker_map[method]

            ax.plot(
                grp['Acc'], grp[pg_col],
                color=color, linewidth=1.5, linestyle='-', alpha=0.6, zorder=1,
                marker=marker, markersize=7, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.8,
            )

            for _, row in grp.iterrows():
                if np.isnan(row['Param']):
                    continue
                prefix = _PARAM_SYMBOLS.get(method, '?')
                param_val = (
                    int(row['Param']) if row['Param'] == int(row['Param']) else row['Param']
                )
                ax.annotate(
                    f"{prefix}={param_val}",
                    xy=(row['Acc'], row[pg_col]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8, alpha=0.6),
                    zorder=3,
                )

        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_ylabel('Privacy Gain', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=10)

    legend_handles = [
        Line2D(
            [0], [0],
            color=color_map[m], marker=marker_map[m],
            linewidth=1.5, markersize=8, label=m,
        )
        for m in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(min(len(methods), 5), 1),
        fontsize=10,
        title='Method',
        title_fontsize=10,
        framealpha=0.9,
    )
    plt.tight_layout()
    return fig
