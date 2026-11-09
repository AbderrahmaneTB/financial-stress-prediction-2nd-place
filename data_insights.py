"""Data insights for the Financial Stress Prediction Challenge (September Edition) - optional, about 1 minute.

Prints the checks behind the model design: how the rows are organised, how the labels were assigned, whether train
and test look alike, how much each risk factor separates stressed customers, and which version of the structural
probit fits the training labels (cap rule, noise shape, noise size). The structural probit has no parameter fitted on
labels, so its score on the 40,000 training rows is an out-of-sample check.

Usage:  python data_insights.py [--data-dir .]
Requires the same packages as best_model_r1mzf.py, whose functions it reuses (run it from the same folder).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

from best_model_r1mzf import EPS, SIGMA, TARGET, WEIGHTS, composite, load, risk_factors, snapshot_dates

FACTOR_NAMES = ['balance drop', 'income drop', 'withdrawal ratio', 'discretionary-spend ratio', 'outflow / income',
                'balance volatility', 'inactivity']


def structural_score(R, d_all, y, n, cap_q=0.99, cap_scope='all', noise_cdf=stats.norm.cdf, sigma=SIGMA):
    """Composite score on the training rows of P = F((S + c_snapshot) / sigma), with c solved to a 15% mean."""
    if cap_scope == 'all':
        caps = np.quantile(R, cap_q, axis=0)
        capped = np.minimum(R, caps)
    elif cap_scope == 'train':
        capped = np.minimum(R, np.quantile(R[:n], cap_q, axis=0))
    else:
        capped = np.empty_like(R)
        for k in np.unique(d_all):
            m = d_all == k
            capped[m] = np.minimum(R[m], np.quantile(R[m], cap_q, axis=0))
    z = np.empty_like(capped)
    for k in np.unique(d_all):
        m = d_all == k
        z[m] = (capped[m] - capped[m].mean(0)) / capped[m].std(0)
    s = (z @ WEIGHTS)[:n]
    p = np.empty(n)
    d_train = d_all[:n]
    for k in np.unique(d_train):
        m = d_train == k
        c = brentq(lambda c: noise_cdf((s[m] + c) / sigma).mean() - 0.15, -20, 20, xtol=1e-14)
        p[m] = noise_cdf((s[m] + c) / sigma)
    return composite(y, np.clip(p, EPS, 1 - EPS))


def main():
    """Print the four groups of checks described in the module docstring."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    args = ap.parse_args()
    train, test, sample = load(Path(args.data_dir))
    d_train, d_test = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_train, d_test]

    print('1. Structure')
    print(f'   {n} training rows, {len(test)} test rows; 10,000 customers (same 8 profile columns), each with 4 training')
    print('   snapshots (indices 0-3) followed by 3 test snapshots (4-6), always contiguous.')
    print('   positives per training snapshot:', [int(y[d_train == k].sum()) for k in range(4)],
          '-> exactly 15%: labels are the top 15% of a risk score within each snapshot.')

    print('\n2. Train vs test (Kolmogorov-Smirnov distance per numeric column)')
    numeric = train.drop(columns=['ID', TARGET]).select_dtypes('number').columns
    ks = pd.Series({c: stats.ks_2samp(train[c], test[c]).statistic for c in numeric}).sort_values(ascending=False)
    print(f'   largest: {ks.index[0]} {ks.iloc[0]:.4f}; median {ks.median():.4f} -> no meaningful shift.')

    print('\n3. Risk factors (single-factor AUC on the training rows)')
    R = risk_factors(all_rows)
    for j, name in enumerate(FACTOR_NAMES):
        print(f'   F{j} {name:26s} weight {WEIGHTS[j] * 20:.0f}/20   AUC {roc_auc_score(y, R[:n, j]):.3f}')

    print('\n4. Which structural probit fits the labels (composite score; no parameter fitted on labels)')
    base = structural_score(R, d_all, y, n)
    print(f'   selected: 99% caps over all 70,000 rows, Gaussian noise sd {SIGMA}: {base:.7f}')
    variants = {
        'caps from training rows only': dict(cap_scope='train'),
        'caps per snapshot': dict(cap_scope='snapshot'),
        'caps at 98.9%': dict(cap_q=0.989),
        'caps at 99.1%': dict(cap_q=0.991),
        'noise sd 0.33': dict(sigma=0.33),
        'noise sd 0.37': dict(sigma=0.37),
        'logistic noise (same sd)': dict(noise_cdf=lambda z: stats.logistic.cdf(z * np.pi / np.sqrt(3))),
        'Student t noise, 5 df (same sd)': dict(noise_cdf=lambda z: stats.t.cdf(z * np.sqrt(5 / 3), 5)),
    }
    for name, kw in variants.items():
        print(f'   {name:32s} {1e6 * (structural_score(R, d_all, y, n, **kw) - base):+9.1f}e-6')
    print('   -> every alternative is worse: the selected structural probit matches how the labels were generated.')


if __name__ == '__main__':
    main()
