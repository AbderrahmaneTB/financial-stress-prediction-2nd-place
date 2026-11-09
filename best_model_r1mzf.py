"""Full pipeline of my Financial Stress Prediction Challenge (September Edition) solution - standalone, deterministic.

Rebuilds the two selected submissions byte for byte:
    python best_model_r1mzf.py --seeds 50 --month-scale rank                    -> 75o4hiai, v14_all_MZF_s50_avg5_mrank.csv
    python best_model_r1mzf.py --seeds 50 --month-scale rank --round-scale 2.0  -> QQZviRgL, v14_all_MZF_s50_avg5_mrank_t200.csv
When the original file is in <data-dir>/submissions/, the script prints IDENTICAL or DIFFERENT after writing its output.

Steps
-----
1. Snapshot index of every row: the rows of one customer (same 8 profile columns) are contiguous, 4 in Train
   (snapshots 0-3) and 3 in Test (snapshots 4-6). No labels are used.
2. Seven liquidity-risk factors per row (M1 = most recent month; recent = M1-M3, older = M4-M6): balance drop,
   income drop, withdrawal ratio, discretionary spending ratio, outflow / income, balance volatility and
   inactivity. Each factor is capped at its 99% quantile over all 70,000 rows.
3. Structural probit: each capped factor is z-scored within its snapshot, S = Z @ [5, 3, 3, 2, 4, 2, 1] / 20 and
   P0 = Phi(c_snapshot + S / 0.35); nothing is fitted in this step.
4. LightGBM residual: init score = logit(P0), 5 parameter configurations x N seeds (--seeds, 50 for the
   submission). The tree count of each configuration is the median early-stopping point over the
   leave-one-month-out folds (trained on two months, stopped on the most recent remaining one).
   With --month-scale rank every feature is first replaced by its percentile rank within its snapshot.
   With --round-scale F every tree count is multiplied by F.
5. Calibration: each test snapshot's intercept is solved so its mean prediction is exactly 15%, the stress rate of
   every training snapshot. Probabilities are written raw (%.17g), with no rounding or thresholds.

Usage:  python best_model_r1mzf.py [--seeds 10] [--month-scale none|rank|z] [--round-scale 1] [--data-dir .] [--out submissions/best_model_r1mzf.csv]
Runtime: about 17 minutes with 50 seeds on 12 CPU threads (about 4 minutes with 10 seeds).
Requires numpy, pandas, scipy, scikit-learn and lightgbm.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, logit, ndtr, ndtri
from sklearn.metrics import log_loss, roc_auc_score

TARGET = 'liquidity_stress_next_30d'
PROFILE = ['arpu', 'age', 'gender', 'region', 'smartphone', 'segment', 'earning_pattern', 'x_90_d_activity_rate']
CATS = ['gender', 'region', 'smartphone', 'segment', 'earning_pattern']
WEIGHTS = np.array([5., 3., 3., 2., 4., 2., 1.]) / 20
RATE = 0.15
SIGMA = 0.35
EPS = 1e-15
PARAMS = dict(objective='binary', learning_rate=0.02, num_leaves=7, max_depth=3, min_data_in_leaf=300,
              lambda_l2=50.0, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              seed=909, deterministic=True, force_row_wise=True, num_threads=12, verbose=-1)
BASE = copy.deepcopy(PARAMS)
CONFIGS = [{}, dict(num_leaves=15, max_depth=4), dict(min_data_in_leaf=100), dict(learning_rate=0.01),
           dict(learning_rate=0.05)]
SEEDS = (909, 11, 22, 33, 44, 1001, 1002, 1003, 1004, 1005)   # --seeds N: (909, 11, 22, 33, 44) + 1001 .. 1000 + N - 5
MAX_ROUNDS = 600


# ---------------------------------------------------------------- data
def find_file(root, stem):
    """Path of <stem>.csv in root; a single downloaded copy such as 'Train (1).csv' is accepted too."""
    exact = root / f'{stem}.csv'
    if exact.is_file():
        return exact
    matches = sorted(root.glob(f'{stem} (*).csv'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {stem}*.csv in {root}, found {matches}')
    return matches[0]


def load(root):
    """Read Train, Test and SampleSubmission, and check their shapes and the order of the test IDs."""
    train = pd.read_csv(find_file(root, 'Train'))
    test = pd.read_csv(find_file(root, 'Test'))
    sample = pd.read_csv(find_file(root, 'SampleSubmission'))
    assert train.shape == (40000, 184) and test.shape == (30000, 183)
    assert (sample.ID.to_numpy() == test.ID.to_numpy()).all(), 'Sample order differs from test'
    return train, test, sample


def snapshot_dates(train, test):
    """Snapshot index of every row: customers are found from the 8 profile columns, train rows get 0-3, test rows 4-6."""
    both = pd.concat([train[PROFILE], test[PROFILE]], ignore_index=True)
    customer = both.groupby(PROFILE, sort=False).ngroup().to_numpy()
    assert (customer[:len(train)] == np.repeat(np.arange(10000), 4)).all()
    assert (customer[len(train):] == np.repeat(np.arange(10000), 3)).all()
    d_train, d_test = np.tile(np.arange(4), 10000), np.tile(np.arange(3), 10000) + 4
    assert (pd.Series(train[TARGET].to_numpy()).groupby(d_train).sum() == 1500).all()
    return d_train, d_test


# ---------------------------------------------------------------- structural score
def risk_factors(df):
    """The seven liquidity-risk factors of every row (recent = months 1-3, older = months 4-6)."""
    def months(stem):
        """The six monthly columns m1..m6 of one variable as a float array."""
        return df[[f'm{i}_{stem}' for i in range(1, 7)]].to_numpy(np.float64)
    bal = months('daily_avg_bal')
    withdraw = months('withdraw_total_value')
    income = months('received_total_value') + months('deposit_total_value') + months('transfer_from_bank_total_value')
    spend = months('paybill_total_value') + months('merchantpay_total_value')
    outflow = withdraw + spend + months('mm_send_total_value')
    recent, older = slice(0, 3), slice(3, 6)
    R = np.column_stack([
        np.maximum(1 - (bal[:, recent].mean(1) + 1) / (bal[:, older].mean(1) + 1), 0),
        np.maximum(1 - (income[:, recent].mean(1) + 1) / (income[:, older].mean(1) + 1), 0),
        (withdraw[:, recent].mean(1) + 1) / (withdraw[:, older].mean(1) + 1),
        (spend[:, recent].mean(1) + 1) / (spend[:, older].mean(1) + 1),
        (outflow[:, recent].sum(1) + 1) / (income[:, recent].sum(1) + 1),
        bal.std(axis=1),
        1 - df['x_90_d_activity_rate'].to_numpy(np.float64),
    ])
    assert np.isfinite(R).all()
    return R


def risk_score(R):
    """Structural score with the capped factors z-scored over all rows (only used for the reference baseline)."""
    caps = np.array([np.quantile(R[:, j], 0.99) for j in range(R.shape[1])])
    clipped = np.minimum(R, caps)
    Z = (clipped - clipped.mean(0)) / clipped.std(0)
    return Z @ WEIGHTS


def solve_intercept(lin, rate=RATE):
    """Intercept c such that the mean of Phi(c + lin) equals the target stress rate (15%)."""
    return brentq(lambda c: ndtr(c + lin).mean() - rate, -10, 10, xtol=1e-14)


def _probit_nll(w, X, y, offset):
    """Negative log-likelihood of a probit model and its gradient, for scipy.optimize.minimize."""
    eta = X @ w + offset
    lp, lq = log_ndtr(eta), log_ndtr(-eta)
    log_phi = -0.5 * eta ** 2 - 0.5 * np.log(2 * np.pi)
    grad_eta = np.where(y == 1, -np.exp(log_phi - lp), np.exp(log_phi - lq))
    return -(y * lp + (1 - y) * lq).sum(), X.T @ grad_eta


def fit_probit(s, y, dates):
    """Probit with one intercept per date and a common fitted slope; returns the slope."""
    levels = np.unique(dates)
    X = np.c_[(dates[:, None] == levels[None, :]).astype(float), s]
    w0 = np.r_[np.full(len(levels), -1.56), 2.87]
    res = minimize(_probit_nll, w0, args=(X, y, np.zeros(len(s))), jac=True, method='L-BFGS-B',
                   options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 10000})
    assert res.success, res.message
    return res.x[-1]


MZ_SLOPE = 2.857142857142857  # 1 / 0.35


def month_z_score(R, d_all):
    """Risk score with every capped factor (global 99% caps) z-scored within its snapshot month."""
    C = np.minimum(R, np.quantile(R, 0.99, axis=0))
    Z = np.empty_like(C)
    for k in np.unique(d_all):
        m = d_all == k
        Z[m] = (C[m] - C[m].mean(0)) / C[m].std(0)
    return Z @ WEIGHTS


def linear_index(s_mz, d_all, y, fit_idx):
    """MZF: month-z score times the fixed slope 1/0.35 (no labels used)."""
    return MZ_SLOPE * s_mz


def init_of(lin, d_all, idx):
    """Calibrated probit index of the rows idx (one intercept per snapshot) and its logit, the LightGBM init score."""
    eta = np.empty(len(idx))
    for k in np.unique(d_all[idx]):
        m = d_all[idx] == k
        eta[m] = solve_intercept(lin[idx][m]) + lin[idx][m]
    return logit(np.clip(ndtr(eta), 1e-15, 1 - 1e-15)), eta


# ---------------------------------------------------------------- LightGBM residual
def features(all_rows, R):
    """LightGBM inputs: sign-log numeric columns, one-hot categorical columns and the seven capped factors."""
    num = all_rows.drop(columns=['ID'] + CATS)
    X = np.sign(num.to_numpy(float)) * np.log1p(np.abs(num.to_numpy(float)))
    cats = pd.get_dummies(all_rows[CATS], dtype=float).to_numpy()
    caps = np.quantile(R, 0.99, axis=0)
    return np.c_[X, cats, np.minimum(R, caps)].astype(np.float32)


def month_scaled(X, d_all, how):
    """Features re-expressed relative to their own snapshot month (label-free): z-scores or percentile ranks."""
    out = np.empty(X.shape, np.float32)
    for k in np.unique(d_all):
        m = d_all == k
        if how == 'z':
            sd = np.nanstd(X[m], 0)
            out[m] = (X[m] - np.nanmean(X[m], 0)) / np.where(sd > 0, sd, 1)
        else:
            out[m] = pd.DataFrame(X[m]).rank(pct=True).to_numpy()
    return out


def use_config(cfg):
    """Reset the LightGBM parameters to the base set updated with one variant."""
    PARAMS.clear()
    PARAMS.update({**BASE, **cfg})


def early_stopping_rounds(X, y, init, Xv, yv, iv):
    """Number of trees chosen by early stopping on a validation set (patience 60 rounds)."""
    ds = lgb.Dataset(X, y, init_score=init, free_raw_data=False)
    dv = lgb.Dataset(Xv, yv, init_score=iv, reference=ds)
    booster = lgb.train(PARAMS, ds, num_boost_round=MAX_ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
    return max(booster.best_iteration, 1)


def bag(X, y, init, Xp, rounds):
    """Average of the raw LightGBM corrections over all seeds, each model trained with a fixed number of trees."""
    raws = []
    for sd in SEEDS:
        PARAMS['seed'] = sd
        raws.append(lgb.train(PARAMS, lgb.Dataset(X, y, init_score=init), num_boost_round=rounds)
                    .predict(Xp, raw_score=True))
    PARAMS['seed'] = BASE['seed']
    return np.mean(raws, 0)


def probit_eta(p):
    """Probit index (inverse normal distribution function) of probabilities p."""
    return ndtri(np.clip(p, 1e-15, 1 - 1e-15))


def per_date(eta, d):
    """Calibrate a probit index so that the mean probability of each snapshot is 15%."""
    out = np.empty(len(eta))
    for k in np.unique(d):
        m = d == k
        out[m] = ndtr(solve_intercept(eta[m]) + eta[m])
    return np.clip(out, EPS, 1 - EPS)


def composite(y, p):
    """Competition metric: 0.6 * (1 - log loss / 0.595) + 0.4 * AUC."""
    return 0.6 * (1 - log_loss(y, p) / 0.595) + 0.4 * roc_auc_score(y, p)


def main():
    """Leave-one-month-out validation, training on all labelled months, and the submission file."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--out', default='submissions/best_model_r1mzf.csv')
    ap.add_argument('--seeds', type=int, default=10, help='bagging seeds per configuration (at least 5)')
    ap.add_argument('--month-scale', default='none', choices=['none', 'rank', 'z'],
                    help='residual features as within-month percentile ranks (rank) or z-scores (z)')
    ap.add_argument('--round-scale', type=float, default=1.0, help='multiply every tree count by this factor')
    args = ap.parse_args()
    scaled = lambda r: int(r) if args.round_scale == 1.0 else max(1, int(round(args.round_scale * r)))
    global SEEDS
    SEEDS = SEEDS[:5] + tuple(range(1001, 1001 + args.seeds - 5))
    train, test, sample = load(Path(args.data_dir))
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    s = risk_score(R)            # global-z score, only for the structural-B reference below
    s_mz = month_z_score(R, d_all)
    X = features(all_rows, R)
    if args.month_scale != 'none':
        X = month_scaled(X, d_all, args.month_scale)

    # leave-one-month-out: tree count per configuration and validation score
    p_b, p_m, rounds = np.empty(n), np.empty(n), [[] for _ in CONFIGS]
    for k in range(4):
        a, b = np.flatnonzero(d_tr != k), np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit, val = a[d_tr[a] != ik], a[d_tr[a] == ik]
        lin_in = linear_index(s_mz, d_all, y, fit)
        lin = linear_index(s_mz, d_all, y, a)
        init_b, _ = init_of(lin, d_all, b)
        eta_b = np.empty(len(b))  # structural B (S / 0.35) on the held-out month, for reference
        eta_b[:] = solve_intercept(1 / SIGMA * s[b]) + 1 / SIGMA * s[b]
        p_b[b] = per_date(eta_b, d_tr[b])
        raws = []
        for c, cfg in enumerate(CONFIGS):
            use_config(cfg)
            best = early_stopping_rounds(X[fit], y[fit], init_of(lin_in, d_all, fit)[0],
                                         X[val], y[val], init_of(lin_in, d_all, val)[0])
            rounds[c].append(best)
            raws.append(bag(X[a], y[a], init_of(lin, d_all, a)[0], X[b], scaled(best)))
        p_m[b] = per_date(probit_eta(1 / (1 + np.exp(-(init_b + np.mean(raws, 0))))), d_tr[b])
        print(f'month {k} done', flush=True)
    gain = composite(y, p_m) - composite(y, p_b)
    per = [round(1e6 * (composite(y[d_tr == k], p_m[d_tr == k]) - composite(y[d_tr == k], p_b[d_tr == k])), 1)
           for k in range(4)]
    print(f'leave-one-month-out gain vs structural B: {gain * 1e6:+.1f}e-6, per month {per}; trees {rounds}')

    # final model on all training rows
    tr, te = np.arange(n), np.arange(n, n + len(test))
    lin = linear_index(s_mz, d_all, y, tr)
    init_te, _ = init_of(lin, d_all, te)
    raws = []
    for c, cfg in enumerate(CONFIGS):
        use_config(cfg)
        raws.append(bag(X[:n], y, init_of(lin, d_all, tr)[0], X[n:], scaled(np.median(rounds[c]))))
    use_config({})
    p = per_date(probit_eta(1 / (1 + np.exp(-(init_te + np.mean(raws, 0))))), d_te)

    p = np.asarray(p, np.float64)
    assert len(p) == len(sample) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    out = sample.copy()
    out['Target'] = p
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format='%.17g')
    check = pd.read_csv(path, float_precision='round_trip')
    assert check.ID.equals(sample.ID) and np.array_equal(check.Target.to_numpy(), p)
    print(f'{path} written, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:12]}, '
          f'mean per test month {[round(float(p[d_te == k].mean()), 12) for k in (4, 5, 6)]}')
    suffix = ('' if args.month_scale == 'none' else f'_m{args.month_scale}') + \
        ('' if args.round_scale == 1.0 else f'_t{int(round(100 * args.round_scale))}')
    ref = Path(args.data_dir) / 'submissions' / f'v14_all_MZF_s{args.seeds}_avg5{suffix}.csv'
    if ref.exists():
        a, b = hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(ref.read_bytes()).hexdigest()
        print(f'standalone vs {ref.name}: {"IDENTICAL" if a == b else "DIFFERENT"} {a[:12]} {b[:12]}')


if __name__ == '__main__':
    main()
