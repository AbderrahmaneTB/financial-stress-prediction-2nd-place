"""Structural models MZ / MZF / AVG - standalone, deterministic, no machine learning library needed.

Financial Stress Prediction Challenge (September Edition). This is the structural part of my solution on its own
(the starting point of the LightGBM model in best_model_r1mzf.py), kept as a simple fallback. It reproduces my
earlier submissions byte for byte:
    --sigma 0.35   Zindi Pyr6nvQx (v16_MZF_structural.csv), public 0.744516108
    --sigma fit    Zindi 1i9HJFxZ (v16_MZ_structural.csv),  public 0.744505364
    --variant AVG  Zindi p1LMiEYW (v16_AVG_structural.csv), public 0.744481546
AVG is the probit-space average of three scalings of the same score
(global z with a fitted slope, global z standardised within each month, month z), each calibrated
to 15% per month first, then recalibrated: a model average over how the factors were scaled.

Why this model
--------------
The labels behave exactly like this generator (checked on the training months):
    latent = S + N(0, 0.35^2), independently for every row; label = top 15% of the latent per month.
S is a weighted sum of seven liquidity factors, each capped at its 99% quantile over all rows and
z-scored *within its snapshot month*. Freely fitted weights come out at 5:3:3:2:4:2:1 (within 1.3
standard errors); the probit link fits best (logit, t and skewed links are worse); there is no
curvature, no profile effect, no customer effect across months; caps at 98.9% or 99.1% fit clearly
worse; z-scoring within the month beats z-scoring over all rows (likelihood ratio 5.3).

Model
-----
1. Seven factors per snapshot (M1 = most recent month; recent = M1-M3, older = M4-M6):
   F0 balance drop          max(1 - (bal_recent + 1) / (bal_older + 1), 0)
   F1 income drop           same, income = received + deposit + transfer_from_bank
   F2 withdrawal ratio      (withdraw_recent + 1) / (withdraw_older + 1)
   F3 discretionary ratio   (paybill + merchantpay recent + 1) / (older + 1)
   F4 outflow / income      (recent outflow + 1) / (recent income + 1),
                            outflow = withdraw + paybill + merchantpay + mm_send
   F5 balance volatility    std of the six monthly average balances
   F6 inactivity            1 - x_90_d_activity_rate
2. Snapshot month of every row: rows of one customer (same 8 profile columns) are contiguous and in
   date order (train months 0-3, test months 4-6; every train month has exactly 1500 positives).
   No labels are used.
3. Caps at the 99% quantile over all 70,000 rows; each capped factor z-scored within its month;
   S = Z @ [5, 3, 3, 2, 4, 2, 1] / 20.
4. P = Phi(c_month + S / sigma): sigma fixed at 0.35, or (--sigma fit) slope fitted by probit MLE
   with one intercept per training month.
5. Every test month's intercept is solved (brentq) so its mean prediction is exactly 15%.
   Probabilities are written raw (%.17g), no rounding or thresholds.

Usage:  python final_model_mz.py [--variant MZ|AVG] [--sigma 0.35|fit] [--data-dir .] [--out submissions/final_model_mz.csv]
Runtime a few seconds. Requires numpy, pandas, scipy, scikit-learn.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, ndtr, ndtri
from sklearn.metrics import log_loss, roc_auc_score

TARGET = 'liquidity_stress_next_30d'
PROFILE = ['arpu', 'age', 'gender', 'region', 'smartphone', 'segment', 'earning_pattern', 'x_90_d_activity_rate']
WEIGHTS = np.array([5., 3., 3., 2., 4., 2., 1.]) / 20
RATE = 0.15
EPS = 1e-15


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
    return np.column_stack([
        np.maximum(1 - (bal[:, recent].mean(1) + 1) / (bal[:, older].mean(1) + 1), 0),
        np.maximum(1 - (income[:, recent].mean(1) + 1) / (income[:, older].mean(1) + 1), 0),
        (withdraw[:, recent].mean(1) + 1) / (withdraw[:, older].mean(1) + 1),
        (spend[:, recent].mean(1) + 1) / (spend[:, older].mean(1) + 1),
        (outflow[:, recent].sum(1) + 1) / (income[:, recent].sum(1) + 1),
        bal.std(axis=1),
        1 - df['x_90_d_activity_rate'].to_numpy(np.float64),
    ])


def global_z_score(R):
    """Structural score with the capped factors z-scored over all rows."""
    caps = np.quantile(R, 0.99, axis=0)
    C = np.minimum(R, caps)
    return ((C - C.mean(0)) / C.std(0)) @ WEIGHTS


def per_month_std(s, d_all):
    """Score divided by its standard deviation within each snapshot."""
    sd = np.array([s[d_all == k].std() for k in range(d_all.max() + 1)])
    return s / sd[d_all]


def month_z_score(R, d_all):
    """Structural score with every capped factor z-scored within its own snapshot."""
    caps = np.quantile(R, 0.99, axis=0)
    C = np.minimum(R, caps)
    Z = np.empty_like(C)
    for k in np.unique(d_all):
        m = d_all == k
        Z[m] = (C[m] - C[m].mean(0)) / C[m].std(0)
    return Z @ WEIGHTS


def _probit_nll(w, X, y):
    """Negative log-likelihood of a probit model and its gradient, for scipy.optimize.minimize."""
    eta = X @ w
    lp, lq = log_ndtr(eta), log_ndtr(-eta)
    log_phi = -0.5 * eta ** 2 - 0.5 * np.log(2 * np.pi)
    grad_eta = np.where(y == 1, -np.exp(log_phi - lp), np.exp(log_phi - lq))
    return -(y * lp + (1 - y) * lq).sum(), X.T @ grad_eta


def fit_slope(s, y, dates):
    """Probit MLE with one intercept per date and a common slope on s."""
    levels = np.unique(dates)
    X = np.c_[(dates[:, None] == levels[None, :]).astype(float), s]
    res = minimize(_probit_nll, np.r_[np.full(len(levels), -1.56), 2.87], args=(X, y), jac=True,
                   method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 10000})
    assert res.success, res.message
    return res.x[-1]


def solve_intercept(lin):
    """Intercept c such that the mean of Phi(c + lin) equals the target stress rate (15%)."""
    return brentq(lambda c: ndtr(c + lin).mean() - RATE, -10, 10, xtol=1e-14)


def calibrated(lin, d):
    """Probabilities Phi(c + lin) with one intercept per snapshot, so that each snapshot averages 15%."""
    p = np.empty(len(lin))
    for k in np.unique(d):
        m = d == k
        p[m] = ndtr(solve_intercept(lin[m]) + lin[m])
    return np.clip(p, EPS, 1 - EPS)


def composite(y, p):
    """Competition metric: 0.6 * (1 - log loss / 0.595) + 0.4 * AUC."""
    ll, auc = log_loss(y, p), roc_auc_score(y, p)
    return 0.6 * (1 - ll / 0.595) + 0.4 * auc


def main():
    """Validation score on the training months, test predictions and the submission file."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='MZ', choices=['MZ', 'AVG'])
    ap.add_argument('--sigma', default='0.35', help="MZ only: noise sd (0.35, the generator's value) or 'fit'")
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--out', default='submissions/final_model_mz.csv')
    args = ap.parse_args()
    train, test, sample = load(Path(args.data_dir))
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    if args.variant == 'AVG':
        s_b = global_z_score(R)
        parts = [(s_b, None), (per_month_std(s_b, d_all), None), (month_z_score(R, d_all), None)]   # slopes fitted
    else:
        parts = [(month_z_score(R, d_all), None if args.sigma == 'fit' else 1 / float(args.sigma))]

    def predict(fit_rows, pred_idx, d_pred):
        """Each part calibrated to 15% per month, probit-averaged, then recalibrated."""
        etas, slopes = [], []
        for s, fixed in parts:
            slope = fixed if fixed is not None else fit_slope(s[:n][fit_rows], y[fit_rows], d_tr[fit_rows])
            etas.append(ndtri(calibrated(slope * s[pred_idx], d_pred)))
            slopes.append(slope)
        return calibrated(np.mean(etas, axis=0), d_pred), slopes

    # leave-one-month-out check on the training months (no labels of the held-out month are used)
    p_v = np.empty(n)
    for k in range(4):
        fit_rows, held = d_tr != k, d_tr == k
        p_v[held] = predict(fit_rows, np.flatnonzero(held), d_tr[held])[0]
    print(f'leave-one-month-out composite score {composite(y, p_v):.7f}, by month '
          f'{[round(composite(y[d_tr == k], p_v[d_tr == k]), 6) for k in range(4)]}')

    pf, slopes = predict(np.ones(n, bool), np.arange(n, len(all_rows)), d_te)
    slope = slopes if len(slopes) > 1 else slopes[0]
    out = sample.copy()
    out['Target'] = pf
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format='%.17g')
    check = pd.read_csv(args.out, float_precision='round_trip')
    assert check.ID.equals(sample.ID) and np.array_equal(check.Target.to_numpy(), pf)
    sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f'{args.out} written, slope(s) {np.round(slope, 6)}, sha256 {sha[:12]}, mean per test month '
          f'{[round(float(pf[d_te == k].mean()), 6) for k in (4, 5, 6)]}')
    ref = Path(args.data_dir) / 'submissions' / ('v16_AVG_structural.csv' if args.variant == 'AVG' else
                                                 'v16_MZ_structural.csv' if args.sigma == 'fit' else 'v16_MZF_structural.csv')
    if ref.exists() and (args.variant == 'AVG' or args.sigma in ('fit', '0.35')):
        ref_sha = hashlib.sha256(ref.read_bytes()).hexdigest()
        print(f'standalone vs {ref.name}: {"IDENTICAL" if ref_sha == sha else "DIFFERENT"} {sha[:12]} {ref_sha[:12]}')


if __name__ == '__main__':
    main()
