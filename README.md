# Financial Stress Prediction Challenge (September Edition) - 2nd place solution

This is my solution to the
[Beginner Track: Financial Stress Prediction Challenge (September Edition)](https://zindi.world/competitions/financial-stress-prediction-challenge-2026-09-01)
on Zindi. It finished **2nd on the private leaderboard** (username Abdo_VDL).

| Submission | Public leaderboard | Private leaderboard |
|---|---|---|
| Final submission (75o4hiai) | 0.744624874 | **0.735747898 (2nd place)** |

## The problem

The task is to predict, for each customer snapshot, the probability of liquidity stress in the 30 days after the
observation date, using six months of mobile money history. The score is 0.6 x (1 - log loss / 0.595) + 0.4 x AUC.
That means calibrated probabilities matter as much as a good ranking.

## My approach

1. **Structural model.** I compute seven liquidity-risk factors per row: balance drop, income drop, withdrawal
   ratio, discretionary spending ratio, outflow / income, balance volatility and inactivity. Each factor is capped at
   its 99% quantile and standardised within its month. The factors are combined with weights 5:3:3:2:4:2:1, and a
   probit P0 = Phi(c + S / 0.35) gives the probability. My analysis showed the labels follow this model very closely,
   so it is a strong, explainable baseline on its own.
2. **LightGBM correction.** LightGBM starts from the structural probability (`init_score = logit(P0)`) and learns
   only the small corrections it misses. All columns are used as within-month percentile ranks. Five parameter sets
   with 50 seeds each are averaged, and the number of trees comes from leave-one-month-out early stopping.
3. **Calibration.** Each test month is calibrated so that its average prediction is 15%, the stress rate of every
   training month.

```
Train.csv + Test.csv
  -> snapshot index from row order
  -> 7 risk factors, capped at the 99% quantile
  -> structural probit  P0 = Phi(c_month + S / 0.35)
  -> LightGBM correction (init_score = logit(P0)), 5 settings x 50 seeds
  -> calibration: mean prediction of each test month = 15%
  -> submission CSV
```

## Repository structure

```
best_model_r1mzf.py   full pipeline, reproduces the final submission byte for byte
final_model_mz.py     the structural model alone (no machine learning library needed)
data_insights.py      data checks behind the model design
requirements.txt      library versions
submissions/          my two selected submission files
docs/                 the full documentation I sent for the Zindi code review (Markdown and PDF)
```

## How to run

1. Download `Train.csv`, `Test.csv` and `SampleSubmission.csv` from the
   [competition data page](https://zindi.world/competitions/financial-stress-prediction-challenge-2026-09-01/data).
   The data is not included in this repository. Put the three files in the repository folder.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run:
   ```
   python best_model_r1mzf.py --seeds 50 --month-scale rank
   ```

The run takes about 15 to 17 minutes on a laptop CPU. The script prints the validation score of each held-out
month, writes `submissions/best_model_r1mzf.csv` and compares it with my original file. On Windows it prints
`IDENTICAL` (sha256 starting with `270ef799f124`).

On Linux or macOS, pandas writes the CSV with LF line endings instead of Windows CRLF, so the hash differs even when
every value is the same. To compare the content only:

```
python -c "a=open('submissions/best_model_r1mzf.csv').read().splitlines(); b=open('submissions/v14_all_MZF_s50_avg5_mrank.csv').read().splitlines(); print('same values' if a == b else 'different')"
```

Optional runs:

- `python final_model_mz.py --sigma 0.35` for the structural model alone (a few seconds);
- `python data_insights.py` for the data checks (a few seconds).

## Results and findings

| Model | Validation (leave-one-month-out) | Public | Private |
|---|---|---|---|
| Structural model alone | 0.7404748 | 0.744516 | 0.735738 |
| Final submission (structural + LightGBM) | 0.7405629 | 0.744625 | 0.735748 |

- In every training month exactly 15% of customers are stressed, so the labels are the top 15% of a risk score
  within each month.
- Train and test look the same (largest Kolmogorov-Smirnov distance 0.011).
- The structural model matches how the labels were made. Changing the caps, the noise level or the noise shape
  always made it worse.
- Most of the LightGBM correction comes from a few single-month count columns. One of them, the month-5 deposit
  pattern, was strong in the training months but did not hold in the test months: a version without those columns
  scored 0.735802 on the private leaderboard. That is a good reminder to re-check month-specific effects on new data.

More details are in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Environment

Everything ran on my laptop: HP Pavilion 15, Intel Core i7-1255U, 16 GB RAM, Windows 11, Python 3.12.3, CPU only.

## Notes

- The competition data belongs to Zindi and the challenge host, so it is not included here.
- Under the challenge rules, the copyright of prize-winning solution code is assigned to Zindi. This repository is
  shared for learning and reference.
- Thanks to Zindi and the challenge host for organising the competition.
