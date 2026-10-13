# Football market-value predictor

Two linear regression models estimate attacking players' published market values
across Europe's five major leagues. The comparison tests whether prior top-five
league and Champions League experience improves estimates based on current-season
performance.

[Open the interactive app](https://danclaytor-player-values.streamlit.app/) to search
for a player, adjust a hypothetical profile and compare estimates. Scenario changes
do not alter the data or evaluation results.

## Method and results

Both models fit ordinary least squares to log market value, then exponentiate the
predictions without a smearing correction or cap.

- **Baseline:** age centred at 25, centred age squared, log(minutes / 1,800), goals
  and assists per 90, league and attacking position.
- **Experience-adjusted:** the baseline plus log(1 + prior top-five-league minutes)
  and log(1 + prior Champions League minutes), recorded before the season begins.
- **Training:** 1,331 player-seasons from 2022/23–2024/25.
- **Evaluation:** 489 players from 2025/26. Players can appear in both periods.

| Model | Mean absolute error | Median absolute error |
| --- | ---: | ---: |
| Baseline | €10.39m | €4.53m |
| Experience-adjusted | €8.70m | €3.96m |

Mean absolute error falls by 16.3%; 57.7% of players have lower individual error.
The paired player bootstrap gives a 95% interval of €1.11m to €2.33m for the MAE
reduction, conditional on the fitted models. It is not a player prediction interval.

**This is a development evaluation.** The 2025/26 errors motivated the experience
features, although that season never enters coefficient fitting. An untouched later
season is still needed. The target is a Transfermarkt valuation, not a transfer fee.

The [notebook](notebooks/market_value_analysis.ipynb)
contains the full analysis, including player examples, league breakdowns and
earlier-season checks. Full player comparisons can be downloaded from the app.

## Data and limitations

Eligible players are attackers aged 18–36 with at least 900 domestic league minutes.
The target is the latest positive valuation available 19 days after season end,
provided the latest record is no more than 90 days old. A newer zero or missing
value does not permit reusing an older positive value. Some valuations predate the
final performance observations, so this estimates values around season end rather
than strictly forecasting later updates.

Contract length, wages and injuries are absent. Experience records start in 2012;
position labels come from a profile snapshot. Missing or stale valuations affect
coverage, particularly Ligue 1 in 2022/23 and Serie A in 2024/25. Exponentiating a
log regression does not estimate an unbiased conditional mean or minimise euro MAE.

Data comes from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets).
[data/source_notes.json](data/source_notes.json) records preparation rules, source
hashes and the source repository's CC0-1.0 designation. This independent project is
not endorsed by Transfermarkt. No separate licence has been assigned to the code.
Codex assisted with data preparation, implementation, troubleshooting and testing.

## Run and reproduce

Use Python 3.12 from the project folder:

```bash
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app fits and caches models from the bundled data; no credentials or live
downloads are needed. Its Streamlit Community Cloud deployment uses `main`,
`app.py` and Python 3.12.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check independent OLS agreement, training/evaluation separation, experience
chronology, valuation boundaries and the interactive profile workflow. GitHub
Actions runs them on Python 3.12. To rerun the analysis, install
`requirements-analysis.txt` and run `notebooks/market_value_analysis.ipynb` from
top to bottom in a Jupyter-compatible editor using the project environment.

To rebuild the prepared data, install the analysis requirements and run:

```bash
python preparation/download_data.py --destination raw_data
python preparation/prepare_data.py --raw raw_data
python preparation/add_experience.py --raw raw_data
```

These commands overwrite the prepared data. Rerun the notebook afterwards.
Upstream downloads may change;
the bundled snapshot is the reproducible input for the reported results.
