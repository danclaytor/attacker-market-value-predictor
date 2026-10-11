"""The two fixed OLS specifications; all category levels are learned from training."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
TARGET = 'market_value_in_eur'
BASE = ('age', 'minutes', 'goals90', 'assists90')
EXPERIENCE = ('prior_top5_minutes', 'prior_cl_minutes')
NAMES = ('model', 'extended_model')
LIMITS = {'age': (18, 36), 'minutes': (900, 4000), 'goals90': (0, 2),
          'assists90': (0, 2), 'prior_top5_minutes': (0, 100000),
          'prior_cl_minutes': (0, 30000)}

def load_data():
    data = pd.read_csv(ROOT / 'data/player_seasons.csv')
    if data.duplicated(['player_id', 'season']).any():
        raise ValueError('Duplicate player-season observations.')
    if not data[TARGET].gt(0).all():
        raise ValueError('Market values must be positive.')
    from preparation.prepare_data import POST_SEASON_DAYS, MAX_VALUATION_AGE_DAYS
    as_of = pd.to_datetime(data.valuation_as_of)
    dates = pd.to_datetime(data.valuation_date)
    ages = (as_of - dates).dt.days
    if not (as_of.eq(pd.to_datetime(data.feature_cutoff) + pd.Timedelta(days=POST_SEASON_DAYS)).all()
            and ages.between(0, MAX_VALUATION_AGE_DAYS).all()
            and ages.eq(data.valuation_age_days).all()):
        raise ValueError('Valuations must respect the as-of cutoff and freshness limit.')
    return data

@dataclass
class ValueModel:
    name: str
    estimator: LinearRegression
    leagues: tuple
    positions: tuple
    ranges: dict
    training_rows: int

    def design(self, rows):
        """Build the numeric features and reference-category dummy variables."""
        age = rows.age.astype(float) - 25
        x = pd.DataFrame({'age_minus_25': age, 'age_squared': age ** 2,
                          'log_minutes': np.log(rows.minutes.astype(float) / 1800),
                          'goals_per_90': rows.goals90.astype(float),
                          'assists_per_90': rows.assists90.astype(float)}, index=rows.index)
        for category, levels, prefix in [('league', self.leagues, 'league'),
                                          ('sub_position', self.positions, 'position')]:
            if not rows[category].isin(levels).all():
                raise ValueError(f'Unknown {category}; choose a training category.')
            for level in levels[1:]:
                x[f'{prefix}: {level}'] = rows[category].eq(level).astype(float)
        if self.name == 'extended_model':
            for key in EXPERIENCE:
                x[f'log1p_{key}'] = np.log1p(rows[key].astype(float))
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError('Features must be finite.')
        return x

    def predict(self, rows):
        # Direct exponentiation, without a smearing adjustment or clipping.
        return np.exp(self.estimator.predict(self.design(rows)))

    def predict_profile(self, profile):
        numeric = BASE + (EXPERIENCE if self.name == 'extended_model' else ())
        for key in numeric:
            value = float(profile[key])
            low, high = LIMITS[key]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be between {low:g} and {high:g}.')
        warnings = [key for key in numeric
                    if not self.ranges[key][0] <= float(profile[key]) <= self.ranges[key][1]]
        return float(self.predict(pd.DataFrame([dict(profile)]))[0]), warnings

def fit_model(training, name='model'):
    """Caller supplies training rows only, including for temporal sensitivity checks."""
    if name not in NAMES:
        raise ValueError(f'Choose one of {NAMES}.')
    if training.empty or not training[TARGET].gt(0).all():
        raise ValueError('Training requires positive targets and nonempty rows.')
    numeric = BASE + (EXPERIENCE if name == 'extended_model' else ())
    fitted = ValueModel(name, LinearRegression(), tuple(sorted(training.league.unique())),
                        tuple(sorted(training.sub_position.unique())),
                        {k: (float(training[k].min()), float(training[k].max())) for k in numeric},
                        len(training))
    fitted.estimator.fit(fitted.design(training), np.log(training[TARGET]))
    return fitted

def fit_models(data):
    training = data.loc[data.split.eq('train')]
    model = fit_model(training, 'model')
    extended_model = fit_model(training, 'extended_model')
    return model, extended_model
