"""Shared comparisons and deliberately simple robustness checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
from modelling import TARGET, NAMES, fit_model

def compare(rows, model, extended_model):
    result = rows.copy()
    for fitted in (model, extended_model):
        result[f'{fitted.name}_prediction_eur'] = fitted.predict(rows)
        result[f'{fitted.name}_absolute_error_eur'] = abs(result[f'{fitted.name}_prediction_eur'] - result[TARGET])
    result['error_reduction_eur'] = result.model_absolute_error_eur - result.extended_model_absolute_error_eur
    return result

def metrics(predictions):
    return pd.DataFrame([{'model': name, 'players': len(predictions),
                          'MAE_EUR': mean_absolute_error(predictions[TARGET], predictions[f'{name}_prediction_eur']),
                          'median_AE_EUR': median_absolute_error(predictions[TARGET], predictions[f'{name}_prediction_eur'])}
                         for name in NAMES]).set_index('model')

def subgroup_results(predictions):
    records = []
    def add(label, frame):
        m = metrics(frame)
        records.append({'group': label, 'players': len(frame),
                        'model_MAE_m': m.loc['model', 'MAE_EUR']/1e6,
                        'extended_model_MAE_m': m.loc['extended_model', 'MAE_EUR']/1e6})
    add('All players', predictions)
    ordered = predictions.sort_values([TARGET, 'player_id'], ascending=[False, True])
    add('Exclude 10 highest published values', ordered.iloc[10:])
    for league, frame in predictions.groupby('league', sort=True):
        add(league, frame)
    return pd.DataFrame(records)

def paired_bootstrap(predictions, repetitions=2000, seed=27):
    """Player bootstrap of a paired MAE difference, conditional on these fitted models."""
    if predictions.player_id.duplicated().any():
        raise ValueError('This bootstrap expects one observation per evaluation player.')
    differences = predictions.error_reduction_eur.to_numpy()/1e6
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), size=(repetitions, len(differences)))].mean(axis=1)
    return {'MAE_reduction_m': float(differences.mean()),
            'lower_95_m': float(np.quantile(means, .025)),
            'upper_95_m': float(np.quantile(means, .975)),
            'players_improved_pct': float(100*(differences > 0).mean()),
            'repetitions': repetitions, 'seed': seed}

def temporal_checks(data):
    """Earlier forward-season evaluations; no later observations enter either fit."""
    records = []
    seasons = sorted(data.season.unique())
    for i in range(1, len(seasons)):
        train = data.loc[data.season.isin(seasons[:i])]
        test = data.loc[data.season.eq(seasons[i])]
        model = fit_model(train, 'model')
        extended_model = fit_model(train, 'extended_model')
        result = metrics(compare(test, model, extended_model))
        records.append({'training_seasons': ', '.join(seasons[:i]), 'evaluation_season': seasons[i],
                        'training_rows': len(train), 'evaluation_players': len(test),
                        'model_MAE_m': result.loc['model', 'MAE_EUR']/1e6,
                        'extended_model_MAE_m': result.loc['extended_model', 'MAE_EUR']/1e6})
    return pd.DataFrame(records)

def player_examples(predictions):
    """Include the motivating case and deterministic best/worst changes, without cherry-picking."""
    ordered = predictions.sort_values(['error_reduction_eur', 'player_id'])
    motivating = predictions.loc[predictions.player_id.eq(342229)]  # Mbappe
    selected = pd.concat([motivating, ordered.tail(3).iloc[::-1], ordered.head(3)])
    return selected.drop_duplicates('player_id').reset_index(drop=True)
