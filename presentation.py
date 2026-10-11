"""Display labels and notebook tables; analytical values are left unchanged."""
import pandas as pd
MODEL_LABELS = {'model': 'Baseline Model', 'extended_model': 'Experience-Adjusted Model'}

def present(frame):
    def label(value):
        text = str(value)
        for key in ('extended_model', 'model'):
            text = text.replace(key, MODEL_LABELS[key])
        return text.replace('_prediction_eur', ' prediction (€)').replace('_absolute_error_eur', ' absolute error (€)').replace('_MAE_m', ' MAE (€m)')
    result = frame.rename(columns=label, index=lambda value: MODEL_LABELS.get(value, value))
    if result.index.name == 'model':
        result.index.name = 'Model'
    return result


LABELS = {
    'name': 'Player', 'season': 'Season', 'split': 'Sample',
    'market_value_in_eur': 'Published value (€m)',
    'model_prediction_eur': 'Baseline Model prediction (€m)',
    'extended_model_prediction_eur': 'Experience-Adjusted Model prediction (€m)',
    'signed_error_eur': 'Prediction error (€m)',
    'absolute_error_eur': 'Absolute error (€m)',
    'model_absolute_error_eur': 'Baseline Model absolute error (€m)',
    'extended_model_absolute_error_eur': 'Experience-Adjusted Model absolute error (€m)',
    'error_reduction_eur': 'Reduction in absolute error (€m)',
    'MAE_EUR': 'MAE (€m)', 'median_AE_EUR': 'Median absolute error (€m)',
    'median_value_eur': 'Median published value (€m)',
    'player_seasons': 'Player-seasons', 'players': 'Evaluation players',
    'training_rows': 'Training player-seasons', 'evaluation_players': 'Evaluation players',
    'group': 'Subgroup', 'model_MAE_m': 'Baseline Model MAE (€m)',
    'extended_model_MAE_m': 'Experience-Adjusted Model MAE (€m)',
    'training_seasons': 'Training seasons', 'evaluation_season': 'Evaluation season',
}

def notebook_table(frame):
    displayed = frame.copy()
    for column in displayed.columns:
        if str(column).lower().endswith('_eur'):
            displayed[column] = displayed[column] / 1e6
    displayed = displayed.rename(columns=LABELS)
    if isinstance(displayed.index, pd.MultiIndex):
        displayed.index = displayed.index.set_names([LABELS.get(n, n) for n in displayed.index.names])
        if 'Sample' in displayed.index.names:
            displayed = displayed.rename(index={'train': 'Training', 'test': 'Evaluation'}, level='Sample')
    else:
        displayed = displayed.rename(index=MODEL_LABELS)
        displayed.index.name = 'Model' if displayed.index.name == 'model' else displayed.index.name
    if any(name is not None for name in displayed.index.names):
        displayed = displayed.reset_index()
    displayed.columns.name = None
    formats = {c: '{:,.2f}' for c in displayed.select_dtypes(include='number').columns}
    for c in ['Player-seasons', 'Training player-seasons', 'Evaluation players']:
        if c in displayed.columns:
            formats[c] = '{:,.0f}'
    return displayed.style.format(formats).hide(axis='index')

def bootstrap_table(result):
    return pd.DataFrame({
        'MAE reduction (€m)': [result['MAE_reduction_m']],
        '95% bootstrap interval (€m)': [f"{result['lower_95_m']:.2f} to {result['upper_95_m']:.2f}"],
        'Players improved (%)': [result['players_improved_pct']],
    }).style.format({'MAE reduction (€m)': '{:.2f}', 'Players improved (%)': '{:.1f}'}).hide(axis='index')
