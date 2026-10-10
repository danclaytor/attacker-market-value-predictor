"""Reader-facing names, independent of model identifiers and data schemas."""
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
