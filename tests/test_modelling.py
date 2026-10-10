import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from modelling import load_data, fit_models, TARGET, ROOT
from evaluation import compare, paired_bootstrap, temporal_checks, player_examples

@pytest.fixture(scope='module')
def bundle():
    data = load_data()
    model, extended_model = fit_models(data)
    predictions = compare(data.loc[data.split.eq('test')], model, extended_model)
    return data, model, extended_model, predictions

def test_independent_ols_and_shared_specification(bundle):
    data, model, extended_model, predictions = bundle
    training = data.loc[data.split.eq('train')]
    evaluation = data.loc[data.split.eq('test')]
    for fitted in (model, extended_model):
        def design(rows):
            columns = [np.ones(len(rows)), rows.age-25, (rows.age-25)**2,
                       np.log(rows.minutes/1800), rows.goals90, rows.assists90]
            for key in ['league','sub_position']:
                columns.extend(rows[key].eq(level).astype(float) for level in sorted(training[key].unique())[1:])
            if fitted.name == 'extended_model':
                columns.extend(np.log1p(rows[key]) for key in ['prior_top5_minutes','prior_cl_minutes'])
            return np.column_stack(columns)
        coef = np.linalg.lstsq(design(training), np.log(training[TARGET]), rcond=None)[0]
        np.testing.assert_allclose(fitted.predict(evaluation), np.exp(design(evaluation)@coef), rtol=1e-9, atol=.001)
    assert model.training_rows == extended_model.training_rows == len(training)
    x = model.design(data.loc[data.split.eq('train')])
    xe = extended_model.design(data.loc[data.split.eq('train')])
    pd.testing.assert_frame_equal(x, xe.iloc[:, :-2])

def test_evaluation_values_cannot_change_fit(bundle):
    data, model, extended_model, _ = bundle
    changed = data.copy()
    changed.loc[changed.split.eq('test'), [TARGET, 'prior_cl_minutes', 'goals90']] = 987654
    new_model, new_extended_model = fit_models(changed)
    for a,b in [(model,new_model),(extended_model,new_extended_model)]:
        np.testing.assert_array_equal(a.estimator.coef_, b.estimator.coef_)

def test_experience_chronology_and_sample(bundle):
    data, *_ = bundle
    for field in ['latest_prior_top5_appearance','latest_prior_cl_appearance']:
        present = data[field].notna()
        assert (data.loc[present, field] <= data.loc[present, 'prior_minutes_cutoff']).all()
    assert (data.prior_minutes_cutoff < data.feature_cutoff).all()
    assert not data.duplicated(['player_id','season']).any()

@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1])
def test_invalid_experience_rejected(bundle, bad):
    data, _, extended_model, _ = bundle
    profile = data.iloc[0].to_dict()
    profile['prior_cl_minutes'] = bad
    with pytest.raises(ValueError): extended_model.predict_profile(profile)

def test_warning_does_not_block_kane(bundle):
    _, _, extended_model, predictions = bundle
    kane = predictions.loc[predictions.name.eq('Harry Kane')].iloc[0]
    value, warnings = extended_model.predict_profile(kane)
    assert ('goals90' in warnings) == (not extended_model.ranges['goals90'][0] <= kane.goals90 <= extended_model.ranges['goals90'][1])
    assert value > 0

def test_checks_are_reproducible_and_show_failures(bundle):
    data, _, _, predictions = bundle
    assert paired_bootstrap(predictions) == paired_bootstrap(predictions)
    examples = player_examples(predictions)
    assert examples.error_reduction_eur.min() < 0 < examples.error_reduction_eur.max()
    temporal = temporal_checks(data)
    counts = data.groupby('season').size()
    assert temporal.training_rows.tolist() == counts.cumsum().iloc[:-1].tolist()
    assert temporal.evaluation_players.tolist() == counts.iloc[1:].tolist()

def test_asof_timing_and_league_coverage(bundle):
    data, *_ = bundle
    ages = (pd.to_datetime(data.valuation_as_of)-pd.to_datetime(data.valuation_date)).dt.days
    assert ages.between(0,90).all()
    assert ages.eq(data.valuation_age_days).all()
    counts = pd.crosstab(data.season,data.league)
    assert counts.loc['2023/24','Premier League'] > 80
    assert counts.loc['2024/25','Serie A'] > 20
    assert data.valuation_before_performance_end.any()

def test_latest_asof_boundaries_and_no_resurrection():
    from preparation.prepare_data import select_asof_valuations
    asof = pd.Timestamp('2026-06-12')
    rows = [(1,0,100),(1,-1,999),(2,90,200),(3,91,300),
            (4,10,400),(4,1,0),(5,10,500),(5,1,np.nan),(6,20,600)]
    values = pd.DataFrame([(pid,asof-pd.Timedelta(days=age),value) for pid,age,value in rows],
                          columns=['player_id','date','market_value_in_eur'])
    selected = select_asof_valuations(values,asof).set_index('player_id')
    assert set(selected.index) == {1,2,6}
    assert selected.loc[1,'market_value_in_eur'] == 100
    assert selected.loc[2,'valuation_age_days'] == 90
