from streamlit.testing.v1 import AppTest
from modelling import ROOT, load_data, fit_models

def test_player_search_scenario_and_reset():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox(key='player_id').value == 132098
    assert app.metric[0].label == 'Experience-Adjusted Model estimate'
    published = app.metric[2].value
    initial = app.metric[0].value
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == initial
    app.number_input(key='input_goals').set_value(.8)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value != initial
    assert app.metric[2].value == published
    app.button[1].click().run()
    assert not app.exception
    assert app.metric[0].value == initial
    app.number_input(key='input_goals').set_value(.8)
    app.button[0].click().run()
    app.selectbox(key='player_id').set_value(342229).run()
    assert not app.exception
    data = load_data()
    mbappe = data.loc[data.split.eq('test') & data.player_id.eq(342229)]
    expected = f'€{fit_models(data)[1].predict(mbappe)[0]/1e6:,.1f}m'
    assert app.metric[0].value == expected
    assert app.metric[2].value == '€180.0m'
    assert 'scenario' not in app.session_state
    app.radio(key='player_chart_model').set_value('model').run()
    assert not app.exception
    assert app.metric[0].value == expected
