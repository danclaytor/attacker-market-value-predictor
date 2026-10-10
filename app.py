"""Local entry point: streamlit run app.py."""
import pandas as pd
import streamlit as st
from modelling import load_data, fit_models, TARGET, ROOT
from evaluation import compare, metrics, subgroup_results, paired_bootstrap, temporal_checks, player_examples
from charts import interactive_accuracy
from presentation import MODEL_LABELS, present

st.set_page_config(page_title='WHAT IS HE WORTH?', page_icon='⚽', layout='wide')

@st.cache_data
def dataset(data_version):
    """Reload when the prepared data file changes."""
    return load_data()

@st.cache_resource
def fitted_models(data):
    return fit_models(data)

@st.cache_data
def analysis(data):
    model, extended_model = fitted_models(data)
    predictions = compare(data.loc[data.split.eq('test')], model, extended_model)
    return predictions, subgroup_results(predictions), paired_bootstrap(predictions), temporal_checks(data)

def money(value):
    return f'€{value/1e6:,.1f}m'

def reset_profile():
    st.session_state.pop('scenario', None)
    for key in list(st.session_state):
        if key.startswith('input_'):
            del st.session_state[key]

def comparison_table(frame):
    display = frame[['name', TARGET, 'model_prediction_eur', 'extended_model_prediction_eur', 'error_reduction_eur']].copy()
    display.iloc[:, 1:] = display.iloc[:, 1:] / 1e6
    display.columns = ['Player', 'Published (€m)', 'Baseline Model (€m)', 'Experience-Adjusted Model (€m)', 'Error reduction (€m)']
    st.dataframe(display, hide_index=True, width='stretch',
                 column_config={c: st.column_config.NumberColumn(format='%.2f') for c in display.columns[1:]})

data = dataset((ROOT/'data/player_seasons.csv').stat().st_mtime_ns)
model, extended_model = fitted_models(data)
predictions, groups, bootstrap, temporal = analysis(data)
scores = metrics(predictions)

st.markdown('**FOOTBALL ANALYTICS**')
st.title('WHAT IS HE WORTH?')
st.markdown('Because nothing settles a pub argument like linear regression.')
st.caption(f'2025/26 development evaluation · Five major European leagues · {len(predictions):,} eligible attackers')

player_tab, performance_tab, method_tab = st.tabs(['Player explorer', 'Model performance', 'Approach & limits'])
with player_tab:
    ordered = predictions.sort_values('name')
    ids = ordered.player_id.tolist()
    labels = {int(r.player_id): f'{r["name"]} · {r.club}' for _, r in ordered.iterrows()}
    player_id = st.selectbox('Find a player', ids, index=ids.index(132098) if 132098 in ids else 0,
                             format_func=lambda pid: labels[pid], key='player_id', on_change=reset_profile,
                             help='Type a player or club name. Only eligible 2025/26 players are included.')
    player = predictions.loc[predictions.player_id.eq(player_id)].iloc[0]
    st.subheader(player['name'])
    st.caption(f'{player.club}  /  {player.league}  /  {player.sub_position}  /  Age {player.age:.1f}')
    scenario = st.session_state.get('scenario')
    profile = scenario if scenario is not None else player.to_dict()
    first, _ = model.predict_profile(profile)
    second, warnings = extended_model.predict_profile(profile)
    a, b, c = st.columns([1.3, 1, 1])
    a.metric('Experience-Adjusted Model estimate', money(second))
    b.metric('Baseline Model estimate', money(first))
    c.metric('Published market value', money(player[TARGET]))
    st.caption(f'Valuation dated {player.valuation_date} · {int(player.valuation_age_days)} days old at the {player.valuation_as_of} cutoff.')
    if player.valuation_before_performance_end:
        st.info('This published valuation predates the end of the performance season. The estimate is a season-end comparison, not a forecast of a later update.')
    if scenario is not None:
        st.info('Hypothetical profile · published value and evaluation results remain fixed.')
    else:
        change = abs(first-player[TARGET])-abs(second-player[TARGET])
        st.caption(f'For this player, the Experience-Adjusted Model reduces absolute error by {money(change)}.' if change >= 0
                   else f'For this player, the Experience-Adjusted Model increases absolute error by {money(-change)}.')
    if warnings:
        friendly = {'goals90': 'Goals per 90', 'assists90': 'Assists per 90', 'prior_top5_minutes': 'Prior top-five minutes',
                    'prior_cl_minutes': 'Prior Champions League minutes', 'age': 'Age', 'minutes': 'Season minutes'}
        st.warning('Beyond the observed training range: '+', '.join(friendly[k] for k in warnings)+'. Treat this estimate with extra caution.')
    with st.container(border=True):
        st.markdown('**Player profile**')
        a,b,c,d = st.columns(4)
        a.metric('Season minutes', f'{profile["minutes"]:,.0f}')
        b.metric('Goals / assists per 90', f'{profile["goals90"]:.2f} / {profile["assists90"]:.2f}')
        c.metric('Prior top-five minutes', f'{player.prior_top5_minutes:,.0f}')
        d.metric('Prior Champions League minutes', f'{player.prior_cl_minutes:,.0f}')
        st.caption('Experience counts recorded minutes before the season began. Both inputs use log(1 + minutes).')
    with st.expander('Explore a hypothetical profile'):
        st.caption('Change performance or player context. Prior experience stays attached to the selected player.')
        with st.form('scenario_form'):
            a,b,c,d = st.columns(4)
            age = a.number_input('Age', 18., 36., float(player.age), step=.1, key='input_age')
            minutes = b.number_input('Season minutes', 900, 4000, int(player.minutes), key='input_minutes')
            goals = c.number_input('Goals per 90', 0., 2., float(player.goals90), step=.01, format='%.3f', key='input_goals')
            assists = d.number_input('Assists per 90', 0., 2., float(player.assists90), step=.01, format='%.3f', key='input_assists')
            a,b = st.columns(2)
            league = a.selectbox('League', model.leagues, index=model.leagues.index(player.league), key='input_league')
            position = b.selectbox('Position', model.positions, index=model.positions.index(player.sub_position), key='input_position')
            submitted = st.form_submit_button('Estimate profile', type='primary')
        if submitted:
            candidate = {**player.to_dict(), 'age': age, 'minutes': minutes, 'goals90': goals,
                         'assists90': assists, 'league': league, 'sub_position': position}
            try:
                extended_model.predict_profile(candidate)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state.scenario = candidate
                st.rerun()
        st.button('Restore observed profile', on_click=reset_profile)
    st.markdown('#### Where this player sits')
    chart_model = st.radio('Prediction to display', ['extended_model', 'model'], horizontal=True, format_func=MODEL_LABELS.get, key='player_chart_model')
    scale = st.radio('Chart scale', ['linear', 'log'], horizontal=True, format_func=str.title, help='Log scale makes lower-value players easier to distinguish.')
    st.plotly_chart(interactive_accuracy(predictions, chart_model, player_id, scale), width='stretch')
    st.caption('Recorded profiles only. Above the line: overestimation; below the line: underestimation. A log axis changes the display, not the predictions.')

with performance_tab:
    st.subheader('Does prior experience improve the estimates?')
    a,b,c = st.columns(3)
    a.metric('Baseline Model · mean absolute error', f'€{scores.loc["model", "MAE_EUR"]/1e6:.2f}m')
    b.metric('Experience-Adjusted Model · mean absolute error', f'€{scores.loc["extended_model", "MAE_EUR"]/1e6:.2f}m')
    improvement = 100*(1-scores.loc['extended_model', 'MAE_EUR']/scores.loc['model', 'MAE_EUR'])
    c.metric('Reduction in mean absolute error', f'{improvement:.1f}%')
    st.caption(f'Same {len(predictions):,} evaluation players; fitted on {model.training_rows:,} player-seasons from 2022/23–2024/25. Lower error is better.')
    st.markdown('#### Improvements and setbacks')
    st.caption('Mbappé motivated the hypothesis. The other examples are the three largest reductions and three largest increases in absolute error across the full evaluation sample.')
    comparison_table(player_examples(predictions))
    st.markdown('#### How stable is the improvement?')
    st.write(f'The paired player bootstrap gives a 95% interval of €{bootstrap["lower_95_m"]:.2f}m to '
             f'€{bootstrap["upper_95_m"]:.2f}m for the reduction in MAE. '
             f'{bootstrap["players_improved_pct"]:.1f}% of individual players have lower absolute error.')
    st.caption('2,000 resamples of these evaluation players with models held fixed. This is not a prediction interval or a guarantee for a new season; shared league effects are not captured.')
    table_labels = {
        'group': 'Subgroup',
        'players': 'Evaluation players',
        'model_MAE_m': 'Baseline Model MAE (€m)',
        'extended_model_MAE_m': 'Experience-Adjusted Model MAE (€m)',
        'training_seasons': 'Training seasons',
        'evaluation_season': 'Evaluation season',
        'training_rows': 'Training player-seasons',
        'evaluation_players': 'Evaluation players',
    }
    st.dataframe(groups.round(2).rename(columns=table_labels), hide_index=True, width='stretch')
    st.markdown('#### Earlier-season checks')
    st.dataframe(temporal.round(2).rename(columns=table_labels), hide_index=True, width='stretch')
    st.caption('Each row trains only on earlier seasons. These are sensitivity checks, not alternative models selected to maximise the 2025/26 result.')
    st.download_button('Download all player comparisons', present(predictions).to_csv(index=False).encode('utf-8'),
                       'player_comparisons.csv', 'text/csv')

with method_tab:
    st.subheader('From attacking output to market value')
    st.markdown('''**Baseline Model** uses age, age squared, log season minutes, goals per 90, assists per 90, league and attacking position.

**Experience-Adjusted Model** adds log(1 + prior top-five-league minutes) and log(1 + prior Champions League minutes). These measure recorded high-level experience, not reputation directly.

Both use scikit-learn ordinary least squares to predict log market value, then exponentiate the result. Neither uses previous market value. The target is a **published Transfermarkt valuation**, not a completed transfer fee.

The sample covers attackers aged 18–36 with at least 900 domestic league minutes. Each target is the latest recorded valuation available 19 days after season end, provided it is positive and no more than 90 days old at that cutoff. The 90-day freshness limit was chosen for recency, not to maximise model accuracy.

Some valuations precede the final performance observations. This estimates published values around season end; it is not a strict forecast of later valuation updates. Missing or stale records still exclude players, so search is not a complete football directory.

**What the comparison can establish**

2025/26 never enters coefficient fitting, but its errors motivated the experience features. It is therefore a development evaluation, not a pristine final test. A genuinely untouched future season remains the next test.

Contract length, wages, injury history and other valuation drivers are absent. Experience records begin in 2012 and may omit earlier careers. Position labels come from a profile snapshot. Coverage and valuation-update selection can affect results. Log regression does not directly minimise euro MAE; its exponentiated output is not an unbiased mean-value forecast.

Source: [Transfermarkt datasets](https://github.com/dcaribou/transfermarkt-datasets). The prepared snapshot is bundled so the app runs without credentials or live downloads.''')
    st.markdown('#### Coverage by league and season')
    st.dataframe(
        pd.crosstab(data.season, data.league)
        .rename_axis(index="Season", columns="League"),
        width="stretch"
    )
    st.caption('Historical source gaps remain, particularly Ligue 1 in 2022/23 and Serie A in 2024/25.')
    coverage = pd.read_csv(ROOT/'data/valuation_coverage.csv')
    st.markdown('#### How the freshness limit affects coverage')
    st.dataframe(
        coverage.groupby(["season", "max_valuation_age_days"])
        .included_players.sum()
        .unstack()
        .rename(columns=lambda days: f"{days}-day limit")
        .rename_axis(index="Season", columns="Freshness limit"),
        width="stretch"
    )
    st.caption('Player counts under alternative freshness limits. These are coverage checks, not model scores or claims that one limit is optimal.')
