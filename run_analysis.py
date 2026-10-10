"""Reproduce comparison tables and figures: python run_analysis.py."""
import json
from modelling import ROOT, load_data, fit_models
from evaluation import compare, metrics, subgroup_results, paired_bootstrap, temporal_checks, player_examples
from charts import accuracy_figure, robustness_figure, player_figure

def main():
    data = load_data()
    model, extended_model = fit_models(data)
    predictions = compare(data.loc[data.split.eq('test')], model, extended_model)
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    data.groupby(['season', 'league']).size().unstack(fill_value=0).to_csv(reports/'league_season_counts.csv')
    data.groupby('season').agg(players=('player_id','size'),
        median_valuation_age_days=('valuation_age_days','median'),
        max_valuation_age_days=('valuation_age_days','max'),
        valuations_before_performance_end=('valuation_before_performance_end','sum')).to_csv(reports/'valuation_timing.csv')
    predictions.to_csv(reports/'player_comparisons.csv', index=False)
    scores = metrics(predictions)
    scores.to_csv(reports/'metrics.csv')
    groups = subgroup_results(predictions)
    groups.to_csv(reports/'subgroup_checks.csv', index=False)
    temporal_checks(data).to_csv(reports/'temporal_checks.csv', index=False)
    (reports/'bootstrap.json').write_text(json.dumps(paired_bootstrap(predictions), indent=2), encoding='utf-8')
    examples = player_examples(predictions)
    examples.to_csv(reports/'player_examples.csv', index=False)
    accuracy_figure(predictions).savefig(reports/'accuracy.png', dpi=170)
    robustness_figure(groups).savefig(reports/'robustness.png', dpi=170)
    player_figure(examples).savefig(reports/'player_examples.png', dpi=170)
    print(scores.to_string())
    print(groups.to_string(index=False))
    print(temporal_checks(data).to_string(index=False))
    print(paired_bootstrap(predictions))

if __name__ == '__main__':
    main()
