"""Build a small, dated player-season dataset from public Transfermarkt CSVs.

Usage: python preparation/prepare_data.py --raw raw_data
The raw folder must contain players, appearances, player_valuations and clubs.csv.gz.
Only actual historical appearances determine league and club; current club is not used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {'GB1': 'Premier League', 'L1': 'Bundesliga', 'ES1': 'La Liga', 'IT1': 'Serie A', 'FR1': 'Ligue 1'}
POST_SEASON_DAYS = 19
MAX_VALUATION_AGE_DAYS = 90


def select_asof_valuations(values, as_of, max_age_days=MAX_VALUATION_AGE_DAYS):
    """Latest record known by the cutoff, then reject stale/nonpositive records.

    Do not resurrect an older positive value after a newer zero/missing value.
    Equal-date conflicts use the highest value, using a deterministic tie rule.
    Age is measured at the as-of date; both endpoints are inclusive.
    """
    if max_age_days < 0:
        raise ValueError('Maximum valuation age must be nonnegative.')
    values = values.copy()
    values['date'] = pd.to_datetime(values['date'])
    as_of = pd.Timestamp(as_of)
    latest = (values.loc[values.date.le(as_of)]
              .sort_values(['date', 'market_value_in_eur'])
              .drop_duplicates('player_id', keep='last').copy())
    latest['valuation_age_days'] = (as_of - latest.date).dt.days
    return latest.loc[latest.valuation_age_days.le(max_age_days)
                      & latest.market_value_in_eur.gt(0)]


def build(raw):
    players = pd.read_csv(raw / 'players.csv.gz', usecols=['player_id', 'name', 'date_of_birth', 'sub_position', 'position', 'url'])
    appearances = pd.read_csv(raw / 'appearances.csv.gz')
    values = pd.read_csv(raw / 'player_valuations.csv.gz', usecols=['player_id', 'date', 'market_value_in_eur'])
    clubs = pd.read_csv(raw / 'clubs.csv.gz', usecols=['club_id', 'name'])
    assert not players.player_id.duplicated().any()
    assert not clubs.club_id.duplicated().any()
    duplicate_appearances = int(appearances.duplicated(['game_id', 'player_id']).sum())
    appearances = appearances.drop_duplicates(['game_id', 'player_id'])
    appearances.date = pd.to_datetime(appearances.date)
    values.date = pd.to_datetime(values.date)
    players.date_of_birth = pd.to_datetime(players.date_of_birth, errors='coerce')
    seasons, audit, coverage = [], [], []
    for year in [2022, 2023, 2024, 2025]:
        season_games = appearances.loc[appearances.date.between(f'{year}-07-01', f'{year+1}-06-15') & appearances.competition_id.isin(LEAGUES)]
        cutoff = season_games.date.max()
        target_end = cutoff + pd.Timedelta(days=POST_SEASON_DAYS)
        a = appearances.loc[appearances.date.between(f'{year}-07-01', cutoff) & appearances.competition_id.isin(LEAGUES)].copy()
        invalid = a[['minutes_played', 'goals', 'assists']].isna().any(axis=1) | (a[['minutes_played', 'goals', 'assists']] < 0).any(axis=1) | a.minutes_played.gt(130)
        bad_count = int(invalid.sum())
        a = a.loc[~invalid]
        # One row per player and season, including any mid-season moves within these leagues.
        totals = a.groupby('player_id').agg(minutes=('minutes_played', 'sum'), goals=('goals', 'sum'), assists=('assists', 'sum'), matches=('game_id', 'nunique'), last_match=('date', 'max')).reset_index()
        last_club = a.sort_values(['date', 'game_id']).groupby('player_id').tail(1)[['player_id', 'player_club_id']]
        main_league = a.groupby(['player_id', 'competition_id']).minutes_played.sum().reset_index().sort_values(['minutes_played', 'competition_id']).groupby('player_id').tail(1)[['player_id', 'competition_id']]
        d = totals.merge(players, on='player_id', validate='one_to_one').merge(last_club, on='player_id', validate='one_to_one').merge(main_league, on='player_id', validate='one_to_one')
        d['age'] = (cutoff - d.date_of_birth).dt.days / 365.25
        eligible = d.sub_position.isin(['Centre-Forward', 'Second Striker', 'Left Winger', 'Right Winger', 'Attacking Midfield']) & d.age.between(18, 36) & d.minutes.ge(900) & d.sub_position.notna()
        excluded = int((~eligible).sum())
        d = d.loc[eligible].copy()
        for max_age in [30, 90, 180]:
            selected = select_asof_valuations(values, target_end, max_age)
            for league, eligible_rows in d.groupby('competition_id'):
                kept = eligible_rows.merge(selected[['player_id', 'valuation_age_days']], on='player_id')
                coverage.append({'season': f'{year}/{str(year+1)[-2:]}', 'league': LEAGUES[league],
                                 'max_valuation_age_days': max_age, 'eligible_players': len(eligible_rows),
                                 'included_players': len(kept), 'excluded_players': len(eligible_rows)-len(kept),
                                 'median_valuation_age_days': float(kept.valuation_age_days.median()) if len(kept) else None})
        v = select_asof_valuations(values, target_end)
        d = d.merge(v.rename(columns={'date': 'valuation_date'}), on='player_id', how='left', validate='one_to_one')
        no_value = int(d.market_value_in_eur.isna().sum())
        d = d.dropna(subset=['market_value_in_eur'])
        d = d.merge(clubs.rename(columns={'club_id': 'player_club_id', 'name': 'club'}), on='player_club_id', how='left', validate='many_to_one')
        d['club'] = d.club.fillna('Club not recorded')
        d['league'] = d.competition_id.map(LEAGUES)
        d['season'] = f'{year}/{str(year+1)[-2:]}'
        d['feature_cutoff'] = cutoff.strftime('%Y-%m-%d')
        d['valuation_as_of'] = target_end.strftime('%Y-%m-%d')
        d['valuation_before_performance_end'] = d.valuation_date.lt(cutoff)
        d['goals90'] = d.goals * 90 / d.minutes
        d['assists90'] = d.assists * 90 / d.minutes
        d['split'] = 'test' if year == 2025 else 'train'
        assert (d.valuation_date <= target_end).all()
        assert d.valuation_age_days.between(0, MAX_VALUATION_AGE_DAYS).all()
        assert not d.player_id.duplicated().any()
        audit.append({'season': f'{year}/{str(year+1)[-2:]}', 'players_with_appearances': len(totals), 'excluded_scope': excluded, 'excluded_missing_stale_or_nonpositive_value': no_value, 'invalid_appearance_rows': bad_count, 'included': len(d), 'valuations_before_performance_end': int(d.valuation_before_performance_end.sum()), 'max_valuation_age_days_observed': int(d.valuation_age_days.max()), 'performance_end': str(cutoff.date()), 'valuation_as_of': str(target_end.date()), 'games_by_league': {str(k):int(v) for k,v in a.groupby('competition_id').game_id.nunique().items()}})
        seasons.append(d)
    data = pd.concat(seasons, ignore_index=True)
    keep = ['player_id', 'name', 'club', 'league', 'sub_position', 'position', 'season', 'split', 'age', 'minutes', 'goals', 'assists', 'goals90', 'assists90', 'matches', 'feature_cutoff', 'last_match', 'valuation_date', 'valuation_as_of', 'valuation_age_days', 'valuation_before_performance_end', 'market_value_in_eur', 'url']
    (ROOT / 'data').mkdir(exist_ok=True)
    data[keep].to_csv(ROOT / 'data/player_seasons.csv', index=False)
    pd.DataFrame(coverage).to_csv(ROOT / 'data/valuation_coverage.csv', index=False)
    report = {'source': 'https://github.com/dcaribou/transfermarkt-datasets',
              'license': 'CC0-1.0 per source repository',
              'duplicate_appearances_removed': duplicate_appearances, 'seasons': audit,
              'valuation_rule': {'as_of_days_after_performance_end': POST_SEASON_DAYS,
                                 'max_age_days': MAX_VALUATION_AGE_DAYS,
                                 'selection': 'Latest record on or before the as-of date, then require positive value and age <= 90 days.',
                                 'interpretation': 'Contemporaneous estimation of published values around season end; not a strict forecast of later valuation updates.',
                                 'freshness_rationale': 'A transparent roughly quarterly freshness limit, chosen for data recency rather than model accuracy. Coverage at 30 and 180 days is also reported.'},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.csv.gz')},
              'notes': ['Domestic league appearances only; season end is the last observed match across the five leagues.',
                        'Target: latest available valuation as of 19 days after season end, retained only if positive and no more than 90 days old at that date.',
                        'The 19-day offset is an as-of cutoff, not a requirement for a new post-season update. It reflects the final available 2025/26 valuation date and is applied consistently to earlier seasons.',
                        'The 90-day limit prevents very stale targets; missing, stale and nonpositive latest records are excluded. League-season coverage and 30/90/180-day freshness sensitivity are in valuation_coverage.csv.',
                        'Some target valuations precede the final performance observations. This is contemporaneous estimation, not a strict forward valuation forecast.',
                        'Positions are from the profile snapshot, not dated position histories. Retrospective sample selection is a limitation.',
                        'Age 18-36; at least 900 minutes; centre-forwards, second strikers, wingers and attacking midfielders.',
                        '2025/26 French league appearances cover 305 distinct games, compared with 306 in a complete 18-team double round-robin. Coverage is near complete, not perfect.',
                        'No earlier market valuation, club value or current market valuation is used as an input.']}
    (ROOT / 'data/source_notes.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    build(p.parse_args().raw)
