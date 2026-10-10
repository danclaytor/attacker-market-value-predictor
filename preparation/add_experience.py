"""Append observed pre-season experience without changing the original sample.

Run from the project folder: python preparation/add_experience.py --raw raw_data
The notebook and Streamlit app use the prepared CSV.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOP5 = ['GB1','ES1','L1','IT1','FR1']
EXTRA = ['prior_top5_minutes','prior_cl_minutes','prior_minutes_cutoff',
         'latest_prior_top5_appearance','latest_prior_cl_appearance']

def build(raw):
    path = ROOT/'data/player_seasons.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        fields=[key for key in reader.fieldnames if key not in EXTRA]
        rows=list(reader)
    source=raw/'appearances.csv.gz'
    a=pd.read_csv(source,usecols=['game_id','player_id','competition_id','date','minutes_played'])
    a=a.drop_duplicates(['game_id','player_id'])
    a=a[a.competition_id.isin(TOP5+['CL'])].copy()
    earliest=str(a.date.min())
    invalid=a.minutes_played.isna() | ~a.minutes_played.between(0,130)
    invalid_count=int(invalid.sum())
    a=a[~invalid]
    for season in sorted({r['season'] for r in rows}):
        year=int(season[:4])
        cutoff=f'{year}-07-01'
        history=a[a.date.lt(cutoff)]
        groups=[]
        for competitions in [TOP5,['CL']]:
            group=history[history.competition_id.isin(competitions)].groupby('player_id').agg(
                minutes=('minutes_played','sum'),last=('date','max'))
            groups.append(group.to_dict('index'))
        for row in rows:
            if row['season']!=season:continue
            pid=int(row['player_id'])
            for group,key,last_key in zip(groups,EXTRA[:2],EXTRA[3:]):
                record=group.get(pid)
                row[key]=int(record['minutes']) if record else 0
                row[last_key]=str(record['last']) if record else ''
                assert not row[last_key] or row[last_key]<cutoff
            row['prior_minutes_cutoff']=f'{year}-06-30'
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+EXTRA)
        writer.writeheader();writer.writerows(rows)
    notes=json.loads((ROOT/'data/source_notes.json').read_text(encoding='utf-8'))
    notes['experience']={
        'definition':'Sum of valid recorded minutes before 1 July at the start of each player-season.',
        'top_five_competitions':TOP5,
        'champions_league_competitions':['CL'],
        'qualifying_rounds_included':False,
        'earliest_relevant_appearance':earliest,
        'invalid_history_records_excluded':invalid_count,
        'appearance_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'sample_changed':False,
        'transformation':'log1p(minutes) for both additional regression inputs',
        'limitations':['Observed history starts in 2012; these are not guaranteed full-career totals.',
                       'Zero means no qualifying minutes recorded, not proof of no real experience.',
                       'Current-season minutes never enter either prior-experience total.',
                       'Experience proxies high-level exposure and selection; it does not directly measure reputation.']}
    (ROOT/'data/source_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('Added pre-season experience to',len(rows),'unchanged player-season records.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    build(parser.parse_args().raw)
