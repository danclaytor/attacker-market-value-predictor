"""Download the four public source tables using requests; no API key required.

Usage: python preparation/download_data.py --destination raw_data
The source is a published snapshot, so the included cleaned CSV remains the
reproducible input for this version even if the upstream files later change.
"""
import argparse
from pathlib import Path
import requests

BASE = 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path('raw_data'))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers['User-Agent'] = 'Mozilla/5.0'
        for name in ['players', 'appearances', 'player_valuations', 'clubs']:
            path = args.destination / f'{name}.csv.gz'
            temporary = path.with_suffix('.part')
            with session.get(BASE + path.name, timeout=(15, 120), stream=True) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            temporary.replace(path)
            print(f'Downloaded {path.name} ({path.stat().st_size:,} bytes)')
