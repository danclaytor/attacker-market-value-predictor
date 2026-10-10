"""Render the one-page analytical summary after running run_analysis.py."""
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from modelling import ROOT
import pandas as pd
import json

def main():
    report = ROOT/'reports'
    scores = pd.read_csv(report/'metrics.csv', index_col=0)
    check = json.loads((report/'bootstrap.json').read_text())
    data = pd.read_csv(ROOT/'data/player_seasons.csv')
    training_rows = int(data.split.eq('train').sum())
    evaluation_rows = int(data.split.eq('test').sum())
    improvement = 100*(1-scores.loc['extended_model','MAE_EUR']/scores.loc['model','MAE_EUR'])
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Deck', fontName='Helvetica', fontSize=11, leading=15, textColor=colors.HexColor('#486373'), spaceAfter=12))
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 12
    styles['BodyText'].spaceAfter = 6
    styles['Heading2'].fontSize = 12
    styles['Heading2'].textColor = colors.HexColor('#087f72')
    story = [Paragraph('TOUCHLINE / PLAYER ANALYTICS', styles['Deck']),
             Paragraph('What is an attacker worth?', styles['Title']),
             Paragraph('Performance, prior experience and published market value', styles['Deck']),
             Paragraph('The initial <b>Baseline Model</b> uses current-season attacking performance and player context. Its errors suggested that established high-value players were sometimes underestimated. <b>Experience-Adjusted Model</b> tests whether recorded high-level experience improves the estimates, without using previous market value.', styles['BodyText'])]
    table = [['2025/26 evaluation', 'Baseline Model', 'Experience-Adjusted Model'],
             ['Mean absolute error', *[f'EUR {scores.loc[n,"MAE_EUR"]/1e6:.2f}m' for n in scores.index]],
             ['Median absolute error', *[f'EUR {scores.loc[n,"median_AE_EUR"]/1e6:.2f}m' for n in scores.index]]]
    t=Table(table, colWidths=[190,130,165], hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#182c3b')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                          ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#eef4f4')),('FONTSIZE',(0,0),(-1,-1),10),
                          ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    story += [t, Spacer(1,8), Paragraph(f'Same {evaluation_rows:,} evaluation players; {training_rows:,} training player-seasons from 2022/23-2024/25. MAE falls by {improvement:.1f}%.',styles['BodyText']),
              Image(str(report/'accuracy.png'),width=385,height=168),
              Paragraph('The specification',styles['Heading2']),
              Paragraph('Both models use scikit-learn linear regression on log value: age, age squared, log season minutes, goals and assists per 90, league and attacking position. The extension adds log(1 + prior top-five-league minutes) and log(1 + prior Champions League minutes). Prior means before the season starts.',styles['BodyText']),
              Paragraph('What holds up - and what does not',styles['Heading2']),
              Paragraph(f'The benefit persists after excluding the ten highest-value players and in two earlier-season checks. The paired bootstrap interval for the reduction in MAE is EUR {check["lower_95_m"]:.2f}m-{check["upper_95_m"]:.2f}m. Bundesliga MAE rises slightly; {check["players_improved_pct"]:.1f}% of players improve. Mbappe remains underestimated; Olise, Igor Thiago and Joao Pedro have larger errors.',styles['BodyText']),
              Paragraph('Revised valuation selection',styles['Heading2']),
              Paragraph('Take the latest record as of 19 days after season end; retain it only if positive and no more than 90 days old. A new post-season update is not required. Coverage improves, but gaps remain: Ligue 1 has 36 players in 2022/23 and Serie A has 41 in 2024/25. The notebook shows coverage under 30-, 90- and 180-day limits.',styles['BodyText']),
              Paragraph('Interpretation',styles['Heading2']),
              Paragraph('Some valuations predate the final performance observations: this estimates published values around season end, not strictly future updates. The 2025/26 errors informed development; an untouched final test is still needed. The bootstrap holds models fixed. Missing contract, injury and wage data, uneven coverage and incomplete careers remain limitations. Published value is not a transfer fee.',styles['BodyText']),
              Paragraph('Source: Transfermarkt datasets (github.com/dcaribou/transfermarkt-datasets). See the accompanying notebook.',styles['BodyText'])]
    SimpleDocTemplate(str(report/'Attacker value - summary.pdf'),pagesize=(595.28,841.89),
                      leftMargin=55,rightMargin=55,topMargin=34,bottomMargin=30,
                      title='Attacker value - model comparison',author='').build(story)

if __name__ == '__main__':
    main()
