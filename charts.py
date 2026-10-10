"""Consistent chart styling for the notebook and local app."""
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from modelling import TARGET, NAMES
from presentation import MODEL_LABELS

COLORS = {'model': '#64849b', 'extended_model': '#087f72', 'Published value': '#182c3b'}

def accuracy_figure(predictions):
    fig = Figure(figsize=(11, 4.8), layout='constrained', facecolor='white')
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    cap = max(predictions[TARGET].max(), *(predictions[f'{n}_prediction_eur'].max() for n in NAMES))/1e6*1.04
    for ax, name in zip(axes, NAMES):
        ax.scatter(predictions[TARGET]/1e6, predictions[f'{name}_prediction_eur']/1e6,
                   s=20, alpha=.55, color=COLORS[name], edgecolors='none')
        ax.plot([0, cap], [0, cap], '--', color='#a3afb7', linewidth=1, label='Exact agreement')
        ax.set(xlim=(0, cap), ylim=(0, cap), title=MODEL_LABELS[name], xlabel='Published market value (€m)')
        ax.set_aspect('equal', adjustable='box')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Predicted market value (€m)')
    fig.suptitle(f'2025/26 • predicted versus published value | {len(predictions):,} attackers', fontsize=14)
    return fig

def robustness_figure(table):
    fig = Figure(figsize=(10, 4.6), layout='constrained')
    ax = fig.subplots()
    y = np.arange(len(table))
    ax.barh(y-.17, table.model_MAE_m, height=.30, color=COLORS['model'], label=MODEL_LABELS['model'])
    ax.barh(y+.17, table.extended_model_MAE_m, height=.30, color=COLORS['extended_model'], label=MODEL_LABELS['extended_model'])
    ax.set_yticks(y, [f'{r.group} (n={r.players})' for r in table.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel('Mean absolute error (€m) · lower is better')
    ax.set_title('Does the improvement extend beyond the highest-value players?', loc='left', pad=16)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    return fig

def player_figure(examples):
    fig = Figure(figsize=(11, 5.3), layout='constrained')
    ax = fig.subplots()
    y = np.arange(len(examples))
    for offset, name, col in [(-.22, 'model', 'model_prediction_eur'),
                              (0, 'extended_model', 'extended_model_prediction_eur'),
                              (.22, 'Published value', TARGET)]:
        ax.barh(y+offset, examples[col]/1e6, height=.20, color=COLORS[name], label=MODEL_LABELS.get(name, name))
    ax.set_yticks(y, examples.name)
    ax.invert_yaxis()
    ax.set_xlabel('Market value (€m)')
    ax.set_title('Player comparisons • improvements and setbacks', loc='left', pad=15)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    return fig

def interactive_accuracy(predictions, name='extended_model', player_id=None, scale='linear'):
    x, y = predictions[TARGET]/1e6, predictions[f'{name}_prediction_eur']/1e6
    fig = go.Figure(go.Scatter(x=x, y=y, mode='markers',
                    marker=dict(color=COLORS[name], size=7, opacity=.58), name='2025/26 attackers',
                    text=predictions['name'],
                    hovertemplate='%{text}<br>Published: €%{x:.1f}m<br>Prediction: €%{y:.1f}m<extra></extra>'))
    cap = max(predictions[TARGET].max(), *(predictions[f'{n}_prediction_eur'].max() for n in NAMES))/1e6*1.04
    start = .1 if scale == 'log' else 0
    fig.add_trace(go.Scatter(x=[start, cap], y=[start, cap], mode='lines',
                             line=dict(color='#9caeb6', dash='dash'), name='Exact agreement', hoverinfo='skip'))
    selected = predictions.loc[predictions.player_id.eq(player_id)]
    if len(selected):
        fig.add_trace(go.Scatter(x=selected[TARGET]/1e6, y=selected[f'{name}_prediction_eur']/1e6,
                                 text=selected['name'], mode='markers', name=selected.iloc[0]['name'],
                                 marker=dict(color='#d78b32', size=14, line=dict(color='white',width=2)),
                                 hovertemplate='%{text}<br>Published: €%{x:.1f}m<br>Prediction: €%{y:.1f}m<extra></extra>'))
    axis_range = [-1, np.log10(cap)] if scale == 'log' else [0, cap]
    fig.update_layout(template='plotly_white', height=470, margin=dict(t=15,b=30,l=20,r=20),
                      xaxis_title='Published market value (€m)', yaxis_title='Predicted market value (€m)',
                      legend=dict(orientation='h',y=-.22), font=dict(family='Arial',color='#182c3b'),
                      xaxis=dict(type=scale,range=axis_range), yaxis=dict(type=scale,range=axis_range))
    return fig
