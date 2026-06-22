import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Guarantee target export directory structure exists
SAVE_DIR = "eda_visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── STUDIO-GRADE CYBER-INDUSTRIAL STYLE TEMPLATE CONFIGURATION ────────────────
BG_COLOR = '#0f1117'
AXIS_COLOR = '#1a1d27'
TEXT_COLOR = '#ffffff'
MUTE_TEXT = '#8b949e'
BORDER_COLOR = '#30363d'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': AXIS_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': MUTE_TEXT,
    'ytick.color': MUTE_TEXT,
    'axes.edgecolor': BORDER_COLOR,
    'grid.color': BORDER_COLOR,
    'font.family': 'sans-serif'
})

CORE_SIGNALS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'power',
    'temp_diff'
]

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63']
FAIL_PALETTE = {'normal': '#2196F3', 'fail': '#F44336'}


# ─────────────────────────────────────────────────────────────────────────────
def plot_sensor_trends(df):
    """
    Plots raw time series of 6 core sensor signals over the full operational
    timeline (Timestamp index). Failure events are marked as red vertical lines.
    """
    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True)
    fig.suptitle('Core Sensor Readings Over Time', fontsize=16, fontweight='bold', y=0.99, color=TEXT_COLOR)

    failure_times = df[df['Machine failure'] == 1].index

    for ax, col, color in zip(axes, CORE_SIGNALS, COLORS):
        ax.plot(df.index, df[col], color=color, linewidth=0.6, alpha=0.85)
        
        if len(failure_times) > 0:
            ax.vlines(failure_times, ymin=df[col].min(), ymax=df[col].max(), 
                      color=FAIL_PALETTE['fail'], alpha=0.15, linewidth=0.4, zorder=1)
            
        ax.set_ylabel(col.replace(' [K]', '').replace(' [rpm]', '').replace(' [Nm]', ''), 
                      fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', labelsize=9)

    axes[-1].set_xlabel('Timestamp', fontsize=11, labelpad=10)
    
    legend_elements = [
        Line2D([0], [0], color=COLORS[0], lw=1.5, label='Sensor Signal'),
        Line2D([0], [0], color=FAIL_PALETTE['fail'], lw=1.5, label='Failure Event')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), facecolor=AXIS_COLOR, edgecolor=BORDER_COLOR)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'sensor_trends.png')
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'✅ Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_stats(df):
    """
    For each core signal, plots raw readings alongside rolling mean (w=5 and w=15)
    to show how the global smoothed baseline tracks the signal over time.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Rolling Mean vs Raw Signal (Global Windows: 5 & 15)', fontsize=15, fontweight='bold', color=TEXT_COLOR)
    axes = axes.flatten()

    for ax, col, color in zip(axes, CORE_SIGNALS, COLORS):
        ax.plot(df.index, df[col], color=color, alpha=0.25, linewidth=0.5, label='Raw')

        # Check or dynamically calculate window 5
        mean5_col = f'{col}_global_mean_5'
        if mean5_col in df.columns:
            ax.plot(df.index, df[mean5_col], color='#ffffff', linewidth=1.0, label='Mean w=5')
        else:
            r5 = df[col].rolling(window=5, min_periods=1).mean()
            ax.plot(df.index, r5, color='#ffffff', linewidth=1.0, label='Mean w=5')

        # Check or dynamically calculate window 15
        mean15_col = f'{col}_global_mean_15'
        if mean15_col in df.columns:
            ax.plot(df.index, df[mean15_col], color='#ffeb3b', linewidth=1.2, label='Mean w=15')
        else:
            r15 = df[col].rolling(window=15, min_periods=1).mean()
            ax.plot(df.index, r15, color='#ffeb3b', linewidth=1.2, label='Mean w=15')

        ax.set_title(col, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Timestamp', fontsize=9)
        ax.legend(fontsize=8, facecolor=AXIS_COLOR, edgecolor=BORDER_COLOR, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'rolling_stats.png')
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'✅ Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def plot_tool_cycle_behavior(df):
    """
    Analyzes tool wear across all tool cycles: sawtooth profiles, maximum lifespans, 
    and event counts per segment.
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    fig.suptitle('Tool Cycle Behavior Analysis', fontsize=16, fontweight='bold', color=TEXT_COLOR)

    # Plot 1 — Tool wear sawtooth
    axes[0].plot(df.index, df['Tool wear [min]'], color='#FF9800', linewidth=0.8)
    resets = df[df['Tool wear [min]'].diff() < -50].index
    if len(resets) > 0:
        axes[0].vlines(resets, ymin=df['Tool wear [min]'].min(), ymax=df['Tool wear [min]'].max(),
                      color=FAIL_PALETTE['fail'], alpha=0.5, linewidth=0.8, linestyle=':', label='Tool Reset')
    axes[0].set_title('Tool Wear Over Time (Visual Component: Mechanical Sawtooth)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel('Tool wear [min]', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Plot 2 & 3 — Requires tool_cycle array segment identifier logic
    if 'tool_cycle' not in df.columns:
        # Fallback generation step based on reset bounds if column is missing
        cycle_series = (df['Tool wear [min]'].diff() < -50).cumsum()
        df['tool_cycle'] = cycle_series

    # Plot 2 — Cycle Lifespans
    cycle_lifespan = df.groupby('tool_cycle')['Tool wear [min]'].max()
    axes[1].bar(cycle_lifespan.index, cycle_lifespan.values, color='#2196F3', alpha=0.7, width=0.8, edgecolor=BORDER_COLOR)
    axes[1].set_title('Max Tool Wear (Lifespan) per Cycle', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Tool Cycle', fontsize=10)
    axes[1].set_ylabel('Max Wear [min]', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.5, axis='y')

    # Plot 3 — Cycle Failures
    cycle_failures = df.groupby('tool_cycle')['Machine failure'].sum()
    axes[2].bar(cycle_failures.index, cycle_failures.values, color=FAIL_PALETTE['fail'], alpha=0.7, width=0.8, edgecolor=BORDER_COLOR)
    axes[2].set_title('Machine Failures per Tool Cycle', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('Tool Cycle', fontsize=10)
    axes[2].set_ylabel('Failure Count', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.5, axis='y')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'tool_cycle_behavior.png')
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'✅ Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def plot_lag_analysis(df):
    """
    Compares current vs lagged values for Torque, RPM, and Power.
    Scatter plots of current vs lag_1 and lag_2, colored by failure state.
    """
    lag_signals = ['Torque [Nm]', 'Rotational speed [rpm]', 'power']
    lags = [1, 2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Lag Feature Analysis — Phase Space Boundary Divergence', fontsize=15, fontweight='bold', color=TEXT_COLOR)

    colors_array = np.where(df['Machine failure'] == 1, FAIL_PALETTE['fail'], FAIL_PALETTE['normal'])

    for col_idx, col in enumerate(lag_signals):
        for lag_idx, lag in enumerate(lags):
            lag_col = f'{col}_lag_{lag}'
            ax = axes[lag_idx][col_idx]

            # Use column value series directly if it exists, otherwise generate dynamically on-the-fly
            if lag_col in df.columns:
                x_vals = df[lag_col]
            else:
                x_vals = df[col].shift(lag)

            ax.scatter(x_vals, df[col], c=colors_array, alpha=0.25, s=2.5, zorder=2)
            ax.set_xlabel(f'{col} (t-{lag})', fontsize=9)
            ax.set_ylabel(f'{col} (t)', fontsize=9)
            ax.set_title(f'{col.split("[")[0].strip()} Profile — Lag {lag}', fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, linestyle=':', alpha=0.4)

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=FAIL_PALETTE['normal'], markersize=6, label='Nominal State'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=FAIL_PALETTE['fail'], markersize=6, label='Anomalous Breakdown')
            ]
            ax.legend(handles=legend_elements, fontsize=8, facecolor=AXIS_COLOR, edgecolor=BORDER_COLOR)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'lag_analysis.png')
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'✅ Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def plot_failure_timeline(df):
    """
    Shows when failures occurred over the timeline:
    - Failure events plotted over tool wear signal
    - Failure type breakdown (TWF, HDF, PWF, OSF, RNF) over time
    - Failure density (rolling count)
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Failure Event Temporal Matrix', fontsize=16, fontweight='bold', color=TEXT_COLOR)

    # Plot 1 — Tool wear with failure overlay
    axes[0].plot(df.index, df['Tool wear [min]'], color='#FF9800', linewidth=0.7, alpha=0.5, label='Tool wear Matrix')
    failures = df[df['Machine failure'] == 1]
    axes[0].scatter(failures.index, failures['Tool wear [min]'], color=FAIL_PALETTE['fail'], s=15, zorder=5, label='Failure State Trigger')
    axes[0].set_title('Failures on Tool Wear Timeline', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel('Tool wear [min]', fontsize=10)
    axes[0].legend(fontsize=9, facecolor=AXIS_COLOR, edgecolor=BORDER_COLOR)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Plot 2 — Failure types over time
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    ft_colors = ['#F44336', '#FF9800', '#2196F3', '#9C27B0', '#4CAF50']
    for ft, fc in zip(failure_types, ft_colors):
        if ft in df.columns:
            ft_times = df[df[ft] == 1].index
            axes[1].scatter(ft_times, [ft] * len(ft_times), color=fc, s=14, alpha=0.8, label=ft, edgecolors=BG_COLOR, linewidths=0.3)
    axes[1].set_title('Discrete Mode Failure Flags Distribution Over Time', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_ylabel('Failure Mode Axis', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.4)

    # Plot 3 — Rolling failure density (window=200)
    df_temp = df.copy()
    df_temp['failure_rolling'] = df_temp['Machine failure'].rolling(window=200, min_periods=1).sum()
    axes[2].fill_between(df_temp.index, df_temp['failure_rolling'], color=FAIL_PALETTE['fail'], alpha=0.2)
    axes[2].plot(df_temp.index, df_temp['failure_rolling'], color=FAIL_PALETTE['fail'], linewidth=1.2)
    axes[2].set_title('Rolling Local Volatility & Stress Accumulation Window (w=200)', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('Timestamp Index Sequence', fontsize=10, labelpad=8)
    axes[2].set_ylabel('Aggregated Incidents', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'failure_timeline.png')
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'✅ Saved: {save_path}')


# ─────────────────────────────────────────────────────────────────────────────
def run_time_series_eda(df):
    """
    Master pipeline controller—executes the newly styled EDA scripts in sequence.
    """
    print(f"🚀 Initializing Structural Time Series EDA under theme {BG_COLOR}...\n")
    plot_sensor_trends(df)
    plot_rolling_stats(df)
    plot_tool_cycle_behavior(df)
    plot_lag_analysis(df)
    plot_failure_timeline(df)
    print(f"\n📦 Operational Assets Generated Successfully. Outputs saved to directory: '{SAVE_DIR}/'")