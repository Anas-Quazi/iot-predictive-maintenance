import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Core signals used across all plots
CORE_SIGNALS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'power',
    'temp_diff'
]

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']


# ─────────────────────────────────────────────────────────────────────────────
def plot_sensor_trends(df):
    """
    Plots raw time series of 6 core sensor signals over the full operational
    timeline (Timestamp index). Failure events are marked as red vertical lines.

    Saves: sensor_trends.png
    """
    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True)
    fig.suptitle('Core Sensor Readings Over Time', fontsize=15, fontweight='bold', y=0.98)

    failure_times = df[df['Machine failure'] == 1].index

    for ax, col, color in zip(axes, CORE_SIGNALS, COLORS):
        ax.plot(df.index, df[col], color=color, linewidth=0.6, alpha=0.85)
        for ft in failure_times:
            ax.axvline(ft, color='red', alpha=0.15, linewidth=0.4)
        ax.set_ylabel(col.replace(' [K]', '').replace(' [rpm]', '')
                      .replace(' [Nm]', ''), fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', labelsize=7)

    axes[-1].set_xlabel('Timestamp', fontsize=9)
    fig.legend(['Signal', 'Failure event'], loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('sensor_trends.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('✅ Saved: sensor_trends.png')


# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_stats(df):
    """
    For each core signal, plots raw readings alongside rolling mean (w=5 and w=15)
    to show how the global smoothed baseline tracks the signal over time.

    Saves: rolling_stats.png
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Rolling Mean vs Raw Signal (Global Windows: 5 & 15)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for ax, col, color in zip(axes, CORE_SIGNALS, COLORS):
        ax.plot(df.index, df[col], color=color, alpha=0.3, linewidth=0.5, label='Raw')

        mean5_col  = f'{col}_global_mean_5'
        mean15_col = f'{col}_global_mean_15'

        if mean5_col in df.columns:
            ax.plot(df.index, df[mean5_col],  color='white',  linewidth=1.0, label='Mean w=5')
        if mean15_col in df.columns:
            ax.plot(df.index, df[mean15_col], color='yellow', linewidth=1.2, label='Mean w=15')

        ax.set_title(col, fontsize=9, fontweight='bold')
        ax.set_xlabel('Timestamp', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', labelsize=6)

    plt.tight_layout()
    plt.savefig('rolling_stats.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('✅ Saved: rolling_stats.png')


# ─────────────────────────────────────────────────────────────────────────────
def plot_tool_cycle_behavior(df):
    """
    Analyzes tool wear across all 120 tool cycles:
    - Tool wear pattern within each cycle (sawtooth)
    - Average lifespan per cycle
    - Failure count per cycle

    Saves: tool_cycle_behavior.png
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Tool Cycle Behavior Analysis', fontsize=14, fontweight='bold')

    # Plot 1 — Tool wear over full timeline with cycle resets
    axes[0].plot(df.index, df['Tool wear [min]'], color='#FF9800', linewidth=0.7)
    # Mark resets
    resets = df[df['Tool wear [min]'].diff() < -50].index
    for r in resets:
        axes[0].axvline(r, color='red', alpha=0.4, linewidth=0.6)
    axes[0].set_title('Tool Wear Over Time (Red = Tool Reset)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Tool wear [min]')
    axes[0].grid(True, alpha=0.2)

    # Plot 2 — Lifespan (max tool wear) per cycle
    cycle_lifespan = df.groupby('tool_cycle')['Tool wear [min]'].max()
    axes[1].bar(cycle_lifespan.index, cycle_lifespan.values, color='#2196F3', alpha=0.8, width=0.8)
    axes[1].set_title('Max Tool Wear (Lifespan) per Cycle', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Tool Cycle')
    axes[1].set_ylabel('Max Wear [min]')
    axes[1].grid(True, alpha=0.2, axis='y')

    # Plot 3 — Failure count per cycle
    cycle_failures = df.groupby('tool_cycle')['Machine failure'].sum()
    bars = axes[2].bar(cycle_failures.index, cycle_failures.values, color='#F44336', alpha=0.8, width=0.8)
    axes[2].set_title('Machine Failures per Tool Cycle', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Tool Cycle')
    axes[2].set_ylabel('Failure Count')
    axes[2].grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig('tool_cycle_behavior.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('✅ Saved: tool_cycle_behavior.png')


# ─────────────────────────────────────────────────────────────────────────────
def plot_lag_analysis(df):
    """
    Compares current vs lagged values for Torque, RPM, and Power.
    Scatter plots of current vs lag_1 and lag_2, colored by failure.

    Saves: lag_analysis.png
    """
    lag_signals = ['Torque [Nm]', 'Rotational speed [rpm]', 'power']
    lags = [1, 2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Lag Feature Analysis — Current vs Lagged Values', fontsize=14, fontweight='bold')

    colors_map = df['Machine failure'].map({0: '#2196F3', 1: '#F44336'})

    for col_idx, col in enumerate(lag_signals):
        for lag_idx, lag in enumerate(lags):
            lag_col = f'{col}_lag_{lag}'
            ax = axes[lag_idx][col_idx]

            if lag_col in df.columns:
                ax.scatter(df[lag_col], df[col],
                           c=colors_map, alpha=0.2, s=2)
                ax.set_xlabel(f'{col} lag_{lag}', fontsize=8)
                ax.set_ylabel(f'{col} current', fontsize=8)
                ax.set_title(f'{col.split("[")[0].strip()} — lag {lag}',
                             fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.2)

                # Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=6, label='No Failure'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=6, label='Failure')
                ]
                ax.legend(handles=legend_elements, fontsize=7)

    plt.tight_layout()
    plt.savefig('lag_analysis.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('✅ Saved: lag_analysis.png')


# ─────────────────────────────────────────────────────────────────────────────
def plot_failure_timeline(df):
    """
    Shows when failures occurred over the timeline:
    - Failure events plotted over tool wear signal
    - Failure type breakdown (TWF, HDF, PWF, OSF, RNF) over time
    - Failure density (rolling count)

    Saves: failure_timeline.png
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Failure Event Timeline', fontsize=14, fontweight='bold')

    # Plot 1 — Tool wear with failure overlay
    axes[0].plot(df.index, df['Tool wear [min]'], color='#FF9800',
                 linewidth=0.6, alpha=0.7, label='Tool wear')
    failures = df[df['Machine failure'] == 1]
    axes[0].scatter(failures.index, failures['Tool wear [min]'],
                    color='red', s=10, zorder=5, label='Failure', alpha=0.8)
    axes[0].set_title('Failures on Tool Wear Timeline', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Tool wear [min]')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Plot 2 — Failure types over time (stacked area)
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    ft_colors = ['#F44336', '#FF9800', '#2196F3', '#9C27B0', '#4CAF50']
    for ft, fc in zip(failure_types, ft_colors):
        ft_times = df[df[ft] == 1].index
        axes[1].scatter(ft_times, [ft] * len(ft_times),
                        color=fc, s=8, alpha=0.7, label=ft)
    axes[1].set_title('Failure Types Over Time', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Failure Type')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.2)

    # Plot 3 — Rolling failure density (window=200)
    df_temp = df.copy()
    df_temp['failure_rolling'] = df_temp['Machine failure'].rolling(window=200, min_periods=1).sum()
    axes[2].fill_between(df_temp.index, df_temp['failure_rolling'],
                         color='#F44336', alpha=0.6)
    axes[2].plot(df_temp.index, df_temp['failure_rolling'],
                 color='#F44336', linewidth=1)
    axes[2].set_title('Rolling Failure Density (window=200)', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Timestamp')
    axes[2].set_ylabel('Failure Count')
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('failure_timeline.png', dpi=130, bbox_inches='tight')
    plt.show()
    print('✅ Saved: failure_timeline.png')


# ─────────────────────────────────────────────────────────────────────────────
def run_time_series_eda(df):
    """
    Master function — runs all time series EDA plots in sequence.
    Call this from the notebook.
    """
    print("Starting Time Series EDA...\n")
    plot_sensor_trends(df)
    plot_rolling_stats(df)
    plot_tool_cycle_behavior(df)
    plot_lag_analysis(df)
    plot_failure_timeline(df)
    print("\n✅ All Time Series EDA plots done.")
