import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def create_statistical_features(df):
    """
    Creates Z-Score statistical features on top of physical features.
    Requires: power, temp_diff already in df (from physics features).

    Features created:
        rpm_zscore, torque_zscore, power_zscore,
        wear_zscore, temp_diff_zscore

    Returns: df with new statistical feature columns added
    """

    df['rpm_zscore']       = (df['rpm']       - df['rpm'].mean())       / df['rpm'].std()
    df['torque_zscore']    = (df['torque']    - df['torque'].mean())    / df['torque'].std()
    df['power_zscore']     = (df['power']     - df['power'].mean())     / df['power'].std()
    df['wear_zscore']      = (df['tool_wear'] - df['tool_wear'].mean()) / df['tool_wear'].std()
    df['temp_diff_zscore'] = (df['temp_diff'] - df['temp_diff'].mean()) / df['temp_diff'].std()

    zscore_cols = ['rpm_zscore', 'torque_zscore', 'power_zscore', 'wear_zscore', 'temp_diff_zscore']

    print("=" * 60)
    print("Z-SCORE FEATURES CREATED")
    print("=" * 60)
    print(f"  {'Feature':<25} {'Mean':>8} {'Std':>8} {'Extreme (>3σ)':>15}")
    print("  " + "-" * 58)
    for col in zscore_cols:
        extreme = (df[col].abs() > 3).sum()
        print(f"  {col:<25} {df[col].mean():>8.2f} {df[col].std():>8.2f} {extreme:>15}")

    # Z-Score Distribution Plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Z-Score Distributions', fontsize=13, fontweight='bold')

    labels = ['RPM', 'Torque', 'Power', 'Tool Wear', 'Temp Diff']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for ax, col, label, color in zip(axes, zscore_cols, labels, colors):
        ax.hist(df[col], bins=50, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(-3, color='red', linestyle='--', linewidth=1, label='±3σ')
        ax.axvline( 3, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{label} Z-Score', fontsize=10, fontweight='bold')
        ax.set_xlabel('Z-Score', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zscore_distributions.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("\nPlot saved: zscore_distributions.png")

    return df


def create_risk_flag_features(df):
    """
    Creates binary general risk flag features based on threshold conditions.
    Requires: power, temp_diff already in df (from physics features).

    Flags created:
        high_wear_flag     — 1 if tool_wear > 200
        high_torque_flag   — 1 if torque > 60
        low_rpm_flag       — 1 if rpm < 1400
        high_temp_flag     — 1 if temp_diff > 15
        power_anomaly_flag — 1 if power is beyond mean ± 2std

    Returns: df with new risk flag columns added
    """

    # General Risk Flags
    df['high_wear_flag']   = (df['tool_wear'] > 200).astype(int)
    df['high_torque_flag'] = (df['torque'] > 60).astype(int)
    df['low_rpm_flag']     = (df['rpm'] < 1400).astype(int)
    df['high_temp_flag']   = (df['temp_diff'] > 15).astype(int)

    power_mean = df['power'].mean()
    power_std  = df['power'].std()
    df['power_anomaly_flag'] = (
        (df['power'] < power_mean - 2 * power_std) |
        (df['power'] > power_mean + 2 * power_std)
    ).astype(int)

    flag_cols = [
        'high_wear_flag', 'high_torque_flag', 'low_rpm_flag',
        'high_temp_flag', 'power_anomaly_flag'
    ]

    print("=" * 60)
    print("RISK FLAG FEATURES CREATED")
    print("=" * 60)
    print(f"  {'Flag':<25} {'Flagged':>8} {'% Data':>8} {'Actual Fail':>12} {'Precision':>10}")
    print("  " + "-" * 66)
    for col in flag_cols:
        count     = df[col].sum()
        pct       = count / len(df) * 100
        actual    = df[df[col] == 1]['Machine failure'].sum()
        precision = actual / count * 100 if count > 0 else 0
        print(f"  {col:<25} {count:>8} {pct:>7.1f}% {actual:>12} {precision:>9.0f}%")

    # Risk Flag Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Risk Flag Analysis', fontsize=13, fontweight='bold')

    short_labels = [c.replace('_flag', '').replace('_risk', '') for c in flag_cols]
    counts       = [df[col].sum() for col in flag_cols]

    bars = axes[0].barh(short_labels, counts, color='#FF5722', alpha=0.8)
    axes[0].set_title('Rows Flagged per Feature', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Count', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='x')
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     str(count), va='center', fontsize=8)

    precisions = []
    for col in flag_cols:
        count  = df[col].sum()
        actual = df[df[col] == 1]['Machine failure'].sum()
        precisions.append(actual / count * 100 if count > 0 else 0)

    bar_colors = ['#4CAF50' if p == 100 else '#2196F3' if p > 30 else '#FF9800' for p in precisions]
    bars2 = axes[1].bar(short_labels, precisions, color=bar_colors, alpha=0.85)
    axes[1].set_title('Precision — % Flagged That Actually Failed', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Precision (%)', fontsize=9)
    axes[1].set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylim(0, 115)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars2, precisions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('risk_flags_analysis.png', dpi=130, bbox_inches='tight')
    plt.show()
    print("\nPlot saved: risk_flags_analysis.png")

    return df
