#!/usr/bin/env python3
"""
Compare relaxation logs between two KSRME runs to evaluate model quality.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_fire_log(log_path):
    """Parse FIRE optimization log and extract metrics."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    steps = []
    energies = []
    fmax_values = []

    for line in lines:
        if line.startswith('FIRE:'):
            parts = line.split()
            try:
                step = int(parts[1])
                energy = float(parts[3])
                fmax = float(parts[4])
                steps.append(step)
                energies.append(energy)
                fmax_values.append(fmax)
            except (ValueError, IndexError):
                continue

    if not steps:
        return None

    return {
        'num_steps': len(steps),
        'initial_energy': energies[0],
        'final_energy': energies[-1],
        'energy_change': energies[-1] - energies[0],
        'initial_fmax': fmax_values[0],
        'final_fmax': fmax_values[-1],
        'energies': energies,
        'fmax_values': fmax_values,
        'converged': fmax_values[-1] < 0.05,  # Typical threshold
    }


def compute_convergence_smoothness(energies):
    """
    Compute how smoothly energy descends during relaxation.
    Lower is better (fewer oscillations).
    """
    if len(energies) < 3:
        return 0.0

    # Count energy increases (non-monotonic steps)
    increases = sum(1 for i in range(1, len(energies)) if energies[i] > energies[i-1])

    # Compute energy oscillation magnitude
    energy_diffs = np.diff(energies)
    oscillation = np.std(energy_diffs) if len(energy_diffs) > 1 else 0.0

    return {
        'num_increases': increases,
        'oscillation_std': oscillation,
    }


def compare_directories(dir1, dir2, sample_structures=None):
    """Compare relaxation logs from two directories."""

    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    # Get all log files
    logs1 = sorted(dir1_path.glob('relax_*.log'))
    logs2 = sorted(dir2_path.glob('relax_*.log'))

    # Extract material IDs
    mat_ids1 = {log.stem.replace('relax_', ''): log for log in logs1}
    mat_ids2 = {log.stem.replace('relax_', ''): log for log in logs2}

    # Find common structures
    common_ids = set(mat_ids1.keys()) & set(mat_ids2.keys())

    if sample_structures is not None:
        common_ids = list(common_ids)[:sample_structures]

    print(f"Found {len(common_ids)} common structures")

    results = []

    for mat_id in sorted(common_ids):
        log1 = mat_ids1[mat_id]
        log2 = mat_ids2[mat_id]

        data1 = parse_fire_log(log1)
        data2 = parse_fire_log(log2)

        if data1 is None or data2 is None:
            continue

        smooth1 = compute_convergence_smoothness(data1['energies'])
        smooth2 = compute_convergence_smoothness(data2['energies'])

        results.append({
            'material_id': mat_id,
            'model1_steps': data1['num_steps'],
            'model2_steps': data2['num_steps'],
            'model1_final_fmax': data1['final_fmax'],
            'model2_final_fmax': data2['final_fmax'],
            'model1_energy_change': data1['energy_change'],
            'model2_energy_change': data2['energy_change'],
            'model1_oscillations': smooth1['oscillation_std'],
            'model2_oscillations': smooth2['oscillation_std'],
            'model1_increases': smooth1['num_increases'],
            'model2_increases': smooth2['num_increases'],
            'model1_converged': data1['converged'],
            'model2_converged': data2['converged'],
        })

    return pd.DataFrame(results)


def print_comparison_summary(df, name1, name2):
    """Print detailed comparison summary."""

    print("\n" + "="*80)
    print(f"RELAXATION COMPARISON: {name1} vs {name2}")
    print("="*80)

    print(f"\nTotal structures compared: {len(df)}")

    print("\n--- CONVERGENCE ---")
    print(f"{name1} converged: {df['model1_converged'].sum()}/{len(df)} ({df['model1_converged'].mean()*100:.1f}%)")
    print(f"{name2} converged: {df['model2_converged'].sum()}/{len(df)} ({df['model2_converged'].mean()*100:.1f}%)")

    print("\n--- STEPS TO CONVERGENCE (fewer is better) ---")
    print(f"{name1}: mean={df['model1_steps'].mean():.1f}, median={df['model1_steps'].median():.0f}, std={df['model1_steps'].std():.1f}")
    print(f"{name2}: mean={df['model2_steps'].mean():.1f}, median={df['model2_steps'].median():.0f}, std={df['model2_steps'].std():.1f}")
    better = name1 if df['model1_steps'].mean() < df['model2_steps'].mean() else name2
    print(f"→ Winner: {better}")

    print("\n--- FINAL MAX FORCE (lower is better) ---")
    print(f"{name1}: mean={df['model1_final_fmax'].mean():.6f}, median={df['model1_final_fmax'].median():.6f}")
    print(f"{name2}: mean={df['model2_final_fmax'].mean():.6f}, median={df['model2_final_fmax'].median():.6f}")
    better = name1 if df['model1_final_fmax'].mean() < df['model2_final_fmax'].mean() else name2
    print(f"→ Winner: {better}")

    print("\n--- CONVERGENCE SMOOTHNESS (lower oscillation is better) ---")
    print(f"{name1}: mean_osc={df['model1_oscillations'].mean():.6f}, mean_increases={df['model1_increases'].mean():.2f}")
    print(f"{name2}: mean_osc={df['model2_oscillations'].mean():.6f}, mean_increases={df['model2_increases'].mean():.2f}")
    better = name1 if df['model1_oscillations'].mean() < df['model2_oscillations'].mean() else name2
    print(f"→ Winner: {better}")

    print("\n--- DETAILED STRUCTURE-BY-STRUCTURE ---")

    # Count wins
    steps_win1 = (df['model1_steps'] <= df['model2_steps']).sum()
    steps_win2 = (df['model2_steps'] < df['model1_steps']).sum()

    fmax_win1 = (df['model1_final_fmax'] <= df['model2_final_fmax']).sum()
    fmax_win2 = (df['model2_final_fmax'] < df['model1_final_fmax']).sum()

    print(f"Steps: {name1} wins {steps_win1}/{len(df)}, {name2} wins {steps_win2}/{len(df)}")
    print(f"Fmax: {name1} wins {fmax_win1}/{len(df)}, {name2} wins {fmax_win2}/{len(df)}")

    print("\n--- WORST CASES (structures with most steps) ---")
    worst = df.nlargest(5, 'model1_steps')[['material_id', 'model1_steps', 'model2_steps', 'model1_final_fmax', 'model2_final_fmax']]
    print(worst.to_string(index=False))

    print("\n--- BEST CASES (structures with fewest steps) ---")
    best = df.nsmallest(5, 'model1_steps')[['material_id', 'model1_steps', 'model2_steps', 'model1_final_fmax', 'model2_final_fmax']]
    print(best.to_string(index=False))

    print("\n" + "="*80)
    print("OVERALL RECOMMENDATION:")

    # Simple scoring
    score1 = 0
    score2 = 0

    if df['model1_steps'].mean() < df['model2_steps'].mean():
        score1 += 1
    else:
        score2 += 1

    if df['model1_final_fmax'].mean() < df['model2_final_fmax'].mean():
        score1 += 1
    else:
        score2 += 1

    if df['model1_oscillations'].mean() < df['model2_oscillations'].mean():
        score1 += 1
    else:
        score2 += 1

    if score1 > score2:
        print(f"✓ {name1} is BETTER overall (score: {score1}/3)")
    elif score2 > score1:
        print(f"✓ {name2} is BETTER overall (score: {score2}/3)")
    else:
        print(f"≈ Models are COMPARABLE (score: {score1}/3 each)")

    print("="*80)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare relaxation logs between two KSRME runs')
    parser.add_argument('--dir1', required=True, help='First directory path')
    parser.add_argument('--dir2', required=True, help='Second directory path')
    parser.add_argument('--name1', default='Model 1', help='Name for first model')
    parser.add_argument('--name2', default='Model 2', help='Name for second model')
    parser.add_argument('--sample', type=int, default=None, help='Number of structures to sample')
    parser.add_argument('--output', default=None, help='Output CSV file for detailed results')

    args = parser.parse_args()

    df = compare_directories(args.dir1, args.dir2, args.sample)
    df_summary = print_comparison_summary(df, args.name1, args.name2)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")
