#!/usr/bin/env python3
"""
Detailed analysis of energy oscillation for a specific structure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_fire_log(log_path):
    """Parse FIRE log and extract energies."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    steps = []
    energies = []

    for line in lines:
        if line.startswith('FIRE:'):
            parts = line.split()
            try:
                step = int(parts[1])
                energy = float(parts[3])
                steps.append(step)
                energies.append(energy)
            except (ValueError, IndexError):
                continue

    return steps, energies


def analyze_oscillation(steps, energies, label):
    """Detailed oscillation analysis."""

    energies = np.array(energies)

    # Energy differences (step-to-step changes)
    energy_diffs = np.diff(energies)

    # Oscillation std (what we reported)
    oscillation_std = np.std(energy_diffs)

    # Count upward steps
    num_increases = np.sum(energy_diffs > 0)

    # Total energy change
    total_change = energies[-1] - energies[0]

    # Median step size
    median_step = np.median(np.abs(energy_diffs))

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Total steps:          {len(energies)}")
    print(f"Total energy change:  {total_change:.6f} eV")
    print(f"Oscillation std:      {oscillation_std:.6f} eV")
    print(f"Median step size:     {median_step:.6f} eV")
    print(f"Upward steps:         {num_increases} / {len(energy_diffs)} ({100*num_increases/len(energy_diffs):.1f}%)")
    print(f"\nEnergy differences (first 10):")
    for i, diff in enumerate(energy_diffs[:10]):
        direction = "↓" if diff < 0 else "↑"
        print(f"  Step {i}→{i+1}: {diff:+.6f} eV {direction}")

    return steps, energies, energy_diffs, oscillation_std


if __name__ == "__main__":
    # Compare mp-252 from both models

    log1 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/pet-omat-xs-v1.0.0-2026-06-02-non_conservative-is_plusminus/relax_mp-252.log"
    log2 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-10/12-43-47/pet-omat-xs-v1.0.0-2026-06-11-non_conservative-is_plusminus/relax_mp-252.log"

    steps1, energies1 = parse_fire_log(log1)
    steps2, energies2 = parse_fire_log(log2)

    s1, e1, diff1, osc1 = analyze_oscillation(steps1, energies1, "2026-06-02 Model (BETTER)")
    s2, e2, diff2, osc2 = analyze_oscillation(steps2, energies2, "2026-06-11 Model (WORSE)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy vs Step
    ax = axes[0, 0]
    ax.plot(s1, e1, 'o-', label='2026-06-02 (osc_std={:.6f})'.format(osc1), linewidth=2, markersize=4)
    ax.plot(s2, e2, 's-', label='2026-06-11 (osc_std={:.6f})'.format(osc2), linewidth=2, markersize=4)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Trajectory During Relaxation (mp-252)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy differences (step-to-step)
    ax = axes[0, 1]
    ax.plot(range(len(diff1)), diff1, 'o-', label='2026-06-02', alpha=0.7)
    ax.plot(range(len(diff2)), diff2, 's-', label='2026-06-11', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy Change (eV)')
    ax.set_title('Step-to-Step Energy Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram of energy changes
    ax = axes[1, 0]
    ax.hist(diff1, bins=20, alpha=0.5, label='2026-06-02', density=True)
    ax.hist(diff2, bins=20, alpha=0.5, label='2026-06-11', density=True)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Energy Change (eV)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Energy Changes')
    ax.legend()

    # Plot 4: Cumulative energy change
    ax = axes[1, 1]
    cumsum1 = np.cumsum(diff1)
    cumsum2 = np.cumsum(diff2)
    ax.plot(range(len(cumsum1)), cumsum1, 'o-', label='2026-06-02', linewidth=2)
    ax.plot(range(len(cumsum2)), cumsum2, 's-', label='2026-06-11', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Energy Change (eV)')
    ax.set_title('Cumulative Energy Descent')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('oscillation_analysis_mp252.png', dpi=150)
    print(f"\n{'='*60}")
    print("Plot saved to: oscillation_analysis_mp252.png")
    print(f"{'='*60}")

    # Key insight
    print(f"\n{'='*60}")
    print("KEY INSIGHT:")
    print(f"{'='*60}")
    print(f"Oscillation ratio: {osc2/osc1:.1f}x worse in 2026-06-11 model")
    print(f"\nThis means the 2026-06-11 model's forces are {osc2/osc1:.1f}x more inconsistent")
    print("between optimization steps, leading to:")
    print("  1. More oscillatory relaxation")
    print("  2. Noisier force constant calculations")
    print("  3. Higher KSRME error (1.12 vs 0.69)")
