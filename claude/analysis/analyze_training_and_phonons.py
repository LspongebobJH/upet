#!/usr/bin/env python3
"""
Comprehensive analysis:
1. Training log comparison (loss curves, force MAE/RMSE)
2. Phonon/force constant error analysis from KSRME JSON files
"""

import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def analyze_training_log(log_path):
    """Extract training metrics from train.log"""

    print(f"\n{'='*80}")
    print(f"Analyzing training log: {log_path}")
    print(f"{'='*80}")

    if not Path(log_path).exists():
        print(f"WARNING: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    val_losses = []
    energy_rmse = []
    energy_mae = []
    force_rmse = []
    force_mae = []

    for line in lines:
        if 'Epoch:' in line and 'validation' in line:
            try:
                parts = line.split('|')
                epoch = int(parts[0].split(':')[1].strip())

                # Extract validation metrics
                for part in parts:
                    if 'validation loss:' in part:
                        val_loss = float(part.split(':')[1].strip())
                        val_losses.append(val_loss)
                    elif 'validation energy RMSE (per atom):' in part:
                        e_rmse = float(part.split(':')[1].strip().split()[0])
                        energy_rmse.append(e_rmse)
                    elif 'validation energy MAE (per atom):' in part:
                        e_mae = float(part.split(':')[1].strip().split()[0])
                        energy_mae.append(e_mae)
                    elif 'non_conservative_forces RMSE' in part:
                        f_rmse = float(part.split(':')[1].strip().split()[0])
                        force_rmse.append(f_rmse)
                    elif 'non_conservative_forces MAE' in part:
                        f_mae = float(part.split(':')[1].strip().split()[0])
                        force_mae.append(f_mae)

                epochs.append(epoch)
            except Exception as e:
                continue

    if not epochs:
        print("WARNING: Could not parse any training metrics")
        return None

    results = {
        'epochs': epochs,
        'val_loss': val_losses,
        'energy_rmse': energy_rmse,
        'energy_mae': energy_mae,
        'force_rmse': force_rmse,
        'force_mae': force_mae,
    }

    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Epoch range: {min(epochs)} - {max(epochs)}")
    print(f"\n  Final metrics (epoch {epochs[-1]}):")
    print(f"    Validation loss:   {val_losses[-1]:.6f}")
    print(f"    Energy RMSE:       {energy_rmse[-1]:.6f} eV/atom")
    print(f"    Energy MAE:        {energy_mae[-1]:.6f} eV/atom")
    print(f"    Force RMSE:        {force_rmse[-1]:.6f} eV/Å")
    print(f"    Force MAE:         {force_mae[-1]:.6f} eV/Å")

    # Check for signs of overfitting or instability
    if len(val_losses) > 5:
        recent_var = np.std(val_losses[-5:])
        overall_var = np.std(val_losses)
        print(f"\n  Training stability:")
        print(f"    Overall val loss std:  {overall_var:.6f}")
        print(f"    Recent val loss std:   {recent_var:.6f}")
        if recent_var > 0.5 * overall_var:
            print(f"    ⚠ WARNING: High variance in recent epochs (potential instability)")

        # Check if force error is increasing
        force_trend = np.polyfit(range(len(force_mae)), force_mae, 1)[0]
        if force_trend > 0:
            print(f"    ⚠ WARNING: Force MAE is increasing (slope: {force_trend:.6f})")
        else:
            print(f"    ✓ Force MAE is stable/decreasing")

    return results


def analyze_phonon_data(json_dir):
    """Analyze phonon/force constant data from KSRME JSON files"""

    print(f"\n{'='*80}")
    print(f"Analyzing phonon data: {json_dir}")
    print(f"{'='*80}")

    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob('*kappa*.json.gz'))

    if not json_files:
        print(f"WARNING: No JSON files found in {json_dir}")
        return None

    print(f"Found {len(json_files)} JSON files")

    # Aggregate data from all files
    all_data = {}

    for json_file in json_files:
        try:
            with gzip.open(json_file, 'rt') as f:
                data = json.load(f)

            # Merge data
            for key in data:
                if key not in all_data:
                    all_data[key] = {}
                all_data[key].update(data[key])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    # Analyze aggregated data
    n_structures = len(all_data.get('material_id', {}))
    print(f"\nTotal structures: {n_structures}")

    # Count failures and issues
    has_imag_count = sum(1 for v in all_data.get('has_imag_ph_modes', {}).values() if v)
    broken_symm_count = sum(1 for v in all_data.get('broken_symmetry', {}).values() if v)
    has_errors = sum(1 for v in all_data.get('errors', {}).values() if v)

    print(f"\nStructure quality:")
    print(f"  Imaginary frequencies:  {has_imag_count}/{n_structures} ({100*has_imag_count/n_structures:.1f}%)")
    print(f"  Broken symmetry:        {broken_symm_count}/{n_structures} ({100*broken_symm_count/n_structures:.1f}%)")
    print(f"  Computation errors:     {has_errors}/{n_structures} ({100*has_errors/n_structures:.1f}%)")

    # Analyze max stress
    max_stresses = []
    for idx, stress_data in all_data.get('max_stress', {}).items():
        if isinstance(stress_data, list) and len(stress_data) > 0:
            max_stresses.append(abs(stress_data[0]))

    if max_stresses:
        print(f"\nRelaxation stress:")
        print(f"  Mean max stress:   {np.mean(max_stresses):.6e} eV/Å³")
        print(f"  Median max stress: {np.median(max_stresses):.6e} eV/Å³")
        print(f"  Max stress:        {np.max(max_stresses):.6e} eV/Å³")

    # Analyze thermal conductivity (kappa)
    kappa_values = []
    for idx, kappa_data in all_data.get('kappa_tot_rta', {}).items():
        if isinstance(kappa_data, list) and len(kappa_data) > 0:
            # Take diagonal average (isotropic part)
            if isinstance(kappa_data[0], list) and len(kappa_data[0]) >= 3:
                kappa_avg = np.mean([kappa_data[0][0], kappa_data[0][1], kappa_data[0][2]])
                kappa_values.append(kappa_avg)

    if kappa_values:
        print(f"\nThermal conductivity (kappa):")
        print(f"  Computed for:      {len(kappa_values)}/{n_structures} structures")
        print(f"  Mean kappa:        {np.mean(kappa_values):.4f} W/(m·K)")
        print(f"  Median kappa:      {np.median(kappa_values):.4f} W/(m·K)")
        print(f"  Std kappa:         {np.std(kappa_values):.4f} W/(m·K)")
        print(f"  Range:             [{np.min(kappa_values):.4f}, {np.max(kappa_values):.4f}]")

    return {
        'n_structures': n_structures,
        'has_imag_rate': has_imag_count / n_structures if n_structures > 0 else 0,
        'broken_symm_rate': broken_symm_count / n_structures if n_structures > 0 else 0,
        'error_rate': has_errors / n_structures if n_structures > 0 else 0,
        'max_stresses': max_stresses,
        'kappa_values': kappa_values,
    }


def compare_two_runs(log1, json_dir1, name1, log2, json_dir2, name2):
    """Compare two training runs"""

    print("\n" + "="*80)
    print(f"COMPARING: {name1} vs {name2}")
    print("="*80)

    # Training comparison
    print("\n" + "─"*80)
    print("PART 1: TRAINING METRICS")
    print("─"*80)

    train1 = analyze_training_log(log1)
    train2 = analyze_training_log(log2)

    if train1 and train2:
        print(f"\n{'='*80}")
        print("TRAINING COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Metric':<30} {name1:>20} {name2:>20} {'Winner':>10}")
        print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*10}")

        f1_mae = train1['force_mae'][-1]
        f2_mae = train2['force_mae'][-1]
        f_winner = name1 if f1_mae < f2_mae else name2
        print(f"{'Force MAE (eV/Å)':<30} {f1_mae:>20.6f} {f2_mae:>20.6f} {f_winner:>10}")

        f1_rmse = train1['force_rmse'][-1]
        f2_rmse = train2['force_rmse'][-1]
        f_rmse_winner = name1 if f1_rmse < f2_rmse else name2
        print(f"{'Force RMSE (eV/Å)':<30} {f1_rmse:>20.6f} {f2_rmse:>20.6f} {f_rmse_winner:>10}")

        e1_mae = train1['energy_mae'][-1]
        e2_mae = train2['energy_mae'][-1]
        e_winner = name1 if e1_mae < e2_mae else name2
        print(f"{'Energy MAE (eV/atom)':<30} {e1_mae:>20.6f} {e2_mae:>20.6f} {e_winner:>10}")

        print(f"\nForce error ratio: {name2} is {f2_mae/f1_mae:.2f}x vs {name1}")

    # Phonon comparison
    print("\n" + "─"*80)
    print("PART 2: PHONON/KSRME METRICS")
    print("─"*80)

    phonon1 = analyze_phonon_data(json_dir1)
    phonon2 = analyze_phonon_data(json_dir2)

    if phonon1 and phonon2:
        print(f"\n{'='*80}")
        print("PHONON COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Metric':<35} {name1:>18} {name2:>18} {'Winner':>10}")
        print(f"{'-'*35} {'-'*18} {'-'*18} {'-'*10}")

        im1 = phonon1['has_imag_rate'] * 100
        im2 = phonon2['has_imag_rate'] * 100
        im_winner = name1 if im1 < im2 else name2
        print(f"{'Imaginary freq rate (%)':<35} {im1:>18.2f} {im2:>18.2f} {im_winner:>10}")

        bs1 = phonon1['broken_symm_rate'] * 100
        bs2 = phonon2['broken_symm_rate'] * 100
        bs_winner = name1 if bs1 < bs2 else name2
        print(f"{'Broken symmetry rate (%)':<35} {bs1:>18.2f} {bs2:>18.2f} {bs_winner:>10}")

        if phonon1['max_stresses'] and phonon2['max_stresses']:
            s1 = np.median(phonon1['max_stresses'])
            s2 = np.median(phonon2['max_stresses'])
            s_winner = name1 if s1 < s2 else name2
            print(f"{'Median max stress (eV/Å³)':<35} {s1:>18.6e} {s2:>18.6e} {s_winner:>10}")

        if phonon1['kappa_values'] and phonon2['kappa_values']:
            k1_std = np.std(phonon1['kappa_values'])
            k2_std = np.std(phonon2['kappa_values'])
            k_winner = name1 if k1_std < k2_std else name2
            print(f"{'Kappa std dev (W/(m·K))':<35} {k1_std:>18.4f} {k2_std:>18.4f} {k_winner:>10}")
            print(f"\n  ↳ Lower kappa variance indicates more reliable thermal conductivity predictions")

    # Final diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'='*80}")

    if train1 and train2:
        force_ratio = f2_mae / f1_mae
        if force_ratio > 1.1:
            print(f"\n⚠ {name2} has {force_ratio:.2f}x higher force MAE")
            print(f"  This explains the noisy relaxation and poor KSRME performance.")
            print(f"\n  Possible causes:")
            print(f"    1. Insufficient training (check if training converged)")
            print(f"    2. Different hyperparameters (learning rate, batch size)")
            print(f"    3. Different training data or augmentation")
            print(f"    4. Model architecture changes")
            print(f"    5. Overfitting (check validation curve stability)")


if __name__ == "__main__":
    # Define paths
    log1 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-03/18-54-13/train.log"
    json_dir1 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/pet-omat-xs-v1.0.0-2026-06-02-non_conservative-is_plusminus"
    name1 = "2026-06-02 (KSRME=0.69)"

    log2 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-10/12-43-47/train.log"
    json_dir2 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-10/12-43-47/pet-omat-xs-v1.0.0-2026-06-11-non_conservative-is_plusminus"
    name2 = "2026-06-11 (KSRME=1.12)"

    compare_two_runs(log1, json_dir1, name1, log2, json_dir2, name2)
