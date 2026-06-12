#!/usr/bin/env python3
"""
Compare phonon/force constant data for specific structures between two models.
"""

import json
import gzip
import numpy as np
from pathlib import Path


def load_all_json_data(json_dir):
    """Load and merge all JSON files from a directory."""
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob('*kappa*.json.gz'))

    all_data = {}
    for json_file in json_files:
        with gzip.open(json_file, 'rt') as f:
            data = json.load(f)

        for key in data:
            if key not in all_data:
                all_data[key] = {}
            all_data[key].update(data[key])

    return all_data


def compare_structure(mat_id, data1, data2, name1, name2):
    """Compare a single structure between two models."""

    # Find the index for this material_id
    idx1 = None
    idx2 = None

    for idx, mid in data1['material_id'].items():
        if mid == mat_id:
            idx1 = idx
            break

    for idx, mid in data2['material_id'].items():
        if mid == mat_id:
            idx2 = idx
            break

    if idx1 is None or idx2 is None:
        print(f"Material {mat_id} not found in one or both datasets")
        return

    print(f"\n{'='*80}")
    print(f"Structure: {mat_id} ({data1['formula'][idx1]})")
    print(f"{'='*80}")

    # Compare relaxation metrics
    print(f"\n--- Relaxation Quality ---")
    stress1 = data1['max_stress'][idx1][0] if isinstance(data1['max_stress'][idx1], list) else None
    stress2 = data2['max_stress'][idx2][0] if isinstance(data2['max_stress'][idx2], list) else None

    print(f"  Max stress:")
    print(f"    {name1}: {stress1:.6e} eV/Å³")
    print(f"    {name2}: {stress2:.6e} eV/Å³")
    print(f"    Ratio: {abs(stress2)/abs(stress1) if stress1 else 'N/A'}x")

    broken1 = data1.get('broken_symmetry', {}).get(idx1, False)
    broken2 = data2.get('broken_symmetry', {}).get(idx2, False)
    print(f"  Broken symmetry: {name1}={broken1}, {name2}={broken2}")

    # Compare phonon properties
    print(f"\n--- Phonon Properties ---")
    imag1 = data1.get('has_imag_ph_modes', {}).get(idx1, None)
    imag2 = data2.get('has_imag_ph_modes', {}).get(idx2, None)
    print(f"  Imaginary modes: {name1}={imag1}, {name2}={imag2}")

    # Get phonon frequencies if available
    if 'ph_freqs' in data1 and idx1 in data1['ph_freqs']:
        freqs1 = np.array(data1['ph_freqs'][idx1])
        freqs2 = np.array(data2['ph_freqs'][idx2]) if 'ph_freqs' in data2 and idx2 in data2['ph_freqs'] else None

        print(f"  Number of phonon modes: {len(freqs1)}")
        print(f"  {name1} freq range: [{np.min(freqs1):.3f}, {np.max(freqs1):.3f}] THz")
        if freqs2 is not None:
            print(f"  {name2} freq range: [{np.min(freqs2):.3f}, {np.max(freqs2):.3f}] THz")

            # Count imaginary (negative) frequencies
            n_imag1 = np.sum(freqs1 < 0)
            n_imag2 = np.sum(freqs2 < 0)
            print(f"  Negative frequencies: {name1}={n_imag1}, {name2}={n_imag2}")

    # Compare thermal conductivity
    print(f"\n--- Thermal Conductivity ---")
    kappa_computed1 = 'kappa_tot_rta' in data1 and idx1 in data1['kappa_tot_rta']
    kappa_computed2 = 'kappa_tot_rta' in data2 and idx2 in data2['kappa_tot_rta']

    print(f"  Computed: {name1}={kappa_computed1}, {name2}={kappa_computed2}")

    if kappa_computed1 and kappa_computed2:
        kappa1 = np.array(data1['kappa_tot_rta'][idx1])[0]  # First temperature
        kappa2 = np.array(data2['kappa_tot_rta'][idx2])[0]

        # Take diagonal average (isotropic average)
        kappa1_avg = np.mean([kappa1[0], kappa1[1], kappa1[2]])
        kappa2_avg = np.mean([kappa2[0], kappa2[1], kappa2[2]])

        print(f"  Kappa (300K, avg): {name1}={kappa1_avg:.3f} W/(m·K), {name2}={kappa2_avg:.3f} W/(m·K)")
        print(f"  Difference: {abs(kappa1_avg - kappa2_avg):.3f} W/(m·K) ({100*abs(kappa1_avg - kappa2_avg)/kappa1_avg:.1f}%)")

    # Check for errors
    errors1 = data1.get('errors', {}).get(idx1, [])
    errors2 = data2.get('errors', {}).get(idx2, [])

    if errors1 or errors2:
        print(f"\n--- Errors ---")
        if errors1:
            print(f"  {name1}: {errors1}")
        if errors2:
            print(f"  {name2}: {errors2}")


def main():
    print("="*80)
    print("PHONON/FORCE CONSTANT COMPARISON")
    print("="*80)

    # Paths
    dir1 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/pet-omat-xs-v1.0.0-2026-06-02-non_conservative-is_plusminus"
    dir2 = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-10/12-43-47/pet-omat-xs-v1.0.0-2026-06-11-non_conservative-is_plusminus"

    name1 = "June 2 (KSRME=0.69)"
    name2 = "June 11 (KSRME=1.12)"

    print(f"\nLoading data from:")
    print(f"  {name1}: {dir1}")
    print(f"  {name2}: {dir2}")

    data1 = load_all_json_data(dir1)
    data2 = load_all_json_data(dir2)

    # Find common structures
    mats1 = set(data1['material_id'].values())
    mats2 = set(data2['material_id'].values())
    common = sorted(mats1 & mats2)

    print(f"\nFound {len(common)} common structures")

    # Compare a few interesting cases
    print(f"\n{'='*80}")
    print("DETAILED STRUCTURE COMPARISONS")
    print(f"{'='*80}")

    # Case 1: A structure that both succeeded on
    good_structures = []
    for mat_id in common:
        idx1 = [i for i, m in data1['material_id'].items() if m == mat_id][0]
        idx2 = [i for i, m in data2['material_id'].items() if m == mat_id][0]

        imag1 = data1.get('has_imag_ph_modes', {}).get(idx1, True)
        imag2 = data2.get('has_imag_ph_modes', {}).get(idx2, True)

        if not imag1 and not imag2:
            good_structures.append(mat_id)

    if good_structures:
        print(f"\n>>> Example: Structure both models succeeded on")
        compare_structure(good_structures[0], data1, data2, name1, name2)

    # Case 2: A structure where June 11 failed (imaginary modes)
    failed_structures = []
    for mat_id in common:
        idx1 = [i for i, m in data1['material_id'].items() if m == mat_id][0]
        idx2 = [i for i, m in data2['material_id'].items() if m == mat_id][0]

        imag1 = data1.get('has_imag_ph_modes', {}).get(idx1, False)
        imag2 = data2.get('has_imag_ph_modes', {}).get(idx2, False)

        if not imag1 and imag2:
            failed_structures.append(mat_id)

    if failed_structures:
        print(f"\n>>> Example: Structure where {name2} FAILED (imaginary modes)")
        compare_structure(failed_structures[0], data1, data2, name1, name2)

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Count successes
    n_imag1 = sum(1 for mat_id in common
                  for idx in [i for i, m in data1['material_id'].items() if m == mat_id]
                  if data1.get('has_imag_ph_modes', {}).get(idx, True))
    n_imag2 = sum(1 for mat_id in common
                  for idx in [i for i, m in data2['material_id'].items() if m == mat_id]
                  if data2.get('has_imag_ph_modes', {}).get(idx, True))

    print(f"\nCommon structures: {len(common)}")
    print(f"  {name1} with imaginary modes: {n_imag1} ({100*n_imag1/len(common):.1f}%)")
    print(f"  {name2} with imaginary modes: {n_imag2} ({100*n_imag2/len(common):.1f}%)")

    print(f"\nKEY INSIGHT:")
    print(f"  {name2} has {n_imag2 - n_imag1} MORE structures with dynamical instability")
    print(f"  This directly contributes to the {(1.12-0.69)/0.69*100:.0f}% worse KSRME score")


if __name__ == "__main__":
    main()
