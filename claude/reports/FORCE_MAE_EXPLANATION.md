# Understanding "Force MAE" in Your Models

## What is "Force MAE"?

**Force MAE = Mean Absolute Error of Forces**

It measures how accurately the model predicts atomic forces:

```
Force MAE = (1/N) × Σ |F_predicted - F_true|
```

Where:
- N = total number of force components (3 × number of atoms across all structures)
- F_predicted = force predicted by your ML model
- F_true = reference force (from DFT calculations)
- Units: eV/Å (energy per unit distance)

## "non_conservative_force" 

In your training logs, you see **`non_conservative_force MAE`**:

```
validation non_conservative_force MAE (per atom): 0.168 eV/A
```

**What "non_conservative" means:**
- Your model predicts forces that may not satisfy energy conservation
- Standard neural networks predict forces independently from energy
- "Non-conservative" = forces are NOT guaranteed to be derivatives of energy
- This is OK for many applications, but can cause issues for some downstream tasks

(There's also "conservative" forces: F = -∇E, derived from energy, always energy-conserving)

## What Dataset?

Based on your config file (`options_restart.yaml`):

### **Validation Set**
```yaml
validation_set:
  read_from: /mnt/shared-storage-user/lijiahang/datasets/non_equi_test_data/data.aselmdb
```

**Dataset**: `non_equi_test_data` 
- Contains structures that are **NOT at equilibrium**
- These are perturbed/displaced structures (intentionally off equilibrium)
- Good for testing force predictions under perturbations
- Likely from Materials Project or similar DFT database

### **Training Set**
(From your earlier files, likely from configs like `pet-omat-xs-v1.0.0-lr-1e-3.yaml`)

Probably trained on:
- **OMat24** dataset (based on file names `pet-omat-xs`)
- or Materials Project structures
- Large-scale DFT calculations with energies, forces, and stresses

## Your Two Models Compared

### June 2 Model (2026-06-03)
```
validation non_conservative_force MAE (per atom): 0.687 eV/Å
```
- **Higher mean error** - on average, forces are off by 0.687 eV/Å
- But: **More consistent** (lower variance)

### June 11 Model (2026-06-10)  
```
validation non_conservative_force MAE (per atom): 0.168 eV/Å
```
- **Lower mean error** - on average, forces are off by only 0.168 eV/Å (4x better!)
- But: **Less consistent** (higher variance)

## Why Better Force MAE → Worse KSRME?

This is the KEY insight:

### Traditional Metric (Force MAE)
```python
# Measures average error across all atoms in validation set
errors = [|F_pred[i] - F_true[i]| for all atoms i]
Force_MAE = mean(errors)
```
- **What it captures**: Average accuracy
- **What it misses**: Consistency, worst-case behavior, outliers

### What KSRME Needs (Force Consistency)
```python
# For finite differences: F(x+δ) must be consistent with F(x-δ)
FC = [F(x+δ) - F(x-δ)] / (2δ)
# If F has random noise ε, then FC has noise ~ε/δ (amplified!)
```

**The Problem**:
- June 11 model: **Low mean error** BUT **high variance**
- When you displace atoms by δ=0.03Å for KSRME:
  - Small random errors in F get amplified in force constants
  - This creates imaginary phonon frequencies
  - Leads to worse KSRME

### Concrete Example

**June 2 Model:**
```
Force predictions on similar structures:
  Structure A: F = 1.52 eV/Å  (error: +0.68)
  Structure A': F = 1.48 eV/Å (error: +0.64)
  Difference: 0.04 eV/Å (consistent!)
  → Force constant calculation: stable
```

**June 11 Model:**
```
Force predictions on similar structures:
  Structure A: F = 0.85 eV/Å  (error: +0.01)
  Structure A': F = 1.12 eV/Å (error: -0.26)
  Difference: 0.27 eV/Å (inconsistent!)
  → Force constant calculation: unstable → imaginary modes
```

Both structures have similar **mean absolute error**, but June 11 has terrible **consistency**.

## Summary Table

| Metric | June 2 | June 11 | What It Means |
|--------|--------|---------|---------------|
| **Force MAE** | 0.687 eV/Å | 0.168 eV/Å | Average error on validation set |
| **Validation Set** | `non_equi_test_data` | `non_equi_test_data` | Same dataset |
| **Relaxation Oscillation** | 0.000004 eV | 0.000542 eV | Force consistency in practice (122x worse!) |
| **Imaginary Freq Rate** | 7.7% | 23.8% | Catastrophic failures in KSRME |
| **KSRME Score** | 0.69 ✓ | 1.12 ✗ | Overall thermal conductivity error |

## Key Takeaway

**Force MAE measures ACCURACY but not ROBUSTNESS/CONSISTENCY**

For downstream tasks like KSRME that use finite differences:
- Need: **Consistent forces** on perturbed structures
- Don't need: Lowest mean error

Your June 11 model optimized for the wrong metric!

## What To Do?

1. **Short-term**: Use hyperparameter tuning (increase `atom_disp`)
2. **Long-term**: Add consistency metrics to training:
   ```python
   # Add to validation
   consistency_loss = |F(x+δ) + F(x-δ) - 2*F(x)|²
   ```
   Or validate on KSRME-like tasks, not just static force MAE.
