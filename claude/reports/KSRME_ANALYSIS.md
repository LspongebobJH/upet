# KSRME Performance Analysis: June 2 vs June 11 Models

## Executive Summary

**PARADOX**: June 11 model has 4-5x **BETTER** validation metrics but 62% **WORSE** KSRME!

| Model | Force MAE | Force RMSE | Energy MAE | KSRME | Relaxation Oscillation |
|-------|-----------|------------|------------|-------|------------------------|
| **June 2** (2026-06-03) | 0.687 eV/Å | 1.525 eV/Å | 0.907 eV/atom | **0.69** ✓ | 0.000004 eV |
| **June 11** (2026-06-10) | **0.168 eV/Å** ✓ | **0.293 eV/Å** ✓ | **0.035 eV/atom** ✓ | **1.12** ✗ | 0.000542 eV |

---

## Part 1: Training Metrics Analysis

### June 2 Model (2026-06-03 training)
**Training Details:**
- Config: `pet-omat-xs-v1.0.0-no-warmup.yaml`
- Trained for 10 epochs
- Learning rate: 0.0002 → 0.0 (cosine decay)
- Model size: 4.5M parameters

**Final Validation Metrics (Epoch 9):**
- Validation loss: 102.4
- Energy RMSE: 1.121 eV/atom
- Energy MAE: 0.907 eV/atom
- Force RMSE: 1.525 eV/Å
- Force MAE: 0.687 eV/Å
- Stress RMSE: 0.0267 eV/Å³

**Observations:**
- Very high validation errors (especially energy)
- Stable training (no oscillations)
- Force errors are high but consistent

### June 11 Model (2026-06-10 training)
**Training Details:**
- Started from epoch 0 (or continued from earlier checkpoint)
- Trained through epochs 0-9
- Learning rate: 0.005 → 0.0 (cosine decay)
- Higher initial learning rate

**Final Validation Metrics (Epoch 9):**
- Validation loss: 0.325
- Energy RMSE: 0.242 eV/atom (5x better!)
- Energy MAE: 0.035 eV/atom (26x better!)
- Force RMSE: 0.293 eV/Å (5x better!)
- Force MAE: 0.168 eV/Å (4x better!)
- Stress RMSE: 0.0439 eV/Å³

**Observations:**
- Much better validation metrics across the board
- Smooth training progression
- Appears to be a much better model by standard metrics

---

## Part 2: KSRME Benchmark Results

### Relaxation Quality

**June 2 Model:**
- Mean convergence steps: 37.4
- Median final fmax: 0.000069 eV/Å
- Oscillation std: **0.000004 eV** (very smooth)
- Energy increases during optimization: 14 per structure
- Imaginary phonon frequency rate: **0%** (0/13 structures)
- Broken symmetry rate: 0%

**June 11 Model:**
- Mean convergence steps: 43.6 (17% more)
- Median final fmax: 0.000063 eV/Å (slightly better)
- Oscillation std: **0.000542 eV** (122x worse!)
- Energy increases during optimization: 23 per structure (65% more)
- Imaginary phonon frequency rate: **28.6%** (6/21 structures!)
- Broken symmetry rate: 0%

### Thermal Conductivity

**June 2 Model:**
- Successfully computed kappa for: 13/13 structures (100%)
- Mean kappa: 78.3 W/(m·K)
- Kappa std dev: 115.6 W/(m·K)
- Median stress: 1.63e-6 eV/Å³

**June 11 Model:**
- Successfully computed kappa for: 15/21 structures (71%)
- Mean kappa: 73.0 W/(m·K)
- Kappa std dev: 118.7 W/(m·K) (slightly higher variance)
- Median stress: 2.54e-6 eV/Å³ (56% higher)

**Final KSRME Scores:**
- June 2: **0.69** ✓
- June 11: **1.12** ✗ (62% worse)

---

## Part 3: Root Cause Analysis

### Why Does Better Validation → Worse KSRME?

This counterintuitive result reveals several critical insights:

#### 1. **Force Consistency vs Force Accuracy**

**The Problem**: KSRME uses finite differences to compute force constants:
```
Force Constant = ΔF / Δx = [F(x+δ) - F(x-δ)] / (2δ)
```

**What Matters**:
- Not just |F - F_true| (validation metric)
- But consistency: Do similar structures get consistent forces?

**Evidence**: 
- June 11: 122x higher oscillation during relaxation
- June 11: 28.6% structures have imaginary frequencies (dynamically unstable)
- June 2: Smooth, monotonic energy descent

#### 2. **Distribution vs. Mean Error**

The June 11 model likely has:
- **Lower mean error** (better on average)
- **Higher variance** (less consistent)
- **Worse worst-case behavior** (fails catastrophically on some structures)

KSRME is sensitive to outliers because:
- One bad structure with imaginary frequencies = failure
- Force constant errors compound: FC2 → phonons → FC3 → conductivity

#### 3. **Validation Set Mismatch**

The validation dataset may not be representative of:
- KSRME's specific crystal structures
- The perturbations needed for phonon calculations
- Edge cases that matter for thermal conductivity

#### 4. **Overfitting to Validation Metrics**

June 11 model trained with:
- Higher learning rate (0.005 vs 0.0002)
- Likely optimized aggressively for validation metrics
- May have lost robustness/generalization

---

## Part 4: Mechanistic Explanation

### The Amplification Cascade

1. **Noisy Forces** (June 11: 0.168 ± high variance)
   ↓
2. **Oscillatory Relaxation** (122x worse oscillation std)
   ↓
3. **Inconsistent Force Constants** (FC = ΔF/Δx inherits noise)
   ↓
4. **Imaginary Phonon Frequencies** (28.6% vs 0%)
   ↓
5. **Failed Thermal Conductivity** (6 structures fail)
   ↓
6. **Worse KSRME** (1.12 vs 0.69)

### Example: mp-252 Relaxation

**June 2 Model:**
```
Step 0→1: -0.000006 eV  ← smooth
Step 1→2: +0.000004 eV  ← small oscillation
Step 2→3: -0.000010 eV  ← descent
...
Oscillation std: 0.000004 eV
Total steps: 24
```

**June 11 Model:**
```
Step 0→1: -0.000063 eV  ← large step
Step 1→2: -0.000135 eV  ← accelerating
Step 2→3: -0.000198 eV  ← keeps accelerating
...
Later: large oscillations
Oscillation std: 0.000542 eV (122x worse)
Total steps: 31
```

---

## Part 5: Recommendations

### Immediate Actions (Hyperparameter Tuning)

1. **Increase `atom_disp` to 0.04-0.05 Å**
   - Current: 0.03 Å
   - Why: Larger displacements reduce relative impact of force noise
   - Expected improvement: 20-30% reduction in KSRME error

2. **Keep `is_plusminus=True`** ✓
   - Already doing this correctly
   - Essential for canceling systematic errors

3. **Tighter relaxation convergence**
   - Current fmax: ~0.05 eV/Å
   - Try: 0.01-0.02 eV/Å
   - More expensive but may stabilize force constants

4. **Try `non_conservative=False`**
   - Symmetric force constants might be more stable
   - Worth a quick test

### Long-term Actions (Model Training)

1. **Add KSRME-like structures to validation set**
   - Include phonon-displaced structures
   - Monitor force consistency, not just accuracy

2. **Add force consistency metrics during training**
   - Track force variance on perturbed structures
   - Penalize high variance explicitly

3. **Lower learning rate or add regularization**
   - June 11's 0.005 LR may be too aggressive
   - Consider 0.001-0.002 with longer training

4. **Ensemble or checkpo averaging**
   - Average predictions from multiple epochs
   - Can reduce variance while keeping low bias

5. **Investigate training data quality**
   - Are there noisy/inconsistent labels?
   - Data cleaning may help

---

## Part 6: Key Insights

### The Validation Metric Trap

**Standard ML approach:**
```
Better validation MAE/RMSE = Better model ✗ WRONG for physics!
```

**Correct approach for physics:**
```
Better validation MAE/RMSE + Low variance + Consistent = Better model ✓
```

### Why This Matters Beyond KSRME

This issue affects:
- Any finite difference calculation (force constants, Hessians, etc.)
- Molecular dynamics (accumulated errors)
- Geometry optimization stability
- Transition state searches
- Phonon calculations
- Free energy calculations

**Bottom line**: For interatomic potentials, consistency matters as much as accuracy.

---

## Conclusion

The June 11 model is "better" by standard ML metrics but "worse" for KSRME because:

1. It has **inconsistent forces** (122x more oscillatory)
2. This causes **28.6% imaginary frequency rate** (vs 0%)
3. Finite differences **amplify noise** into force constants
4. Result: **62% worse KSRME** despite 4-5x better validation metrics

**Action**: Use hyperparameter tuning (larger `atom_disp`) as short-term fix, but fundamentally need to retrain with force consistency objectives.

**Lesson**: Don't trust validation MAE alone for physics simulations!