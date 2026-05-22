# DIFF_KSRME

Compared files:

- `scripts/ksrme-pet/ksrme-oam-pet-imp.sh`
- `scripts/ksrme-pet/test_pet_kappa.py`
- `scripts/ksrme-pet/calc_kappa.py`
- `scripts/ksrme-pet/thermal_conductivity.py`
- `scripts/ksrme-pet/ana_ksrme_pet_imp.py`
- `scripts/ksrme-oam.sh`
- `benchmark/ksrme.py`
- `benchmark/ana_ksrme.py`
- `src/upet/calculator.py`

## Short answer

The two paths are **not running the same KSRME evaluation pipeline**. The biggest metric-changing differences are:

1. **PET test path forces conductivity to be computed even when imaginary phonons are present** and then also forces `has_imag_ph_modes=False` before metric calculation.
2. **The two paths use different model-loading / calculator stacks** (`.pt` + `MetatomicCalculator` in the PET test path vs `.ckpt` + `UPETCalculator` in the benchmark path).
3. **The PET test path forces float64**, while the benchmark path does not set a dtype.
4. **The PET test path uses a different phonon helper implementation**, including `is_plusminus=True` for displacements.
5. **The PET test script prints metrics on per-worker / per-slice partial data**, while the benchmark path computes metrics only after aggregating results.

So if you are seeing different KSRME numbers, that is expected from the current code.

## 1. Metric policy is different

### PET test path

In `scripts/ksrme-pet/test_pet_kappa.py:104`, the script sets:

- `ignore_imaginary_freqs = True`

That value is passed into `scripts/ksrme-pet/calc_kappa.py:170`, and inside `scripts/ksrme-pet/calc_kappa.py:211-217` it makes:

- `ltc_condition = True`

That means the PET test path will still proceed to FC3 and conductivity even when imaginary phonon modes are present.

Then, before computing metrics, `scripts/ksrme-pet/test_pet_kappa.py:202-205` does:

- `df_kappa["has_imag_ph_modes"] = False`

The same forced override also appears in `scripts/ksrme-pet/ana_ksrme_pet_imp.py:32-35`.

### Benchmark path

In `benchmark/ksrme.py:233-257`, conductivity is only computed when:

- no imaginary frequencies are found, and
- symmetry is not broken (unless `conductivity_broken_symm` were enabled, which it is not).

So the benchmark path is stricter.

### Why this changes KSRME

This is the **largest and clearest source** of metric differences.

The PET test path includes structures that the benchmark path would skip or mark as problematic. It also explicitly clears the imaginary-mode flag before metric computation. That changes which samples contribute valid conductivity values and therefore changes mean SRME.

## 2. The calculators and model artifacts are different

### PET test path

`scripts/ksrme-pet/test_pet_kappa.py:83-86` loads:

- `/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/pet-oam-xl-v1.0.0.pt`
- through `load_atomistic_model(...)`
- wrapped by `metatomic.torch.ase_calculator.MetatomicCalculator`

### Benchmark path

`scripts/ksrme-oam.sh:11-12` passes:

- `./checkpoints/pet-oam-xl-v1.0.0.ckpt`
- to `benchmark/ksrme.py`

Then `benchmark/ksrme.py:94-100` builds a `UPETCalculator`, and `src/upet/calculator.py:121-181` shows that this wraps the model through the `UPETCalculator` / `metatomic_ase.MetatomicCalculator` path.

### Why this changes KSRME

Even if both artifacts nominally correspond to `pet-oam-xl-v1.0.0`, they are **not loaded through the same execution path**:

- `.pt` exported atomistic model vs `.ckpt` checkpoint path
- `metatomic.torch.ase_calculator.MetatomicCalculator` vs `src/upet/calculator.py`'s `UPETCalculator`
- different wrapper codepaths for forces/stresses

That can change predicted forces, relaxation trajectories, phonon force constants, and finally KSRME.

## 3. Precision is different

### PET test path

`scripts/ksrme-pet/test_pet_kappa.py:79-85` explicitly does:

- `precision = "float64"`
- `dtype = torch.float64`
- `model.capabilities().dtype = precision`
- `model = model.to(dtype=dtype, device=device)`

### Benchmark path

`benchmark/ksrme.py:94-100` does not pass `dtype` into `UPETCalculator`, and `src/upet/calculator.py:167-172` shows dtype conversion only happens when `dtype is not None`.

### Why this changes KSRME

Relaxation and phonon calculations are numerically sensitive. Running one path in float64 and the other in default model precision can change:

- optimizer steps,
- final relaxed geometry,
- force constants,
- imaginary-mode detection,
- conductivity values.

This is another strong reason for different KSRME.

## 4. The phonon helper implementation is different

### PET test path

`scripts/ksrme-pet/calc_kappa.py` imports local helpers from `scripts/ksrme-pet/thermal_conductivity.py`.

Important differences there:

- `scripts/ksrme-pet/calc_kappa.py:169` passes `is_plusminus=True`
- `scripts/ksrme-pet/thermal_conductivity.py:19` defaults `is_plusminus=False`, but the PET test path overrides it to `True`
- `scripts/ksrme-pet/thermal_conductivity.py:52-54` therefore generates plus/minus displacements
- `scripts/ksrme-pet/thermal_conductivity.py:81-103` and `106-151` compute FC2/FC3 using batched `calculator.compute_energy(...)`

### Benchmark path

`benchmark/ksrme.py:37-39` imports `matbench_discovery.phonons.thermal_conductivity as ltc`.

`benchmark/ksrme.py:209-223` calls that package implementation, which:

- does **not** pass `is_plusminus=True`
- uses the package helper's default displacement generation
- uses the package helper implementation for FC2/FC3, not the local `scripts/ksrme-pet/thermal_conductivity.py`

### Why this changes KSRME

Different displacement generation and different FC2/FC3 force-evaluation code can change the derived force constants and therefore conductivity. This is a real algorithmic difference, not just a logging difference.

## 5. Metrics are computed at different aggregation scopes

### PET test path

The launcher `scripts/ksrme-pet/ksrme-oam-pet-imp.sh:21-36` runs two dataset slices (`0_52` and `52_103`) and each slice is further split over `WORLD_SIZE=8` workers.

But `scripts/ksrme-pet/test_pet_kappa.py:197-213` computes metrics **inside each worker process**, using only that worker's `df_kappa` subset.

So the printed `kappa_srme` from that script is **not the full 103-structure metric**. It is a partial metric on one worker's shard.

The separate aggregation script `scripts/ksrme-pet/ana_ksrme_pet_imp.py:20-38` is the one that computes the combined metric from all shard outputs.

### Benchmark path

`benchmark/ksrme.py` does **not** compute KSRME inline. The aggregation is deferred to `benchmark/ana_ksrme.py:25-54`, which loads all result files first and only then computes the metric.

### Why this changes KSRME

If you are comparing:

- the `kappa_srme` printed by `test_pet_kappa.py`
- against the aggregated benchmark KSRME

then you are comparing **partial-shard metrics** to **full-dataset metrics**.
That alone can produce different numbers even if all per-structure predictions were identical.

## 6. Dataset sharding is different

### PET test path

`scripts/ksrme-pet/test_pet_kappa.py:117-123` sorts structures by size before slicing:

- `atoms_list = sorted(atoms_list, key=len)`

Then it applies:

- manual slice (`0_52` or `52_103`), and
- per-worker split within that slice (`scripts/ksrme-pet/test_pet_kappa.py:127-134`)

### Benchmark path

`benchmark/ksrme.py:319-344` does **not** sort the structures before distributed slicing.

### Why this usually does **not** change final KSRME

If aggregation is done by `material_id`, sorting only changes which worker handles which structures, not the final metric.

However, it **does** change:

- per-worker composition,
- per-worker printed partial metrics,
- workload distribution.

So it matters if you compare per-process outputs.

## 7. Launcher differences that can affect execution conditions

### PET launcher

`scripts/ksrme-pet/ksrme-oam-pet-imp.sh:25-36`

- `NPROC_PER_NODE=8`
- defaults `NODE_COUNT=1`, `NODE_RANK=0`
- runs workers in background with `&`

### Benchmark launcher

`scripts/ksrme-oam.sh:16-26`

- `NPROC_PER_NODE=2`
- relies on external `NODE_COUNT` / `NODE_RANK`
- runs Python commands sequentially inside the loop (no `&`)

### Why this matters

Parallelism itself should not change the mathematically intended metric, but it changes execution topology and the exact worker subsets. Combined with the PET script's per-worker inline metric printing, it can make the observed `kappa_srme` outputs look very different.

## 8. Differences that are probably not the main cause

These differ, but are less likely to be the primary reason for KSRME mismatch:

- `save_forces=True` in `scripts/ksrme-pet/test_pet_kappa.py:101` vs `save_forces=False` in `benchmark/ksrme.py:88`
- different output directory naming
- different environment activation style in the shell scripts
- PET path uses `formula_getter=lambda a: a.info.get("name", ...)` while benchmark path uses `atoms.get_chemical_formula()` for labeling

These mainly affect outputs/logging, not the metric definition itself.

## Bottom line

If you want one sentence:

> The PET test path is more permissive and uses a different numerical stack: it loads a different model artifact through a different calculator path, runs in float64, uses a different phonon helper with `is_plusminus=True`, and computes metrics after forcing imaginary-mode structures to count as valid.

That combination is enough to explain different KSRME values.

## Most likely causes ranked

1. **Imaginary-frequency handling + forced `has_imag_ph_modes=False` override**
2. **Different calculator/model-loading stack (`.pt` vs `.ckpt`, direct `MetatomicCalculator` vs `UPETCalculator`)**
3. **float64 in PET test path vs default precision in benchmark path**
4. **Different phonon helper implementation and `is_plusminus=True`**
5. **Comparing per-worker PET metrics to aggregated benchmark metrics**
