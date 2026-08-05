# pyDEBtox2019 — deep code quality & correctness review

Date: 2026-08-05
Scope: `pydebtox2019/` (`models.py`, `readin.py`, `parspace.py`, `debtox2019api.py`) plus the
top-level driver/test scripts and shipped input files.
Environment used for verification: miniconda env `pyiba` (numpy 1.25.2, scipy 1.11.4, numba 0.58.1).

Findings marked **[verified]** were reproduced by executing code; the rest are from reading.
The previous sweep (`Claude_sweep_review.txt`) is not repeated here except where an item was only
partially closed.

---

## 1. Bugs

### B1. `Lf` feeding-threshold term uses `3*L` instead of `L^3` — **[verified]**
`models.py:96`

```python
if Lf>0:
    f = f/(1+(Lf*Lf*Lf)/(y[1]+y[1]+y[1]))
```

`y[1]+y[1]+y[1]` is `3*L`, not `L**3`. BYOM's DEBtox2019 derivatives use the hyperbolic
relation `f / (1 + (Lf/L)^3)`. The numerator was correctly written as a repeated product
(`Lf*Lf*Lf`) and the denominator was written with `+` instead of `*` — a straight typo, but
it changes the physics, not just the magnitude: the correction becomes linear in `L` rather
than cubic, and is dimensionally inconsistent with `Lf^3`.

Reproduced at `L=2.0, Lf=1.5, f=1.0`: coded value `0.640`, BYOM value `0.703` (≈10% off, and
the discrepancy grows/shrinks non-monotonically with `L`).

Silent: `Lf` is fixed at 0 in all three shipped `input_pars*.json`, so the branch is dead in
the current test set and the error will only surface for a user who actually enables the
feeding threshold.

Fix: `f = f/(1 + (Lf*Lf*Lf)/(y[1]*y[1]*y[1]))`.

**!!!DONE!!!**

### B2. The derivative function mutates the solver's state vector in place — **[verified]**
`models.py:94`

```python
y[1] = max(1e-3*L0, y[1])   # to avoid numerical issues with length = 0
```

MATLAB's `L = max(1e-3*L0, X(2))` writes to a *local*; here it writes back into the array the
integrator handed in. `scipy.integrate.solve_ivp` passes its own `self.y` to the RHS on the
first evaluation of each step (`RK45.__init__` / `_step_impl`), and `odeint` reuses its work
array, so this clamp is not confined to the RHS evaluation — it can silently rewrite the
integrator's stored state.

Verified: calling `_DEBtox2019_derivatives_core` with `y = [0, 1e-6, 0, 1]` returns with the
caller's array changed to `[0, 8.8e-4, 0, 1]`.

Today this mostly coincides with the intended clamp, so it is hard to notice; but it makes the
trajectory dependent on scipy-internal buffer reuse, will differ between the `RK45` and `LSODA`
paths, and is exactly the kind of thing that turns into an unreproducible fit later.

Fix: work on a local, e.g. `L = max(1e-3*L0, y[1])` and use `L` throughout the body.

**!!!DONE!!!**

### B3. `breaktime=True` is silently ignored whenever `Tbp == 0`
`models.py:494-503`

`calc_model` shortcuts out before the segmented solve:

```python
if not (hasattr(self, "Tbp") and self.Tbp and self.Tbp > 0):
    return calc_DEBresults(...)      # <- self.breaktime never consulted
...
if self.breaktime:                   # only reachable when Tbp > 0
```

`breaktime` is documented in `__init__` as an independent option ("solve the ODE separately on
each segment between concentration time points — needed for some renewal/pulsed exposure
designs"), and `testing_script.py:135-137` constructs a model with `Tbp=0, breaktime=1`
expecting exactly that. As written, that model silently integrates straight through the
renewal discontinuities.

This matters for the shipped data: `Test_Cdata.txt` has repeated time points at t=5 and t=15
(instantaneous concentration steps), which is precisely the design `breaktime` exists for.

Fix: hoist the breaktime branch out so it applies for both `Tbp == 0` and `Tbp > 0`.

Related, in the same block: the segment loop `for t in range(len(C)-1)` iterates over the raw
concentration rows, so a duplicated time point produces a zero-length segment
(`timextr[t] == timextr[t+1]`) and calls the solver with a degenerate `t_span`. Worth skipping
zero-length segments explicitly.

**!!!DONE!!!**

### B4. `completedataset.subset()` leaves survival-derived arrays stale and misaligned — **[verified]**
`readin.py:934-991`

`_slice_endpoint` reslices `dataarray`, `weights`, `treatmentsnames`, `timetreat`,
`dataarray_cumulative`, and rebuilds `flatdataclean/flatweightsclean/indfintable`. It does not
touch the per-treatment lists `survdataclass` builds in its constructor
(`deatharraytreat`, `survarrtreat`, `survprobstreat`, `lowlimtreat`, `upplimtreat`,
`meanvalstransf`) nor re-run `calc_mean_and_ci` for length/reproduction.

`deatharraytreat` is read directly by the likelihood
(`models.py:1091`: `self.survstruct_list[nd].deatharraytreat[i]`), so a subset selection that is
not a leading prefix of the treatments produces a **wrong survival log-likelihood**, with no
error.

Verified on the shipped data, dropping the solvent control (`labels != 0.1`, keeping 0/1/2/3):

| | |
|---|---|
| subset labels | `[0, 1, 2, 3]` |
| subset `dataarray` rows | 4 (correct) |
| subset `deatharraytreat` | still **5** entries, unsliced |
| entry `[1]` | belongs to treatment `0.1`, but is consumed as treatment `1.0` |
| `meanvalstransf.shape` | `(5, 12)` while `uniquetreats` has 4 entries |

`efsa_criteria` reads `survprobstreat`/`survarrtreat` through `get_survival_data`, so its
R²/NRMSE are wrong on subsets too; `plot_data(wmeans=True)` will raise or mis-index on the
`(5,12)` vs 4-treatment mismatch.

The current scripts get away with it because `control_type='both'` happens to select the two
*leading* treatments. `build_dataset_variants(..., control_type='control'|'solvent')` builds a
complement subset that is **not** a prefix, and that path is live in the API.

Fix: rather than patching each list, have `_slice_endpoint` re-run the endpoint's constructor
logic on the sliced arrays (or give each dataclass a `_rebuild_derived()` that `__init__` and
`subset()` both call). The current "deepcopy then patch a hand-picked subset of attributes"
approach will keep drifting every time a derived attribute is added.

**!!!DONE!!!**

### B5. Shipped `input_pars.json` fails its own validation — **[verified]**
`input_pars.json` + `debtox2019api.py:1681`

```
ValueError: Parameter 'bb': value (24.9) is outside the declared bounds [0.01, 1.0].
```

`bb` is `islog: true` with `value: 24.90` but `min: 0.01, max: 1.0`. Either the value is linear
and the bounds are log10, or vice versa — the two are not in the same scale. (`kd` in the same
file is more plausibly consistent: `value 0.03253` in `[-2, 2]`, though `0.03253` also reads
like a linear rate constant that was never converted.)

`input_pars_grp.json` and `input_pars_tbp3.json` load fine. Since the stricter validation was
added in commit `0340a1a`, `input_pars.json` has been dead weight; it should be corrected or
removed so it doesn't get picked up as a template.

Note also that `preset_toxlimits` overwrites `bb/bs/kd/zb/zs` bounds immediately afterwards in
every driver script — so the JSON bounds for tox parameters are, in the normal workflow,
write-only. That is worth documenting, because it makes a wrong bound in the JSON both fatal
(at construction) and irrelevant (at fit time), which is a confusing combination.

**!!!NOT REAL BUG!!!**

### B6. `savefig=True` raises `AttributeError` — `model.variant` does not exist
`parspace.py:948, 951`

```python
plt.savefig(figbasename+"_"+self.model.variant+extension)
```

`self.model` is a `DEBtox2019models`; there is no `variant` attribute anywhere in the package
(grep confirms the only other `variant` is the unrelated `build_dataset_variants` function).
Every `savefig=True` call path in `run_parspace` / `replot_results` fails. Leftover from the
GUTS codebase this was ported from.



### B7. `pbest` captured before it is written during profiling
`parspace.py:671-676`

```python
if mll_tst < mll:            # found a better minimum
    pbest = parprof[i_g,:]   # parprof[i_g] is still all zeros here
    ...
parprof[i_g,:] = np.concatenate((pmat_tst, [mll_tst]), axis=0)   # written afterwards
```

In the forward profiling loop, `parprof[i_g,:]` is only assigned at the end of the iteration
(line 680), so when a better optimum is found `pbest` is set to the still-zero row. The reverse
loop (line 699) has the same statement but *after* the assignment, so it is correct — which is
what makes the forward one look like an oversight rather than intent.

Downstream, `run_parspace` does
`self.pbest = pbest[np.argmin([p[-1] for p in pbest])]` and then `if self.pbest[-1] < mll:` —
a zero row has `mll = 0.0`, which for a negative best log-likelihood will win the `argmin` and
trigger a spurious "profiling found a new minimum" restart from an all-zero parameter vector.

Fix: `pbest = np.concatenate((pmat_tst, [mll_tst]))`.

### B8. `scaled_loglikelihood`'s missing-data filter is broken — **[verified]**
`models.py:222-245`

```python
ind_fin = np.isfinite(lengths) & (weights>0)
weights = weights[ind_fin]     # only weights is subset
n = np.sum(ind_fin)
...
res = lengths**transf - model**transf     # full-length
wssq = np.dot(res*weights, res)           # shape mismatch
```

`lengths` and `model` are never filtered by `ind_fin`, so as soon as any observation is
non-finite or zero-weighted the function raises:

```
ValueError: unable to broadcast argument 1 to output array
```

(verified for both a NaN observation and a zero weight; the all-valid case returns normally).

It does not fire today only because `flatten_and_clean` already dropped those points upstream —
i.e. the guard inside the likelihood is dead code that would crash if it ever became live.
Either delete it (and document the precondition) or apply `ind_fin` to all three arrays.

Two smaller things in the same function:
- `mn`, `res_tot`, `wssq_tot` are computed and never used.
- `mn` is an unweighted mean of the transformed data while everything around it is
  weight-aware; if `res_tot` is ever revived for an R²-style statistic this will be wrong.

### B9. Duplicate concentration time points corrupt `concslopes` / `concconst` / `concarray`
`readin.py:158-167` — **[verified as warnings on the shipped data]**

`Test_Cdata.txt` legitimately repeats t=5 and t=15 to encode instantaneous steps. The
constructor then computes

```python
self.concslopestr[i,:-1] = np.diff(self.concarraytr[i]) / np.diff(self.time)   # /0
```

producing `RuntimeWarning: divide by zero` / `invalid value` (observed) and inf/NaN slopes.
Downstream:

- `self.time = np.unique(self.time)` shrinks the time vector, and the reshape
  `tmpslopes.reshape((self.ntreats, len(self.time)))` only survives because the number of
  non-finite entries per treatment happens to equal the number of dropped duplicates. It is an
  accidental invariant, not an enforced one.
- `tmparray = self.concarraytr[np.isfinite(self.concslopestr)]` selects *concentrations* using a
  *slope* mask — so `concarray` silently becomes "the value after each jump", which is not what
  the docstring says it is.
- The last column of `concslopestr` is never assigned (stays 0 from `zeros_like`) and is then
  treated as a real slope.
- `concconst` is never set for a genuinely constant treatment when duplicates are present:
  `0/0 → NaN`, so `np.all(slopes == 0)` is `False` even for the all-zero control.

`concslopes`/`concconst`/`conctwa` are currently unused outside `subset()`, so the blast radius
is limited — but `concarray` *is* used (`debtox2019api.py:88,91`) and is being sliced in
`subset()`, so the mis-selection propagates.

Also `conctwa = trapz(c, t)/self.time[-1]` should divide by `t[-1] - t[0]` to be a time-weighted
average for a profile not starting at 0.

### B10. `concclass(..., focus=True)` reads data rows as treatment names
`readin.py:107-117`

With `focus=True` the header row is *not* stripped from `self.concdata`, but
`self.conctreatsnames = concdata[0,1:]` still reads row 0 (now a data row) as the labels, and
`self.timetr`/`self.time` include the header value as a time point. `focus=True` is the
documented way to feed a raw exposure profile into `calc_epx` via
`_resolve_exposure_profile` (`debtox2019api.py:1145`), so this is on a live path.

Either strip consistently, or make `focus=True` a separate constructor
(`concclass.from_profile(time, conc)`) that never pretends to have a header.

### B11. `makerepro_ind` case 2 mutates the raw data array
`readin.py:687-693`

```python
Rtmp = self.dataarray[i,:]      # a view, not a copy
...
Rtmp[ind_eggs] = 0              # writes into self.dataarray
```

Cases 0 and 1 both use `.copy()`; case 2 does not. The `-1` first-egg markers are therefore
destroyed in `self.dataarray`, which is what `plot_data()` (non-cumulative) and
`flatten_and_clean` for the raw array would show. Inconsistent with the sibling branches and
with the "keep raw, derive cumulative" design of the class.

### B12. `_print_results` mutates `coll_all` — reprinting corrupts the sample
`parspace.py:799`

```python
def _print_results(self, profile=None):
    ...
    for i in range(self.npars):
        self.coll_all = np.append(self.coll_all, profile[i], axis=0)
```

A method named `_print_results` permanently appends the profile rows to the stored sample and
re-sorts it. `reprint_results()` is a public, documented "just show me the results again"
entry point — calling it twice appends the profiles twice, inflating `coll_all`, shifting
`ind_prop1/ind_prop2`, and thus changing the reported CIs and the propagation set on every
call. The merge (which is legitimate — BYOM does fold profiles into the cloud) belongs in
`run_parspace`, once.

### B13. Off-by-one / inconsistency in the propagation-set slice
`parspace.py:1267` vs `1422`

```python
self.propagationset = self.coll_all[ind_prop1:ind_prop2, :-1]      # non-profile branch
self.propagationset = self.coll_all[ind_prop1:ind_prop2+1, :-1]    # profile branch
```

Two different conventions for the same quantity. Additionally `ind_prop1` is
`argwhere(L < mll + 0.5*crit_prop[0]).max()` — the last index *below* the lower edge — so the
slice starts one set outside the intended band in both branches. Should be `ind_prop1 + 1`
(and consistent `+1` on the upper end).

### B14. Hard-coded 10-element tables break above 10 free parameters
`parspace.py:197-215, 983-984`

`crit_table`, `n_ok`, and `n_conf_all` all have exactly 10 rows and are indexed
`[self.npars-1]`. With 11+ free parameters (entirely reachable — grouped/dataset-specific
expansion multiplies the parameter count) this raises `IndexError` deep inside `run_parspace`
rather than failing fast with a clear message. Add an explicit check, or clamp to the last row
the way `crit_table[5:] = crit_table[4]` already does conceptually.

### B15. Age-dependent background hazard is singular at t=0 for `a < 1`
`models.py:91`

```python
hb = a * hb**a * t**(a-1)
```

Applied unconditionally. For `a == 1` (all shipped configs) it reduces to `hb`, fine. For
`a < 1`, `t**(a-1)` at `t = 0` is `+inf`, which makes `dydt[3]` non-finite at the very first
evaluation. BYOM guards this branch (`if a ~= 1`) and the model is only meaningful for `t > 0`.
Since `a` has bounds `[0, 10]` in the JSON and could be freed, add a small floor on `t` or gate
the branch on `a != 1`.

### B16. `calc_model` Tbp fallback path can raise or silently drop values
`models.py:546-565`

- `idx_targets = np.searchsorted(timeext, target_times)` can return `len(timeext)`, and
  `timeext[idx_targets]` on the next line is evaluated *before* the clipping in the fallback
  branch — an `IndexError` waiting on a `target_time` beyond the grid.
- The tolerance in the fallback is `np.finfo(float).eps * 10` — an *absolute* tolerance around
  1e-15, which is meaningless for `t ≈ 21` (where 1 ULP is already ~3.6e-15) and hopeless for
  larger time units. Use a relative tolerance (`np.isclose`).
- When a target is not close enough, its reproduction entry is left at the `0.0` written a few
  lines earlier — a silent zero, not a NaN or an error. Given that the previous review's #1 was
  exactly this class of silent Tbp corruption, this deserves to fail loudly.

### B17. `log_likelihood` assumes all replicates of a treatment share one time vector
`models.py:1098-1099, 1106-1107`

```python
commontime = np.array([...])                       # per-replicate index arrays
modelvector = np.tile(modelsol[1,:][commontime[0]], len(commontime))[...]
```

Only `commontime[0]` is used, tiled `len(commontime)` times. That is correct only if every
replicate of the treatment was observed at the same times — which the parent `dataclass` does
enforce (`timetreat[i] = self.time` for all `i`), but nothing checks it, and `survdataclass`
deliberately breaks that invariant for its own endpoint. If `commontime` is empty (a treatment
present in the concentration file but absent from the length file) this is an `IndexError`.
An explicit assertion or a clear error message would be cheap.

---

## 2. Architecture / quality

### A1. Dataset classes conflate "raw data", "derived data", and "presentation"
`dataclass` and its subclasses currently hold: the parsed table, per-treatment reshaped views,
fit-ready flattened/cleaned vectors, weighted means with CIs, *and* matplotlib plotting. The
constructors compute all of it eagerly, in a fixed order, with no way to recompute — which is
the direct cause of B4 (subset leaves derived state stale) and of the "you must remember to
call `calc_mean_and_ci()` first" trap noted in the previous review.

Suggested shape: keep the parsed table + a `_rebuild_derived()` that recomputes everything
downstream from it, call it from `__init__` and from `subset()`, and move plotting to free
functions or a thin mixin. The endpoint classes would then differ only in
`_rebuild_derived()`, which is the actual variation between them.

### A2. `subset()` is a 145-line method that reimplements construction by hand
`readin.py:880-1025`

It allocates with `completedataset.__new__` to skip `__init__`, then re-does — in a different
order and with different code — what `__init__` does, plus a nested closure that patches
attributes by name (`if hasattr(ep_new, 'lengthtreat')`, `if hasattr(ep_new, 'reprocumtreat')`,
…). It also rebuilds `labels`/`mask` twice (lines 898-914 and again 993-1001). This is where
B4 lives, and any new derived attribute silently escapes it. It should delegate to the same
construction path as `__init__`.

### A3. `hasattr`-based endpoint presence, three different ways
- `completedataset` signals presence by *conditionally creating* the attribute
  (`if type(lendata) is lengthdataclass: self.lengthdata = ...`), so absence is
  "attribute missing".
- `DEBtox2019models` signals presence by a `None` in a per-dataset list.
- `compile_dataset_dict` (`readin.py:34`) builds a third, dict-based representation of the same
  thing — and appears to be entirely unused (its endpoint codes 1/2/3 don't even match the
  `ENDPOINTS` registry, where survival is 0, not 3).

Pick one. The `ENDPOINTS` registry added in `models.py` is the right idea and should be the
single source of truth for `completedataset` too; `compile_dataset_dict` looks like dead code
that should be deleted before someone uses it and inherits the wrong endpoint codes.

Also `type(x) is concclass` should be `isinstance(...)` — the strict identity check silently
*drops* a user's subclass instead of erroring, which is the worst of both worlds.

### A4. `type(...) is` / silent-drop failure mode in `completedataset.__init__`
Passing a wrong-typed `lendata` (say, a raw ndarray) is not an error: the attribute is simply
never set, `complete_timevec` is built without it, and the model quietly fits without that
endpoint. Raise instead.

### A5. Parameter name → index resolution is string-matching in the hot loop
`models.py:427-480`

`build_dataset_parameters` is called once per dataset per likelihood evaluation, and for each
of the 21 canonical parameters does `np.where(self.full_base_names == pname)` (a full
object-array string comparison) plus a Python loop over the matches. The resolution is
*static* — it depends only on `full_base_names` and `par_dataset_map`, neither of which changes
after construction.

This should be resolved once in `__init__` into an `(ndatasets, 21)` integer index array; then
`build_dataset_parameters` becomes `expanded_parvals[self._par_index[nd]]`. That also converts
the three `RuntimeError` cases from per-call surprises into construction-time validation, which
is where they belong. See O1.

### A6. `base_name()` splits on substrings, not suffixes
`debtox2019api.py:1513-1514`

```python
def base_name(name):
    return name.split("_g")[0].split("_ds")[0]
```

This is a substring split, so any parameter whose name contains `_g` or `_ds` anywhere is
silently truncated (`kd_gut` → `kd`, `f_ds_ratio` → `f`). Since base names drive parameter
identity everywhere (`build_dataset_parameters`, `set_freefix_parameters`, `validation`'s
cross-model matching), a collision here is a quiet mis-assignment. Use a regex anchored at the
end: `re.sub(r'_(g\d+|ds\d+)$', '', name)`.

### A7. `validation()` mutates its caller's `DEBparameters` and never restores it
`debtox2019api.py:434-458`

`fixfree_tox_pars`, `full_list[mask] = val`, `set_fixfree_all(False)`,
`set_freefix_parameters_list(..., True)` all write through to the object the caller passed in.
After `validation()` returns, the caller's parameter set is in a different state than it went
in, with no indication. Given the function's own docstring warns about ordering
("should be called only after the physiological model has already been refitted"), operating on
a `deepcopy` would remove a whole class of ordering foot-guns.

The rest of `validation()` — the base-name matching, the duplicate detection, the log-scale
consistency checks, the propagation-set permutation — is genuinely good, careful code and is
the strongest part of the dataset-parameter feature. It is worth extracting the
"remap free-parameter columns from model A to model B by base name" logic into its own tested
helper, since it is the piece most likely to be needed again (e.g. for multi-study workflows).

### A8. `_startp` is mutated as scratch space throughout profiling
`parspace.py:627-628, 643, 687, 732, 758`

```python
allpars = self._startp        # alias, not a copy
allpars[self.posfree] = pmat_tst
```

`self._startp` is documented as "a copy of the initial parameters of the model" and is the base
vector `_applylog` uses for the *fixed* parameters. Five separate places alias it and write
through. It happens to be safe today (every free slot is rewritten before use, and the pool
workers each get their own copy), but "safe because every writer happens to overwrite every
slot" is a fragile invariant to rely on in a 250-line method. Use `allpars = self._startp.copy()`.

The same pattern is in `log_likelihood` (`models.py:1043`: `DEBallpars[posfree] = theta` writes
into the caller's array before copying it on the next line).

### A9. `parspace.py` core routines are untestable monoliths
Reiterating the prior review because nothing changed and it is now blocking: `run_parspace`
(~470 lines), `_parameter_profile_sub` (~250), `_test_profile` (~110). B7, B12, B13 and B14 are
all bugs that a smaller, named unit would have made obvious. The natural seams are already
marked by the openGUTS comments: "select continuation set for next round", "compute stopping
criteria", "extend profile to bound", "merge profile into cloud".

### A10. Leftover / dead code
- `models.py:390` `self.endpoints = np.array([0,1,2,3])` — set, never read (already
  self-documented as leftover; delete it, it also conflicts with the `ENDPOINTS` codes).
- `models.py:376-379` commented-out log-bound conversion.
- `readin.py:223-225` `_modify_treatmentsnames` writes `self.trrateatsnames` (typo) and is
  never called.
- `readin.py:34-70` `compile_dataset_dict` — unused, and its endpoint codes contradict
  `ENDPOINTS` (see A3).
- `parspace.py:386` `_random_mutations` still ignores its `l_bounds`/`u_bounds` arguments
  (flagged previously, unchanged).
- `parspace.py:1280` `parprofile=[0]*n_cores` — overwritten immediately.
- `parspace.py:1090-1121` the ~30-line commented-out GUTS slow-kinetics block; if it is not
  coming back, delete it (it is referenced by four `SettingParspace.slowkin_*` options that are
  therefore also dead).
- `models.py:1061-1066, 1078-1086` commented-out `newtimeext`/mask code superseded by the
  precomputation.
- `readin.py:441-444` debug `print("mask: ", ...)` left in `survdataclass.add_plotdata` — this
  fires on every plot.

### A11. `efsa_criteria` recomputes what the model already cached
`debtox2019api.py:265-276` rebuilds `newtimeext` with the exact expression that
`DEBtox2019models.__init__` already evaluated into `self.newtimeext[nd]` (that precomputation
was the previous review's optimization item). Use the cached one — otherwise the two can drift.

### A12. Docstring/reality drift
- `models.py` class docstring documents `endpoints` as "currently unused… kept as-is pending
  cleanup" — good honesty, but the right move is to delete it (A10).
- `readin.py:75` still says "quantities needed for the GUTS model fits" (flagged previously).
- `concclass` docstring describes `concarray` as "concentration data reshaped to match the
  number of treatments and unique time points", which is not what B9 shows it actually contains.
- `makerepro_ind` prints "Values that are allowed for case are 0, 1, 2, 3" then validates
  `optcase not in [0,1,2]`, and the `match` statement has no `case _` — an invalid `optcase`
  prints a help message and then silently does nothing, leaving `dataarray_cumulative` as a raw
  copy. Should raise `ValueError`.

---

## 3. Optimization opportunities (lower priority than the above)

### O1. Precompute the parameter index map (biggest single win)
Per A5: `build_dataset_parameters` performs 21 full object-array string comparisons plus Python
loops on *every* likelihood evaluation, of which `run_parspace` does tens to hundreds of
thousands. Replacing it with a precomputed `(ndatasets, 21)` index array makes the whole
function one fancy-index operation. This is pure overhead removal with no behavioural change.

### O2. Precompute the `commontime` selections in `log_likelihood`
`models.py:1098` and `1106` build, per treatment, per endpoint, per call:

```python
commontime = np.array([self.indexcommon_length[nd][j]
                       for j in range(len(...))
                       if self.lengthstruct_list[nd].treatmentsnames[j] == ...conctreatsnames[i]])
```

These depend only on the dataset structure. Hoist into `__init__` next to `newtimeext` (the
same treatment that was already applied there).

### O3. `mp.Pool` per `_applylog` call
`parspace.py:362` creates and tears down a pool on every `_applylog` invocation, i.e. once per
mutation round *and* inside every profiling worker. On Windows (spawn), each pool start
re-imports the package and re-JITs nothing but still costs ~100 ms+ per pool. A single
long-lived pool owned by `PyParspace` (created in `run_parspace`, closed at the end) would
remove that.

Note the interaction with `allow_multiprocessing`: it is set to `False` only inside the
`npars == 1` branch of `_parameter_profile_sub` and unconditionally reset to `True` at the end
of that method — so a `PyParspace` whose profiling ran in a pool comes back with the flag
flipped regardless of what it was. Worth making it a local/context manager instead of instance
state.

### O4. `_epx_window_task` pickles the whole model per window
`models.py:954` dispatches a *bound method* through `pool.starmap`, so every window task
pickles `self` — including all dataset structures — in addition to `modelpars`. For a long
profile with many windows this dominates. Making it a module-level function taking only the
arrays it needs (it already uses no shared state, as the docstring proudly notes) would cut
that.

### O5. `_window_profile` is rebuilt inside the bisection loop
`models.py:814`: `t_list, c_list = self._window_profile(...)` sits *inside* `f(logMF)`, so it is
recomputed on every bisection iteration (up to `max_expand + brentq` iterations) even though it
does not depend on `MF`. Hoist it above the closure.

### O6. `_prune_windows_mask` calls `_window_profile` once per candidate window
`models.py:791-798` — fine, but it only needs the min and max concentration in each window,
which can be had from a cumulative min/max or a sliding-window scan over the profile without
materializing each window's full `(t, c)` pair.

### O7. `np.trapz` is removed in NumPy 2.0
`readin.py:159`. The env is on 1.25.2 so it works today; `np.trapezoid` is the forward-compatible
name (with a `getattr(np, 'trapezoid', np.trapz)` shim if 1.x support is needed).

### O8. `calc_ecx_core` re-solves the control for every effect level
`models.py:687` computes `control = response(0.0, t)` once per time — good. But `response()`
itself re-solves the full ODE from `t=0` for every bisection probe at every `x`. Since the
untreated control is `x`-independent and the bisection over `x` for the *same* `t` walks a
monotone curve, seeding each `x`'s bracket from the previously solved `x` (they are computed in
order) would cut the expansion loop substantially — `calc_dose_response` runs this 49 times per
endpoint by default.

---

## 4. Notes on the dataset-parameter feature (reviewed on its own terms)

The shared / grouped (`_g{i}`) / dataset-specific (`_ds{d}`) expansion in `DEBparameters` and
its resolution in `build_dataset_parameters` is internally consistent, and the resolution rules
are the right ones: an explicit group/dataset match wins, ambiguity is an error, shared is the
fallback, unresolvable is an error. Specific observations:

- **Good:** ambiguity (`len(selected) > 1`) and unresolvable are both hard errors rather than
  silent picks. That is the correct call for a feature where a wrong pick is invisible.
- **Good:** `validation()` matches across parameter sets by base name and *verifies* log-scale
  agreement and uniqueness before remapping the propagation set. That is more careful than the
  rest of the package.
- **Timing:** the three `RuntimeError`s in `build_dataset_parameters` are raised from inside the
  likelihood, i.e. potentially thousands of evaluations into a fit. Nothing about them is
  data-dependent — they should be checked once at `DEBtox2019models.__init__` (this falls out
  naturally from O1).
- **Fragility:** `par_dataset_map` is `np.array(full_owner, dtype=object)` where entries are
  either `-1` or a list. This is a 1-D ragged object array *only because* the global parameters
  always contribute scalar `-1` entries first. A configuration with no global parameters and
  uniformly-sized groups would produce a 2-D object array and change the meaning of
  `self.par_dataset_map[idx]` in `build_dataset_parameters`. Store it as an explicit
  `list` of `int | list[int]`, or normalize every entry to a list (`[-1]` for shared).
- **Coverage gap:** nothing validates that every dataset index in `0..ndatasets-1` is covered by
  exactly one group for a grouped parameter. Declaring `"groups": [[0],[1]]` with
  `ndatasets=3` passes construction and only fails at the first likelihood evaluation for
  dataset 2.
- **Naming:** grouped names are `{p}_g{ig}` where `ig` is the *position in the groups list*, so
  the printed parameter table (`models.py:420-424`) shows `kd_g0`, `kd_g1` with no indication of
  which datasets each covers. Including the dataset membership in that printout would make the
  setup far easier to verify by eye — which matters, since this feature has no MATLAB
  counterpart to cross-check against.

---

## 5. Suggested order of work

1. **B1** (`Lf` typo) and **B2** (in-place state mutation) — model physics, cheap fixes. **DONE!**
2. **B4** (`subset()` stale survival arrays) — silently wrong likelihoods on a live API path. **DONE!**
3. **B3** (`breaktime` ignored) — a documented option that does nothing, on data that needs it. **DONE!**
4. **B5** (broken `input_pars.json`), **B6** (`model.variant`) — trivially reproducible breakage. **DONE!**
5. **B7**, **B12**, **B13** — parspace correctness; then **A9** (split the monoliths) so the
   next round of these is findable.
6. **A5 / O1** — precompute the parameter index map; unblocks both the performance win and
   construction-time validation of the dataset-parameter feature.
7. Everything else.
