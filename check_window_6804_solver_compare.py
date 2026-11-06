# Quick, single-window diagnostic: compare LSODA vs RK45 on the specific
# window that's been failing, across a range of trial multiplication
# factors, without running the full moving-window calc_epx search.
#
# Edit the SETUP section below - it mirrors example_script.py's own model
# construction (no PyParspace/pickle/CI needed for this). The COMPARISON
# section at the bottom should not need to change.

import numpy as np
import pandas as pd
import json

import pydebtox2019.models as mm
import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin


if __name__ == "__main__":

    # =========================================================================
    # SETUP - same pattern as example_script.py's model construction.
    # Edit paths/values below to match your own data.
    # =========================================================================
    parameterpath = "./testingfiles/parameters/"  # <-- edit
    datapath      = "./testingfiles/datafiles/"   # <-- edit

    with open(parameterpath + 'input_pars_tbp3.json') as json_file:
        DEBpars = json.load(json_file)
    debparameters = dt2019.DEBparameters(DEBpars)

    moas   = np.array([1, 0, 0, 0, 0])   # assimilation-only ("10000") - confirm this matches your real run
    feedbs = np.array([0, 0, 0, 0])      # <-- edit to match your model

    Cdata = pd.read_csv(datapath + "Test_Cdata.txt", sep=r"\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(), "", "ug/L")
    Ldata = pd.read_csv(datapath + "Test_Ldata.txt", sep=r"\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())
    Rdata = pd.read_csv(datapath + "Test_Rdata_opt1.txt", sep=r"\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual', optcase=1)
    Sdata = pd.read_csv(datapath + "Test_Sdata.txt", sep=r"\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())

    full_ds, control_ds, _ = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    debparameters.preset_toxlimits(moas, feedbs, ccl)

    model = mm.DEBtox2019models(
        [full_ds], debparameters, moas, feedbs, Tbp=3, breaktime=0, solver='LSODA',  # <-- edit Tbp/breaktime/solver to match your setup
    )
    # model.parvals is already set from your JSON's fitted values at this
    # point (same as input_pars_tbp3.json above) - no separate assignment
    # needed, as long as your JSON encodes the fitted values/islog/bounds
    # per parameter the same way input_pars_tbp3.json does.

    dataset = 0        # <-- edit if needed
    zero_hb = True      # <-- match whatever calc_epx call you were using

    # your exposure profile - edit to load however you normally do
    profile_raw = pd.read_csv("PATH/TO/YOUR/profile.txt", sep=r"\s+", header=None).to_numpy(dtype=float)  # <-- edit
    exposure_time = profile_raw[:, 0]
    exposure_conc = profile_raw[:, 1]

    window_start = 7899.021  # the failing window (moved here after reverting
                              # to the unprotected original code - profile
                              # has a peak of 0.5852 starting at 7899.396)
    Twin = 21.0              # your Twin
    endpoints = (0, 1, 2)    # <-- edit to match your real calc_epx call (0=survival,1=length,2=repro)
    X = (10, 50)             # <-- edit to match your real calc_epx call
    MF_bounds = (1e-3, 1e3)  # <-- matches what you reported
    max_expand = 60          # <-- edit if you changed it from the default
    xtol = 1e-5              # <-- matches what you reported
    plateau_tol = 1e-6       # <-- edit if you changed it from the default

    # =========================================================================
    # COMPARISON - replays _epx_window_task/_bisect_log EXACTLY (not a hand-
    # picked MF grid), with every (MF, effect) pair it evaluates logged. This
    # is the only way to see the actual sequence brentq homes in on, since
    # that sequence is data-dependent and can land on values an external
    # grid - however dense - would skip right over.
    # =========================================================================
    t_list, c_list = model._window_profile(
        np.asarray(exposure_time, dtype=float), np.asarray(exposure_conc, dtype=float),
        window_start, Twin,
    )
    modelpars = model.resolve_dataset_parameters(dataset, zero_hb=zero_hb)
    timeext_dense = np.linspace(0.0, Twin, 50)

    print("window_start=%.4f  Twin=%.4f  n_profile_points=%d" % (window_start, Twin, len(t_list)))
    print("first few (t, c) pairs in the window profile (sanity check vs. the")
    print("described peak: rise to 0.5852 starting at 7899.396, i.e. t=%.4f in window-local time):"
          % (7899.396 - window_start))
    for tt, cc in list(zip(t_list, c_list))[:8]:
        print("  t=%10.4f  c=%10.6f" % (tt, cc))

    def run_bisection_traced(ep, x, solver_name):
        si = model.ENDPOINT_STATE_IDX[ep]
        control = model.calc_model(np.zeros(2), np.array([0.0, Twin]), modelpars,
                                    model.moa, model.feedb, timeext=timeext_dense)[:, -1]
        control_val = control[si]
        target = x / 100.0
        trace = []

        def f(logMF):
            MF = 10 ** logMF
            try:
                sol = model.calc_model(MF * c_list, t_list, modelpars, model.moa, model.feedb,
                                        timeext=timeext_dense)
                finite = bool(np.all(np.isfinite(sol)))
                effect = 1.0 - sol[si, -1] / control_val
                val = effect - target
                trace.append((MF, val, finite, None))
                return val
            except Exception as e:
                trace.append((MF, None, False, "%s: %s" % (type(e).__name__, e)))
                raise

        try:
            root = model._bisect_log(f, MF_bounds[0], MF_bounds[1], max_expand, xtol,
                                      increasing=True, plateau_tol=plateau_tol)
            outcome = "root=%.6g" % root if np.isfinite(root) else "NaN (no valid root found)"
        except Exception as e:
            outcome = "RAISED %s: %s" % (type(e).__name__, e)

        print("\n[%s] endpoint=%d X=%d%%  control_val=%.6g  -> %s  (%d f() evaluations)"
              % (solver_name, ep, x, control_val, outcome, len(trace)))
        for MF, val, finite, err in trace:
            if err is not None:
                print("    MF=%12.6f  EXCEPTION: %s" % (MF, err))
            elif not finite:
                print("    MF=%12.6f  SILENT NON-FINITE result (f() returned, but sol had NaN/inf)" % MF)
            else:
                print("    MF=%12.6f  effect-target=%12.6g" % (MF, val))

    original_solver = model.solver
    try:
        for solver_name in ('LSODA', 'RK45'):
            model.solver = solver_name
            for ep in endpoints:
                for x in X:
                    run_bisection_traced(ep, x, solver_name)
    finally:
        model.solver = original_solver  # restore, in case the model object is reused afterward

    print("\nDONE")
