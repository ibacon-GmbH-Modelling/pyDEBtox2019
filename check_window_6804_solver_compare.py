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

    moas   = np.array([0, 0, 1, 1, 0])   # <-- edit to match your model
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
        [full_ds], debparameters, moas, feedbs, Tbp=3, breaktime=1, solver='LSODA',  # <-- edit Tbp/breaktime/solver to match your setup
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
    # We don't know the exact before/after-gap MF bracket for this window
    # yet, so sweep broadly (log-spaced) instead of needing it pinned down.
    trial_MFs = tuple(sorted(set(
        (1, 5, 10, 20, 29, 40, 50, 60, 69, 100, 300, 1000)
        + tuple(np.round(np.logspace(0, 3, 120), 4))
    )))

    # =========================================================================
    # COMPARISON - should not need editing below this line
    # =========================================================================
    t_list, c_list = model._window_profile(
        np.asarray(exposure_time, dtype=float), np.asarray(exposure_conc, dtype=float),
        window_start, Twin,
    )
    modelpars = model.resolve_dataset_parameters(dataset, zero_hb=zero_hb)
    timeext = np.linspace(0.0, Twin, 50)

    print("window_start=%.4f  Twin=%.4f  n_profile_points=%d" % (window_start, Twin, len(t_list)))
    print("first few (t, c) pairs in the window profile (sanity check vs. the")
    print("described peak: rise to 0.5852 starting at 7899.396, i.e. t=%.4f in window-local time):"
          % (7899.396 - window_start))
    for tt, cc in list(zip(t_list, c_list))[:8]:
        print("  t=%10.4f  c=%10.6f" % (tt, cc))
    print("%-10s %-10s %-10s" % ("MF", "LSODA", "RK45"))

    original_solver = model.solver
    try:
        for MF in trial_MFs:
            model.solver = 'LSODA'
            try:
                res = model.calc_model(MF * c_list, t_list, modelpars, model.moa, model.feedb, timeext=timeext)
                if np.all(np.isfinite(res)):
                    lsoda_ok = "OK (final=%s)" % np.array2string(res[:, -1], precision=4)
                else:
                    lsoda_ok = "SILENT NON-FINITE (no exception raised, but result has NaN/inf)"
            except Exception as e:
                lsoda_ok = "FAIL (%s: %s)" % (type(e).__name__, e)

            model.solver = 'RK45'
            try:
                res = model.calc_model(MF * c_list, t_list, modelpars, model.moa, model.feedb, timeext=timeext)
                if np.all(np.isfinite(res)):
                    rk45_ok = "OK (final=%s)" % np.array2string(res[:, -1], precision=4)
                else:
                    rk45_ok = "SILENT NON-FINITE (no exception raised, but result has NaN/inf)"
            except Exception as e:
                rk45_ok = "FAIL (%s: %s)" % (type(e).__name__, e)

            print("%-10g %-10s %-10s" % (MF, lsoda_ok, rk45_ok))
    finally:
        model.solver = original_solver  # restore, in case the model object is reused afterward

    print("\nDONE")
