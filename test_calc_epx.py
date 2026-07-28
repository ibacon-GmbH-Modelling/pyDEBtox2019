# testing script for calc_epx with a realistic, high-resolution exposure profile

import json
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pydebtox2019.models as mm
import pydebtox2019.parspace as ps
import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin


if __name__ == "__main__":

    with open('input_pars_tbp3.json') as json_file:
        DEBpars = json.load(json_file)

    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False)

    # growth + reproduction mode of action, so length (in addition to
    # reproduction and survival) also responds to damage
    moas = np.array([0, 0, 1, 1, 0])
    feedbs = np.array([0, 0, 0, 0])

    Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(), "", "ug/L")

    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())

    Rdata = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual', optcase=1)

    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())

    full_ds, control_ds, ph = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')

    debparameters.preset_toxlimits(moas, feedbs, ccl)

    debmodeltest = mm.DEBtox2019models(
        [full_ds], debparameters, moas, feedbs, Tbp=0, breaktime=0, solver='LSODA'
    )

    # --- a realistic, high-resolution exposure profile (hourly, ~1 year) ---
    # built as a concclass instance (focus=True: the file has no treatment
    # header row, all rows are actual time/concentration data), so it can
    # also be sanity-checked with plot_exposure().
    profile_raw = pd.read_csv("apple_R1_pond.txt", sep="\s+", header=None).to_numpy(dtype=float)
    epcl = readin.concclass(profile_raw, "apple_R1_pond", "ug/L", focus=True)
    epcl.plot_exposure()
    print("Profile duration: %.2f days, %d points" % (
        epcl.timetr[-1] - epcl.timetr[0], len(epcl.timetr)
    ))

    # EPx/LPx with the moving time window advanced 1 day at a time (Tstep=1.0).
    # Window starts now also range *before* the profile's first time point
    # (zero-padded there), so the sliding window also probes the initial
    # rise of the exposure, not only its (already zero-padded) tail-off.
    #
    # prune_win=True skips window positions that provably cannot be the
    # worst case (pyDEBtox2019 equivalent of prune_windows.m) before running
    # any bisection on them - safe here since feedbs are all off. Only X=50
    # and a single window length are used to keep the runtime of this demo
    # reasonable: with Tstep=1 on a full-year hourly profile, every
    # (endpoint, X) combination scans ~390 window positions without pruning.
    res = dt2019.calc_epx(
        debmodeltest,
        epcl,
        Twin=21,
        X=[50],
        dataset=0,
        Tstep=1.0,
        prune_win=True,
        verbose=True,
    )

    print("\nRaw results dict:")
    for k, v in res.items():
        print(k, v)

    # --- worst-case window time, and the two diagnostic plots ---
    # results[endpoint][x] is the EPx/LPx value (as before); the window
    # start time at which that worst case was found is now available under
    # results[endpoint]['{x}_worst_time'], and the full per-window critical
    # multiplication-factor curve under results[endpoint]['{x}_curve']
    # (used internally by plot_epx_results, but also directly inspectable).
    for endpoint in ('length', 'reproduction', 'survival'):
        print("Worst-case window start for %s EP/LP50: %.4g" % (
            endpoint, res[endpoint]['50_worst_time'][0]
        ))

    # Figure 1: critical multiplication factor as a function of the window
    # start time along the profile (worst case marked).
    # Figure 2: for that worst-case window, the (scaled) exposure on the
    # left, and the endpoint trajectory vs. the unexposed control on the right.
    fig_mf, fig_window = dt2019.plot_epx_results(
        debmodeltest, epcl, res, endpoint='reproduction', x=50, dataset=0, twin_index=0,
    )
    plt.show()

    # --- single-endpoint selection, to save computation time ---
    # a bare endpoint name or code is accepted (no need to wrap it in a list)
    res_single = dt2019.calc_epx(
        debmodeltest, epcl, Twin=21, X=[50],
        endpoints='reproduction', dataset=0, Tstep=1.0, prune_win=True, verbose=False,
    )
    print("\nSingle endpoint by name ('reproduction'):")
    print(res_single)

    # --- subset of endpoints (2 out of 3) ---
    res_subset = dt2019.calc_epx(
        debmodeltest,
        epcl,
        Twin=21,
        X=[50],
        endpoints=['length', 'reproduction'],
        dataset=0,
        Tstep=1.0,
        prune_win=True,
        verbose=True,
    )
    print("\nSubset endpoints ['length', 'reproduction']:")
    print(list(res_subset.keys()))

    # --- CI smoke test, bypassing the slow full parameter-space search ---
    # (in a real workflow, parspace.propagationset comes from parspace.run_parspace();
    # a small propagation set + multicore is used here to keep this demo's
    # runtime reasonable)
    parspace = ps.PyParspace(ps.SettingParspace(0, 1), debmodeltest)
    base = debmodeltest.parvals[debmodeltest.posfree]
    rng = np.random.default_rng(0)
    parspace.propagationset = base + rng.normal(scale=0.02, size=(3, len(base)))

    res_ci = dt2019.calc_epx(
        debmodeltest,
        epcl,
        Twin=21,
        X=[50],
        endpoints='reproduction',
        dataset=0,
        Tstep=1.0,
        prune_win=True,
        ci=True,
        parspace=parspace,
        multicore=True,
        verbose=False,
    )
    print("\nCI results (reproduction):")
    print(res_ci['reproduction'])

    # =========================================================================
    # Parallelization (multicore) and window pruning: demonstrated on a
    # SUBSET of the apple pond profile (first 180 days), so all four
    # variants below can be run back to back in a reasonable time.
    #
    # - prune_win=True (pyDEBtox2019 equivalent of prune_windows.m): skips
    #   window positions that provably cannot be the worst case before
    #   running any bisection on them at all.
    # - multicore=True: distributes the per-window bisections themselves
    #   across worker processes (unlike BYOM's MATLAB implementation, which
    #   cannot parallelize across windows because it stores the exposure
    #   scenario in a global variable - our implementation has no such
    #   shared/global state, so this works even though MATLAB's could not).
    #
    # All four variants must (and do) agree on the resulting EP50 and
    # worst-case time; only the runtime differs.
    # =========================================================================
    subset_mask = profile_raw[:, 0] <= 180.0
    etime_sub = profile_raw[subset_mask, 0]
    econc_sub = profile_raw[subset_mask, 1]
    print("\nSubset profile for parallel/pruning demo: %.0f days, %d points" % (
        etime_sub[-1] - etime_sub[0], len(etime_sub)
    ))

    subset_kwargs = dict(
        Twin=21, X=[50], endpoints='reproduction', dataset=0, Tstep=1.0, verbose=False,
    )

    t0 = time.time()
    res_serial = dt2019.calc_epx(
        debmodeltest, (etime_sub, econc_sub), multicore=False, prune_win=False, **subset_kwargs
    )
    print("serial,   no pruning : %6.2fs, EP50=%.6g at t=%.4g" % (
        time.time() - t0, res_serial['reproduction'][50][0],
        res_serial['reproduction']['50_worst_time'][0],
    ))

    t0 = time.time()
    res_multi = dt2019.calc_epx(
        debmodeltest, (etime_sub, econc_sub), multicore=True, prune_win=False, **subset_kwargs
    )
    print("multicore, no pruning: %6.2fs, EP50=%.6g at t=%.4g" % (
        time.time() - t0, res_multi['reproduction'][50][0],
        res_multi['reproduction']['50_worst_time'][0],
    ))

    t0 = time.time()
    res_prune = dt2019.calc_epx(
        debmodeltest, (etime_sub, econc_sub), multicore=False, prune_win=True, **subset_kwargs
    )
    print("serial,   pruned     : %6.2fs, EP50=%.6g at t=%.4g" % (
        time.time() - t0, res_prune['reproduction'][50][0],
        res_prune['reproduction']['50_worst_time'][0],
    ))

    t0 = time.time()
    res_both = dt2019.calc_epx(
        debmodeltest, (etime_sub, econc_sub), multicore=True, prune_win=True, **subset_kwargs
    )
    print("multicore, pruned    : %6.2fs, EP50=%.6g at t=%.4g" % (
        time.time() - t0, res_both['reproduction'][50][0],
        res_both['reproduction']['50_worst_time'][0],
    ))
