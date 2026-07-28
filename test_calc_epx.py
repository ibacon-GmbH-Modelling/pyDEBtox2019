# testing script for calc_epx with a realistic, high-resolution exposure profile

import json
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
    # Only X=50 and a single window length are used here to keep the runtime
    # of this demo reasonable: with Tstep=1 on a full-year hourly profile,
    # every (endpoint, X) combination already scans ~390 window positions,
    # each requiring its own bisection on the multiplication factor.
    res = dt2019.calc_epx(
        debmodeltest,
        epcl,
        Twin=21,
        X=[50],
        dataset=0,
        Tstep=1.0,
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
        endpoints='reproduction', dataset=0, Tstep=1.0, verbose=False,
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
        ci=True,
        parspace=parspace,
        multicore=True,
        verbose=False,
    )
    print("\nCI results (reproduction):")
    print(res_ci['reproduction'])
