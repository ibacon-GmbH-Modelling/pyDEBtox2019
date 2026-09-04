# testing script for calc_dose_response

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # --- dose-response curve for a single endpoint ---
    res_single = dt2019.calc_dose_response(
        debmodeltest, Tend=21, endpoints='reproduction', dataset=0,
    )
    print("Single-endpoint keys:", list(res_single.keys()))
    print("x (effect levels, %):", res_single['reproduction']['x'])
    print("conc (ECx):", res_single['reproduction']['conc'])
    print("response (% of control):", res_single['reproduction']['response'])

    # --- dose-response curves for all active endpoints: one subplot each ---
    res_all = dt2019.calc_dose_response(debmodeltest, Tend=21, dataset=0)
    print("\nAll-endpoints keys:", list(res_all.keys()))

    # --- CI band on the dose-response curve, bypassing the slow full
    # parameter-space search (a small propagation set is used here to keep
    # this demo's runtime reasonable) ---
    parspace = ps.PyParspace(ps.SettingParspace(0, 1), debmodeltest)
    base = debmodeltest.parvals[debmodeltest.posfree]
    rng = np.random.default_rng(0)
    parspace.propagationset = base + rng.normal(scale=0.02, size=(5, len(base)))

    res_ci = dt2019.calc_dose_response(
        debmodeltest, Tend=21, endpoints='reproduction', dataset=0,
        n_points=25, ci=True, parspace=parspace, multicore=True,
    )
    print("\nCI band keys:", list(res_ci['reproduction'].keys()))

    # --- plateau_tol: length has a hard floor in the model (growth is not
    # allowed to shrink below half the starting length), so its dose-response
    # curve plateaus and never reaches high effect levels such as 99%.
    # Without plateau_tol, every such (unreachable) x exhausts the full
    # concentration-bracket expansion budget (max_expand) before giving up;
    # plateau_tol detects the plateau and gives up early instead. Both give
    # the same curve (NaN beyond the plateau) - only the runtime differs.
    t0 = time.time()
    res_length_slow = dt2019.calc_dose_response(
        debmodeltest, Tend=21, endpoints='length', dataset=0,
        n_points=25, plateau_tol=None, plot=False,
    )
    t_slow = time.time() - t0

    t0 = time.time()
    res_length_fast = dt2019.calc_dose_response(
        debmodeltest, Tend=21, endpoints='length', dataset=0,
        n_points=25, plateau_tol=1e-6, plot=True,
    )
    t_fast = time.time() - t0

    n_nan = np.sum(~np.isfinite(res_length_fast['length']['conc']))
    print("\nplateau_tol demo (endpoint 'length', which cannot reach high effect levels):")
    print("  plateau_tol=None (exhausts max_expand every time): %.2fs" % t_slow)
    print("  plateau_tol=1e-6 (stops early once plateaued):     %.2fs (%.1fx faster)" % (
        t_fast, t_slow / t_fast
    ))
    print("  %d / %d effect levels are unreachable (NaN) for length at t=21" % (
        n_nan, len(res_length_fast['length']['x'])
    ))
    print("  concentrations match between the two settings:", np.allclose(
        res_length_slow['length']['conc'], res_length_fast['length']['conc'], equal_nan=True
    ))

    plt.show()
