# testing script for calc_ecx

import json
import numpy as np
import pandas as pd

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

    # ECx/LCx at the initial (unfitted) parameter values, no CI.
    # All three active endpoints are computed (the default: endpoints=None
    # falls back to model.active_endpoints[dataset]).
    res = dt2019.calc_ecx(
        debmodeltest,
        Tend=[7, 21],
        X=[10, 50],
        dataset=0,
        verbose=True,
    )

    print("\nRaw results dict:")
    for k, v in res.items():
        print(k, v)

    # --- single-endpoint selection, to save computation time ---
    # a bare endpoint name or code is accepted (no need to wrap it in a list)
    res_single_name = dt2019.calc_ecx(
        debmodeltest, Tend=[21], X=[50], endpoints='reproduction', dataset=0, verbose=False
    )
    print("\nSingle endpoint by name ('reproduction'):")
    print(res_single_name)

    res_single_code = dt2019.calc_ecx(
        debmodeltest, Tend=[21], X=[50], endpoints=0, dataset=0, verbose=False
    )
    print("\nSingle endpoint by code (0 = survival):")
    print(res_single_code)

    # --- subset of endpoints (2 out of 3) ---
    res_subset = dt2019.calc_ecx(
        debmodeltest,
        Tend=[7, 21],
        X=[10, 50],
        endpoints=['length', 'reproduction'],
        dataset=0,
        verbose=True,
    )
    print("\nSubset endpoints ['length', 'reproduction']:")
    print(list(res_subset.keys()))

    # --- CI smoke test, bypassing the slow full parameter-space search ---
    # (in a real workflow, parspace.propagationset comes from parspace.run_parspace())
    parspace = ps.PyParspace(ps.SettingParspace(0, 1), debmodeltest)
    base = debmodeltest.parvals[debmodeltest.posfree]
    rng = np.random.default_rng(0)
    parspace.propagationset = base + rng.normal(scale=0.02, size=(5, len(base)))

    res_ci = dt2019.calc_ecx(
        debmodeltest,
        Tend=[7, 21],
        X=[10, 50],
        dataset=0,
        ci=True,
        parspace=parspace,
        multicore=False,
        verbose=False,
    )
    print("\nCI results (reproduction):")
    print(res_ci['reproduction'])
    print("\nCI results (survival):")
    print(res_ci['survival'])
