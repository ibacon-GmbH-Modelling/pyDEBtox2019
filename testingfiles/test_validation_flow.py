# testing script for the full calibration -> validation flow
#
# Exercises DEBparameters -> DEBtox2019models -> PyParspace -> validation()
# end-to-end, without running the (slow) full parameter-space search: the
# "calibration" propagation set is faked with a small random perturbation
# around the point estimate - the same shortcut used for the CI smoke test
# in test_calc_ecx.py (see around its call to ps.PyParspace there).

import matplotlib
#matplotlib.use("Agg")  # keep this script runnable headless / non-interactively

import json
import numpy as np
import pandas as pd

import pydebtox2019.models as mm
import pydebtox2019.parspace as ps
import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin


def _read_raw_components():
    """Read the shared Test_*data.txt fixtures into the four raw endpoint structures."""
    Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(), "", "ug/L")

    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())

    Rdata = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual', optcase=1)

    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())

    return ccl, lcl, rcl, scl


def _read_full_dataset():
    """Read the shared Test_*data.txt fixtures into a fresh completedataset."""
    ccl, lcl, rcl, scl = _read_raw_components()
    full_ds, _, _ = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    return full_ds, ccl


if __name__ == "__main__":

    # growth + reproduction mode of action, so length (in addition to
    # reproduction and survival) also responds to damage
    moas = np.array([0, 0, 1, 1, 0])
    feedbs = np.array([0, 0, 0, 0])

    # ------------------------------------------------------------------
    # 1) "Calibration" step, following the same three-stage sequence a
    #    real BYOM-style calibration would use (see e.g.
    #    testing_script_grprepro.py):
    #      1a) fit the background mortality parameter hb on survival data
    #          from the controls only;
    #      1b) fit the physiological model parameters (Lp, Lm, rB, Rm) on
    #          the full control dataset, with hb fixed at its fitted value;
    #      1c) fit the toxicity parameters (kd, bb, zb, bs, zs) on the
    #          full dataset, with all physiology (including hb) fixed.
    #    The actual (slow) optimizations are not run here - every
    #    PyParspace.run_parspace() call is left commented out, and each
    #    parameter block is simply fixed at its current point estimate
    #    once its stage is "done", exactly as a real run would fix it at
    #    its fitted value. Only stage 1c's result (the toxicity
    #    calibration) is actually needed downstream, so only its
    #    propagation set is faked (see below).
    # ------------------------------------------------------------------
    with open('input_pars_tbp3.json') as json_file:
        DEBpars_calib = json.load(json_file)
    debparameters_tox = dt2019.DEBparameters(DEBpars_calib)

    ccl_calib, lcl_calib, rcl_calib, scl_calib = _read_raw_components()
    full_ds_calib, control_ds_calib, _ = dt2019.build_dataset_variants(
        ccl_calib, lcl_calib, rcl_calib, scl_calib, control_type='solvent'
    )
    _, hbonly_calib, _ = dt2019.build_dataset_variants(
        ccl=ccl_calib, lcl=None, rcl=None, scl=scl_calib, control_type='solvent'
    )

    debparameters_tox.preset_toxlimits(moas, feedbs, ccl_calib)

    # --- 1a) fit hb on the survival-only control data ---
    debparameters_tox.set_free_onlyone('hb', isfree=True)
    debmodel_hb = mm.DEBtox2019models(
        [hbonly_calib], debparameters_tox, moas, feedbs, Tbp=0, solver='LSODA'
    )
    parspace_hb = ps.PyParspace(ps.SettingParspace(0, 1), debmodel_hb)
    # parspace_hb.run_parspace()  # <- the actual (slow) fit; intentionally not run
    debparameters_tox.full_list = parspace_hb.model.parvals  # carry the (pretend-)fitted values forward
    debparameters_tox.set_freefix_parameters("hb", isfree=False)  # fix hb at its (pretend-)fitted value

    # --- 1b) fit the physiological parameters on the control dataset ---
    debparameters_tox.set_freefix_parameters_list(["lp", "lm", "rb", "rm"], isfree=True)
    debmodel_ctrl = mm.DEBtox2019models(
        [control_ds_calib], debparameters_tox, moas, feedbs, Tbp=0, solver='LSODA'
    )
    parspace_ctrl = ps.PyParspace(ps.SettingParspace(0, 1), debmodel_ctrl)
    # parspace_ctrl.run_parspace()  # <- the actual (slow) fit; intentionally not run
    debparameters_tox.full_list = parspace_ctrl.model.parvals
    debparameters_tox.set_freefix_parameters_list(["lp", "lm", "rb", "rm"], isfree=False)

    # --- 1c) fit the toxicity parameters on the full dataset, with all
    # physiology fixed - this is "the previous calibration of the
    # toxicity parameters" that validation() later reuses ---
    debparameters_tox.fixfree_physio_pars(isfree=False)
    debparameters_tox.fixfree_tox_pars(isfree=True)

    debmodel_tox = mm.DEBtox2019models(
        [full_ds_calib], debparameters_tox, moas, feedbs, Tbp=0, breaktime=0, solver='LSODA'
    )

    parspace_tox = ps.PyParspace(ps.SettingParspace(0, 1), debmodel_tox)
    # parspace_tox.run_parspace()  # <- the actual (slow) fit; intentionally not run

    # --- shortcut: fake the propagation set instead of running the full,
    # slow parameter-space search (PyParspace.run_parspace) ---
    base = debmodel_tox.parvals[debmodel_tox.posfree]
    rng = np.random.default_rng(0)
    parspace_tox.propagationset = base + rng.normal(scale=0.02, size=(5, len(base)))

    print("Faked propagation set for the tox calibration:")
    print("  free parameters:", debmodel_tox.parnames[debmodel_tox.posfree])
    print("  propagation set shape:", parspace_tox.propagationset.shape)

    # ------------------------------------------------------------------
    # 2) "New dataset" step: an independently-parsed DEBparameters/dataset
    #    pair, standing in for a second experiment whose physiology has
    #    already been refit (here just nudged a bit, then fixed) and whose
    #    toxicity parameters are still at placeholder JSON values, to be
    #    overwritten by validation() using the calibration above.
    # ------------------------------------------------------------------
    with open('input_pars_tbp3.json') as json_file:
        DEBpars_val = json.load(json_file)
    debparameters_val = dt2019.DEBparameters(DEBpars_val)

    # pretend the physiological model was refit on the new dataset, giving
    # slightly different values, and is now fixed at that refit
    for pname, factor in [("lp", 1.05), ("lm", 0.95), ("rb", 1.1), ("rm", 0.9)]:
        mask = debparameters_val.full_base_names == pname
        debparameters_val.full_list[mask] *= factor
    debparameters_val.set_freefix_parameters_list(["lp", "lm", "rb", "rm"], isfree=False)

    full_ds_val, _ = _read_full_dataset()

    # ------------------------------------------------------------------
    # 3) Run the validation flow: overwrites the tox parameters on
    #    debparameters_val with the values from parspace_tox (matched by
    #    base name), remaps the fake propagation set onto the new
    #    parameter ordering, plots the result and prints the EFSA
    #    R2/NRMSE criteria.
    # ------------------------------------------------------------------
    dt2019.validation(
        full_ds_val, debparameters_val, parspace_tox, CI=True, multicore=True, wmeans=False
    )

    # ------------------------------------------------------------------
    # sanity checks
    # ------------------------------------------------------------------
    tox_names = ("kd", "bb", "zb", "bs", "zs")
    for name in tox_names:
        mask = debparameters_val.full_base_names == name
        assert mask.sum() == 1, "expected exactly one '%s' in the new dataset" % name
        assert np.isfinite(debparameters_val.full_list[mask][0]), "'%s' was not updated to a finite value" % name

    assert np.all(np.isin(list(tox_names), debparameters_val.full_base_names[debparameters_val.full_isfree])), \
        "not all tox parameters ended up free on the new dataset after validation()"

    print("\nValidation flow executed successfully.")
