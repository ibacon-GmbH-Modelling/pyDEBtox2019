# testing script for PyParspace._plot_samples(savefig=True)
#
# Regression test for review item B6 ("savefig=True raises AttributeError -
# model.variant does not exist"): _plot_samples built the saved filename as
# figbasename + "_" + self.model.variant + extension, but DEBtox2019models
# has no `variant` attribute anywhere in the package (a leftover from the
# GUTS codebase this was ported from), so every savefig=True call path in
# run_parspace/replot_results raised AttributeError before ever writing a
# file. The fix drops the nonexistent attribute from the filename.
#
# Running an actual parameter-space search just to reach _plot_samples would
# be slow and is not what's under test here, so this builds a minimal,
# synthetic (but shape/ordering-valid) coll_all directly and calls
# _plot_samples(savefig=True) on it - exercising exactly the code path that
# used to crash, for both the batchmode=True and batchmode=False branches
# (both had the bug).

import json
import os
import tempfile

import numpy as np

import pydebtox2019.models as mm
import pydebtox2019.parspace as ps
import pydebtox2019.debtox2019api as dt2019

from test_validation_flow import _read_raw_components


if __name__ == "__main__":

    with open('input_pars_tbp3.json') as json_file:
        DEBpars_json = json.load(json_file)
    debparameters = dt2019.DEBparameters(DEBpars_json)
    debparameters.set_fixfree_all(isfree=False)
    debparameters.set_freefix_parameters_list(["hb", "kd"], isfree=True)

    moas = np.array([0, 0, 0, 1, 0], dtype=float)
    feedbs = np.array([0, 0, 0, 0], dtype=float)

    ccl, lcl, rcl, scl = _read_raw_components()
    full_ds, _, _ = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')

    debmodel = mm.DEBtox2019models([full_ds], debparameters, moas, feedbs, Tbp=0, solver='LSODA')

    parspace = ps.PyParspace(ps.SettingParspace(rough=1, profile=0), debmodel)
    assert parspace.npars == 2, "expected 2 free parameters (hb, kd), got %d" % parspace.npars

    # Synthetic coll_all: (n, npars+1), sorted so the first row is the best
    # (lowest) likelihood value and later rows are progressively worse -
    # exactly the invariant _plot_samples relies on for its argwhere(...).max()
    # calls, without needing a real (slow) parameter-space search.
    rng = np.random.default_rng(0)
    n = 30
    mll = -50.0
    coll_all = np.zeros((n, parspace.npars + 1))
    coll_all[:, :-1] = rng.uniform(0.5, 1.5, size=(n, parspace.npars))
    coll_all[:, -1] = mll + np.linspace(0.0, 20.0, n)
    parspace.coll_all = coll_all

    tmpdir = tempfile.mkdtemp(prefix="pydebtox2019_test_")
    figbase = os.path.join(tmpdir, "fit")

    for batchmode in (True, False):
        outfile = figbase + ".png"
        if os.path.exists(outfile):
            os.remove(outfile)
        parspace._plot_samples(batchmode=batchmode, savefig=True, figbasename=figbase, extension=".png")
        assert os.path.exists(outfile), (
            "batchmode=%s: savefig=True did not produce %s - B6 not fixed" % (batchmode, outfile))
        assert os.path.getsize(outfile) > 0, "batchmode=%s: %s is empty" % (batchmode, outfile)
        print("batchmode=%-5s savefig=True -> wrote %s (%d bytes)" % (
            batchmode, outfile, os.path.getsize(outfile)))

    print("\nB6 fixed: savefig=True no longer raises AttributeError on model.variant,")
    print("for both the batchmode=True and batchmode=False code paths.")
