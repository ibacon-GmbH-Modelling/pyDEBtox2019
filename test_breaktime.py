# testing script for calc_model's breaktime x Tbp handling
#
# Regression test for review item B3 ("breaktime=True is silently ignored
# whenever Tbp == 0"): calc_model used to shortcut to a single, unsegmented
# solve whenever Tbp == 0, without ever consulting self.breaktime. After the
# fix, calc_model consults breaktime independently of Tbp, in all four
# combinations:
#   (breaktime=False, Tbp=0), (breaktime=True, Tbp=0),
#   (breaktime=False, Tbp>0), (breaktime=True, Tbp>0)
#
# breaktime doesn't change the physics (it's the same ODE, just solved
# piecewise instead of continuously - matching BYOM's option for speeding up
# very pulsed profiles), so the primary check here is *mechanical*: does
# calc_model actually call the solver once per real segment when
# breaktime=True, instead of once overall? That's verified by counting
# calls to calc_DEBresults. As a secondary check, the two ways of solving
# should agree closely, and zero-length segments (the duplicated t=5/t=15
# rows in Test_Cdata.txt, encoding an instantaneous concentration step)
# must not raise - that's the second bug fixed alongside B3.

import json

import numpy as np

import pydebtox2019.models as mm
import pydebtox2019.debtox2019api as dt2019

from test_validation_flow import _read_raw_components
from test_debsolvererror import _dummy_pars


def _count_calc_DEBresults_calls(fn):
    """Wrap calc_DEBresults so calls made *through calc_model* (which looks
    it up as a plain module-global at call time) can be counted, then patch
    it into the models module in place of the original."""
    count = {"n": 0}

    def _wrapped(*args, **kwargs):
        count["n"] += 1
        return fn(*args, **kwargs)

    return count, _wrapped


if __name__ == "__main__":

    moas = np.array([0, 0, 1, 1, 0], dtype=float)
    feedbs = np.array([0, 0, 0, 0], dtype=float)
    DEBpars = _dummy_pars()

    with open('input_pars_tbp3.json') as json_file:
        DEBpars_json = json.load(json_file)
    debparameters = dt2019.DEBparameters(DEBpars_json)

    ccl, lcl, rcl, scl = _read_raw_components()
    full_ds, _, _ = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    # build_dataset_variants' first return is always the *unfiltered* dataset
    # (all 5 treatments), which is what we want here to reach treatment "1".
    assert list(np.array(full_ds.concdata.conctreatsnames).astype(float)) == [0.0, 0.1, 1.0, 2.0, 3.0]

    idx_pulse = 2      # treatment "1": has the duplicated-time C pulse
    idx_control = 0    # treatment "0": C is constant zero throughout

    # Test_Cdata.txt's raw time vector is [0, 5, 5, 15, 15, 21]: 3 genuine
    # intervals ([0,5], [5,15], [15,21]) plus 2 zero-length duplicate pairs
    # used to encode the instantaneous concentration steps.
    timetr_full = np.array(full_ds.concdata.timetr, dtype=float)
    n_real_segments = int(np.sum(np.diff(timetr_full) > 0))
    n_zero_segments = int(np.sum(np.diff(timetr_full) == 0))
    print("Test_Cdata.txt raw time vector:", timetr_full)
    print("-> %d real segment(s), %d zero-length (duplicated-time) segment(s)" % (
        n_real_segments, n_zero_segments))
    assert n_real_segments == 3 and n_zero_segments == 2

    orig_calc_DEBresults = mm.calc_DEBresults
    call_count, wrapped = _count_calc_DEBresults_calls(orig_calc_DEBresults)
    mm.calc_DEBresults = wrapped

    results = {}
    try:
        for Tbp in (0, 3):
            for breaktime in (False, True):
                debmodel = mm.DEBtox2019models(
                    [full_ds], debparameters, moas, feedbs, Tbp=Tbp, breaktime=breaktime, solver='LSODA')
                cstruct = debmodel.concstruct_list[0]
                timeext = debmodel.newtimeext[0]

                for label, idx in (('pulse', idx_pulse), ('control', idx_control)):
                    call_count["n"] = 0
                    sol = debmodel.calc_model(
                        cstruct.concarraytr[idx], cstruct.timetr, DEBpars, moas, feedbs, timeext)
                    n_calls = call_count["n"]

                    assert sol.shape == (4, len(timeext)), (
                        "Tbp=%s breaktime=%s %s: unexpected shape %s" % (Tbp, breaktime, label, sol.shape))
                    assert np.all(np.isfinite(sol)), (
                        "Tbp=%s breaktime=%s %s: solution is not all-finite" % (Tbp, breaktime, label))

                    # --- core B3 check: the solver call count must reflect
                    #     breaktime, regardless of Tbp. Before the fix,
                    #     breaktime=True at Tbp=0 still made exactly 1 call
                    #     (breaktime was never even consulted). ---
                    expected_calls = n_real_segments if breaktime else 1
                    assert n_calls == expected_calls, (
                        "Tbp=%s breaktime=%s %s: calc_DEBresults was called %d time(s), "
                        "expected %d - breaktime is not taking effect (or the "
                        "zero-length-segment skip regressed)" % (
                            Tbp, breaktime, label, n_calls, expected_calls))

                    results[(Tbp, breaktime, label)] = sol
                    print("Tbp=%s breaktime=%-5s %-7s -> shape %s, %d solver call(s) (expected %d)" % (
                        Tbp, breaktime, label, sol.shape, n_calls, expected_calls))
    finally:
        mm.calc_DEBresults = orig_calc_DEBresults

    print()
    # --- breaktime doesn't change the physics: the segmented and
    #     continuous solves should agree closely (same ODE, solved
    #     piecewise vs continuously) ---
    for Tbp in (0, 3):
        for label in ('pulse', 'control'):
            sol_off = results[(Tbp, False, label)]
            sol_on = results[(Tbp, True, label)]
            maxdiff = np.max(np.abs(sol_off - sol_on))
            assert np.allclose(sol_off, sol_on, atol=1e-3), (
                "Tbp=%d %s: breaktime=True/False solutions disagree by more than "
                "expected (max abs diff = %.3g)" % (Tbp, label, maxdiff))
            print("Tbp=%d %-7s: breaktime on/off agree (max abs diff = %.3g)" % (Tbp, label, maxdiff))

    # --- Tbp>0 still delays reproduction as before: state row 2
    #     (reproduction) must be exactly zero up to and including Tbp,
    #     for both breaktime settings ---
    REPRO_ROW = 2
    for breaktime in (False, True):
        sol = results[(3, breaktime, 'pulse')]
        timeext = mm.DEBtox2019models(
            [full_ds], debparameters, moas, feedbs, Tbp=3, breaktime=breaktime, solver='LSODA'
        ).newtimeext[0]
        pre_tbp = timeext <= 3
        assert np.allclose(sol[REPRO_ROW, pre_tbp], 0.0), (
            "breaktime=%s: reproduction should be exactly 0 up to Tbp=3" % breaktime)
        assert np.any(sol[REPRO_ROW, ~pre_tbp] > 0), (
            "breaktime=%s: reproduction should become nonzero after Tbp=3" % breaktime)
    print("Tbp=3: reproduction delay is correctly applied for both breaktime settings")

    print("\nB3 fixed: breaktime is honored independently of Tbp, in all four combinations,")
    print("and zero-length (duplicated-time) segments no longer reach the solver.")
