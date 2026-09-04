# testing script for the shared-core refactor of the DEB derivative functions
#
# DEBtox2019_derivatives_odeint and DEBtox2019_derivatives_solveivp used to
# be two fully duplicated ~60-line numba-jitted functions, differing only in
# the order of their first two arguments (odeint's func(y, t, *args) vs.
# solve_ivp's fun(t, y, *args)). They now both delegate to one shared
# _DEBtox2019_derivatives_core(y, t, *args). This script checks that the
# refactor is behavior-preserving: both public wrappers must return
# identical derivatives for the same physical state, just addressed via
# their respective argument orders.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import pydebtox2019.models as mm


def _dummy_pars():
    # a plausible DEBtox2019 parameter vector (linear scale), same as used
    # in test_debsolvererror.py
    FBV, KRV, kap, yP, Lm_ref = 0.02, 1.0, 0.8, 0.64, 5.0
    L0, Lp, Lm, rB, Rm, f, hb, a = 0.88, 1.8, 3.1, 0.14, 10.0, 1.0, 0.0048, 1.0
    Lf, Lj, Tlag = 0.0, 0.0, 0.0
    kd, bb, zb, bs, zs = 0.03, 20.0, 0.05, 1.0, 1.0
    return np.array([FBV, KRV, kap, yP, Lm_ref, L0, Lp, Lm, rB, Rm, f, hb, a,
                      Lf, Lj, Tlag, kd, bb, zb, bs, zs])


if __name__ == "__main__":

    moa = np.array([0, 0, 1, 1, 0], dtype=float)
    feedb = np.array([0, 0, 0, 0], dtype=float)
    C = np.array([10.0, 10.0])
    timextr = np.array([0.0, 21.0])
    DEBpars = _dummy_pars()
    L0 = DEBpars[5]

    # a handful of (t, y) states, including edge cases the model explicitly
    # special-cases: t=0 (Tlag boundary), y[1] at/under the half-L0 floor,
    # and a mid-simulation "typical" state
    cases = [
        ("t=0, y=y0", 0.0, np.array([0.0, L0, 0.0, 1.0])),
        ("mid-simulation", 10.0, np.array([0.05, 2.5, 3.0, 0.9])),
        ("length at half-L0 floor", 5.0, np.array([0.02, 0.5 * L0, 1.0, 0.95])),
        ("large damage (near/above zb, zs)", 15.0, np.array([2.0, 3.0, 5.0, 0.5])),
    ]

    all_ok = True
    for label, t, y in cases:
        # each call gets its own fresh copy of y, since the derivative
        # functions mutate y[1] in place (the 1e-3*L0 floor)
        dydt_odeint = mm.DEBtox2019_derivatives_odeint(y.copy(), t, C, timextr, DEBpars, moa, feedb)
        dydt_solveivp = mm.DEBtox2019_derivatives_solveivp(t, y.copy(), C, timextr, DEBpars, moa, feedb)
        dydt_core = mm._DEBtox2019_derivatives_core(y.copy(), t, C, timextr, DEBpars, moa, feedb)

        match_wrappers = np.array_equal(dydt_odeint, dydt_solveivp)
        match_core = np.array_equal(dydt_odeint, dydt_core)
        all_ok &= match_wrappers and match_core

        status = "OK" if (match_wrappers and match_core) else "MISMATCH"
        print("[%s] %-32s dydt=%s (%s)" % (status, label, dydt_odeint, "t=%.4g" % t))
        if not (match_wrappers and match_core):
            print("    odeint:    ", dydt_odeint)
            print("    solve_ivp: ", dydt_solveivp)
            print("    core:      ", dydt_core)

    assert all_ok, "odeint/solve_ivp wrappers and the shared core disagree - see mismatches above"
    print("\nBoth wrappers and the shared core agree on every case: derivatives refactor verified.")
