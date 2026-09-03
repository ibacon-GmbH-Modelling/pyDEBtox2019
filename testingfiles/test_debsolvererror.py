# testing script for the DEBSolverError handling in models.py
#
# Confirms that:
#   1) calc_DEBresults raises models.DEBSolverError (not some incidental
#      exception several calls downstream) when the ODE solver cannot
#      produce a usable, finite trajectory - for both the RK45 (solve_ivp)
#      and LSODA (odeint) code paths;
#   2) an unrelated error (an unrecognized solver name) still raises a
#      plain ValueError as before, i.e. is NOT reclassified as a
#      DEBSolverError;
#   3) a normal, well-posed model evaluation is unaffected by the new
#      finiteness/shape checks (no false positives);
#   4) DEBtox2019models.log_likelihood catches DEBSolverError specifically
#      and returns np.inf, using a real model/dataset built the same way
#      as in test_validation_flow.py.

import json

import numpy as np

import pydebtox2019.models as mm
import pydebtox2019.debtox2019api as dt2019

from test_validation_flow import _read_raw_components


def _dummy_pars():
    # a plausible DEBtox2019 parameter vector (linear scale), in the order
    # expected by calc_DEBresults / DEBtox2019_derivatives_*
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
    timeext = np.linspace(0.0, 21.0, 20)
    DEBpars = _dummy_pars()
    L0 = DEBpars[5]
    y0_ok = np.array([0.0, L0, 0.0, 1.0])

    # NaN in a *rate* parameter (kd), not in y0: scipy's solvers validate
    # y0's finiteness upfront and would raise their own plain ValueError
    # before ever reaching the ODE right-hand side (confirmed while writing
    # this test - solve_ivp raises "All components of the initial state y0
    # must be finite" immediately for a NaN y0). Injecting the NaN into a
    # DEBpars rate constant instead lets integration start normally and
    # only turn non-finite partway through, which is what the new
    # mid-integration finiteness check in calc_DEBresults is there to catch.
    DEBpars_nan = DEBpars.copy()
    DEBpars_nan[16] = np.nan  # kd

    # ------------------------------------------------------------------
    # 1) & 3): direct calc_DEBresults checks, for both solver backends
    # ------------------------------------------------------------------
    for solver in ("RK45", "LSODA"):
        try:
            mm.calc_DEBresults(C, timextr, y0_ok, DEBpars_nan, moa, feedb, timeext, solver=solver)
        except mm.DEBSolverError:
            print("[%s] NaN rate parameter correctly raised DEBSolverError" % solver)
        else:
            raise AssertionError("[%s] expected DEBSolverError for a NaN rate parameter" % solver)

        sol = mm.calc_DEBresults(C, timextr, y0_ok, DEBpars, moa, feedb, timeext, solver=solver)
        assert sol.shape == (4, len(timeext)), "[%s] unexpected shape %s" % (solver, sol.shape)
        assert np.all(np.isfinite(sol)), "[%s] normal evaluation should be all-finite" % solver
        print("[%s] normal evaluation still returns a finite %s trajectory" % (solver, sol.shape))

    # ------------------------------------------------------------------
    # 2) an unrelated error must not be reclassified as DEBSolverError
    # ------------------------------------------------------------------
    try:
        mm.calc_DEBresults(C, timextr, y0_ok, DEBpars, moa, feedb, timeext, solver="bogus")
    except mm.DEBSolverError:
        raise AssertionError("an unrecognized solver name must not be reported as DEBSolverError")
    except ValueError:
        print("Unrecognized solver name still raises a plain ValueError, as before")

    # ------------------------------------------------------------------
    # 4) log_likelihood: catches DEBSolverError specifically -> np.inf,
    #    and is otherwise unaffected for a normal parameter point
    # ------------------------------------------------------------------
    with open('input_pars_tbp3.json') as json_file:
        DEBpars_json = json.load(json_file)
    debparameters = dt2019.DEBparameters(DEBpars_json)
    debparameters.set_fixfree_all(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)

    ccl, lcl, rcl, scl = _read_raw_components()
    full_ds, _, _ = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')

    debmodel = mm.DEBtox2019models([full_ds], debparameters, moa, feedb, Tbp=0, solver='LSODA')

    theta_ok = debmodel.parvals[debmodel.posfree].copy()
    llik_ok = debmodel.log_likelihood(theta_ok.copy(), debmodel.parvals.copy(), debmodel.posfree)
    assert np.isfinite(llik_ok), "log_likelihood should be finite for a normal parameter point"
    print("log_likelihood at a normal parameter point:", llik_ok)

    # force a solver failure by injecting a NaN into one of the free tox
    # parameters (kd) - this should surface as DEBSolverError inside
    # calc_model/calc_DEBresults, which log_likelihood must turn into np.inf
    theta_bad = theta_ok.copy()
    kd_pos = list(debmodel.parnames[debmodel.posfree]).index('kd')
    theta_bad[kd_pos] = np.nan
    llik_bad = debmodel.log_likelihood(theta_bad, debmodel.parvals.copy(), debmodel.posfree)
    assert llik_bad == np.inf, "log_likelihood should return np.inf on a solver failure, got %r" % llik_bad
    print("log_likelihood at a NaN parameter point correctly returned np.inf")

    print("\nDEBSolverError handling verified successfully.")
