# testing script for build_dataset_variants() / completedataset.subset()
#
# Regression test for review item B4 ("subset() leaves survival-derived
# arrays stale and misaligned"): the previous _slice_endpoint patched a
# hand-picked subset of derived attributes by name (lengthtreat,
# reprocumtreat, the flattened fit vectors), but never touched
# deatharraytreat, survprobstreat, lowlimtreat/upplimtreat or
# meanvalstransf. Those kept their *unsliced*, original-order values
# whenever the subset was not a leading prefix of the treatments - which is
# exactly what control_type='control' and 'solvent' produce for their
# complement dataset (dropping treatment 0.0 or 0.1 out of [0, 0.1, 1, 2, 3]
# is not a leading prefix). control_type='both' happened to select a
# leading prefix, which is why it never surfaced the bug.
#
# This test exercises all three control_type variants and checks, for every
# endpoint of both the control subset and its complement:
#   1) every per-treatment/per-replicate derived list or array has exactly
#      as many entries as the subset's own ntreats/dataarray rows;
#   2) deatharraytreat is recomputable from (and matches) the *subset's own*
#      dataarray rows, not the original unsliced ones;
#   3) the whole endpoint matches an independently rebuilt endpoint made by
#      filtering the raw Test_*data.txt tables to the same treatments
#      *before* construction - a ground truth that does not go through
#      subset() at all.

import numpy as np
import pandas as pd

import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin

from test_validation_flow import _read_raw_components


def _check_endpoint_consistency(ep, label, keep_labels):
    """Invariants that must hold for any sliced dataclass-like endpoint."""
    n = ep.ntreats
    assert ep.dataarray.shape[0] == n, "%s: dataarray rows (%d) != ntreats (%d)" % (
        label, ep.dataarray.shape[0], n)
    assert ep.weights.shape[0] == n, "%s: weights rows != ntreats" % label
    assert ep.treatmentsnames.shape[0] == n, "%s: treatmentsnames length != ntreats" % label
    assert np.all(np.isin(ep.treatmentsnames.astype(float), keep_labels)), (
        "%s: treatmentsnames contains a label that should have been dropped" % label)
    assert len(ep.uniquetreats) == ep.meanvalstransf.shape[0], (
        "%s: meanvalstransf has %d rows for %d unique treatments" % (
            label, ep.meanvalstransf.shape[0], len(ep.uniquetreats)))

    if hasattr(ep, 'deatharraytreat'):
        assert len(ep.deatharraytreat) == n, "%s: deatharraytreat has %d entries, expected %d" % (
            label, len(ep.deatharraytreat), n)
        for i in range(n):
            tmpsurv = ep.dataarray[i, ~np.isnan(ep.dataarray[i])]
            expected = np.append(-(np.diff(tmpsurv).astype(float)), tmpsurv[-1])
            got = ep.deatharraytreat[i]
            assert np.allclose(expected, got), (
                "%s: deatharraytreat[%d] does not match its own dataarray row "
                "(expected %s, got %s) - stale/misaligned reslicing" % (label, i, expected, got))
        assert len(ep.survprobstreat) == n
        assert ep.lowlimtreat.shape[0] == n
        assert ep.upplimtreat.shape[0] == n

    if hasattr(ep, 'lengthtreat'):
        assert len(ep.lengthtreat) == n

    if hasattr(ep, 'reprocumtreat'):
        assert len(ep.reprocumtreat) == n
        assert ep.dataarray_cumulative.shape[0] == n


def _ground_truth_endpoint(raw_df, ep_full, keep_labels, builder):
    """
    Rebuild an endpoint directly from a raw table pre-filtered (by column)
    to the replicates of `ep_full` whose treatment is in `keep_labels`, i.e.
    independently of completedataset.subset()'s own bookkeeping.
    """
    keep_cols_rows = np.isin(ep_full.treatmentsnames.astype(float), keep_labels)
    cols = np.concatenate(([True], keep_cols_rows))
    filtered = raw_df.to_numpy()[:, cols]
    return builder(filtered)


def _check_against_ground_truth(sub_ds, keep_labels, lcl_full, rcl_full, scl_full,
                                 Ldata, Rdata, Sdata, label):
    lcl_gt = _ground_truth_endpoint(Ldata, lcl_full, keep_labels, readin.lengthdataclass)
    rcl_gt = _ground_truth_endpoint(
        Rdata, rcl_full, keep_labels,
        lambda t: readin.reproclass(t, reprocase='individual', optcase=1))
    scl_gt = _ground_truth_endpoint(Sdata, scl_full, keep_labels, readin.survdataclass)

    assert np.array_equal(sub_ds.lengthdata.dataarray, lcl_gt.dataarray, equal_nan=True), \
        "%s: lengthdata.dataarray does not match ground truth" % label
    assert np.allclose(sub_ds.lengthdata.meanvalstransf, lcl_gt.meanvalstransf, equal_nan=True), \
        "%s: lengthdata.meanvalstransf does not match ground truth" % label

    assert np.allclose(sub_ds.survdata.dataarray, scl_gt.dataarray, equal_nan=True), \
        "%s: survdata.dataarray does not match ground truth" % label
    for i in range(scl_gt.ntreats):
        assert np.allclose(sub_ds.survdata.deatharraytreat[i], scl_gt.deatharraytreat[i]), \
            "%s: survdata.deatharraytreat[%d] does not match ground truth" % (label, i)
    assert np.allclose(sub_ds.survdata.meanvalstransf, scl_gt.meanvalstransf, equal_nan=True), \
        "%s: survdata.meanvalstransf does not match ground truth" % label

    assert np.allclose(sub_ds.reprodata.dataarray_cumulative, rcl_gt.dataarray_cumulative,
                        equal_nan=True), "%s: reprodata.dataarray_cumulative does not match ground truth" % label
    assert np.allclose(sub_ds.reprodata.meanvalstransf, rcl_gt.meanvalstransf, equal_nan=True), \
        "%s: reprodata.meanvalstransf does not match ground truth" % label


if __name__ == "__main__":

    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    Rdata = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)

    for control_type in ("both", "control", "solvent"):
        ccl, lcl, rcl, scl = _read_raw_components()

        full_ds, control_ds, full_ds_compl = dt2019.build_dataset_variants(
            ccl, lcl, rcl, scl, control_type=control_type
        )

        print("\n=== control_type=%r ===" % control_type)
        control_labels = np.array(control_ds.concdata.conctreatsnames).astype(float)
        print("control_ds labels:", control_labels)

        for ep_name in ("lengthdata", "reprodata", "survdata"):
            _check_endpoint_consistency(
                getattr(control_ds, ep_name), "%s/control_ds.%s" % (control_type, ep_name), control_labels)
        _check_against_ground_truth(
            control_ds, control_labels, lcl, rcl, scl, Ldata, Rdata, Sdata,
            "%s/control_ds" % control_type)
        print("control_ds (%d treatments) matches ground truth" % control_ds.survdata.ntreats)

        if full_ds_compl is not None:
            compl_labels = np.array(full_ds_compl.concdata.conctreatsnames).astype(float)
            print("full_ds_compl labels:", compl_labels)
            for ep_name in ("lengthdata", "reprodata", "survdata"):
                _check_endpoint_consistency(
                    getattr(full_ds_compl, ep_name), "%s/full_ds_compl.%s" % (control_type, ep_name), compl_labels)
            _check_against_ground_truth(
                full_ds_compl, compl_labels, lcl, rcl, scl, Ldata, Rdata, Sdata,
                "%s/full_ds_compl" % control_type)
            print("full_ds_compl (%d treatments) matches ground truth" % full_ds_compl.survdata.ntreats)

    print("\nAll control_type variants ('both', 'control', 'solvent') produce correctly")
    print("resliced and aligned derived data for every endpoint - B4 regression check passed.")
