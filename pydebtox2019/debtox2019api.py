'''
classes and functions for the DEBtox2019 handling of data and parameters
'''

import numpy as np
import matplotlib.pyplot as plt
from .readin import completedataset
from . import models as mm
from . import parspace as ps

import multiprocessing as mp
import psutil
n_cores = psutil.cpu_count(logical=False) # to have the number of physical cores only


def plot_DEBresults(parspaceres, CI=True, multicore=True, ds = -1, wmeans=False):
    if ds == -1:
        for dataset in range(parspaceres.model.ndatasets):
            plot_DEBresults_ds(parspaceres, CI=CI, multicore=multicore, dataset=dataset, wmeans=wmeans)
    else:
        plot_DEBresults_ds(parspaceres, CI=CI, multicore=multicore, dataset=ds, wmeans=wmeans)

def plot_DEBresults_ds(parspaceres, CI=True, multicore=True, dataset=0, wmeans=False):
    print("plotting the results")
    # assumes a single dataset for now
    ccl = parspaceres.model.concstruct_list[dataset]
    lcl = parspaceres.model.lengthstruct_list[dataset]
    rcl = parspaceres.model.reprostruct_list[dataset]
    scl = parspaceres.model.survstruct_list[dataset]
    sol=[]
    allpars = np.copy(parspaceres.model.parvals)
    allpars[parspaceres.model.islog] = 10 ** allpars[parspaceres.model.islog]
    modelpars = parspaceres.model.build_dataset_parameters(allpars, dataset)
    #print("model parameters: ", modelpars, "  ", len(modelpars))
    #treatmentnames = dataset.concdata.treatmentnames
    treatmentnames = parspaceres.model.concstruct_list[dataset].conctreatsnames
    tevals = np.linspace(np.min(parspaceres.model.concstruct_list[dataset].time),
                         np.max(parspaceres.model.concstruct_list[dataset].time),100)
    #print("tevals: ", tevals, "  ", len(tevals))
    fig = plt.figure()
    lenendpoints = np.sum([1 for cl in [ccl,lcl,rcl,scl] if cl is not None])
    # print("number of endpoints to plot: ", lenendpoints)
    # print("number of treatments to plot: ", len(treatmentnames))
    ax = fig.subplots(lenendpoints,len(treatmentnames),squeeze=False)
    for i in range(len(treatmentnames)):
        # print("i: ", i)
        # print("treatment: ", treatmentnames[i])
        sol.append(parspaceres.model.calc_model(parspaceres.model.concstruct_list[dataset].concarraytr[i],
                                      parspaceres.model.concstruct_list[dataset].timetr,
                                      modelpars,
                                      parspaceres.model.moa,
                                      parspaceres.model.feedb,
                                      tevals))
        if CI:
            solci=np.zeros((len(parspaceres.propagationset),len(tevals),4))                
            # ---- prepare constant arguments
            parvals   = parspaceres.model.parvals
            posfree   = parspaceres.posfree
            concarray = parspaceres.model.concstruct_list[dataset].concarraytr[i]
            time      = parspaceres.model.concstruct_list[dataset].timetr
            islog     = parspaceres.model.islog
            moa       = parspaceres.model.moa
            feedb     = parspaceres.model.feedb
            # ---- build starmap argument list
            args = [(pars, parvals, posfree, concarray, time, islog, moa, feedb, tevals,dataset) for pars in parspaceres.propagationset]
            # ---- run in parallel
            if multicore:
                with mp.Pool(n_cores) as pool:
                    results = pool.starmap(parspaceres.model.worker_DEBresults, args)
            else:
                results = [parspaceres.model.worker_DEBresults(*arg) for arg in args]
            # ---- fill output array
            solci[:] = results
            # find max and min for each time point
            low = solci.min(axis=0)
            upp = solci.max(axis=0)
        ### concentration and damage plot
        ax[0,i].plot(ccl.timetr,ccl.concarraytr[i])
        ax[0,i].plot(tevals,sol[i][0])
        ax[0,i].set_title('T %s'%(treatmentnames[i]))
        if CI:
            ax[0,i].fill_between(tevals, low[:,0], upp[:,0], color='gray', alpha=0.5)
        if (ccl.concarray==0).all():
            ax[0,i].set_ylim(0.0,1.1)
        else:
            ax[0,i].set_ylim(0,ccl.concarray.max()*1.1)
        if i == 0:
            ax[0,i].set_ylabel("Concentration (%s)"%(ccl.concunits))
        k=1 # mark endpoint
        ### lengths
        if lcl is not None:
            lcl.add_plotdata(ax[k,i],treatmentnames[i],wmeans=wmeans)
            ax[k,i].plot(tevals,sol[0][1],'k--')
            ax[k,i].plot(tevals,sol[i][1])
            ax[k,i].set_ylim(0,np.nanmax(lcl.dataarray)*1.1)
            if CI:
                ax[k,i].fill_between(tevals, low[:,1], upp[:,1], color='gray', alpha=0.5)
            if i == 0:
                ax[k,i].set_ylabel("Length (mm)")
            k+=1
        ### reproduction
        if rcl is not None:
            rcl.add_plotdata(ax[k,i],treatmentnames[i],wmeans=wmeans)
            ax[k,i].plot(tevals,sol[0][2],'k--')
            ax[k,i].plot(tevals,sol[i][2])
            ax[k,i].set_ylim(0,np.nanmax(rcl.dataarray_cumulative)*1.1)
            if CI:
                ax[k,i].fill_between(tevals, low[:,2], upp[:,2], color='gray', alpha=0.5)
            if i == 0:
                ax[2,i].set_ylabel("Reproduction (#/female)")
            k+=1
        ### survival
        if scl is not None:
            scl.add_plotdata(ax[k,i],ntreat=treatmentnames[i], scaleto1=True,wmeans=wmeans)
            ax[k,i].plot(tevals,sol[0][3], 'k--')
            ax[k,i].plot(tevals,sol[i][3])
            ax[k,i].set_ylim([0,1.1])
            if CI:
                ax[k,i].fill_between(tevals, low[:,3], upp[:,3], color='gray', alpha=0.5)
            #ax[k,i].set_xlabel("Time (d)")
            if i == 0:
                ax[k,i].set_ylabel("Survival fraction")
        ax[lenendpoints-1,i].set_xlabel("Time (d)") # add the xlabel only to the last row of plots
    plt.tight_layout()


def get_survival_data(model, modelsolcontainer, nd):

    ntreats = model.concstruct_list[nd].ntreats
    struct = model.survstruct_list[nd]

    mask = np.isin(model.timeext[nd], struct.time)
    modelvals = np.array([
        modelsolcontainer[nd][i][3][mask]
        for i in range(ntreats)
    ])

    datavals = struct.survprobstreat
    counts = struct.survarrtreat

    return modelvals, datavals, counts


def calc_r2_nrmse(data, model):
    valid = ~np.isnan(data)
    data = data[valid]
    model = model[valid]
    nobs = len(data)
    if nobs == 0:
        return np.nan, np.nan
    rss = np.sum((data - model) ** 2)
    tss = np.sum((data - np.mean(data)) ** 2)
    r2 = np.nan if tss == 0 else 1 - rss / tss
    nrmse = np.sqrt(rss / nobs) / np.mean(data)
    return r2, nrmse

def get_endpoint_data(model, modelsolcontainer, nd, endpoint):
    ntreats = model.concstruct_list[nd].ntreats
    if endpoint == 1:
        struct = model.lengthstruct_list[nd]
        state_idx = 1
    elif endpoint == 2:
        struct = model.reprostruct_list[nd]
        state_idx = 2
    else:
        raise ValueError(endpoint)
    mask = np.isin(model.timeext[nd], struct.time)
    modelvals = np.array([
        modelsolcontainer[nd][i][state_idx][mask]
        for i in range(ntreats)
    ])
    return modelvals, struct.meanvalstransf

def calc_survival_metrics(modelvals, surv_probs, surv_counts):

    ssq_fit_num = 0.0
    rss = 0.0

    # collect all valid survival probabilities
    # for calculation of total sum of squares
    tss_data = []

    # for NRMSE denominator
    all_counts = []

    for i in range(len(modelvals)):

        probs_i = np.asarray(surv_probs[i])
        counts_i = np.asarray(surv_counts[i])
        model_i = np.asarray(modelvals[i])

        valid = ~np.isnan(probs_i)

        nmax = counts_i[0]

        # NRMSE based on counts
        ssq_fit_num += np.sum(
            (counts_i[valid] - nmax * model_i[valid]) ** 2
        )

        # R² residual sum of squares
        rss += np.sum(
            (probs_i[valid] - model_i[valid]) ** 2
        )

        # accumulate valid observations
        tss_data.extend(probs_i[valid])

        # accumulate counts for mean denominator
        all_counts.extend(counts_i[valid])

        # ----------------------------------------
        # future treatment-specific calculations
        # can be added here
        #
        # bias_i = np.mean(model_i[valid] - probs_i[valid])
        # metrics_per_treatment.append(...)
        # ----------------------------------------

    tss_data = np.asarray(tss_data)

    if len(tss_data) == 0:
        return np.nan, np.nan

    tss = np.sum(
        (tss_data - np.mean(tss_data)) ** 2
    )

    r2 = np.nan if tss == 0 else 1 - rss / tss

    nrmse = (
        np.sqrt(ssq_fit_num / len(tss_data))
        / np.mean(all_counts)
    )

    return r2, nrmse


def efsa_criteria(model):
    from copy import deepcopy

    model = deepcopy(model)

    basepars = model.parvals.copy()
    basepars[model.islog] = 10 ** basepars[model.islog]

    endpoint_names = {
        0: "survival",
        1: "length",
        2: "reproduction"
    }

    modelsolcontainer = [None] * model.ndatasets

    # ======================================================
    # Compute model solutions
    # ======================================================
    for nd in range(model.ndatasets):

        modelpars = model.build_dataset_parameters(basepars, nd)

        newtime = model.timeext[nd]

        newtimeext = np.unique(
            np.concatenate((
                np.linspace(
                    newtime[0],
                    newtime[-1],
                    max(model.min_t, len(newtime))
                ),
                newtime
            ))
        )

        modelcoltreatcont = [None] * model.concstruct_list[nd].ntreats

        for i in range(model.concstruct_list[nd].ntreats):

            modelsol = model.calc_model(
                model.concstruct_list[nd].concarraytr[i],
                model.concstruct_list[nd].timetr,
                modelpars,
                model.moa,
                model.feedb,
                timeext=newtimeext
            )

            idx_targets = np.searchsorted(
                newtimeext,
                model.timeext[nd]
            )

            modelsol = modelsol[:, idx_targets]
            modelcoltreatcont[i] = modelsol

        modelsolcontainer[nd] = modelcoltreatcont

    # ======================================================
    # Dataset-specific metrics
    # ======================================================
    for nd in range(model.ndatasets):

        print(f"\nCalculating EFSA criteria for dataset {nd}")

        for endpoint in model.active_endpoints[nd]: 
            if endpoint == 0:
                modelvals, datavals, counts = (
                    get_survival_data(
                        model,
                        modelsolcontainer,
                        nd
                    )
                )

                r2, nrmse = calc_survival_metrics(
                    modelvals,
                    datavals,
                    counts
                )

            else:
                modelvals, datavals = get_endpoint_data(
                    model,
                    modelsolcontainer,
                    nd,
                    endpoint
                )

                r2, nrmse = calc_r2_nrmse(
                    datavals,
                    modelvals
                )

            print(
                f"R2 {endpoint_names[endpoint]}: {r2}"
            )

            print(
                f"NRMSE {endpoint_names[endpoint]}: {nrmse}"
            )
    if model.ndatasets > 1:
        print("\n=== Combined metrics across datasets ===")
        print("Note: Only endpoints present in all datasets are considered.")
        # ======================================================
        # Combined metrics across datasets
        # ======================================================
        common_endpoints = set(model.active_endpoints[0])
        for nd in range(1, model.ndatasets):
            common_endpoints &= set(model.active_endpoints[nd])
        for endpoint in sorted(common_endpoints):
            if endpoint == 0:
                all_model = []
                all_data = []
                all_counts = []
                for nd in range(model.ndatasets):
                    modelvals, datavals, counts = (get_survival_data(model,
                                                                     modelsolcontainer,
                                                                     nd))

                    all_model.append(modelvals)
                    all_data.append(datavals)
                    all_counts.append(counts)
                all_model = np.concatenate(all_model, axis=0)
                all_data = np.concatenate(all_data, axis=0)
                all_counts = np.concatenate(all_counts, axis=0)

                r2, nrmse = calc_survival_metrics(all_model,all_data,all_counts)
            else:
                all_model = []
                all_data = []
                for nd in range(model.ndatasets):
                    modelvals, datavals = get_endpoint_data(model,modelsolcontainer,
                                                            nd,endpoint)
                    valid = ~np.isnan(datavals)
                    all_model.append(modelvals[valid])
                    all_data.append(datavals[valid])
                if not all_data:
                    continue

                all_model = np.concatenate(all_model)
                all_data = np.concatenate(all_data)

                r2, nrmse = calc_r2_nrmse(all_data,all_model)
            print(f"Global R2 {endpoint_names[endpoint]}: {r2}")
            print(f"Global NRMSE {endpoint_names[endpoint]}: {nrmse}\n")
    #return modelsolcontainer

def validation(full_ds, debparameterclass, parspace_tox, CI=True, multicore=True, wmeans=False):
    """
    Validation for a new dataset using the toxicity parameters obtained from a
    previous calibration.

    This function should be called only after the physiological model for the
    new dataset has already been refitted (i.e. debparameterclass already
    carries the refit physiological parameter values). The toxicity
    parameters are then overwritten with the values found in the previous
    parspace_tox calibration and held fixed at those values, with physiology
    frozen at its own point estimate. Only the *toxicity*-parameter
    uncertainty from parspace_tox is propagated into the validation CI/EFSA
    metrics - physiological parameters carry no uncertainty band here, since
    they are not re-explored on the new dataset.

    Parameters are matched between parspace_tox and debparameterclass by base
    name (e.g. "kd"), not by full expanded name, since the two may use
    different dataset/group suffixes (or none at all).

    Arguments:
    - full_ds: the full dataset for the new data (including controls)
    - debparameterclass: the DEBparameters instance containing the parameters relative to the new dataset
    - parspace_tox: the PyParspace instance from the previous calibration of the toxicity parameters
    """
    tox_model = parspace_tox.model
    tox_base_names = tox_model.full_base_names[parspace_tox.posfree]
    tox_islog = tox_model.islog[parspace_tox.posfree]
    tox_values = tox_model.parvals[parspace_tox.posfree]

    uniq, counts = np.unique(tox_base_names, return_counts=True)
    dupes = uniq[counts > 1]
    if dupes.size > 0:
        raise ValueError(
            "validation() cannot resolve a single calibrated value for "
            "parameter(s) %s: parspace_tox fitted more than one "
            "(grouped/dataset-specific) instance of it. Resolve which value "
            "to carry forward before calling validation()." % ", ".join(dupes)
        )

    # Fix all tox parameters, then overwrite their central values and free
    # exactly the ones that were free in the calibration - matched by base
    # name, since the new dataset's expanded names need not use the same
    # group/dataset suffixes as the calibration.
    debparameterclass.fixfree_tox_pars(isfree=False)
    for base_name, islog_src, val in zip(tox_base_names, tox_islog, tox_values):
        mask = debparameterclass.full_base_names == base_name
        if not np.any(mask):
            raise ValueError(
                "Calibrated toxicity parameter '%s' not found in the new "
                "dataset's parameter set." % base_name
            )
        if mask.sum() > 1:
            raise ValueError(
                "Toxicity parameter '%s' expands to more than one instance "
                "in the new dataset's parameter set; validation() expects a "
                "single dataset with one instance per parameter." % base_name
            )
        if debparameterclass.full_islog[mask][0] != islog_src:
            raise ValueError(
                "Parameter '%s' is log-scaled in parspace_tox but not in "
                "the new dataset's parameter set (or vice versa); cannot "
                "safely transfer its calibrated value." % base_name
            )
        print("Updating parameter %s to value %f from the calibration" % (base_name, val))
        debparameterclass.full_list[mask] = val

    debparameterclass.set_fixfree_all(isfree=False)
    debparameterclass.set_freefix_parameters_list(tox_base_names, isfree=True)

    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameterclass,
                                       parspace_tox.model.moa,
                                       parspace_tox.model.feedb,
                                       parspace_tox.model.Tbp, solver='LSODA')
    physioparspace = ps.PyParspace(ps.SettingParspace(0, 1), debmodeltest)

    # Remap the calibration's propagation set onto the new dataset's
    # parameter order by base name, rather than assuming the two parameter
    # sets share the same column order (or even the same suffix scheme).
    dst_base_names = physioparspace.model.full_base_names[physioparspace.posfree]
    dst_islog = physioparspace.model.islog[physioparspace.posfree]

    uniq_dst, counts_dst = np.unique(dst_base_names, return_counts=True)
    dupes_dst = uniq_dst[counts_dst > 1]
    if dupes_dst.size > 0:
        raise ValueError(
            "The new dataset's parameter set expands parameter(s) %s into "
            "more than one free instance; validation() expects a single "
            "dataset with one instance per toxicity parameter."
            % ", ".join(dupes_dst)
        )

    name_to_src_col = {name: j for j, name in enumerate(tox_base_names)}
    perm = np.empty(len(dst_base_names), dtype=int)
    for k, name in enumerate(dst_base_names):
        if name not in name_to_src_col:
            raise ValueError(
                "Free parameter '%s' in the new dataset has no matching "
                "calibrated toxicity parameter in parspace_tox." % name
            )
        j = name_to_src_col[name]
        if dst_islog[k] != tox_islog[j]:
            raise ValueError(
                "Parameter '%s' is log-scaled in one parameter set but not "
                "the other; cannot safely remap the propagation set." % name
            )
        perm[k] = j
    physioparspace.propagationset = parspace_tox.propagationset[:, perm]

    plot_DEBresults(physioparspace, CI=CI, multicore=multicore, wmeans=wmeans)
    efsa_criteria(physioparspace.model)


def predict_exposure(
    model,
    concclass,
    dataset=0,
    return_time=None,
    solver_points=1000,
    plot=False,
    ci=False,
    parspace=None,
    figsize=(10, 8),
):
    """
    Predict DEBtox2019 responses for an arbitrary exposure profile.

    Parameters
    ----------
    model : DEBtox2019models
        Fitted DEBtox model.

    concclass : concentration object
        Exposure object containing:
            concarraytr
            timetr

    dataset : int, optional
        Dataset-specific parameterization to use.

    return_time : array-like, optional
        Time points at which numerical predictions are returned.
        If None, concclass.timetr is used.

    solver_points : int, optional
        Number of internal points used for model solution.

    plot : bool, optional
        Produce plots.

    ci : bool, optional
        Calculate confidence intervals.

    parspace : PyParspace, optional
        Required when ci=True.

    figsize : tuple
        Figure size.

    Returns
    -------
    dict
        Dictionary containing predictions and optional CI.
    """

    # ==========================================================
    # exposure profile
    # ==========================================================

    exposure_time = np.asarray(concclass.timetr)

    if hasattr(concclass, "concarraytr"):

        if np.ndim(concclass.concarraytr) > 1:

            if len(concclass.concarraytr) != 1:
                raise ValueError(
                    "Prediction exposure should contain exactly one treatment."
                )

            C = np.asarray(concclass.concarraytr[0])

        else:
            C = np.asarray(concclass.concarraytr)

    else:
        raise AttributeError(
            "concclass must contain attribute 'concarraytr'"
        )

    # ==========================================================
    # fine grid for solving
    # ==========================================================

    solver_time = np.unique(
        np.concatenate(
            (
                np.linspace(
                    exposure_time.min(),
                    exposure_time.max(),
                    solver_points,
                ),
                exposure_time,
            )
        )
    )

    # ==========================================================
    # output time vector
    # ==========================================================

    if return_time is None:
        return_time = exposure_time

    return_time = np.asarray(return_time)

    # ==========================================================
    # parameter handling
    # ==========================================================

    expanded_pars = model.parvals.copy()

    expanded_pars[model.islog] = (
        10 ** expanded_pars[model.islog]
    )

    debpars = model.build_dataset_parameters(
        expanded_pars,
        dataset,
    )

    # ==========================================================
    # main prediction
    # ==========================================================

    fine_sol = model.calc_model(
        C=C,
        timextr=exposure_time,
        DEBpars=debpars,
        moa=model.moa,
        feedb=model.feedb,
        timeext=solver_time,
    )

    output_sol = np.vstack(
        [
            np.interp(
                return_time,
                solver_time,
                fine_sol[i],
            )
            for i in range(4)
        ]
    )

    # ==========================================================
    # confidence intervals
    # ==========================================================

    ci_low = None
    ci_upp = None

    output_low = None
    output_upp = None

    if ci:

        if parspace is None:
            raise ValueError(
                "parspace must be supplied when ci=True"
            )

        nprop = len(parspace.propagationset)

        all_ci = np.zeros(
            (
                nprop,
                len(solver_time),
                4,
            )
        )

        for ip, pars in enumerate(parspace.propagationset):

            expanded = np.copy(model.parvals)

            expanded[parspace.posfree] = pars

            expanded = np.where(
                model.islog,
                10 ** expanded,
                expanded,
            )

            solver_pars = model.build_dataset_parameters(
                expanded,
                dataset,
            )

            sol_ci = model.calc_model(
                C=C,
                timextr=exposure_time,
                DEBpars=solver_pars,
                moa=model.moa,
                feedb=model.feedb,
                timeext=solver_time,
            )

            all_ci[ip] = sol_ci.T

        ci_low = all_ci.min(axis=0)
        ci_upp = all_ci.max(axis=0)

        output_low = np.vstack(
            [
                np.interp(
                    return_time,
                    solver_time,
                    ci_low[:, i],
                )
                for i in range(4)
            ]
        )

        output_upp = np.vstack(
            [
                np.interp(
                    return_time,
                    solver_time,
                    ci_upp[:, i],
                )
                for i in range(4)
            ]
        )

    # ==========================================================
    # output dictionary
    # ==========================================================

    result = {
        "time": return_time,
        "damage": output_sol[0],
        "length": output_sol[1],
        "reproduction": output_sol[2],
        "survival": output_sol[3],
        "raw": output_sol,
        "solver_time": solver_time,
        "fine_solution": fine_sol,
    }

    if ci:

        result["ci_low"] = {
            "damage": output_low[0],
            "length": output_low[1],
            "reproduction": output_low[2],
            "survival": output_low[3],
        }

        result["ci_upp"] = {
            "damage": output_upp[0],
            "length": output_upp[1],
            "reproduction": output_upp[2],
            "survival": output_upp[3],
        }

    # ==========================================================
    # plotting
    # ==========================================================

    if plot:

        fig, ax = plt.subplots(
            2,
            2,
            figsize=figsize,
            sharex=True,
        )

        labels = [
            "Damage",
            "Length",
            "Reproduction",
            "Survival",
        ]

        for i, axi in enumerate(ax.flat):

            # central prediction
            axi.plot(
                solver_time,
                fine_sol[i],
                lw=2,
                color="C0",
                label="Prediction",
            )

            # confidence interval
            if ci:

                axi.fill_between(
                    solver_time,
                    ci_low[:, i],
                    ci_upp[:, i],
                    color="gray",
                    alpha=0.4,
                    label="95% CI",
                )

            # returned values
            axi.plot(
                return_time,
                output_sol[i],
                "o",
                color="C0",
                ms=4,
                label="Returned values",
            )

            axi.set_title(labels[i])
            axi.grid(True)

        ax[1, 0].set_xlabel("Time (d)")
        ax[1, 1].set_xlabel("Time (d)")

        ax[0, 0].set_ylabel("Damage")
        ax[0, 1].set_ylabel("Length")
        ax[1, 0].set_ylabel("Reproduction")
        ax[1, 1].set_ylabel("Survival")

        handles, labels = ax[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper right",
        )

        plt.tight_layout()

        result["figure"] = fig

    return result


def calc_ecx(
    model,
    Tend,
    X=(10, 50),
    endpoints=None,
    dataset=0,
    conc_bounds=None,
    ci=False,
    parspace=None,
    multicore=True,
    max_expand=60,
    xtol=1e-8,
    plateau_tol=1e-6,
    verbose=True,
):
    """
    Calculate ECx / LCx values (effect concentrations) for constant exposure.

    Emulates the functionality of calc_ecx.m from the DEBtox2019/BYOM toolbox:
    for each evaluation time in `Tend` and each effect level in `X`, the
    constant exposure concentration (applied from t=0 to that time) that
    produces an X% effect relative to the untreated control is found by
    bisection. Survival is interpreted as LCx (percentage additional
    mortality); length and reproduction are interpreted as ECx (percentage
    reduction relative to the control).

    Parameters
    ----------
    model : DEBtox2019models
        Model instance holding the (fitted) parameter vector (model.parvals).
    Tend : float or array-like
        Evaluation time(s), in the model's time unit.
    X : iterable of float
        Effect levels in percent, e.g. (10, 50) for EC10/EC50. Must be in [0, 100).
    endpoints : {'survival', 'length', 'reproduction'}, int, or iterable thereof, optional
        Endpoint(s) to evaluate; endpoints not requested are skipped entirely
        (no bisection is run for them, saving computation time). A single
        name/code (e.g. endpoints='reproduction') is accepted, as well as an
        iterable of several. Defaults to those active for `dataset`
        (model.active_endpoints[dataset]).
    dataset : int
        Dataset whose parameterization is used to run the model
        (see model.build_dataset_parameters).
    conc_bounds : tuple(float, float), optional
        Initial (low, high) concentration bracket for the bisection search;
        automatically expanded if it does not bracket the root. Defaults to
        a bracket derived from the dataset's observed concentrations.
    ci : bool
        If True, also propagate the 95% CI on the ECx/LCx estimates using
        parspace.propagationset (same mechanism as predict_exposure).
    parspace : PyParspace, optional
        Required when ci=True.
    multicore : bool
        Use multiprocessing for the CI propagation.
    max_expand, xtol
        Passed to the bisection search (see DEBtox2019models.calc_ecx_core).
    plateau_tol : float or None
        Some endpoints can plateau before reaching a requested effect level
        (e.g. body length has a hard floor - growth cannot shrink below
        half the starting length - so at high enough concentration the
        effect on length stops increasing). Once the effect changes by less
        than `plateau_tol` between two successive concentration-bracket
        expansion steps, the search for that (endpoint, x, time) gives up
        early (NaN) instead of always exhausting `max_expand`. Set to None
        to disable and always use the full `max_expand` budget.
    verbose : bool
        Print a summary table.

    Returns
    -------
    dict
        results['time']: evaluation times.
        results[endpoint_name][x] -> np.ndarray of ECx/LCx values aligned with Tend.
        If ci=True, also results[endpoint_name]['{x}_lo'] / '{x}_up'.
    """
    ENDPOINT_NAMES = {0: 'survival', 1: 'length', 2: 'reproduction'}
    NAME_TO_CODE = {v: k for k, v in ENDPOINT_NAMES.items()}

    Tend_arr = np.atleast_1d(np.asarray(Tend, dtype=float))
    X = tuple(X)

    if endpoints is None:
        endpoint_codes = tuple(model.active_endpoints[dataset])
    else:
        if isinstance(endpoints, (str, int, np.integer)):
            endpoints = (endpoints,)  # allow a single endpoint, e.g. endpoints='reproduction'
        endpoint_codes = tuple(
            ep if isinstance(ep, (int, np.integer)) else NAME_TO_CODE[ep]
            for ep in endpoints
        )

    if conc_bounds is None:
        concmax = model.concstruct_list[dataset].concmax
        chigh = max(concmax.max() * 1e3, 1.0)
        clow = max(concmax.max() * 1e-6, 1e-10)
        conc_bounds = (clow, chigh)

    basepars = model.parvals.copy()
    basepars[model.islog] = 10 ** basepars[model.islog]
    modelpars = model.build_dataset_parameters(basepars, dataset)

    core = model.calc_ecx_core(modelpars, Tend_arr, X, endpoint_codes, conc_bounds, max_expand, xtol,
                                plateau_tol=plateau_tol)

    results = {ENDPOINT_NAMES[ep]: core[ep] for ep in endpoint_codes}
    results['time'] = Tend_arr

    if ci:
        if parspace is None:
            raise ValueError("parspace must be supplied when ci=True")

        args = [
            (pars, model.parvals, parspace.posfree, model.islog, dataset,
             Tend_arr, X, endpoint_codes, conc_bounds, max_expand, xtol, plateau_tol)
            for pars in parspace.propagationset
        ]

        if multicore:
            with mp.Pool(n_cores) as pool:
                allruns = pool.starmap(model.worker_ecx, args)
        else:
            allruns = [model.worker_ecx(*arg) for arg in args]

        for ep in endpoint_codes:
            name = ENDPOINT_NAMES[ep]
            for x in X:
                stacked = np.vstack([run[ep][x] for run in allruns])
                results[name]['%s_lo' % x] = np.nanmin(stacked, axis=0)
                results[name]['%s_up' % x] = np.nanmax(stacked, axis=0)

    if verbose:
        for ep in endpoint_codes:
            name = ENDPOINT_NAMES[ep]
            label = 'LCx' if ep == 0 else 'ECx'
            print(f"\n{label} for endpoint '{name}' (dataset {dataset}):")
            header = "Time".ljust(10) + "".join(("%s%%" % x).ljust(14) for x in X)
            print(header)
            for it, t in enumerate(Tend_arr):
                row = ("%.4g" % t).ljust(10)
                for x in X:
                    row += ("%.4g" % results[name][x][it]).ljust(14)
                print(row)

    return results


def calc_dose_response(
    model,
    Tend,
    endpoints=None,
    dataset=0,
    x_values=None,
    n_points=49,
    conc_bounds=None,
    max_expand=60,
    xtol=1e-8,
    plateau_tol=1e-6,
    ci=False,
    parspace=None,
    multicore=True,
    plot=True,
    verbose=False,
    figsize=None,
):
    """
    Calculate (and, by default, plot) a dose-response curve for one or more
    endpoints, at a single fixed exposure duration.

    This is built directly on top of calc_ecx: instead of a handful of
    named effect levels (e.g. EC10/EC50), it computes ECx/LCx over a fine
    grid of effect levels spanning 1% to 99% effect (by default). Each
    resulting (x, ECx) pair is then reframed as a point on the classical
    dose-response curve: concentration = ECx(x) on the x-axis, response =
    (100 - x)% of the unexposed control on the y-axis. Connecting these
    points (in increasing concentration) traces out the usual sigmoid
    response-vs-concentration curve. If more than one endpoint is
    requested, one subplot is drawn per endpoint.

    Parameters
    ----------
    model : DEBtox2019models
        Model instance holding the (fitted) parameter vector (model.parvals).
    Tend : float
        The single exposure duration at which the dose-response curve is
        evaluated (in the model's time unit).
    endpoints : {'survival', 'length', 'reproduction'}, int, or iterable thereof, optional
        Endpoint(s) to compute/plot. Defaults to those active for `dataset`
        (model.active_endpoints[dataset]).
    dataset : int
        Dataset whose parameterization is used to run the model.
    x_values : array-like of float, optional
        Effect levels (in percent, 0 < x < 100) to evaluate. Defaults to
        `n_points` values evenly spaced between 1% and 99%. Overrides
        `n_points` when given.
    n_points : int
        Number of effect levels between 1% and 99% (inclusive) used to
        build `x_values` when it is not given directly.
    conc_bounds, max_expand, xtol
        Passed through to calc_ecx (see there for details).
    plateau_tol : float or None
        Passed through to calc_ecx. Some endpoints plateau before reaching
        high effect levels (e.g. body length has a hard floor - growth
        cannot shrink below half the starting length - so no concentration
        reaches, say, 99% effect on length). Rather than exhausting the
        full concentration-bracket expansion budget for every such x, the
        search for a given (endpoint, x) gives up early (NaN, correctly
        excluded from the plotted curve) once the effect stops changing
        appreciably as concentration keeps increasing. Set to None to
        disable and always use the full max_expand budget.
    ci : bool
        If True, also compute (and, if plot=True, shade) the 95% CI band
        on the concentration axis, using parspace.propagationset (same
        mechanism as calc_ecx).
    parspace : PyParspace, optional
        Required when ci=True.
    multicore : bool
        Use multiprocessing for the CI propagation (passed to calc_ecx).
    plot : bool
        Produce the dose-response figure (one subplot per endpoint).
    verbose : bool
        Passed through to calc_ecx; left False by default here since a
        per-effect-level table with `n_points` columns is impractically
        wide to print.
    figsize : tuple, optional
        Figure size; defaults to (6 * n_endpoints, 5).

    Returns
    -------
    dict
        results[endpoint_name] -> dict with:
            'x': the effect levels used (percent, ascending),
            'conc': the corresponding ECx/LCx concentrations,
            'response': 100 - x (percent of the unexposed control),
        plus 'conc_lo' / 'conc_up' (CI bounds on the concentration) if
        ci=True. If plot=True, also results['figure'].
    """
    if x_values is None:
        x_values = np.linspace(1.0, 99.0, n_points)
    else:
        x_values = np.asarray(x_values, dtype=float)

    Tend_val = float(Tend)  # a dose-response curve is defined at one fixed duration

    ecx_results = calc_ecx(
        model, Tend_val, X=x_values, endpoints=endpoints, dataset=dataset,
        conc_bounds=conc_bounds, ci=ci, parspace=parspace, multicore=multicore,
        max_expand=max_expand, xtol=xtol, plateau_tol=plateau_tol, verbose=verbose,
    )

    endpoint_names = [name for name in ecx_results.keys() if name != 'time']

    results = {}
    for name in endpoint_names:
        conc = np.array([ecx_results[name][x][0] for x in x_values])
        entry = {
            'x': x_values,
            'conc': conc,
            'response': 100.0 - x_values,
        }
        if ci:
            entry['conc_lo'] = np.array([ecx_results[name]['%s_lo' % x][0] for x in x_values])
            entry['conc_up'] = np.array([ecx_results[name]['%s_up' % x][0] for x in x_values])
        results[name] = entry

    if plot:
        n = len(endpoint_names)
        if figsize is None:
            figsize = (6 * n, 5)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes[0]

        for ax, name in zip(axes, endpoint_names):
            entry = results[name]
            order = np.argsort(entry['conc'])
            conc_sorted = entry['conc'][order]
            resp_sorted = entry['response'][order]
            valid = np.isfinite(conc_sorted)

            ax.plot(conc_sorted[valid], resp_sorted[valid], '-o', ms=3, color='tab:blue')

            if ci:
                lo_sorted = entry['conc_lo'][order]
                up_sorted = entry['conc_up'][order]
                valid_ci = valid & np.isfinite(lo_sorted) & np.isfinite(up_sorted)
                ax.fill_betweenx(resp_sorted[valid_ci], lo_sorted[valid_ci], up_sorted[valid_ci],
                                  color='tab:blue', alpha=0.2)

            ax.set_xscale('log')
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Response (% of control)')
            ax.set_ylim(0, 105)
            ax.set_title("%s (t=%.4g)" % (name, Tend_val))

        fig.tight_layout()
        results['figure'] = fig

    return results


def _resolve_exposure_profile(exposure):
    """
    Normalize an exposure-profile input to (time, concentration) 1D arrays.

    Accepts either:
    - a `concclass` instance (e.g. built with `focus=True` from a raw
      time/concentration file) holding exactly one treatment/profile, so
      the same object can also be used with `concclass.plot_exposure()` to
      inspect the profile before/after calling calc_epx; or
    - a (time, concentration) pair of array-likes.
    """
    if hasattr(exposure, 'concarraytr') and hasattr(exposure, 'timetr'):
        if exposure.ntreats != 1:
            raise ValueError(
                "calc_epx expects a concclass instance with exactly one "
                "treatment/profile (got ntreats=%d); split it first if it "
                "holds several." % exposure.ntreats
            )
        return (np.asarray(exposure.timetr, dtype=float),
                np.asarray(exposure.concarraytr[0], dtype=float))
    exposure_time, exposure_conc = exposure
    return np.asarray(exposure_time, dtype=float), np.asarray(exposure_conc, dtype=float)


def calc_epx(
    model,
    exposure,
    Twin,
    X=(10, 50),
    endpoints=None,
    dataset=0,
    Tstep=1.0,
    MF_bounds=(1e-3, 1e3),
    max_expand=60,
    xtol=1e-8,
    plateau_tol=1e-6,
    prune_win=False,
    ci=False,
    parspace=None,
    multicore=True,
    verbose=True,
):
    """
    Calculate EPx / LPx values (exposure-profile multiplication factors).

    Emulates the functionality of calc_epx.m from the DEBtox2019/BYOM
    toolbox using the moving time window (MTW) method: a (typically long,
    realistic) exposure profile is scanned with a sliding window of length
    `Twin`, stepping the window start across the whole profile - including
    starts before the profile's first time point and up to its last one -
    zero-padding the head/tail whenever a window falls outside the
    recorded profile. This lets a window probe both the initial rise of
    the profile and its tail-off, in addition to the fully-covered middle.
    For every window position, bisection finds the multiplication factor
    (MF) that - applied to the whole profile - makes that single window
    (run from a fresh, undamaged organism) reach exactly X% effect
    relative to an unexposed control. The overall EPx/LPx is the minimum
    of this per-window MF curve (the window that is easiest to push to X%
    effect is the worst case), and the window start at that minimum is the
    worst-case window time (see the `{x}_worst_time` / `{x}_curve` entries
    in the returned dict, and plot_epx_results for visualizing them).
    Survival is interpreted as LPx (percentage additional mortality);
    length and reproduction are interpreted as EPx (percentage reduction
    relative to the control).

    Parameters
    ----------
    model : DEBtox2019models
        Model instance holding the (fitted) parameter vector (model.parvals).
    exposure : concclass or (time, concentration)
        The exposure profile to be scaled and scanned. Either a `concclass`
        instance with a single treatment (allowing e.g. `exposure.plot_exposure()`
        to inspect the profile), or a plain (time, concentration) pair of
        1D array-likes.
    Twin : float or array-like
        Time window length(s) over which the effect is evaluated, in the
        model's time unit.
    X : iterable of float
        Effect levels in percent, e.g. (10, 50) for EP10/EP50. Must be in [0, 100).
    endpoints : {'survival', 'length', 'reproduction'}, int, or iterable thereof, optional
        Endpoint(s) to evaluate; endpoints not requested are skipped
        entirely. Defaults to those active for `dataset`
        (model.active_endpoints[dataset]).
    dataset : int
        Dataset whose parameterization is used to run the model
        (see model.build_dataset_parameters).
    Tstep : float
        Step by which the window start is advanced. Window starts range
        from one that just reaches the profile's first time point (so its
        trailing part probes the initial rise of the profile) up to one
        starting exactly at the profile's last time point (so its leading
        part probes the profile's tail-off); both ends are zero-padded
        where the window falls outside the recorded profile.
    MF_bounds : tuple(float, float), optional
        Initial (low, high) bracket for the multiplication-factor search;
        automatically expanded if it does not bracket the root.
    max_expand, xtol
        Passed to the bisection search (see DEBtox2019models.calc_epx_core).
    plateau_tol : float or None
        Some endpoints can plateau before reaching a requested effect level
        (e.g. body length has a hard floor - growth cannot shrink below
        half the starting length). Once the effect changes by less than
        `plateau_tol` between two successive multiplication-factor bracket
        expansion steps, that window's search gives up early (NaN) instead
        of always exhausting `max_expand`. Set to None to disable.
    prune_win : bool
        If True, skip window positions that provably cannot be the worst
        case before running any bisection on them (pyDEBtox2019 equivalent
        of prune_windows.m from BYOM): a window whose maximum concentration
        is below the largest *minimum* concentration found across all
        windows cannot be the worst case, since some other window has
        exposure at least as high everywhere. Their `mf_curve` entries stay
        NaN. Can substantially cut runtime on long profiles, but - per the
        original implementation - is not recommended when there is a
        feedback on the elimination rate combined with an
        assimilation/maintenance/growth mode of action (see
        DEBtox2019models._prune_windows_mask).
    ci : bool
        If True, also propagate the 95% CI on the EPx/LPx estimates using
        parspace.propagationset (same mechanism as calc_ecx).
    parspace : PyParspace, optional
        Required when ci=True.
    multicore : bool
        Use all physical cores. For the main (point-estimate) calculation
        this parallelizes the per-window bisections themselves (unlike
        BYOM's MATLAB implementation, which stores the exposure scenario
        in a global and therefore cannot parallelize across windows - see
        DEBtox2019models._epx_window_task); for CI propagation (ci=True)
        it parallelizes across the parameter sets in
        parspace.propagationset instead (each of which then runs its
        per-window loop serially, to avoid nested process pools).
    verbose : bool
        Print a summary table.

    Returns
    -------
    dict
        results['window']: evaluation window lengths.
        results[endpoint_name][x] -> np.ndarray of EPx/LPx values aligned with Twin.
        results[endpoint_name]['{x}_worst_time'] -> np.ndarray (aligned with
            Twin) of the window start time at which the worst-case (i.e.
            minimal-MF) window was found.
        results[endpoint_name]['{x}_curve'] -> list (aligned with Twin) of
            (window_starts, mf_curve) tuples: the full per-window critical
            multiplication-factor curve, e.g. for plotting with
            plot_epx_results.
        If ci=True, also results[endpoint_name]['{x}_lo'] / '{x}_up'
        (CI bounds on the EPx/LPx value only).
    """
    ENDPOINT_NAMES = {0: 'survival', 1: 'length', 2: 'reproduction'}
    NAME_TO_CODE = {v: k for k, v in ENDPOINT_NAMES.items()}

    exposure_time, exposure_conc = _resolve_exposure_profile(exposure)
    Twin_arr = np.atleast_1d(np.asarray(Twin, dtype=float))
    X = tuple(X)

    if endpoints is None:
        endpoint_codes = tuple(model.active_endpoints[dataset])
    else:
        if isinstance(endpoints, (str, int, np.integer)):
            endpoints = (endpoints,)  # allow a single endpoint, e.g. endpoints='reproduction'
        endpoint_codes = tuple(
            ep if isinstance(ep, (int, np.integer)) else NAME_TO_CODE[ep]
            for ep in endpoints
        )

    basepars = model.parvals.copy()
    basepars[model.islog] = 10 ** basepars[model.islog]
    modelpars = model.build_dataset_parameters(basepars, dataset)

    core = model.calc_epx_core(modelpars, exposure_time, exposure_conc, Twin_arr, X,
                                endpoint_codes, Tstep, MF_bounds, max_expand, xtol,
                                prune_win=prune_win, multicore=multicore,
                                plateau_tol=plateau_tol)

    results = {}
    for ep in endpoint_codes:
        name = ENDPOINT_NAMES[ep]
        results[name] = {}
        for x in X:
            results[name][x] = core[ep][x]['value']
            results[name]['%s_worst_time' % x] = core[ep][x]['worst_time']
            results[name]['%s_curve' % x] = list(
                zip(core[ep][x]['window_starts'], core[ep][x]['mf_curve'])
            )
    results['window'] = Twin_arr

    if ci:
        if parspace is None:
            raise ValueError("parspace must be supplied when ci=True")

        args = [
            (pars, model.parvals, parspace.posfree, model.islog, dataset,
             exposure_time, exposure_conc, Twin_arr, X, endpoint_codes,
             Tstep, MF_bounds, max_expand, xtol, prune_win, plateau_tol)
            for pars in parspace.propagationset
        ]

        if multicore:
            with mp.Pool(n_cores) as pool:
                allruns = pool.starmap(model.worker_epx, args)
        else:
            allruns = [model.worker_epx(*arg) for arg in args]

        for ep in endpoint_codes:
            name = ENDPOINT_NAMES[ep]
            for x in X:
                stacked = np.vstack([run[ep][x]['value'] for run in allruns])
                results[name]['%s_lo' % x] = np.nanmin(stacked, axis=0)
                results[name]['%s_up' % x] = np.nanmax(stacked, axis=0)

    if verbose:
        for ep in endpoint_codes:
            name = ENDPOINT_NAMES[ep]
            label = 'LPx' if ep == 0 else 'EPx'
            print(f"\n{label} for endpoint '{name}' (dataset {dataset}):")
            header = "Window".ljust(10) + "".join(("%s%%" % x).ljust(14) for x in X)
            print(header)
            for iw, tw in enumerate(Twin_arr):
                row = ("%.4g" % tw).ljust(10)
                for x in X:
                    row += ("%.4g" % results[name][x][iw]).ljust(14)
                print(row)
            worst_header = "Worst t".ljust(10) + "".join(("%s%%" % x).ljust(14) for x in X)
            print(worst_header)
            for iw, tw in enumerate(Twin_arr):
                row = ("%.4g" % tw).ljust(10)
                for x in X:
                    row += ("%.4g" % results[name]['%s_worst_time' % x][iw]).ljust(14)
                print(row)

    return results


def plot_epx_results(model, exposure, results, endpoint, x, dataset=0, twin_index=0,
                      n_fine=300, figsize_mf=(7, 5), figsize_window=(11, 5)):
    """
    Produce the two diagnostic figures for a calc_epx result.

    Figure 1 ("MF curve"): the per-window critical multiplication factor
    (MF) as a function of the time-window start along the exposure
    profile, with the worst-case (minimum, i.e. the EPx/LPx value) window
    marked.

    Figure 2 ("worst-case window"): for that worst-case window, the
    EPx/LPx-scaled exposure profile on the left, and the endpoint
    trajectory over the window compared to the unexposed control on the
    right.

    Parameters
    ----------
    model : DEBtox2019models
        The same model instance passed to calc_epx.
    exposure : concclass or (time, concentration)
        The same exposure profile passed to calc_epx.
    results : dict
        The dict returned by calc_epx (must include the requested
        endpoint/x, and must have been computed with this same
        model/exposure/dataset).
    endpoint : {'survival', 'length', 'reproduction'}
    x : float
        Effect level; must be one of the X values calc_epx was called with.
    dataset : int
        Dataset whose parameterization is used to re-run the model for the
        smooth trajectory in Figure 2 (should match what calc_epx used).
    twin_index : int
        Index into the Twin array calc_epx was called with (0 for a single
        window length).
    n_fine : int
        Number of points used to draw the smooth endpoint-vs-time curves
        in Figure 2 (right panel).

    Returns
    -------
    (fig_mf, fig_window) : the two matplotlib Figure objects.
    """
    ENDPOINT_STATE_IDX = {0: 3, 1: 1, 2: 2}
    NAME_TO_CODE = {'survival': 0, 'length': 1, 'reproduction': 2}
    ep_code = NAME_TO_CODE[endpoint]
    si = ENDPOINT_STATE_IDX[ep_code]
    label = 'LPx' if ep_code == 0 else 'EPx'

    exposure_time, exposure_conc = _resolve_exposure_profile(exposure)

    window_starts, mf_curve = results[endpoint]['%s_curve' % x][twin_index]
    worst_time = results[endpoint]['%s_worst_time' % x][twin_index]
    epx_value = results[endpoint][x][twin_index]
    tw = results['window'][twin_index]

    # --- Figure 1: per-window critical MF as a function of window start ---
    fig_mf, ax_mf = plt.subplots(figsize=figsize_mf)
    ax_mf.plot(window_starts, mf_curve, '-', color='tab:blue')
    if np.isfinite(worst_time):
        ax_mf.plot(worst_time, epx_value, 'o', color='tab:red',
                   label='worst case (%s%s = %.4g)' % (label, x, epx_value))
        ax_mf.legend()
    ax_mf.set_yscale('log')
    ax_mf.set_xlabel('Window start time')
    ax_mf.set_ylabel('Critical multiplication factor')
    ax_mf.set_title("%s%s vs. window start ('%s', Twin=%.4g)" % (label, x, endpoint, tw))
    fig_mf.tight_layout()

    # --- Figure 2: exposure and endpoint response for the worst-case window ---
    fig_window, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize_window)

    if not np.isfinite(worst_time):
        ax_left.set_title('No worst-case window found')
        ax_right.set_title('No worst-case window found')
        fig_window.tight_layout()
        return fig_mf, fig_window

    basepars = model.parvals.copy()
    basepars[model.islog] = 10 ** basepars[model.islog]
    modelpars = model.build_dataset_parameters(basepars, dataset)

    t_list, c_list = model._window_profile(exposure_time, exposure_conc, worst_time, tw)

    ax_left.plot(t_list, epx_value * c_list, color='tab:blue')
    ax_left.fill_between(t_list, epx_value * c_list, color='tab:blue', alpha=0.2)
    ax_left.set_xlabel('Time in window')
    ax_left.set_ylabel('Exposure concentration (x %s%s)' % (label, x))
    ax_left.set_title('Worst-case window exposure (start=%.4g)' % worst_time)

    t_fine = np.linspace(0.0, tw, n_fine)
    sol_worst = model.calc_model(epx_value * c_list, t_list, modelpars, model.moa, model.feedb,
                                  timeext=t_fine)
    sol_control = model.calc_model(np.zeros(2), np.array([0.0, tw]), modelpars, model.moa,
                                    model.feedb, timeext=t_fine)

    ax_right.plot(t_fine, sol_worst[si], color='tab:blue', label='Exposed (x %s%s)' % (label, x))
    ax_right.plot(t_fine, sol_control[si], '--', color='tab:gray', label='Control')
    ax_right.set_xlabel('Time in window')
    ax_right.set_ylabel(endpoint.capitalize())
    ax_right.set_title('%s in worst-case window vs. control' % endpoint.capitalize())
    ax_right.legend()

    fig_window.tight_layout()

    return fig_mf, fig_window


def build_dataset_variants(ccl, lcl, rcl, scl, control_type='both'):
    full_ds   = completedataset(concdata=ccl, lendata=lcl, reprodata=rcl, survdata=scl)
    # controls = by label value; your control values are 0 and/or 0.1 in the first row/headers
    complement_selector = None
    if control_type == 'both':
        control_selector = lambda labels: np.isin(labels.astype(float), [0.0, 0.1])  # adapt if labels are strings
    elif control_type == 'control':
        control_selector = lambda labels: labels.astype(float) == 0.0
        complement_selector = lambda labels: labels.astype(float) != 0.1 # pop out the other control if present
    elif control_type == 'solvent':
        control_selector = lambda labels: labels.astype(float) == 0.1
        complement_selector = lambda labels: labels.astype(float) != 0.0 # pop out the other control if present
    else:
        raise ValueError("control_type must be 'both' | 'control' | 'solvent'.")
    control_ds = full_ds.subset(control_selector)
    if complement_selector is not None:
        full_ds_compl = full_ds.subset(complement_selector)
    return full_ds, control_ds, full_ds_compl if complement_selector is not None else None



class DEBparameters:
    def __init__(self, DEBpars, ndatasets=1):
        """
        Parse DEB parameter specification and expand parameters according to
        sharing rules (shared, dataset-specific, grouped).

        Parameters
        ----------
        DEBpars : dict
            Parsed JSON dictionary of DEB parameters.
        ndatasets : int
            Number of datasets in the model.
        """

        import numpy as np

        self.DEBpars = DEBpars
        self.ndatasets = ndatasets

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def normalize(name):
            return name.lower()

        def base_name(name):
            return name.split("_g")[0].split("_ds")[0]

        # ------------------------------------------------------------------
        # Containers for expanded parameters
        # ------------------------------------------------------------------
        full_vals = []
        full_names = []
        full_isfree = []
        full_islog = []
        full_low = []
        full_high = []
        full_owner = []

        # ------------------------------------------------------------------
        # 1. Global parameters (always shared)
        # ------------------------------------------------------------------
        for pname, val in DEBpars["global_parameters"].items():
            n = normalize(pname)
            full_vals.append(val)
            full_names.append(n)
            full_isfree.append(False)
            full_islog.append(False)
            full_low.append(0.0)
            full_high.append(0.0)
            full_owner.append(-1)

        # ------------------------------------------------------------------
        # Generic block parser (physio / special / tox)
        # ------------------------------------------------------------------
        def parse_block(block):
            for pname, pinfo in block.items():
                pname_n = normalize(pname)

                # ---- CASE 1: grouped parameters ---------------------------
                if "groups" in pinfo:
                    groups = pinfo["groups"]
                    values = pinfo["value"]

                    if len(groups) != len(values):
                        raise ValueError(
                            f"{pname}: length of 'groups' and 'value' must match"
                        )

                    for ig, (ds_group, v) in enumerate(zip(groups, values)):
                        ds_group = list(ds_group)
                        for d in ds_group:
                            if d < 0 or d >= ndatasets:
                                raise ValueError(
                                    f"{pname}: dataset index {d} out of bounds"
                                )

                        full_vals.append(v)
                        full_names.append(f"{pname_n}_g{ig}")
                        full_isfree.append(pinfo["fixed"] == False)
                        full_islog.append(pinfo["islog"] == True)
                        full_low.append(pinfo["min"])
                        full_high.append(pinfo["max"])
                        full_owner.append(ds_group)

                # ---- CASE 2: fully dataset-specific -----------------------
                elif pinfo.get("scope", "shared") == "dataset":
                    values = pinfo["value"]
                    datasets = pinfo.get("datasets", list(range(ndatasets)))

                    if len(values) != len(datasets):
                        raise ValueError(
                            f"{pname}: 'value' and 'datasets' length mismatch"
                        )

                    for d, v in zip(datasets, values):
                        if d < 0 or d >= ndatasets:
                            raise ValueError(
                                f"{pname}: dataset index {d} out of bounds"
                            )

                        full_vals.append(v)
                        full_names.append(f"{pname_n}_ds{d}")
                        full_isfree.append(pinfo["fixed"] == False)
                        full_islog.append(pinfo["islog"] == True)
                        full_low.append(pinfo["min"])
                        full_high.append(pinfo["max"])
                        full_owner.append([d])

                # ---- CASE 3: shared ---------------------------------------
                else:
                    full_vals.append(pinfo["value"])
                    full_names.append(pname_n)
                    full_isfree.append(pinfo["fixed"] == False)
                    full_islog.append(pinfo["islog"] == True)
                    full_low.append(pinfo["min"])
                    full_high.append(pinfo["max"])
                    full_owner.append(-1)

        # ------------------------------------------------------------------
        # 2. Parse parameter blocks
        # ------------------------------------------------------------------
        parse_block(DEBpars["physiological_model"])
        parse_block(DEBpars["special_cases"])
        parse_block(DEBpars["tox_parameters"])

        # ------------------------------------------------------------------
        # 3. Convert to numpy arrays
        # ------------------------------------------------------------------
        self.full_list = np.array(full_vals, dtype=float)
        self.full_names = np.array(full_names, dtype=object)
        self.full_base_names = np.array([base_name(n) for n in self.full_names])
        self.full_isfree = np.array(full_isfree, dtype=bool)
        self.full_islog = np.array(full_islog, dtype=bool)
        self.full_lowlim = np.array(full_low, dtype=float)
        self.full_uplim = np.array(full_high, dtype=float)
        self.par_dataset_map = np.array(full_owner, dtype=object)

        # ------------------------------------------------------------------
        # 4. Convert bounds to log-space where needed
        # ------------------------------------------------------------------
        for i in range(len(self.full_list)):
            if self.full_islog[i]:
                self.full_lowlim[i] = np.log10(self.full_lowlim[i])
                self.full_uplim[i] = np.log10(self.full_uplim[i])

        # ------------------------------------------------------------------
        # 5. Initialize free-parameter positions
        # ------------------------------------------------------------------
        self.update_posfree()

    # =========================================================
    # Parameter‑selection helpers
    # =========================================================
    def set_fixfree_all(self, isfree):
        self.full_isfree[:] = isfree
        self.update_posfree()

    def set_freefix_parameters(self, parname, isfree):
        mask = self.full_base_names == parname
        if not np.any(mask):
            raise ValueError(f"Parameter '{parname}' not found")
        self.full_isfree[mask] = isfree
        self.update_posfree()

    def set_freefix_parameters_list(self, parnames, isfree):
        for name in parnames:
            self.set_freefix_parameters(name, isfree)

    def set_free_onlyone(self, parname, isfree=True):
        self.full_isfree[:] = False
        mask = self.full_base_names == parname
        if not np.any(mask):
            raise ValueError(f"Parameter '{parname}' not found")
        self.full_isfree[mask] = isfree
        self.update_posfree()

    def fixfree_physio_pars(self, isfree=False):
        physio_keys = {k.lower() for k in self.DEBpars["physiological_model"]}
        special_keys = {k.lower() for k in self.DEBpars["special_cases"]}

        physio_names = np.unique([
            n for n in self.full_base_names
            if ((n in physio_keys) or (n in special_keys))
        ])
        for n in physio_names:
            self.set_freefix_parameters(n, isfree)

    def fixfree_tox_pars(self, isfree=False):
        for n in self.DEBpars["tox_parameters"]:
            self.set_freefix_parameters(n, isfree)

    def update_posfree(self):
        self.posfree = np.where(self.full_isfree)[0]

    # =========================================================
    # Limits (fixed using base names)
    # =========================================================
    def preset_toxlimits(self, moa, feedb, concclass):
        """
        Estimate lower and upper bounds for toxicity parameters
        based on exposure data, mode of action (moa) and feedback (feedb).
    
        Limits are applied to *all expanded instances* of a parameter
        (shared and dataset-specific) using base parameter names.
        """
    
        treatments = concclass.concmax[concclass.concmax > 0]
    
        if len(treatments) == 0:
            raise ValueError("Cannot preset toxicity limits: no non-zero concentrations.")
    
        # ------------------------------------------------------------------
        # Helper to set limits by logical (base) parameter name
        # ------------------------------------------------------------------
        touched_names = set()

        def set_limits(parname, low, high):
            mask = self.full_base_names == parname
            self.full_lowlim[mask] = low
            self.full_uplim[mask] = high
            touched_names.add(parname)
    
        # ------------------------------------------------------------------
        # kd (dominant rate constant)
        # ------------------------------------------------------------------
        kdlowlim = 0.01
        kduplim = 10.0
        set_limits("kd", kdlowlim, kduplim)
    
        # ------------------------------------------------------------------
        # zb, zs (damage thresholds)
        # ------------------------------------------------------------------
        zb_low = treatments.min() * (1 - np.exp(-kdlowlim * (4.0 / 24.0)))
        zb_up  = treatments.max() * 0.99
        set_limits("zb", zb_low, zb_up)
        set_limits("zs", zb_low, zb_up)
    
        # Special case: damage amplification feedback
        if feedb[0] == 1 and feedb[1] == 0:
            set_limits("zb", zb_low, 2.0 * treatments.max())
            set_limits("zs", zb_low, 2.0 * treatments.max())
    
        # ------------------------------------------------------------------
        # bs (hazard slope, usually log-scale)
        # ------------------------------------------------------------------
        bslowlim = -np.log(0.9) / (treatments.max() * concclass.time.max())
        bsuplim = (
            (2 ** 2 * 0.95) /
            (0.01 * treatments.max() *
             np.exp(-kdlowlim * concclass.time.max() * 0.5))
        )
    
        set_limits("bs", bslowlim, bsuplim)
    
        # ------------------------------------------------------------------
        # bb (stress slope; depends on MoA)
        # ------------------------------------------------------------------
        if moa[0] == 1:      # assimilation
            bblow = 0.2 / treatments.max()
            bbup  = 200 / (treatments.max() * (1 - np.exp(-kdlowlim * concclass.time.max())))
        elif moa[1] == 1:    # maintenance
            bblow = 0.2 / treatments.max()
            bbup  = 10 / (treatments.max() * (1 - np.exp(-kdlowlim * concclass.time.max())))
        elif moa[2] == 1:    # growth
            bblow = 0.2 / treatments.max()
            bbup  = 10 / (treatments.max() * (1 - np.exp(-kdlowlim * concclass.time.max())))
        elif moa[3] == 1:    # reproduction
            bblow = 0.5 / treatments.max()
            bbup  = 2000 / (treatments.max() * (1 - np.exp(-kdlowlim * concclass.time.max())))
        elif moa[4] == 1:    # hazard
            bblow = 0.2 / treatments.max()
            bbup  = 200 / (treatments.max() * (1 - np.exp(-kdlowlim * concclass.time.max())))
        else:
            # fallback (conservative)
            bblow = 0.1 / treatments.max()
            bbup  = 100 / treatments.max()
    
        set_limits("bb", bblow, bbup)
    
        # ------------------------------------------------------------------
        # Convert limits to log-space where needed. Only the bounds just set
        # above (in linear space) need this; every other parameter's bounds
        # are already in log10-space from __init__, so touching them again
        # here would double-apply log10 and corrupt them.
        # ------------------------------------------------------------------
        for i in range(len(self.full_names)):
            if self.full_islog[i] and self.full_base_names[i] in touched_names:
                self.full_lowlim[i] = np.log10(self.full_lowlim[i])
                self.full_uplim[i] = np.log10(self.full_uplim[i])
