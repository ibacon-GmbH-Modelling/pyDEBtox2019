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
            ax[k,i].set_ylim=[0,1.1]
            if CI:
                ax[k,i].fill_between(tevals, low[:,3], upp[:,3], color='gray', alpha=0.5)
            #ax[k,i].set_xlabel("Time (d)")
            if i == 0:
                ax[k,i].set_ylabel("Survival fraction")
        ax[lenendpoints-1,i].set_xlabel("Time (d)") # add the xlabel only to the last row of plots
    plt.tight_layout()


def get_survival_data(model, modelsolcontainer, nd):

    ntreats = model.concstruct_list[nd].ntreats

    modelvals = np.array([
        modelsolcontainer[nd][i][3]
        for i in range(ntreats)
    ])

    datavals = model.survstruct_list[nd].survprobstreat
    counts = model.survstruct_list[nd].survarrtreat

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

def validation(full_ds, debparameterclass, parspace_tox, CI=True, multicore=True,wmeans=False):
    """
    Validation for a new dataset using the parameters obtained from a previous calibration.
    This function should be called only after the physiological model for the new dataset has
    been refitted.
    Arguments:
    - full_ds: the full dataset for the new data (including controls)
    - debparameterclass: the DEBparameters instance containing the parameters relative to the new dataset
    - parspace_tox: the PyParspace instance from the previous calibration of the toxicity parameters
    """
    # refit physiological model to new data
    debparameterclass.fixfree_tox_pars(isfree=False) # to make sure tox paramters remain fixed
    # copy the tox parameters from the parspace_tox to the debparameterclass.
    changedpars = parspace_tox.model.parvals[parspace_tox.model.isfree]
    namechangedpars = parspace_tox.model.parnames[parspace_tox.model.isfree]
    for name, val in zip(namechangedpars, changedpars):
        print("Updating parameter %s to value %f from the calibration"%(name,val))
        debparameterclass.full_list[debparameterclass.full_names == name] = val
    debparameterclass.set_fixfree_all(isfree=False)
    debparameterclass.set_freefix_parameters_list(namechangedpars, isfree=True)
    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameterclass,
                                       parspace_tox.model.moa,
                                       parspace_tox.model.feedb,
                                       parspace_tox.model.Tbp,solver='LSODA')
    physioparspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)
    # copy the propagation set into the new instance of parspace.
    physioparspace.propagationset = parspace_tox.propagationset # DANGER!!! NEEDS TESTING!!
    plot_DEBresults(physioparspace,CI=CI,multicore=multicore,wmeans=wmeans)
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
        physio_names = np.unique([
            n for n in self.full_base_names
            if n in self.DEBpars["physiological_model"]
            or n in self.DEBpars["special_cases"]
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
        def set_limits(parname, low, high):
            mask = self.full_base_names == parname
            self.full_lowlim[mask] = low
            self.full_uplim[mask] = high
    
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
        # Convert limits to log-space where needed
        # ------------------------------------------------------------------
        for i in range(len(self.full_names)):
            if self.full_islog[i]:
                self.full_lowlim[i] = np.log10(self.full_lowlim[i])
                self.full_uplim[i] = np.log10(self.full_uplim[i])
