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


def plot_DEBresults(parspaceres, CI=True, multicore=True, ds = -1):
    if ds == -1:
        for dataset in range(parspaceres.model.ndatasets):
            plot_DEBresults_ds(parspaceres, CI=CI, multicore=multicore, dataset=dataset)
    else:
        plot_DEBresults_ds(parspaceres, CI=CI, multicore=multicore, dataset=ds)

def plot_DEBresults_ds(parspaceres, CI=True, multicore=True, dataset=0):
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
            lcl.add_plotdata(ax[k,i],treatmentnames[i])
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
            rcl.add_plotdata(ax[k,i],treatmentnames[i])
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
            scl.add_plotdata(ax[k,i],ntreat=treatmentnames[i], scaleto1=True)
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



def efsa_basic_metrics(y_obs, y_pred, eps=1e-12):
    # rewrite these equations with the correct ones
    residuals = y_obs - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    nrmse = rmse / (np.mean(y_obs) + eps)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2 = 1 - ss_res / (ss_tot + eps)

    return dict(R2=r2, NRMSE=nrmse)


def efsa_criteria(model):
    from copy import deepcopy
    model = deepcopy(model)
    basepars = model.parvals.copy()
    basepars[model.islog] = 10 ** basepars[model.islog]
    modelsolcontainer = [None]*model.ndatasets
    for nd in range(model.ndatasets):
        modelpars = model.build_dataset_parameters(basepars, nd)
        newtime = model.timeext[nd]
        # need to make sure that there are enough time points being calculated
        # ensure precision of the ODE solution
        newtimeext = np.unique(np.concatenate((np.linspace(newtime[0],newtime[-1],max(model.min_t,len(newtime))),newtime)))
        modelcoltreatcont = [None]*model.concstruct_list[nd].ntreats
        for i in range(model.concstruct_list[nd].ntreats):
            modelsol = model.calc_model(model.concstruct_list[nd].concarraytr[i],
                                                 model.concstruct_list[nd].timetr,
                                                 modelpars,
                                                 model.moa,
                                                 model.feedb,
                                                 timeext=newtimeext)
            idx_targets = np.searchsorted(newtimeext, model.timeext[nd])
            modelsol = modelsol[:,idx_targets]
            modelcoltreatcont[i] = modelsol
        modelsolcontainer[nd] = modelcoltreatcont
    # now all the model solutions have been calculated and stored in modelsolcontainer, 
    # we can calculate the metrics for each endpoint and treatment.
    # now need to get the data element and do the calculatations of the criteria
    #return(modelsolcontainer)
    for endpoint in model.active_endpoints[nd]:
        ntreats = model.concstruct_list[nd].ntreats
        if endpoint == 0:
            # survival
            survmodelvals = np.array([modelsolcontainer[nd][x][3] for x in range(ntreats)]) ## to check properly if the timing is right
            survdatavals = model.survstruct_list[nd].survprobstreat
            # DEBUG:
            # print("model")
            # print(survmodelvals)
            # print("data")
            # print(survdatavals)
        elif endpoint == 1:
            # length
            mask = np.isin(model.timeext[nd], model.lengthstruct_list[nd].time)
            lengthmodelvals = np.array([modelsolcontainer[nd][x][1][mask] for x in range(ntreats)])
            lengthdatavals = model.lengthstruct_list[nd].meanvalstransf
            nobs = np.sum(~np.isnan(lengthdatavals))
            # DEBUG:
            # print("model")
            # print(lengthmodelvals)
            # print("data")
            # print(lengthdatavals)
            res = lengthdatavals - lengthmodelvals
            restot = lengthdatavals - np.nanmean(lengthdatavals,axis=1, keepdims=True)
            # print(np.nanmean(lengthdatavals,axis=1))
            # print(restot)
            r2 = 1 - np.nansum(res**2)/np.nansum(restot**2)
            nrmse = (np.sqrt((np.nansum(res**2))/nobs))/np.nanmean(lengthdatavals)
            print("R2 length: ", r2)
            print("NRMSE length: ", nrmse)
        elif endpoint == 2:
            # reproduction
            mask = np.isin(model.timeext[nd], model.reprostruct_list[nd].time)
            repromodelvals = np.array([modelsolcontainer[nd][x][2][mask] for x in range(ntreats)])
            reprodatavals = model.reprostruct_list[nd].meanvalstransf
            # ###DEBUG:
            # print("model")
            # print(repromodelvals)
            # print("data")
            # print(reprodatavals)
            # print("mean observations")
            # print(np.nanmean(reprodatavals))
            # print("number of observations")
            # print(np.sum(~np.isnan(reprodatavals)))
            nobs = np.sum(~np.isnan(reprodatavals))
            res = reprodatavals - repromodelvals
            restot = reprodatavals - np.nanmean(reprodatavals,axis=1, keepdims=True)
            r2 = 1 - np.nansum(res**2)/np.nansum(restot**2)
            # this needs to be fixed because the sizes are not correct
            nrmse = np.sqrt(np.nansum(res**2/nobs))/np.nanmean(reprodatavals)
            print("R2 repro: ", r2)
            print("NRMSE repro: ", nrmse)
    return(modelsolcontainer)



def efsa_survival(model, dataset):
    # observed survival probabilities
    survdata = model.survstruct_list[dataset]
    for i in range(survdata.ntreats):
        S_obs = survdata.survprobstreat[i]

    # predicted survival trajectory
    # S_pred = ... model_output_survival_at_data_times

    # SPPE (EFSA-specific, keep original formula)
    # sppe = ...

    basic = efsa_basic_metrics(S_obs, S_pred)

    return {**basic, "SPPE": sppe}



def efsa_length(model, dataset, treatment_index):
    ldata = model.lengthstruct_list[dataset]

    y_obs = ldata.flatdataclean[treatment_index]
    weights = ldata.flatweightsclean[treatment_index]

    y_pred = simulated_length_values_at_matching_times

    metrics = efsa_basic_metrics(y_obs, y_pred)

    return metrics



def efsa_repro(model, dataset, treatment_index):
    rdata = model.reprostruct_list[dataset]

    y_obs = rdata.flatdataclean[treatment_index]
    y_pred = simulated_reproduction_values

    metrics = efsa_basic_metrics(y_obs, y_pred)

    return metrics



def EFSA_quality_criteria_DEB(model, dataset):
    results = {}

    for tr in range(model.concstruct_list[dataset].ntreats):
        results[tr] = {}

        if model.survstruct_list[dataset] is not None:
            results[tr]["survival"] = efsa_survival(model, dataset, tr)

        if model.lengthstruct_list[dataset] is not None:
            results[tr]["length"] = efsa_length(model, dataset, tr)

        if model.reprostruct_list[dataset] is not None:
            results[tr]["reproduction"] = efsa_repro(model, dataset, tr)

    return results


def validation(full_ds, debparameterclass, parspace_tox):
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
    plot_DEBresults(physioparspace,CI=True,multicore=True)


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






# class DEBparameters:
#     def __init__(self, DEBpars, ndatasets=1):
#         self.DEBpars = DEBpars
#         self.ndatasets = ndatasets
        
#         def parse_parameter_block(block):
#             vals = []
#             names = []
#             isfree = []
#             islog = []
#             low = []
#             high = []
#             owners = []
        
#             for pname, pinfo in block.items():
            
#                 scope = pinfo.get("scope", "shared")
        
#                 if scope == "shared":
#                     vals.append(pinfo["value"])
#                     names.append(pname)
#                     isfree.append(pinfo["fixed"] == 0)
#                     islog.append(pinfo["islog"] == 1)
#                     low.append(pinfo["min"])
#                     high.append(pinfo["max"])
#                     owners.append(-1)
        
#                 elif scope == "dataset":
#                     ds = pinfo["datasets"]
#                     values = pinfo["value"]
        
#                     if len(ds) != len(values):
#                         raise ValueError(f"{pname}: datasets and values must match length")
        
#                     for d, v in zip(ds, values):
#                         vals.append(v)
#                         names.append(f"{pname}_ds{d}")
#                         isfree.append(pinfo["fixed"] == 0)
#                         islog.append(pinfo["islog"] == 1)
#                         low.append(pinfo["min"])
#                         high.append(pinfo["max"])
#                         owners.append(d)
        
#                 else:
#                     raise ValueError(f"Unknown scope '{scope}' for parameter {pname}")
        
#             return vals, names, isfree, islog, low, high, owners

        
#         self.global_parvals = np.array(list(DEBpars["global_parameters"].values()))
#         self.global_parnames = np.array(list(DEBpars["global_parameters"].keys()))
#         self.global_isfree = np.zeros(len(self.global_parvals), dtype=bool)
#         self.global_islog = np.zeros(len(self.global_parvals), dtype=bool)
#         self.global_lowlim = np.zeros(len(self.global_parvals))
#         self.global_uplim = np.zeros(len(self.global_parvals))
#         self.global_owner = np.full(len(self.global_parvals), -1)

#         phys_vals, phys_names, phys_free, phys_log, phys_low, phys_high, phys_owner = \
#             parse_parameter_block(DEBpars["physiological_model"])

#         spec_vals, spec_names, spec_free, spec_log, spec_low, spec_high, spec_owner = \
#             parse_parameter_block(DEBpars["special_cases"])

#         tox_vals, tox_names, tox_free, tox_log, tox_low, tox_high, tox_owner = \
#             parse_parameter_block(DEBpars["tox_parameters"])

        
#         # make complete 
#         self.full_list = np.array(
#             list(self.global_parvals) + phys_vals + spec_vals + tox_vals
#         )

#         self.full_names = np.array(
#             list(self.global_parnames) + phys_names + spec_names + tox_names,
#             dtype=object
#         )

#         self.full_base_names = np.array([self.base_name(n) for n in self.full_names])

#         self.full_isfree = np.concatenate([
#             self.global_isfree,
#             np.array(phys_free),
#             np.array(spec_free),
#             np.array(tox_free)
#         ])

#         self.full_islog = np.concatenate([
#             self.global_islog,
#             np.array(phys_log),
#             np.array(spec_log),
#             np.array(tox_log)
#         ])

#         self.full_lowlim = np.concatenate([
#             self.global_lowlim,
#             np.array(phys_low),
#             np.array(spec_low),
#             np.array(tox_low)
#         ])

#         self.full_uplim = np.concatenate([
#             self.global_uplim,
#             np.array(phys_high),
#             np.array(spec_high),
#             np.array(tox_high)
#         ])

#         self.par_dataset_map = np.concatenate([
#             self.global_owner,
#             np.array(phys_owner),
#             np.array(spec_owner),
#             np.array(tox_owner)
#         ])

    
#     def base_name(self, pname):
#         """
#         Returns logical parameter name without dataset suffix.
#         Lp_ds0 → Lp
#         Lm → Lm
#         """
#         return pname.split("_ds")[0]


#     def set_freefix_parameters(self, parname, isfree):    
#         mask = self.full_base_names == parname
#         if not np.any(mask):
#             raise ValueError(f"Parameter '{parname}' not found.")
#         self.full_isfree[mask] = isfree

#     def set_freefix_parameters_list(self, parname, isfree):
#         for name in parname:
#             self.set_freefix_parameters(name, isfree)


#     def set_free_onlyone(self, parname, isfree=True):
#         self.full_isfree[:] = False
#         mask = self.full_base_names == parname
#         if not np.any(mask):
#             raise ValueError(f"Parameter '{parname}' not found.")
#         self.full_isfree[mask] = isfree

        
#     def fixfree_physio_pars(self,isfree=False):
#         # option 1
#         # self.full_isfree[len(self.global_parvals):len(self.global_parvals)+len(self.physio_parvals)] = False
#         # option 2
#         for i, name in enumerate(self.physio_parnames):
#             self.set_freefix_parameters(name, isfree)
#         for i, name in enumerate(self.special_parnames):
#             self.set_freefix_parameters(name, isfree)

#     def fixfree_tox_pars(self,isfree=False):
#         # option 1
#         # self.full_isfree[-len(self.tox_parvals):] = False
#         # option 2
#         for i, name in enumerate(self.tox_parnames):
#             self.set_freefix_parameters(name, isfree)
    
#     def update_posfree(self):
#         self.posfree = np.where(self.full_isfree)[0]

#     def get_indices_by_param(self, parname):
#         return np.where(self.full_base_names == parname)[0]
    
#     def get_shared_indices(self):
#         return np.where(self.par_dataset_map == -1)[0]
    
#     def get_dataset_indices(self, nd):
#         return np.where(self.par_dataset_map == nd)[0]
    
#     def preset_toxlimits(self, moa, feedb, concclass):
#         '''
#         This function automatically estimates the lower and upper boundary
#         of the toxicity parameters based on exposure and feedback mechanisms 
#         '''
#         # set limits of the parameters
#         treatments=concclass.concmax[concclass.concmax>0]

#         # kd parameter
#         kdlowlim = 0.01
#         kduplim = 10
#         self.full_lowlim[self.full_names=='kd'] = kdlowlim
#         self.full_uplim[self.full_names=='kd'] = kduplim

#         # zb parameter
#         self.full_lowlim[self.full_names=='zb']  = treatments.min()*(1-np.exp(-kdlowlim*(4./24.)))
#         self.full_uplim[self.full_names=='zb']  = treatments.max()*0.99

#         # zs parameter
#         self.full_lowlim[self.full_names=='zs']  = treatments.min()*(1-np.exp(-kdlowlim*(4./24.)))
#         self.full_uplim[self.full_names=='zs']  = treatments.max()*0.99
#         # for this specific combination, damage can be larger than external concentration
#         if feedb[0] == 1 & feedb[1] == 0:
#             self.full_uplim[self.full_names=='zb'] = 2*treatments.max() # so increase the threshold
#             self.full_uplim[self.full_names=='zs']  = 2*treatments.max()

#         # bb and bs parameters. These are usually in log scale, so the limits need to be given in log scale as well.
#         bslowlim = -np.log(0.9) / (treatments.max()*concclass.time.max())
#         bsuplim = (2**2*0.95) /(0.01*treatments.max()*np.exp(-kdlowlim*concclass.time.max()*0.5))
#         self.full_lowlim[self.full_names=='bs']  = bslowlim
#         self.full_uplim[self.full_names=='bs']   = bsuplim
#         # if debparameterclass.full_isfree[debparameterclass.full_names=='bb'] == 1:
#         if moa[0] == 1:
#             bblowlim  = 0.2 / treatments.max()
#             bbuplim = 200 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
#         elif moa[1] == 1:
#             bblowlim = 0.2 / treatments.max()
#             bbuplim = 10 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
#         elif moa[2] == 1:
#             bblowlim = 0.2 / treatments.max()
#             bbuplim = 10 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
#         elif moa[3] == 1:
#             bblowlim = 0.5 / treatments.max()
#             bbuplim = 2000 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
#         elif moa[4] == 1:
#             bblowlim = 0.2 / treatments.max()
#             bbuplim = 200 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
#         self.full_lowlim[self.full_names=='bb'] = bblowlim
#         self.full_uplim[self.full_names=='bb'] = bbuplim
#         # now trasnform in log if the parameter is in log scale
#         for i in range(len(self.full_names)):
#             if self.full_islog[i]:
#                 self.full_lowlim[i] = np.log10(self.full_lowlim[i])
#                 self.full_uplim[i] = np.log10(self.full_uplim[i])