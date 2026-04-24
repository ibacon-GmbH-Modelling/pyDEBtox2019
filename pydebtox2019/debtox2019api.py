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


def plot_DEBresults(parspaceres, CI=True, multicore=True):
    print("plotting the results")
    # assumes a single dataset for now
    ccl = parspaceres.model.concstruct_list[0]
    lcl = parspaceres.model.lengthstruct_list[0]
    rcl = parspaceres.model.reprostruct_list[0]
    scl = parspaceres.model.survstruct_list[0]
    sol=[]
    
    #treatmentnames = dataset.concdata.treatmentnames
    treatmentnames = parspaceres.model.concstruct_list[0].conctreatsnames
    tevals = np.linspace(np.min(parspaceres.model.concstruct_list[0].time),
                         np.max(parspaceres.model.concstruct_list[0].time),100)
    fig = plt.figure()
    lenendpoints = np.sum([1 for cl in [ccl,lcl,rcl,scl] if cl is not None])
    # print("number of endpoints to plot: ", lenendpoints)
    # print("number of treatments to plot: ", len(treatmentnames))
    ax = fig.subplots(lenendpoints,len(treatmentnames),squeeze=False)
    for i in range(len(treatmentnames)):
        # print("i: ", i)
        # print("treatment: ", treatmentnames[i])
        sol.append(parspaceres.model.calc_model(parspaceres.model.concstruct_list[0].concarraytr[i],
                                      parspaceres.model.concstruct_list[0].timetr,
                                      10**(parspaceres.model.parvals)*parspaceres.model.islog + 
                                           parspaceres.model.parvals*(~parspaceres.model.islog),
                                      parspaceres.model.moa,
                                      parspaceres.model.feedb,
                                      tevals))
        if CI:
            solci=np.zeros((len(parspaceres.propagationset),len(tevals),4))                
            # ---- prepare constant arguments
            parvals   = parspaceres.model.parvals
            posfree   = parspaceres.posfree
            concarray = parspaceres.model.concstruct_list[0].concarraytr[i]
            time      = parspaceres.model.concstruct_list[0].timetr
            islog     = parspaceres.model.islog
            moa       = parspaceres.model.moa
            feedb     = parspaceres.model.feedb
            # ---- build starmap argument list
            args = [(pars, parvals, posfree, concarray, time, islog, moa, feedb, tevals) for pars in parspaceres.propagationset]
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


def build_dataset_variants(ccl, lcl, rcl, scl, control_type='both'):
    full_ds   = completedataset(concdata=ccl, lendata=lcl, reprodata=rcl, survdata=scl)
    # controls = by label value; your control values are 0 and/or 0.1 in the first row/headers
    if control_type == 'both':
        control_selector = lambda labels: np.isin(labels.astype(float), [0.0, 0.1])  # adapt if labels are strings
    elif control_type == 'control':
        control_selector = lambda labels: labels.astype(float) == 0.0
    elif control_type == 'solvent':
        control_selector = lambda labels: labels.astype(float) == 0.1
    else:
        raise ValueError("control_type must be 'both' | 'control' | 'solvent'.")
    control_ds = full_ds.subset(control_selector)
    return full_ds, control_ds


def validation(ccl, lcl, rcl, scl, debparameterclass, parspace_tox, refit_physio = True, refit_mortality = True):
    # refit physiological model to new data
    debparameterclass.fixfree_tox_pars(isfree=False) # to make sure tox paramters remain fixed
    if refit_mortality & (scl is not None):
        debparameterclass.set_freefix_parameters("hb", isfree=True)
        pass
    if refit_physio:
        debparameterclass.fixfree_physio_pars(isfree=True)
        full_ds, control_ds = build_dataset_variants(ccl, lcl, rcl, None, control_type='both')
        debmodeltest = mm.DEBtox2019models([control_ds],
                                       debparameterclass,
                                       parspace_tox.model.moas,
                                       parspace_tox.model.feedbs,
                                       Tbp=1,solver='LSODA')
        physioparspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)
        physioparspace.run_parspace()
        plot_DEBresults(physioparspace,CI=True,multicore=True)
        # update the parameters of the physiological model
        debparameterclass.full_list = physioparspace.model.parvals
    # this assumes the same length. Might need a fix later on
    parspace_tox.model.parvals[~parspace_tox.model.is_free] = debparameterclass.full_list[~parspace_tox.model.is_free]
    plot_DEBresults(parspace_tox,CI=True,multicore=True)
    pass



class DEBparameters:
    def __init__(self, DEBpars, ndatasets=1):
        self.ndatasets = ndatasets
        self.DEBpars = DEBpars

        # ---------- helpers ----------
        def base_name(pname):
            return pname.split("_ds")[0]

        def parse_block(block):
            vals, names = [], []
            isfree, islog = [], []
            low, high = [], []
            owners = []

            for pname, pinfo in block.items():
                scope = pinfo.get("scope", "shared")

                if scope == "shared":
                    vals.append(pinfo["value"])
                    names.append(pname)
                    isfree.append(pinfo["fixed"] == 0)
                    islog.append(pinfo["islog"] == 1)
                    low.append(pinfo["min"])
                    high.append(pinfo["max"])
                    owners.append(-1)

                elif scope == "dataset":
                    ds = pinfo["datasets"]
                    values = pinfo["value"]

                    if len(ds) != len(values):
                        raise ValueError(
                            f"{pname}: datasets and values length mismatch"
                        )

                    for d, v in zip(ds, values):
                        if d < 0 or d >= self.ndatasets:
                            raise ValueError(
                                f"{pname}: dataset index {d} out of bounds"
                            )
                        vals.append(v)
                        names.append(f"{pname}_ds{d}")
                        isfree.append(pinfo["fixed"] == 0)
                        islog.append(pinfo["islog"] == 1)
                        low.append(pinfo["min"])
                        high.append(pinfo["max"])
                        owners.append(d)
                else:
                    raise ValueError(f"Unknown scope '{scope}' for {pname}")

            return vals, names, isfree, islog, low, high, owners

        # ---------- global parameters (always shared) ----------
        self.global_parvals = np.array(list(DEBpars["global_parameters"].values()))
        self.global_parnames = np.array(list(DEBpars["global_parameters"].keys()), dtype=object)

        self.global_isfree = np.zeros(len(self.global_parvals), dtype=bool)
        self.global_islog = np.zeros(len(self.global_parvals), dtype=bool)
        self.global_lowlim = np.zeros(len(self.global_parvals))
        self.global_uplim = np.zeros(len(self.global_parvals))
        self.global_owner = np.full(len(self.global_parvals), -1)

        # ---------- parse blocks ----------
        phys_vals, phys_names, phys_free, phys_log, phys_low, phys_high, phys_owner = \
            parse_block(DEBpars["physiological_model"])

        spec_vals, spec_names, spec_free, spec_log, spec_low, spec_high, spec_owner = \
            parse_block(DEBpars["special_cases"])

        tox_vals, tox_names, tox_free, tox_log, tox_low, tox_high, tox_owner = \
            parse_block(DEBpars["tox_parameters"])

        # ---------- build full vectors ----------
        self.full_list = np.array(
            list(self.global_parvals) + phys_vals + spec_vals + tox_vals
        )

        self.full_names = np.array(
            list(self.global_parnames) + phys_names + spec_names + tox_names,
            dtype=object
        )

        self.full_base_names = np.array([base_name(n) for n in self.full_names])

        self.full_isfree = np.concatenate([
            self.global_isfree,
            np.array(phys_free),
            np.array(spec_free),
            np.array(tox_free)
        ])

        self.full_islog = np.concatenate([
            self.global_islog,
            np.array(phys_log),
            np.array(spec_log),
            np.array(tox_log)
        ])

        self.full_lowlim = np.concatenate([
            self.global_lowlim,
            np.array(phys_low),
            np.array(spec_low),
            np.array(tox_low)
        ])

        self.full_uplim = np.concatenate([
            self.global_uplim,
            np.array(phys_high),
            np.array(spec_high),
            np.array(tox_high)
        ])

        self.par_dataset_map = np.concatenate([
            self.global_owner,
            np.array(phys_owner),
            np.array(spec_owner),
            np.array(tox_owner)
        ])

        # ---------- log-scale limits ----------
        for i in range(len(self.full_names)):
            if self.full_islog[i]:
                self.full_lowlim[i] = np.log10(self.full_lowlim[i])
                self.full_uplim[i] = np.log10(self.full_uplim[i])

        self.update_posfree()

    # =========================================================
    # Parameter‑selection helpers
    # =========================================================
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