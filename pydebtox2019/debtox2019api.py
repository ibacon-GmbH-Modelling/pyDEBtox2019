'''
classes and functions for the DEBtox2019 handling of data and parameters
'''

import numpy as np
import matplotlib.pyplot as plt
from .parspace import parspace as ps
from .readin import completedataset

import multiprocessing as mp
import psutil
n_cores = psutil.cpu_count(logical=False) # to have the number of physical cores only


def plot_DEBresults(parspaceres, CI=True, multicore=True):
    print("plotting the results")
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
    ax = fig.subplots(lenendpoints,len(treatmentnames))
    for i in range(len(treatmentnames)):
        sol.append(parspaceres.model.calc_model(parspaceres.model.concstruct_list[0].concarray[i],
                                      parspaceres.model.concstruct_list[0].time,
                                      10**(parspaceres.model.parvals)*parspaceres.model.islog + parspaceres.model.parvals*(~parspaceres.model.islog),
                                      parspaceres.model.moa,
                                      parspaceres.model.feedb,
                                      tevals))
        if CI:
            solci=np.zeros((len(parspaceres.propagationset),len(tevals),4))                
            # ---- prepare constant arguments
            parvals   = parspaceres.model.parvals
            posfree   = parspaceres.posfree
            concarray = parspaceres.model.concstruct_list[0].concarray[i]
            time      = parspaceres.model.concstruct_list[0].time
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
        ax[0,i].plot(ccl.time,ccl.concarray[i])
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
            ax[k,i].set_xlabel("Time (d)")
            if i == 0:
                ax[k,i].set_ylabel("Survival fraction")
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



class DEBparameters:
    def __init__(self, DEBpars):
        self.DEBpars = DEBpars
        self.global_parvals = np.array(list(DEBpars['global_parameters'].values()))
        self.global_parnames = np.array(list(DEBpars['global_parameters'].keys()))
        
        self.physio_parvals = np.array([DEBpars['physiological_model'][key]['value'] for key in DEBpars['physiological_model']])
        self.physio_parnames = np.array(list(DEBpars['physiological_model'].keys()))
        self.physio_isfree = np.array([DEBpars['physiological_model'][key]['fixed']==0 for key in DEBpars['physiological_model']])
        self.physio_islog = np.array([DEBpars['physiological_model'][key]['islog']==1 for key in DEBpars['physiological_model']])
        self.physio_lowlim = np.array([DEBpars['physiological_model'][key]['min'] for key in DEBpars['physiological_model']])
        self.physio_uplim = np.array([DEBpars['physiological_model'][key]['max'] for key in DEBpars['physiological_model']])
        self.special_parvals = np.array([DEBpars['special_cases'][key]['value'] for key in DEBpars['special_cases']])
        self.special_parnames = np.array(list(DEBpars['special_cases'].keys()))
        self.special_isfree = np.array([DEBpars['special_cases'][key]['fixed']==0 for key in DEBpars['special_cases']])
        self.special_islog = np.array([DEBpars['special_cases'][key]['islog']==1 for key in DEBpars['special_cases']])
        self.special_lowlim = np.array([DEBpars['special_cases'][key]['min'] for key in DEBpars['special_cases']])
        self.special_uplim = np.array([DEBpars['special_cases'][key]['max'] for key in DEBpars['special_cases']])
        self.tox_parvals = np.array([DEBpars['tox_parameters'][key]['value'] for key in DEBpars['tox_parameters']])
        self.tox_parnames = np.array(list(DEBpars['tox_parameters'].keys()))
        self.tox_isfree = np.array([DEBpars['tox_parameters'][key]['fixed']==0 for key in DEBpars['tox_parameters']])
        self.tox_islog = np.array([DEBpars['tox_parameters'][key]['islog']==1 for key in DEBpars['tox_parameters']])
        self.tox_lowlim = np.array([DEBpars['tox_parameters'][key]['min'] for key in DEBpars['tox_parameters']])
        self.tox_uplim = np.array([DEBpars['tox_parameters'][key]['max'] for key in DEBpars['tox_parameters']])
        
        # make complete 
        self.full_list = np.concatenate([self.global_parvals,
                                         self.physio_parvals,
                                         self.special_parvals,
                                         self.tox_parvals])
        self.full_names = np.concatenate([self.global_parnames,
                                          self.physio_parnames,
                                          self.special_parnames,
                                          self.tox_parnames])
        self.full_isfree = np.concatenate([np.zeros_like(self.global_parvals, dtype=bool),
                                           self.physio_isfree,
                                           self.special_isfree,
                                           self.tox_isfree])
        self.full_islog = np.concatenate([np.zeros_like(self.global_parvals, dtype=bool),
                                            self.physio_islog,
                                            self.special_islog,
                                            self.tox_islog])
        self.full_lowlim = np.concatenate([np.full_like(self.global_parvals, 0),
                                            self.physio_lowlim,
                                            self.special_lowlim,
                                            self.tox_lowlim])
        self.full_uplim = np.concatenate([np.full_like(self.global_parvals, 0),
                                            self.physio_uplim,
                                            self.special_uplim,
                                            self.tox_uplim])
        
    def set_freefix_parameters(self, parname, isfree):
        idx = np.where(self.full_names == parname)[0][0]
        self.full_isfree[idx] = isfree
    def set_freefix_parameters_list(self, parname, isfree):
        for name in parname:
            idx = np.where(self.full_names == name)[0][0]
            self.full_isfree[idx] = isfree
    def set_free_onlyone(self, parname, isfree):
        # fix all parameters except the one given
        self.full_isfree[:] = False
        idx = np.where(self.full_names == parname)[0][0]
        self.full_isfree[idx] = isfree
    def fixfree_physio_pars(self,isfree=False):
        # option 1
        # self.full_isfree[len(self.global_parvals):len(self.global_parvals)+len(self.physio_parvals)] = False
        # option 2
        for i, name in enumerate(self.physio_parnames):
            self.set_freefix_parameters(name, isfree)
        for i, name in enumerate(self.special_parnames):
            self.set_freefix_parameters(name, isfree)
    def fixfree_tox_pars(self,isfree=False):
        # option 1
        # self.full_isfree[-len(self.tox_parvals):] = False
        # option 2
        for i, name in enumerate(self.tox_parnames):
            self.set_freefix_parameters(name, isfree)
    def update_posfree(self):
        self.posfree = np.where(self.full_isfree)[0]