'''
classes and functions for the DEBtox2019 handling of data and parameters
'''

import numpy as np
import matplotlib.pyplot as plt
from .readin import completedataset

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
    ax = fig.subplots(lenendpoints,len(treatmentnames))
    for i in range(len(treatmentnames)):
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

def preset_toxlimits(debparameterclass, moa, feedb, concclass):
    '''
    This function automatically estimates the lower and upper boundary
    of the toxicity parameters based on exposure and feedback mechanisms 
    '''
    # set limits of the parameters
    treatments=concclass.concmax[concclass.concmax>0]
    
    # kd parameter
    kdlowlim = 0.01
    kduplim = 10
    debparameterclass.full_lowlim[debparameterclass.full_names=='kd'] = kdlowlim
    debparameterclass.full_uplim[debparameterclass.full_names=='kd'] = kduplim
    
    # zb parameter
    debparameterclass.full_lowlim[debparameterclass.full_names=='zb']  = treatments.min()*(1-np.exp(-kdlowlim*(4./24.)))
    debparameterclass.full_uplim[debparameterclass.full_names=='zb']  = treatments.max()*0.99

    # zs parameter
    debparameterclass.full_lowlim[debparameterclass.full_names=='zs']  = treatments.min()*(1-np.exp(-kdlowlim*(4./24.)))
    debparameterclass.full_uplim[debparameterclass.full_names=='zs']  = treatments.max()*0.99
    # for this specific combination, damage can be larger than external concentration
    if feedb[0] == 1 & feedb[1] == 0:
        debparameterclass.full_uplim[debparameterclass.full_names=='zb'] = 2*treatments.max() # so increase the threshold
        debparameterclass.full_uplim[debparameterclass.full_names=='zs']  = 2*treatments.max()
    
    # bb and bs parameters. These are usually in log scale, so the limits need to be given in log scale as well.
    bslowlim = -np.log(0.9) / (treatments.max()*concclass.time.max())
    bsuplim = (2**2*0.95) /(0.01*treatments.max()*np.exp(-kdlowlim*concclass.time.max()*0.5))
    debparameterclass.full_lowlim[debparameterclass.full_names=='bs']  = bslowlim
    debparameterclass.full_uplim[debparameterclass.full_names=='bs']   = bsuplim
    # if debparameterclass.full_isfree[debparameterclass.full_names=='bb'] == 1:
    if moa[0] == 1:
        bblowlim  = 0.2 / treatments.max()
        bbuplim = 200 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
    elif moa[1] == 1:
        bblowlim = 0.2 / treatments.max()
        bbuplim = 10 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
    elif moa[2] == 1:
        bblowlim = 0.2 / treatments.max()
        bbuplim = 10 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
    elif moa[3] == 1:
        bblowlim = 0.5 / treatments.max()
        bbuplim = 2000 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
    elif moa[4] == 1:
        bblowlim = 0.2 / treatments.max()
        bbuplim = 200 / (treatments.max() * (1-np.exp(-kdlowlim*concclass.time.max())))
    debparameterclass.full_lowlim[debparameterclass.full_names=='bb'] = bblowlim
    debparameterclass.full_uplim[debparameterclass.full_names=='bb'] = bbuplim

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
