# testing script

import time
import numpy as np
import pandas as pd
import json

import pydebtox2019.models as mm
import pydebtox2019.parspace as ps
import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin

# read DEB parameters from json file


if __name__ == "__main__":

    # Open JSON file
    with open('input_pars.json') as json_file:
        DEBpars = json.load(json_file)

    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False)

    moas = np.array([0,0,0,1,0])
    feedbs =  np.array([0,0,0,0])
    tevals = np.linspace(0,21,100)

    Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(),"","ug/L")
    ccl.plot_exposure()

    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())

    Rdata = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=2)

    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())

    # isolate the controls    
    full_ds, control_ds = readin.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    
    # take only the survival data from the controls
    _,hbonly = readin.build_dataset_variants(ccl=ccl, lcl=None,rcl=None,scl=scl,control_type='both')

    # if fit_mode == "controls":
    #     ds_list = [control_ds]         # only controls
    # elif fit_mode == "full":
    #     ds_list = [full_ds]            # all treatments
    # elif fit_mode == "both":
    #     ds_list = [control_ds, full_ds]  # (advanced) multi-scenario joint fit
    # else:
    #     raise ValueError("fit_mode must be 'controls' | 'full' | 'both'.")


    # datadict = compile_dataset_dict(ccl, scl, lcl, rcl,1)
    # datadict_c = compile_dataset_dict(ccl_c, scl_c, lcl_c, rcl_c,1)
    # fulldataset = completedataset(concdata=ccl,lendata=lcl,reprodata=rcl,survdata=scl)
    # controldataset = completedataset(concdata=ccl_c,lendata=lcl_c,reprodata=rcl_c,survdata=scl_c)


    # preparation for fit of the physiological model only
    treatments=ccl.concmax[ccl.concmax>0]
    debparameters.full_lowlim[debparameters.full_names=='zb']  = treatments.min()*(1-np.exp(-0.01*(4./24.)))
    debparameters.full_uplim[debparameters.full_names=='zb']  = treatments.max()*0.99
    debparameters.full_lowlim[debparameters.full_names=='zs']  = treatments.min()*(1-np.exp(-0.01*(4./24.)))
    debparameters.full_uplim[debparameters.full_names=='zs']  = treatments.max()*0.99

    # these are in log10 scale
    debparameters.full_lowlim[debparameters.full_names=='bb']  = np.log10(0.2 / treatments.max())
    debparameters.full_uplim[debparameters.full_names=='bb']   = np.log10(200 / (treatments.max() * (1-np.exp(-0.01*ccl.time.max()))))
    debparameters.full_lowlim[debparameters.full_names=='bs']  = np.log10(-np.log(0.9) / (treatments.max()*ccl.time.max()))
    debparameters.full_uplim[debparameters.full_names=='bs']   = np.log10((2**2*0.95) /(0.01*treatments.max()*np.exp(-0.01*ccl.time.max()*0.5)))

    # debparameters.full_list = np.array([
    #     2.0000e-02, 1.0000e+00, 8.0000e-01, 6.4000e-01, 5.0000e+00,
    #     8.8000e-01, 2.3120e+00, 3.1254e+00, 1.416e-01, 9.7293e+00,
    #     1.0000e+00, 4.8140e-03, 1.0000e+00, 0.0000e+00, 0.0000e+00,
    #     0.0000e+00, 3.610e-02, 1.74237e+01, 7.80e-02, 8.375e-01,
    #     1.9070e-01])
    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters.full_list,
                                       debparameters.full_names,
                                       debparameters.full_islog, 
                                       debparameters.full_isfree, 
                                       debparameters.full_lowlim,
                                       debparameters.full_uplim,
                                       moas, feedbs, Tbp=3,
                                       solver='LSODA')
    
    # transform in log scale when needed avoiding nan values due to log(0)
    listparswlog = debparameters.full_list.copy()
    newlistpars = np.zeros_like(listparswlog)
    for i,par in enumerate(listparswlog):
        newlistpars[i] = np.log10(par) if debparameters.full_islog[i] else par
    
    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    lk = debmodeltest.log_likelihood(newlistpars[debmodeltest.posfree],newlistpars,debmodeltest.posfree)
    print(lk)
    dt2019.plot_DEBresults(parspace,CI=False,multicore=False) 
    
    
    debparameters.set_free_onlyone("hb", isfree=True)
    debhbmodel = mm.DEBtox2019models([hbonly],
                                    debparameters.full_list,
                                    debparameters.full_names,
                                    debparameters.full_islog, 
                                    debparameters.full_isfree, 
                                    debparameters.full_lowlim,
                                    debparameters.full_uplim,
                                    moas, feedbs, Tbp=3,solver='LSODA')
    parspacehb = ps.PyParspace(ps.SettingParspace(0,1), debhbmodel)
    parspacehb.profile =1
    startt = time.time()
    parspacehb.run_parspace()
    endt = time.time()
    print("Time for hb fit: ", endt-startt)
    parspacehb.replot_results()

    debparameters.full_list = parspacehb.model.parvals
    debparameters.set_freefix_parameters("hb", isfree=False)
    debparameters.set_freefix_parameters_list(["Lp","Lm","rB","Rm"], isfree=True)


    debmodeltest = mm.DEBtox2019models([control_ds],
                                       debparameters.full_list,
                                       debparameters.full_names,
                                       debparameters.full_islog, 
                                       debparameters.full_isfree, 
                                       debparameters.full_lowlim,
                                       debparameters.full_uplim,
                                       moas, feedbs, Tbp=3,solver='LSODA')

    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    lk = debmodeltest.log_likelihood(debparameters.full_list[debmodeltest.posfree],debparameters.full_list,debmodeltest.posfree)
    print(lk)

    # ## time the likelihood function
    # ## begintime = time.time()
    # ## for i in range(1000):
    # ##     lk = debmodeltest.log_likelihood(debparameters.full_list[debmodeltest.posfree],debparameters.full_list,debmodeltest.posfree)
    # ## print("Time for 1000 likelihood evaluations: ", time.time()-begintime)

    parspace.profile =1
    startt = time.time()
    parspace.run_parspace()
    endt = time.time()
    print("Time for physiological model fit: ", endt-startt)
    dt2019.plot_DEBresults(parspace,CI=True,multicore=True)
    
    '''
    debparameters.full_list = parspace.model.parvals
    debparameters.fixfree_physio_pars(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)

    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters.full_list,
                                       debparameters.full_names,
                                       debparameters.full_islog, 
                                       debparameters.full_isfree, 
                                       debparameters.full_lowlim,
                                       debparameters.full_uplim,
                                       moas, feedbs, Tbp=3,solver='LSODA')
    
    parspace_tox = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)
    parspace_tox.profile =1
    startt = time.time()
    parspace_tox.run_parspace()
    endt = time.time()
    print("Time for tox model fit: ", endt-startt)

    # # time with parallel computation
    # # ~200 seconds (physio part)
    # # 1915.45 seconds (tox part)

    # debparameters.full_list = parspace_tox.model.parvals
    # debparameters.full_list = 10**(debparameters.full_list)*debparameters.full_islog + debparameters.full_list*(~debparameters.full_islog)
    
    # parspace_tox = ps.PyParspace.load_class("test_toxfit.pkl")
    # parspace_tox.model.solver = 'LSODA' # needed because the saved object does not have this attribute
    # # startt = time.time()
    # # plot_DEBresults(parspace_tox, CI=True, multicore=True)
    # # endt = time.time()
    # # print("Time for plotting with CI (parallel): ", endt-startt)

    # startt = time.time()
    # plot_DEBresults(parspace_tox, CI=True, multicore=False)
    # endt = time.time()
    # print("Time for plotting with CI (series): ", endt-startt)
    
    
    # parspacehb = ps.PyParspace.load_class("test_hbfit.pkl")
    # parspacehb.model.solver = 'LSODA'
    # plot_DEBresults(parspacehb, CI=True, multicore=False)
    '''