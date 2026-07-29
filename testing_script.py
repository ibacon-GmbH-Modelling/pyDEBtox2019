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
    with open('input_pars_tbp3.json') as json_file:
        DEBpars = json.load(json_file)

    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False)

    moas = np.array([0,0,0,1,0])
    feedbs =  np.array([0,0,0,0])

    Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(),"","ug/L")
    ccl.plot_exposure()

    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())
    lcl.plot_data()

    Rdata = pd.read_csv("Test_Rdata_opt1.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=1)
    rcl.plot_data_cumulative()

    Rdata2 = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    rcl2 = readin.reproclass(Rdata2.to_numpy(), reprocase='individual',optcase=2)
    rcl2.plot_data_cumulative()

    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())
    scl.plot_data(scaleto1=True, label="suviving fraction")


    # isolate the controls    
    full_ds, control_ds, ph = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    
    # take only the survival data from the controls
    _,hbonly,_ = dt2019.build_dataset_variants(ccl=ccl, lcl=None,rcl=None,scl=scl,control_type='both')

    # set the parameter limits to start the grid search
    debparameters.preset_toxlimits(moas, feedbs, ccl)

    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=3,
                                       breaktime=0,
                                       solver='LSODA')
    
    # transform in log scale when needed avoiding nan values due to log(0)
    # this is needed only here as we are calling the likelihood externally.
    # In the standard workflow, the likelihood is called by the parspace class
    # that takes care internally to transform the parameters in log scale when 
    # needed and to transform them back when calling the model.



    # listparswlog = debparameters.full_list.copy()
    # newlistpars = np.zeros_like(listparswlog)
    # for i,par in enumerate(listparswlog):
    #     newlistpars[i] = np.log10(par) if debparameters.full_islog[i] else par
    
    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    # lk = debmodeltest.log_likelihood(newlistpars[debmodeltest.posfree],newlistpars,debmodeltest.posfree)
    # print(lk)


    dt2019.plot_DEBresults(parspace,CI=False,multicore=False) 
    
    debparameters.set_free_onlyone("hb", isfree=True)
    debhbmodel = mm.DEBtox2019models([hbonly], debparameters, 
                                     moas, feedbs, Tbp=3,solver='LSODA')
    
    
    parspacehb = ps.PyParspace(ps.SettingParspace(0,1), debhbmodel)
    parspacehb.profile =0
    print(debhbmodel.parbound_lower)
    startt = time.time()
    parspacehb.run_parspace()
    endt = time.time()
    print("Time for hb fit: ", endt-startt)
    parspacehb.replot_results()

    debparameters.full_list = parspacehb.model.parvals
    debparameters.set_freefix_parameters("hb", isfree=False)
    debparameters.set_freefix_parameters_list(["lp","lm","rb","rm"], isfree=True)


    debmodeltest = mm.DEBtox2019models([control_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=3,solver='LSODA')

    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    lk = debmodeltest.log_likelihood(debparameters.full_list[debmodeltest.posfree],debparameters.full_list,debmodeltest.posfree)
    print(lk)

    # ## time the likelihood function
    # ## begintime = time.time()
    # ## for i in range(1000):
    # ##     lk = debmodeltest.log_likelihood(debparameters.full_list[debmodeltest.posfree],debparameters.full_list,debmodeltest.posfree)
    # ## print("Time for 1000 likelihood evaluations: ", time.time()-begintime)

    parspace.profile =0
    startt = time.time()
    parspace.run_parspace()
    endt = time.time()
    print("Time for physiological model fit: ", endt-startt)
    dt2019.plot_DEBresults(parspace,CI=True,multicore=True)
    
    
    debparameters.full_list = parspace.model.parvals
    debparameters.fixfree_physio_pars(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)
    
    '''
    print("Starting the tox model fit")
    debparameters.fixfree_physio_pars(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)
    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=0,
                                       breaktime=1,
                                       solver='LSODA')
    
    parspace_tox = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)
    startt = time.time()
    parspace_tox.run_parspace()
    endt = time.time()
    print("Time for tox model fit: ", endt-startt)
    dt2019.plot_DEBresults(parspace_tox, CI=False, multicore=False)
    parspace_tox.replot_results()
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