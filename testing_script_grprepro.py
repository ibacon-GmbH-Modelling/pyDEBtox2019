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
    with open('input_pars_grp.json') as json_file:
        DEBpars = json.load(json_file)

    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False)

    moas = np.array([0,0,0,0,1])
    feedbs =  np.array([0,0,0,0])

    Cdata = pd.read_csv("Test_Cgrp.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(),"","ug/L")
    ccl.plot_exposure()

    Ldata = pd.read_csv("Test_Lgrp.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())
    lcl.calc_mean_and_ci()
    lcl.plot_data(wmeans=True)

    Fdata = pd.read_csv("test_Fgrp.txt", sep="\s+",header=None)
    Fdata = Fdata.to_numpy()

    Sdata = pd.read_csv("test_Sgrp.txt", sep="\s+", header=None)
    Sdata = Sdata.to_numpy()
    scl = readin.survdataclass(Sdata)
    scl.plot_data(scaleto1=True, label="suviving fraction")
    

    Rdata = pd.read_csv("test_Rgrp.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='group',optcase=2,
                            survtable=Sdata, femaletable=Fdata)
    rcl.plot_data_cumulative()
    


    # isolate the controls    
    full_ds, control_ds, ph = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='solvent')
    
    # take only the survival data from the controls
    _,hbonly,_ = dt2019.build_dataset_variants(ccl=ccl, lcl=None,rcl=None,scl=scl,control_type='solvent')

    # set the parameter limits to start the grid search
    debparameters.preset_toxlimits(moas, feedbs, ccl)

    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=0,
                                       breaktime=0,
                                       solver='LSODA')
    
    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    # lk = debmodeltest.log_likelihood(newlistpars[debmodeltest.posfree],newlistpars,debmodeltest.posfree)
    # print(lk)

    dt2019.plot_DEBresults(parspace,CI=False,multicore=False) 

    debparameters.set_free_onlyone('hb', isfree=True)
    debmodelhb = mm.DEBtox2019models([hbonly],
                                       debparameters,
                                       moas, feedbs, Tbp=0,
                                       breaktime=0,
                                       solver='LSODA')
    parspacehb = ps.PyParspace(ps.SettingParspace(0,1), debmodelhb)
    startt = time.time()
    parspacehb.run_parspace()
    endt = time.time()
    print("Time for hb model fit: ", endt-startt)
    dt2019.plot_DEBresults(parspaceres=parspacehb,wmeans=False)


    # debparameters.full_list = parspacehb.model.parvals
    # debparameters.set_freefix_parameters("hb", isfree=False)
    # debparameters.set_freefix_parameters_list(["lp","lm","rb","rm"], isfree=True)

    # debmodelctrl = mm.DEBtox2019models([control_ds],
    #                                    debparameters,
    #                                    moas, feedbs, Tbp=0,solver='LSODA')
    # parspacectrl = ps.PyParspace(ps.SettingParspace(0,1), debmodelctrl)  
    # startt = time.time()
    # parspacectrl.run_parspace()
    # endt = time.time()
    # print("Time for physiological model fit: ", endt-startt)
    # dt2019.plot_DEBresults(parspacectrl,CI=True,multicore=True)

    
    # debparameters.full_list = parspacectrl.model.parvals
    # debparameters.fixfree_physio_pars(isfree=False)
    # debparameters.fixfree_tox_pars(isfree=True)

    # debmodeltox = mm.DEBtox2019models([full_ds],
    #                                    debparameters,
    #                                    moas, feedbs, Tbp=0,
    #                                    breaktime=1,
    #                                    solver='LSODA')
    # parspacetox = ps.PyParspace(ps.SettingParspace(0,1),debmodeltox)
    # startt=time.time()
    # parspacetox.run_parspace()
    # endt=time.time()
    # print("Time for tox fit: ",endt-startt)
    # dt2019.plot_DEBresults(parspacetox,CI=True,multicore=True)