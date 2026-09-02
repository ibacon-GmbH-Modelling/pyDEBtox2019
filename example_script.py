# testing script

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import pydebtox2019.models as mm
import pydebtox2019.parspace as ps
import pydebtox2019.debtox2019api as dt2019
import pydebtox2019.readin as readin

if __name__ == "__main__":

    loadresults = 1 # load example runs
    validate = 1

    # Open JSON file with the parameter values
    # the if a parameter is to be fitted in log10 scale,
    # both value and limits are to be given in log10 scale
    with open('input_pars_tbp3.json') as json_file:
        DEBpars = json.load(json_file)

    # initialize the parameter class
    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False) # fix all tox parameters

    # define mode of action and feedbacks
    # same BYOM codes: here it is growth and repro
    moas = np.array([0,0,1,1,0]) 
    feedbs =  np.array([0,0,0,0])

    # load exposure data. File constructed as for BYOM
    Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(),"","ug/L")
    ccl.plot_exposure()

    # load length data
    Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())
    lcl.plot_data()

    # load reproduction data, and remove 0s (similar way as in BYOM)
    Rdata = pd.read_csv("Test_Rdata_opt1.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=1)
    rcl.plot_data_cumulative()

    # load reproduction data, and do NOT remove 0s (similar way as in BYOM)
    Rdata2 = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    rcl2 = readin.reproclass(Rdata2.to_numpy(), reprocase='individual',optcase=2)
    rcl2.plot_data_cumulative()

    # load survival data
    Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())
    scl.plot_data(scaleto1=True, label="suviving fraction")

    # The plot_* methods build figures but no longer display them - that is
    # the caller's job. One plt.show() puts every figure made so far on
    # screen at once; execution resumes when they are closed. (Under IPython
    # or Jupyter with matplotlib integration this line is a no-op, as the
    # figures have already appeared.)
    plt.show()


    # isolate the controls. control_type to decide if pooling or using either water or solvent control
    full_ds, control_ds, ph = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    
    # isolate control survival data for the calculation of background mortality
    _,hbonly,_ = dt2019.build_dataset_variants(ccl=ccl, lcl=None,rcl=None,scl=scl,control_type='both')

    # set the parameter limits to start the grid search
    debparameters.preset_toxlimits(moas, feedbs, ccl)

    # load model and plot initial conditions
    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=3,
                                       breaktime=1,
                                       solver='LSODA')
    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  
    dt2019.plot_DEBresults(parspace,CI=False,multicore=False) 
    
    ## Now start the proper fit of the parameters

    # free only the background mortality parameter and fit it
    # to the data
    debparameters.set_free_onlyone("hb", isfree=True)
    debhbmodel = mm.DEBtox2019models([hbonly], debparameters, 
                                     moas, feedbs, Tbp=3,
                                     solver='LSODA')
    parspacehb = ps.PyParspace(ps.SettingParspace(0,1), debhbmodel)

    if loadresults:
        parspacehb = ps.PyParspace.load_class("./exampleresults/hb_daphnia_moa1000fb0000.pkl")
    else:
        # fit with the parspace explorer and plot the results
        startt = time.time()
        parspacehb.run_parspace()
        endt = time.time()
        print("Time for hb fit: ", endt-startt)
        parspacehb.replot_results()
        parspacehb.save_sample("./exampleresults/hb_daphnia_moa1000fb0000.pkl")

    # fix the background mortality parameter and free the
    # physiological parameters that are to be fitted
    debparameters.full_list = parspacehb.model.parvals
    debparameters.set_freefix_parameters("hb", isfree=False)
    debparameters.set_freefix_parameters_list(["lp","lm","rb","rm"], isfree=True)

    # reconstruct the model with the new data and new parameter setup
    debmodeltest = mm.DEBtox2019models([control_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=3,
                                       solver='LSODA')
    parspace = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)  

    if loadresults:
        parspace = ps.PyParspace.load_class("./exampleresults/physio_daphnia_moa1000fb0000.pkl")
    else:
        # fit and run. Plot also the results of the fit of the control.
        # including the confidence intervals (not directly available in BYOM)
        startt = time.time()
        parspace.run_parspace()
        endt = time.time()
        print("Time for physiological model fit: ", endt-startt)
        dt2019.plot_DEBresults(parspace,CI=True,multicore=True)
        parspace.save_sample("./exampleresults/physio_daphnia_moa1000fb0000.pkl")

    # update the parameter values with the new ones
    debparameters.full_list = parspace.model.parvals

    # now free only the tox parameters
    debparameters.fixfree_physio_pars(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)
    
    print("Starting the tox model fit")
    debparameters.fixfree_physio_pars(isfree=False)
    debparameters.fixfree_tox_pars(isfree=True)
    debmodeltest = mm.DEBtox2019models([full_ds],
                                       debparameters,
                                       moas, feedbs, Tbp=3,
                                       breaktime=1, solver='LSODA')
    parspace_tox = ps.PyParspace(ps.SettingParspace(0,1), debmodeltest)
    if loadresults:
        parspace_tox = ps.PyParspace.load_class("./exampleresults/tox_daphnia_moa1000fb0000.pkl")
    else:
        startt = time.time()
        parspace_tox.run_parspace()
        endt = time.time()
        print("Time for tox model fit: ", endt-startt)
        dt2019.plot_DEBresults(parspace_tox, CI=True, multicore=True)
        parspace_tox.replot_results()
        parspace_tox.save_sample("./exampleresults/tox_daphnia_moa1000fb0000.pkl")

    # calculate EFSA criteria
    EFSAcal = dt2019.efsa_criteria(parspace_tox.model)

    # calculate ECx values
    EC10 = dt2019.calc_ecx(parspace_tox.model,
                           Tend = 21,
                           X=(10, 50),ci=True,
                           parspace=parspace_tox,
                           multicore=True,
                           verbose=True,)

    # calculate EPx values
    # load a profile
    # profile from EFSA
    profile_raw = pd.read_csv("apple_R1_pond.txt", sep="\s+", header=None).to_numpy(dtype=float)
    epcl = readin.concclass(profile_raw, "apple_R1_pond", "ug/L", focus=True)
    epcl.plot_exposure()
    print("Profile duration: %.2f days, %d points" % (
        epcl.timetr[-1] - epcl.timetr[0], len(epcl.timetr)
    ))

    # EPx/LPx with the moving time window advanced 1 day at a time (Tstep=1.0).
    # prune_win=True skips window positions that provably cannot be the
    # worst case (pyDEBtox2019 equivalent of prune_windows.m) before running
    # any bisection on them - safe here since feedbs are all off.
    # Keep the ci to false if you are running it on a normal laptop
    res = dt2019.calc_epx(
        parspace_tox.model,
        epcl,
        Twin=21,
        X=[50],
        dataset=0,
        Tstep=1.0,
        prune_win=True,
        verbose=True,
        ci=False,
        zero_hb=True)

    for ep in ('survival', 'length', 'reproduction'):
        dt2019.plot_epx_results(model=parspace_tox.model,
                                exposure=epcl,
                                results=res,
                                endpoint=ep,
                                x=50,
                                zero_hb=True)
    
    if validate:
        with open('input_pars_tbp3.json') as json_file:
            DEBpars_val = json.load(json_file)

        debparameters_val = dt2019.DEBparameters(DEBpars_val)
        # load exposure data. File constructed as for BYOM
        Cdata = pd.read_csv("Test_Cdata.txt", sep="\s+", header=None)
        ccl_val = readin.concclass(Cdata.to_numpy(),"","ug/L")
        # load length data
        Ldata = pd.read_csv("Test_Ldata.txt", sep="\s+", header=None)
        lcl_val = readin.lengthdataclass(Ldata.to_numpy())
        # load reproduction data, and remove 0s (similar way as in BYOM)
        Rdata = pd.read_csv("Test_Rdata_opt1.txt", sep="\s+", header=None)
        rcl_val = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=1)  
        # load survival data
        Sdata = pd.read_csv("Test_Sdata.txt", sep="\s+", header=None)
        scl_val = readin.survdataclass(Sdata.to_numpy())
        full_ds_val, _, _ = dt2019.build_dataset_variants(ccl_val, lcl_val, rcl_val, scl_val, control_type='both')

        # validation function
        dt2019.validation(full_ds_val,
                          debparameters_val,
                          parspace_tox, 
                          CI=True, 
                          multicore=True, 
                          wmeans=False)
