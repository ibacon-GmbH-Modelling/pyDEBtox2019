# Example script to run a complete analysis of a DEBtox2019 model with the pyDEBtox2019 package.
# The example data is taken from the original DEBtox2019 model coded in Matlab by Dr. Tjalling Jager.
#
# This script shows the use of the tool to fit a DEBtox2019 model to data, calculate ECx/EPx values
# (including their confidence intervals) and dose-response curves, and validate the model.
# The calibration of the model is preformed on a single dataset, but the code allows the use of multiple datasets.
# Additional example files are in the testingfiles folder

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

    # decide which steps to run.
    # If loadresults=1, the example runs are loaded and the fitting is not done.
    loadresults = 1 # load example runs
    runecx = 1      # run the ECx calculation
    rundoseresponse = 1  # run the ECx-based dose-response curve
    runepx = 1      # run the EPx calculation
    runepx_ci = 1   # also propagate the EPx/LPx confidence interval (slower -
                    # see ci_n_samples below for keeping this laptop-friendly)
    validate = 1    # run the validation step

    # define a few path to files for easiness of reading
    parameterpath = "./testingfiles/parameters/"
    datapath      = "./testingfiles/datafiles/"
    profilepath   = "./testingfiles/profiles/"
    presavedpath  = "./testingfiles/parspaceresults/"

    # Open JSON file with the parameter values
    # the if a parameter is to be fitted in log10 scale,
    # both value and limits are to be given in log10 scale
    with open(parameterpath+'input_pars_tbp3.json') as json_file:
        DEBpars = json.load(json_file)

    # initialize the parameter class
    debparameters = dt2019.DEBparameters(DEBpars)
    debparameters.fixfree_tox_pars(isfree=False) # fix all tox parameters

    # define mode of action and feedbacks
    # same BYOM codes: here it is growth and repro
    moas = np.array([0,0,1,1,0]) 
    feedbs =  np.array([0,0,0,0])

    # load exposure data. File constructed as for BYOM
    Cdata = pd.read_csv(datapath+"Test_Cdata.txt", sep="\s+", header=None)
    ccl = readin.concclass(Cdata.to_numpy(),"","ug/L")
    ccl.plot_exposure()

    # load length data
    Ldata = pd.read_csv(datapath+"Test_Ldata.txt", sep="\s+", header=None)
    lcl = readin.lengthdataclass(Ldata.to_numpy())
    lcl.plot_data()

    # load reproduction data, and remove 0s (similar way as in BYOM)
    Rdata = pd.read_csv(datapath+"Test_Rdata_opt1.txt", sep="\s+", header=None)
    rcl = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=1)
    rcl.plot_data_cumulative()

    # # load reproduction data, and do NOT remove 0s (similar way as in BYOM)
    # Rdata2 = pd.read_csv("Test_Rdata.txt", sep="\s+", header=None)
    # rcl2 = readin.reproclass(Rdata2.to_numpy(), reprocase='individual',optcase=2)
    # rcl2.plot_data_cumulative()

    # load survival data
    Sdata = pd.read_csv(datapath+"Test_Sdata.txt", sep="\s+", header=None)
    scl = readin.survdataclass(Sdata.to_numpy())
    scl.plot_data(scaleto1=True, label="suviving fraction")


    # isolate the controls. control_type to decide if pooling or using either water or solvent control
    full_ds, control_ds, ph = dt2019.build_dataset_variants(ccl, lcl, rcl, scl, control_type='both')
    
    # isolate control survival data for the calculation of background mortality
    _,hbonly,_ = dt2019.build_dataset_variants(ccl=ccl, lcl=None,rcl=None,scl=scl,control_type='both')

    # set the parameter limits to start the grid search
    debparameters.preset_toxlimits(moas, feedbs, ccl)

    # load model and plot it with the initial parameters before fitting
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
        parspacehb = ps.PyParspace.load_class(presavedpath+"hb_daphnia_moa1000fb0000.pkl")
        parspacehb.replot_results()
        dt2019.plot_DEBresults(parspacehb,CI=True,multicore=True)
    else:
        # fit with the parspace explorer and plot the results
        startt = time.time()
        parspacehb.run_parspace()
        endt = time.time()
        print("Time for hb fit: ", endt-startt)
        parspacehb.replot_results()
        parspacehb.save_sample(presavedpath+"hb_daphnia_moa1000fb0000.pkl")

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
        parspace = ps.PyParspace.load_class(presavedpath+"physio_daphnia_moa1000fb0000.pkl")
        parspace.replot_results()
        dt2019.plot_DEBresults(parspace,CI=True,multicore=True)
    else:
        # fit and run. Plot also the results of the fit of the control.
        # including the confidence intervals (not directly available in BYOM)
        startt = time.time()
        parspace.run_parspace()
        endt = time.time()
        print("Time for physiological model fit: ", endt-startt)
        dt2019.plot_DEBresults(parspace,CI=True,multicore=True)
        parspace.save_sample(presavedpath+"physio_daphnia_moa1000fb0000.pkl")

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
        parspace_tox = ps.PyParspace.load_class(presavedpath+"tox_daphnia_moa1000fb0000.pkl")
        parspace_tox.replot_results()
        dt2019.plot_DEBresults(parspace_tox,CI=True,multicore=True)
    else:
        startt = time.time()
        parspace_tox.run_parspace()
        endt = time.time()
        print("Time for tox model fit: ", endt-startt)
        dt2019.plot_DEBresults(parspace_tox, CI=True, multicore=True)
        parspace_tox.replot_results()
        parspace_tox.save_sample(presavedpath+"tox_daphnia_moa1000fb0000.pkl")

    # calculate EFSA criteria
    EFSAcal = dt2019.efsa_criteria(parspace_tox.model)

    # calculate ECx values
    if runecx:
        EC10 = dt2019.calc_ecx(parspace_tox.model,
                               Tend = 21,
                               X=(10, 50),ci=True,
                               parspace=parspace_tox,
                               multicore=True,
                               verbose=True,)

    if rundoseresponse:
        # dose-response curve at a fixed exposure duration, built directly on
        # top of calc_ecx: instead of a handful of named X values (as above),
        # it scans a fine effect-level grid and reframes each (x, ECx) pair as
        # a point on the classical concentration-vs-response sigmoid curve -
        # one subplot per endpoint, with a shaded CI band on the concentration
        # axis (same parspace.propagationset mechanism as calc_ecx above).
        #
        # Cost here scales with n_points x len(propagationset) (one bisection
        # search per grid point per parameter set) - parspace_tox's real
        # propagation set has thousands of rows, so ci_n_samples caps how
        # many are actually used (same subsampling knob as calc_epx above).
        dose_response = dt2019.calc_dose_response(
            parspace_tox.model,
            Tend=21,
            dataset=0,
            n_points=21,
            ci=True,
            parspace=parspace_tox,
            ci_n_samples=15,
            ci_seed=0,
            multicore=True,
            verbose=False,
        )

    if runepx:
        # calculate EPx values
        # load a profile
        # profile from EFSA
        profile_raw = pd.read_csv(profilepath+"apple_R1_pond.txt", sep="\s+", header=None).to_numpy(dtype=float)
        epcl = readin.concclass(profile_raw, "apple_R1_pond", "ug/L", focus=True)
        epcl.plot_exposure()
        print("Profile duration: %.2f days, %d points" % (
            epcl.timetr[-1] - epcl.timetr[0], len(epcl.timetr)
        ))

        # EPx/LPx with the moving time window advanced 1 day at a time (Tstep=1.0).
        # prune_win=True skips window positions that provably cannot be the
        # worst case (pyDEBtox2019 equivalent of prune_windows.m) before running
        # any bisection on them - safe here since feedbs are all off.
        #
        # ci=True propagates the confidence interval through the full
        # moving-window search for every parameter set in
        # parspace_tox.propagationset - by far the most expensive part of
        # EPx/LPx CI propagation, since it reruns that whole search per
        # parameter set. ci_n_samples caps how many of those parameter sets
        # are actually used (randomly subsampled, ci_seed for
        # reproducibility) - the knob to turn if this gets too slow on a
        # normal laptop; set runepx_ci=False above to skip CI entirely.
        res = dt2019.calc_epx(
            parspace_tox.model,
            epcl,
            Twin=21,
            X=[50],
            dataset=0,
            Tstep=1.0,
            prune_win=True,
            verbose=True,
            ci=runepx_ci,
            parspace=parspace_tox if runepx_ci else None,
            ci_n_samples=10,
            ci_seed=0,
            multicore=True,
            zero_hb=True)

        for ep in ('survival', 'length', 'reproduction'):
            # cycle true the endpoints and plot the results of the EPx
            # calculation. Passing parspace here (again subsampled the same
            # way) additionally shades Figure 2's exposed/control trajectory
            # band - the parameter-uncertainty CI at the fixed worst-case MF.
            dt2019.plot_epx_results(model=parspace_tox.model,
                                    exposure=epcl,
                                    results=res,
                                    endpoint=ep,
                                    x=50,
                                    zero_hb=True,
                                    parspace=parspace_tox if runepx_ci else None,
                                    ci_n_samples=10,
                                    ci_seed=0)
    
    if validate:
        with open(parameterpath+'input_pars_tbp3.json') as json_file:
            DEBpars_val = json.load(json_file)

        debparameters_val = dt2019.DEBparameters(DEBpars_val)
        # load exposure data. File constructed as for BYOM
        Cdata = pd.read_csv(datapath+"Test_Cdata.txt", sep="\s+", header=None)
        ccl_val = readin.concclass(Cdata.to_numpy(),"","ug/L")
        # load length data
        Ldata = pd.read_csv(datapath+"Test_Ldata.txt", sep="\s+", header=None)
        lcl_val = readin.lengthdataclass(Ldata.to_numpy())
        # load reproduction data, and remove 0s (similar way as in BYOM)
        Rdata = pd.read_csv(datapath+"Test_Rdata_opt1.txt", sep="\s+", header=None)
        rcl_val = readin.reproclass(Rdata.to_numpy(), reprocase='individual',optcase=1)  
        # load survival data
        Sdata = pd.read_csv(datapath+"Test_Sdata.txt", sep="\s+", header=None)
        scl_val = readin.survdataclass(Sdata.to_numpy())
        full_ds_val, _, _ = dt2019.build_dataset_variants(ccl_val, lcl_val, rcl_val, scl_val, control_type='both')

        # validation function
        dt2019.validation(full_ds_val,
                          debparameters_val,
                          parspace_tox, 
                          CI=True, 
                          multicore=True, 
                          wmeans=False)
        
    plt.show()
