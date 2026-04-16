'''
Class for a paramter space explorer in python.

This implementation is heavily based on the BYOM and openGUTS parameter space 
explorer code written by Dr. Tjalling Jager. The original code is written in 
Matlab and can be found here: https://www.debtox.info/byom.html

The original MATLAB files are located in the engine folder of openGUTS 
or in the engine and engine_par folder of the BYOM release.

The rewriting of this software made use of the GitHub Copilot, which is based
on AI technology.

=======================================================================
Copyright (c) 2018-2024, Tjalling Jager (tjalling@debtox.nl)
Copyright (c) 2025, Carlo Romoli - ibacon GmbH (carlo.romoli AT ibacon.com)
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <https://www.gnu.org/licenses/>

3rd Party Addendum
==================
This program uses the python packages numpy, scipy
which are available under the modified 3 clause BSD license

Copyright (c) 2005-2024, NumPy Developers.
Copyright (c) 2001, 2002 Enthought, Inc.
Copyright (c) 2003-2019 SciPy Developers.


All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

3rd Party Addendum
==================
This program uses the python package matplotlib which is distributed under the 
following license:
Copyright (c) 2003-2024 Matplotlib Development Team

1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
   the Individual or Organization ("Licensee") accessing and otherwise using Python
   3.13.1 software in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
   grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
   analyze, test, perform and/or display publicly, prepare derivative works,
   distribute, and otherwise use Python 3.13.1 alone or in any derivative
   version, provided, however, that PSF's License Agreement and PSF's notice of
   copyright, i.e., "Copyright © 2001-2024 Python Software Foundation; All Rights
   Reserved" are retained in Python 3.13.1 alone or in any derivative version
   prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on or
   incorporates Python 3.13.1 or any part thereof, and wants to make the
   derivative work available to others as provided herein, then Licensee hereby
   agrees to include in any such work a brief summary of the changes made to Python
   3.13.1.

4. PSF is making Python 3.13.1 available to Licensee on an "AS IS" basis.
   PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
   EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
   WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
   USE OF PYTHON 3.13.1 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.13.1
   FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
   MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.13.1, OR ANY DERIVATIVE
   THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material breach of
   its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any relationship
   of agency, partnership, or joint venture between PSF and Licensee.  This License
   Agreement does not grant permission to use PSF trademarks or trade name in a
   trademark sense to endorse or promote products or services of Licensee, or any
   third party.

8. By copying, installing or otherwise using Python 3.13.1, Licensee agrees
   to be bound by the terms and conditions of this License Agreement.
'''

# import of external libraries and packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import multiprocessing as mp  
import scipy.optimize as optim
from functools import partial

import psutil
n_cores = psutil.cpu_count(logical=False) # to have the number of physical cores only

from copy import deepcopy

class SettingParspace:
    '''
    Class to store the settings for the parameter space explorer

    Attributes:
    -----------
    rough : bool 
        flag to use rougher set up for the minimizer
    profile : bool
        flag to calculate profiling
    crit_table : np.array
        levels of chi2 distributions for different number of parameters for 0.95 CI
    crit_prop : np.array
        limits to select the sample for the propagation of the 95% CI
    crit_add : np.array
        extra values on top of the chi2 criterion to continue iterations
    n_tr : np.array
        number of random parameter tries in round 2 and further, for each ok parameter set
    f_d : np.array
        maximum step for random mutations, as factor of initial grid spacing
    n_ok : np.array
        number of ok parameter sets to keep in each round
    n_conf_all : np.array
        number of samples within outer and inner rim
    tries : int
        initial grid points per fitter parameters
    crit_add_extra : int
        ditance from mll for extra sampling
    f_d_extra : float
        maximum step jump for extra sampling
    d_extra : float
        grid spacing for extra sampling (as fraction of parameter)
    gap_extra : float
        maximum distance between profile and sample to trigger resampling
    real_better : float 
        difference between new optimum and old optimum for printout
    n_max : int
        maximum number of rounds for the algorithm
    slowkin_corr : float
        minimum correlation coefficient between kd and zs
    slowkin_pars : float
        closeness of kd and zs to their lower bound (as fraction of total range)
    slowkin_f_mw : float
        factor to multiply current-highest zs to get new upper bound
    slowkin_f_kd
        factor by which to multiply current-highest kd to get new upper bound
    '''
    def __init__(self, rough, profile=0):
        """
        Initialize the settings for the parameter space explorer.
        Parameters:
        -----------
        rough : bool
            If True, use rough settings for the parameter space exploration.
        profile : int, optional
            If true calculate the likelihood profile for each of the free 
            parameters of the model.
        Notes:
        ------
        If `rough` is True, the settings are adjusted for a rough exploration, 
        including fewer initial grid points, adjusted stop criteria, and a reduced 
        maximum number of rounds.
        """
        self.profile = profile
        self.rough = rough
        self.crit_table = np.array([3.8415, 5.9915,7.8147,9.4877,11.07,12.592,14.0671,15.5073,16.9190,18.3070])        
        self.crit_table[5:] = self.crit_table[4] # this copies the option in the BYOM code
        self.crit_prop = np.array([-2*0.3764, 0.3764]) + self.crit_table[0]
        self.crit_add = np.array([15,  7 , 3 ,  2 ,  1.5,   1,  0.5,  0.25]) # extra on top of chi2-criterion to select ok values for next round
        self.n_tr = np.array([np.nan, 60, 40 ,  30,   20,   10 ,  10 ,  10]) # number of random parameter tries in round 2 and further, for each ok parameter set
        self.f_d = np.array([np.nan,  1, 0.65, 0.42, 0.27, 0.18, 0.12, 0.08]) # maximum step for random mutations, as factor of initial grid spacing
        self.n_ok = np.array([5, 50, 200, 200, 400, 800, 800, 800, 800, 800]) # number of ok parameter sets to keep in each round
        # Stop criterion: minimum number of values within the total joint CI and
        # the inner rim, which is made dependent on number of fitted parameters ...
        self.n_conf_all = np.array([[500, 500],
                                    [2500, 1500],
                                    [5000, 2000],
                                    [7000, 3000],
                                    [10000, 5000],
                                    [15000, 7500],
                                    [15000, 7500],
                                    [15000, 7500],
                                    [15000, 7500],
                                    [15000, 7500]])
        
        # self.tries = np.array([14,12,12,8]); # initial grid points per fitted parameter
        self.tries = 10; # initial grid points per fitted parameter CHECK IN BYOM
        # criteria for extra sampling
        self.crit_add_extra = 3; # continue with sets that are within this value from the MLL (this focusses on parameters within, or close to, the inner rim)
        self.f_d_extra      = 1; # initial maximum jump size as fraction of the grid spacing
        self.d_extra        = 0.1; # grid spacing as fraction of the parameter's total range (from joint 95% CI)
        self.gap_extra      = 0.25; # the gap distance between profile and sample that triggers resampling
        self.real_better    = 0.05; # difference between new optimum and old optimum (MLL) that leads to printing on screen
        # Settings for total number of mutation rounds (used in <calc_parspace>).
        self.n_max = 12;  # maximum number of rounds for the algorithm (default 12)
        
        # criteria for the slow kinetic case
        self.slowkin_corr = 0.70 # minimum correlation coefficient between <kd> and <zs>
        self.slowkin_pars = 0.05 # closeness of <kd> and <zs> to their lower bound (as fraction of total range)
        self.slowkin_f_mw = 3    # factor by which to multiply current-highest <zs> to get new upper bound
        self.slowkin_f_kd = 10   # factor by which to multiply current-highest <kd> to get new upper bound
        
        if rough:
            self.tries = 8
            self.n_conf_all = np.array([[100, 100],
                                        [250, 150],
                                        [500, 200],
                                        [700, 300],
                                        [1000, 500],
                                        [1500, 750],
                                        [1500, 750],
                                        [1500, 750],
                                        [1500, 750],
                                        [1500, 750]])
            self.gap_extra  = 2*0.25; # the gap distance between profile and sample that triggers resampling
            self.n_max = 4;  # maximum number of rounds for the algorithm (default 12)

# Auxiliary function for the parallel calculation of the
# likelihood profile for each free parameter of the model
def parameter_profile_sub_wrapper(index, instance):
    return instance._parameter_profile_sub(index)

class PyParspace:
    '''
    Class for a minimizer that uses a parameter space explorer algorithm

    Attributes
    ----------
    coll_all : np.array
        full set of the inner and outer rim of the explorer
    coll_ok : np.array
        set used to start the next exploration (empty at the end)
    fullset : np.array
        complete list of model parameters
    bestaic : float
        best AIC value
    model : ModelSetUp object
        stores the model information
    npars : int
        number of free parameters
    opts : SettingParspace   object 
        stores the settings of the parameter space explorer
    parlabels : np object array
        labels of the free parameters
    pbest : np.array
        best fitting parameter set with likelihood value
    posfree : np.array
        position of the free parameters compared to the complete list of model parameters
    profile : list
        list of np.arrays with parameter likelihood profiling
    propagationset : np.array
        array of parameter set in the region of the parameter space used to propagate the
        95% confidence interval (on single parameters)

    Public Methods
    --------------
    run_parspace : function
        runs the parameter space explorer, plots the results of the exploration
        and returns the best fit parameter set and 95% CI

    replot_results : function
        replots the paramter space explorer results

    reprint_data : function
        reprints on screen the results of the parameter space
        explorer

    plot_data : function
        plot the dataset with optionally the model and the 
        confidence intervals

    save_sample : function
        save results in a pickle file

    load_class : function
        load the class with the results from a pickle file
    '''
    def __init__(self, SettingParspace, ModelSetUp):
        '''
        Initialisation of parameter space explorer

        Arguments:
        ---------
        SettingParspace : SettingParspace   object 
            stores the settings of the parameter space explorer
        ModelSetUp : ModelSetUp object
            stores the model information
        '''
        self.opts = deepcopy(SettingParspace)
        self.model = deepcopy(ModelSetUp)
        self.npars = sum(self.model.isfree)
        self.posfree = np.argwhere(self.model.isfree == 1).flatten()
        self.parlabels = np.copy(self.model.parnames)
        # deal with logarithmic parameters
        for i in range(self.npars):
            if self.model.islog[self.posfree[i]] == 1:
                self.parlabels[self.posfree[i]] = "log10(%s)"%self.parlabels[self.posfree[i]]
        self.parlabels = self.parlabels[self.posfree]
        # object that is a copy of the initial parameters of the model, but I do not care if it gets overwritten
        self.propagationset = np.array([]) #placeholder
        self.fullset = np.copy(self.model.parvals) # stores the full set of parameters. This is because not all parameter might be free
        self._startp = np.copy(self.model.parvals) # needed for internal operations 
        self.allow_multiprocessing = True # flag to allow multiprocessing in likelihood calculations

    # Using here sequential approach as the model is still not computationally heavy
    # enough to offset the overhead of parallelization
    # def _applylog(self,sample_scaled):
    #    llog = np.zeros(sample_scaled.shape[0])
    #    for i in range(sample_scaled.shape[0]):
    #        llog[i]=self.model.log_likelihood(sample_scaled[i],self._startp,self.posfree)
    #    return llog
    
    # mp version much slower for easy models
    # this function should be used in case of computationally
    # heavy models

    def _applylog(self, sample_scaled):
        # If disabled, run sequentially (used during parallel profiling)
        if not self.allow_multiprocessing:
            log_like = []
            for row in sample_scaled:
                val = self.model.log_likelihood(row, self._startp, self.posfree)
                log_like.append(val)
            return(np.array(log_like))
    
        # Normal mode: use multiprocessing
        log_likelihood_partial = partial(self.model.log_likelihood,
                                         DEBallpars=self._startp,
                                         posfree=self.posfree)
        with mp.Pool(n_cores) as pool:
            llog2 = pool.map(log_likelihood_partial, sample_scaled)
        return(np.array(llog2))

    
    # "Private" method to perform a standard optimization using the Nelder-Mead algorithm
    def _NM_minimizer(self, pfit, allpars, posfree, rough=0):
        if self.npars==1:
            bounds=None
        else:
            # keep the bounds pre-defined in the model initialization
            bounds = [(self.model.parbound_lower[i], self.model.parbound_upper[i]) for i in posfree]
        if rough:
            # make here a rougher minimization by changing the options
            res = optim.minimize(self.model.log_likelihood, 
                             x0=pfit, method="Nelder-Mead", args=(allpars,posfree), options={'xatol':1e-3, 'fatol':1e-3},
                             bounds=bounds)
        else:
            res = optim.minimize(self.model.log_likelihood, 
                             x0=pfit, method="Nelder-Mead", args=(allpars,posfree), options={'xatol':1e-6, 'fatol':1e-6},
                             bounds=bounds)
        return(res.x, res.fun)

    #@jit(nopython=True, parallel=True)
    def _random_mutations(self,selectedsample, l_bounds,u_bounds,n_tr_i, f_d_i_dgrid):
        """
        Perform random mutations on the selected sample

        Arguments:
            - selectedsample : selected sample
            - l_bounds : lower bounds of the parameters
            - u_bounds : upper bounds of the parameters
            - n_tr_i : number of tries
            - f_d_i_dgrid : maximum jump size as fraction of the grid spacing
        """
        l_bounds = self.model.parbound_lower[self.posfree]
        u_bounds = self.model.parbound_upper[self.posfree]
        npars = selectedsample.shape[1]
        n_cont = selectedsample.shape[0]
        mutatesample = np.zeros((n_cont*n_tr_i, npars))
        mutatellog = np.inf * np.ones(n_cont*n_tr_i)
        for i in range(n_cont):
            # create new values for the parameters using random mutations
            # f_d_i_dgrid is the maximum jump size as fraction of the grid spacing
            # selectedsample[i] is the current parameter set
            # np.random.uniform(-1,1, (n_tr_i,npars) creates a random array of shape (n_tr_i,npars)
            # with values between -1 and 1
            newvals = selectedsample[i] + f_d_i_dgrid * np.random.uniform(-1,1, (n_tr_i,npars))
            newvals = np.clip(newvals, l_bounds, u_bounds)  # make sure we are not outside the boundaries of the parmeters
            mutatesample[i*n_tr_i:(i+1)*n_tr_i] = newvals   # assign the new values to the mutatesample array
        mutatellog = self._applylog(mutatesample)           # calculate the loglikelihood for all the samples
        sortind = np.argsort(mutatellog)                    # sort the samples by their loglikelihood
        # sorted list of paramters
        mutatesample = mutatesample[sortind]
        mutatellog = mutatellog[sortind]
        # remove NaN values from the loglikelihood and the corresponding samples
        mask = np.isnan(mutatellog)==False # compute this only once
        mutatellog_nonan = mutatellog[mask]
        mutatesample_nonan = mutatesample[mask]
        return (mutatesample_nonan, mutatellog_nonan)
    
    # function to check if the profile is if problematic with respect to the sample
    def _test_profile(self, coll_all, parprofile, settings):
        """
        Check if the profile is good enough with respect to the sample.
        From openGUTS code:
         Testing of the profile likelihood, using the sample of sets. This
         function looks for two situations: 
         1) Points where the MLL of the sample is considerably higher (less good)
            than the MLL of the profile. This indicates that the sampling has been
            poor, so we need to resample later.
         2) Points where the MLL of the sample is considerably lower (better fit)
            than the profile. This indicates that the profiling was not good
            enough and may need to be redone.
        Parameters:
        -----------
        coll_all : np.array
            complete set of the inner and outer rim of the explorer with likelihood information
        parprofile : list of np.array
            profile for the free parameter being tested
        settings : SettingParspace object
            settings for the parameter space explorer
        Returns:
        --------
        flag_profile : list
            [flag, maximum distance between profile and sample]
             - flag = 0 if the profile is good enough, 1 if not
             - maximum distance between profile and sample
        Note:
        -----
        The function updates the value of attribute coll_ok 
        which is the array of ok parameter sets that are used 
        to start the next exploration
        """
        # This function might need some more testing
        # ind_fit = np.copy(self.posfree)
        coll_ok = np.zeros((0,self.npars+1))
        coll_all = np.copy(coll_all)
        flag_profile = [0, 0]
        chicrit_single = 0.5 * settings.crit_table[0]
        mll = coll_all[0,-1]
        for i_p in range(self.npars):  # run over all free parameters
            parprof = np.copy(parprofile[i_p])  # make a hard copy of the profile
            coll_tst = coll_all[coll_all[:,-1] < (mll + chicrit_single+1),:]
            if (min(coll_tst[:,i_p]) < min(parprof[:,i_p])) | (max(coll_tst[:,i_p]) > max(parprof[:,i_p])):
                # the profile is not good enough
                flag_profile[0] = 1
                flag_profile[1] = np.inf
                # this should be checked carefully, see note in Tjalling's code
            parprof = parprof[parprof[:,-1] < (mll + chicrit_single+1),:] # keep important parts
            for i_g in range(1,parprof.shape[0]-1):

                # From openGUTS code: 
                # Find best value from <coll_all> in this slice. This
                # parameter set will be compared to the profile likelihood. Note
                # that <coll_all> is sorted, so the first within this range is
                # automatically the best.
                #
                # Note: the spacing between the profile points is not necessarily
                # the same everywhere. Profiles may hit a boundary (and are then
                # placed on top of that boundary) or may be extended at half the
                # default grid spacing. Therefore, it is better to take half the
                # distance between the adjacent points. The <gridsp> is thus a
                # two-element vector with the half distance to the previous point
                # and to the next one. Since grid spacing is smaller when the
                # profile has been extended, it will probably trigger more gaps
                # than is strictly needed, and thus too much resampling. Only
                # downside is a larger sample and longer calculation times, but
                # that should be acceptable (since this will be a tough parameter
                # space anyway).

                gridsp = 0.5 * np.diff(parprof[[i_g-1,i_g,i_g+1], i_p], axis=0)
                ind_tst_tmp1 = np.argwhere(coll_all[:,i_p] > (parprof[i_g,i_p]-gridsp[0])).flatten()
                ind_tst_tmp2 = np.argwhere(coll_all[:,i_p] < (parprof[i_g,i_p]+gridsp[1])).flatten()
                ind_tst = np.intersect1d(ind_tst_tmp1, ind_tst_tmp2)

                if ind_tst.size < 1:
                    # take the closest point
                    ind_tst = np.argmin(np.abs(coll_all[:,i_p] - parprof[i_g, i_p]))
                    mll_compare = np.inf
                else:
                    ind_tst = np.min(ind_tst)
                    mll_compare = coll_all[ind_tst,-1]
                mll_prof = parprof[i_g,-1]
                if mll_prof < mll_compare:
                    # then the profile is better than the sample (as it should be)
                    if (mll_compare - mll_prof) > self.opts.gap_extra:
                        tmp = np.append([coll_all[ind_tst,:]], [parprof[i_g,:]], axis=0)
                        coll_ok = np.append(coll_ok, tmp, axis=0)
                elif mll_compare < mll_prof:
                    # sample is better then profile, so we will need a new profiling round

                    # From openGUTS code:
                    # The profile is on a limited number of grid points on the
                    # parameter axis, while the sample is anywhere in the slice.
                    # This can be done complex, but it is probably enough to only
                    # look at the minimum of the current profile point, previous
                    # one, and next one. If the sample is below the lowest of the
                    # three, it deserves to be counted as a true deviation.
                    #
                    # Note: one could decide to also put these problem areas into
                    # <coll_ok>. I had that in a previous version as a test, but
                    # decided not to do it anymore (can't remember why, though).
                    min_MLL = min(parprof[[i_g-1,i_g,i_g+1], -1])
                    if mll_compare < min_MLL:
                        # DEBUG
                        # print(flag_profile)
                        # print("this should go here",mll_compare - min_MLL)
                        flag_profile[0] = flag_profile[0]+1
                        flag_profile[1] = max(flag_profile[1], min_MLL-mll_compare)
        self.coll_ok = np.copy(coll_ok) # overwrite the coll_ok sample
        # print(self.coll_ok)
        return flag_profile

    # Perform the profile likelihood for the parameter at index <index>
    def _parameter_profile_sub(self,index):
        """
        Perform the profile likelihood for the parameter at index <index>
        Arguments:
            - index : index of the parameter to profile
        Returns:
            - parprof : np.array
                array with the profile for the parameter at index <index>
        -----------
        Note:
        From openGUTS code:
         Use the collected sample from parameter space to zoom in on the profile
         likelihood for the single parameters. This is useful to extract
         single-parameter CIs, but also to extend the sample with profiled sets.
         The latter may be useful when sampling is poor in some part of parameter
         space. Furthermore, the profiling does a number of optimisations (one
         parameter fixed), which may spot a better value of the optimum.
        
         Strategy is to split up the parameter range for each parameter into 50
         slices of equal width. Within each slice, look for the best-fitting set.
         Use that set as starting point for an optimisation (with the target
         parameter fixed to the middle of the slice). Also do another optimisation
         from the optimised value in the previous slice. Sometimes, extending a
         profile works better than taking the best available sample in the slice.
         From these two optimisations, the best value is optimised again and
         stored. Next, the profiling starts from the other side and calulates the
         next version from the previous profile point. In some cases, extending a
         profile from the other direction can find a better profile.
        
         Additionally, the algorithm is looking for situations where there is a
         substantial difference between the profile and the sample. If that
         happens in a relevant range (where the fit is still acceptable), the
         points are collected and will be used for an additional round of sampling
         (apparently, sampling was not optimal in that case). Note that the entire
         profile will also be added to the total sample as they are relevant
         parameter sets (this happens in <calc_parspace> at the end, when making
         CIs).
        """

        # without np.copy, the assignment is by reference
        # and the original array is modified. This creates
        # problems with the routine
        coll_allL = np.copy(self.coll_all[:,-1])
        coll_all = np.copy(self.coll_all[:,:-1])
        npars = self.npars
        gridsz = 50
        mll = coll_allL[0]

        mll_rem = mll

        chicrit_single = 0.5 * self.opts.crit_table[0]
        chicrit_joint = 0.5 * self.opts.crit_table[self.npars-1]
        chicrit_joint = max(chicrit_joint, chicrit_single+1.5)
        ind_final = np.argwhere(coll_allL < (coll_allL[0]+chicrit_joint)).flatten().max()
        coll_ok = np.ones((gridsz*10,npars)) * np.inf
        coll_okL = np.ones(gridsz*10) * np.inf
        ind_ok =0
        pbest = np.copy(self.coll_all[0,:]) # initialize to the best likelihood value so far

        parprof = np.zeros((gridsz,npars+1))

        parnr = self.posfree[index]
        edges = np.array([min(coll_all[:ind_final+1,index]), max(coll_all[:ind_final+1,index])])
        par_rng = np.linspace(edges[0],edges[1],gridsz)
        gridsp = 0.5 * (edges[1] - edges[0])/(gridsz-1) # why do I need this again when I have par_rng
        # include the best fit position in the grid
        bfitind = np.argwhere(par_rng <= coll_all[0,index]).flatten().max()
        par_rng = par_rng + (coll_all[0,index] - par_rng[bfitind])
        
        # The first thing to do is to run over the grid just made (based on the
        # sample) and optimise in each interval to find the best point (which
        # is the profile likelihood).
        for i_g in range(gridsz):
            # find the best point in the sample within this slice
            # In matlab this all process was done in a one liner, here had to be split
            # into 3 lines
            ind_tst_tmp = np.argwhere(coll_all[:,index]>(par_rng[i_g]-gridsp)).flatten()
            ind_tst_tmp2 = np.argwhere(coll_all[:,index]<(par_rng[i_g]+gridsp)).flatten()
            ind_tst = np.intersect1d(ind_tst_tmp,ind_tst_tmp2)  

            if ind_tst.size < 1:
                # take the closest point as the selection is empty
                ind_tst = np.argmin(np.abs(coll_all[:,index] - par_rng[i_g]))
                mll_compare = np.inf
            else:
                ind_tst = min(ind_tst)
                mll_compare = coll_allL[ind_tst]
        
            pmat_tst1 = coll_all[ind_tst]
            pmat_tst1[index] = par_rng[i_g]
            allpars = self._startp
            allpars[self.posfree]=pmat_tst1
            posfree = np.delete(self.posfree,index) # fix the parameter being profiled
            # do the optimization with the selected point in the slice
            phat_tst1,mll_tst1 = self._NM_minimizer(np.delete(pmat_tst1,index),allpars, posfree, rough=0)

            allpars[self.posfree[index]] = par_rng[i_g]
            allpars[posfree] = phat_tst1
            pmat_tst = allpars[self.posfree]

            # this is done to see if a different starting point has a better chances
            if i_g>0:
                # calling the variable with -2 to discriminate from the previous one
                pmat_tst2 = np.copy(parprof[i_g-1,:-1]) # hard copy of arrays
                pmat_tst2[index] = par_rng[i_g] 
                allpars = self._startp
                allpars[self.posfree]=pmat_tst2
                phat_tst2,mll_tst2 = self._NM_minimizer(np.delete(pmat_tst2,index),allpars, posfree, rough = 0)
                allpars[posfree] = phat_tst2
            else:
                mll_tst2 = np.inf
            
            if (mll_tst2 < mll_tst1):
                pmat_tst = allpars[self.posfree] # take the new values

            # # make a final fit using the best result obtianed (not a rough fit)
            phat_tst,mll_tst = self._NM_minimizer(np.delete(pmat_tst,index),allpars, posfree, rough = 0)
            allpars[posfree] = phat_tst            
            pmat_tst = allpars[self.posfree]

            # check for gaps and better optima
            if mll_tst < mll +chicrit_single+1:
                # only if we are not clearly above the parameter's CI ...
                # We don't care when there is a gap between profile and sample
                # points in parts of parameter space that results in bad fits
                # anyway.
                if mll_compare - mll_tst > self.opts.gap_extra:
                    # here there is a better value found with the profile
                    # put the sample point and the profile point in coll_ok
                    coll_ok[ind_ok,:]=coll_all[ind_tst,:]
                    coll_okL[ind_ok] =coll_allL[ind_tst]
                    coll_ok[ind_ok+1,:]=pmat_tst
                    coll_okL[ind_ok+1] =mll_tst
                    ind_ok = ind_ok+2
                if mll_tst < mll: # found a better minimum
                    pbest = parprof[i_g,:]
                    if ((mll_rem - mll_tst) > self.opts.real_better) & ((mll - mll_tst) > self.opts.real_better):
                        # print on screen that there is a much better minimum
                        print('  Better optimum found when profiling ',parnr, ': ',mll_tst,' (best was ',mll,')')
                    mll = mll_tst


            # make this smarter
            parprof[i_g,:] = np.concatenate((pmat_tst, [mll_tst]),axis=0)

        # running the profiling in reverse to keep optimizing
        for i_g in range(gridsz-2,-1,-1):
            mll_tst1 = parprof[i_g,-1]
            pmat_tst2= np.copy(parprof[i_g+1,:-1])
            pmat_tst2[index] = np.copy(parprof[i_g,index])
            allpars = self._startp
            allpars[self.posfree]=pmat_tst2
            phat_tst2,mll_tst2 = self._NM_minimizer(np.delete(pmat_tst2,index),
                                                    allpars, posfree, rough =0)
            allpars[posfree] = phat_tst2
            if mll_tst2 < mll_tst1:
                parprof[i_g,:] = np.concatenate((allpars[self.posfree],[mll_tst2]),axis=0)
                if mll_tst1-mll_tst2 > self.opts.gap_extra:
                    coll_ok[ind_ok,:]=pmat_tst2
                    coll_okL[ind_ok] =mll_tst2
                    ind_ok = ind_ok+1
                if mll_tst2 < mll:
                    pbest = parprof[i_g,:]
                    if (mll_rem - mll_tst2 > self.opts.real_better) & (mll - mll_tst2 > self.opts.real_better):
                        # print on screen that there is a much better minimum
                        print('  Better optimum found when profiling ',parnr, ': ',mll_tst2,' (best was ',mll,')')
                    mll = mll_tst2

        ## Extending the profile
        # Check and eventually extend the profile if we have not been catching all the space
        #
        # From OpenGUTS code:
        # Note: Extending the profile is done in steps of <gridsp>, which is
        # HALF of the distance between the regular profile points. This implies
        # that extension is done on a finer grid. This was a mistake on my
        # side, but I think it is okay to keep it in: extension is being done
        # without the aid of sample points, so maybe a finer grid is a good
        # idea. Will be a bit slow though. Also note that extension to lower
        # values will generally be triggered when the sample is on the lower
        # edge. This is caused by the fact that I shift the profiling grid
        # slightly to higher values to catch the best value exactly. That is
        # also not problematic.
        #
        # Start with profiling to lower values of the parameter,
        # if needed. If we are not clearly above the parameter's CI yet, and
        # haven't hit the lower bound ... continu further down until we are, or
        # until we hit a boundary
        flag_disp = False
        while (parprof[0,index] > self.model.parbound_lower[self.posfree[index]]) & (parprof[0,-1] < mll+chicrit_single+1):
            if flag_disp == 0:
                print('  Extending profile to lower bound of parameter ',parnr)
                flag_disp = 1
            pmat_tst = np.copy(parprof[0,:-1])  # use first entry of <parprof> to continue with
            pmat_tst[index] = pmat_tst[index] - gridsp # move to lower value for this parameter
            pmat_tst[index] = max(pmat_tst[index], self.model.parbound_lower[self.posfree[index]]) # make sure it is not below lower bound
            allpars = self._startp
            allpars[self.posfree]=pmat_tst
            phat_tst,mll_tst = self._NM_minimizer(np.delete(pmat_tst,index), 
                                                  allpars, posfree, rough=0) # do a rough optimisation first
            phat_tst,mll_tst = self._NM_minimizer(phat_tst, 
                                                  allpars, posfree, rough=0) # do a normal optimisation after
            allpars[posfree] = phat_tst
            parprof = np.append([np.concatenate((allpars[self.posfree],[mll_tst]),axis=0)],parprof,axis=0)
            coll_ok[ind_ok,:]=np.copy(allpars[self.posfree])
            coll_okL[ind_ok] = mll_tst
            ind_ok = ind_ok+1

            if mll_tst < mll:
                pbest = np.copy(parprof[0,:])
                if ((mll_rem - mll_tst) > self.opts.real_better) & ((mll - mll_tst) > self.opts.real_better):
                    print("Better optimum found when extending profile for ", self.parlabels[self.posfree[index]], " down: ", mll_tst, " (best was ", mll, ")")
                mll = mll_tst

        flag_disp = 0
        while (parprof[-1,index] < self.model.parbound_upper[self.posfree[index]]) & (parprof[-1,-1] < mll+chicrit_single+1):
            if flag_disp == 0:
                print('  Extending profile to higher bound of parameter ',parnr)
                flag_disp = 1
            pmat_tst = np.copy(parprof[-1,:-1])
            pmat_tst[index] = pmat_tst[index] + gridsp # move to higher value for this parameter
            pmat_tst[index] = min(pmat_tst[index], self.model.parbound_upper[self.posfree[index]])
            allpars = self._startp
            allpars[self.posfree]=pmat_tst
            # double optimization here as the parameter space becomes difficult
            phat_tst,mll_tst = self._NM_minimizer(np.delete(pmat_tst,index), 
                                                  allpars, posfree, rough=0)
            phat_tst,mll_tst = self._NM_minimizer(phat_tst,
                                                  allpars, posfree, rough=0)
            allpars[posfree] = phat_tst
            parprof = np.append(parprof,[np.concatenate((allpars[self.posfree],[mll_tst]),axis=0)],axis=0)
            coll_ok[ind_ok,:]=np.copy(allpars[self.posfree])
            coll_okL[ind_ok] = mll_tst
            ind_ok = ind_ok+1

            if mll_tst < mll:
                pbest = np.copy(parprof[-1,:])
                if ((mll_rem - mll_tst) > self.opts.real_better) & ((mll - mll_tst) > self.opts.real_better):
                    print("Better optimum found when extending profile for ", self.parlabels[self.posfree[index]], " up: ", mll_tst, " (best was ", mll, ")")
                mll = mll_tst

        if self.npars == 1:
                self.allow_multiprocessing = False
                parprof = np.column_stack((parprof[:,0], self._applylog(parprof[:,0])))
        self.allow_multiprocessing = True
        # in the end coll_ok is returned by the function together with the profile and
        # the best fit parameter set
        coll_ok = np.append(coll_ok, coll_okL[:,None], axis=1)
        coll_ok = coll_ok[0:ind_ok,:]  # also here?
        return parprof, pbest, coll_ok
    
    # print on screen the results of the parameter space explorer.
    # This is a "private" method, not to be called from outside the class
    # There is a wrapper written to reprint the results of the fit
    def _print_results(self, profile=None):
        if self.opts.profile:
            chicrit_single = 0.5 * self.opts.crit_table[0]
            res_parspace = np.zeros((self.npars,6))
            # always show the results in linear scale, by transforming back the parameter values
            res_parspace[:,0] = (10**(self.coll_all[0,:-1])*self.model.islog[self.posfree] + 
                                 self.coll_all[0,:-1]*(1-self.model.islog[self.posfree])).transpose() # best fit values
            print("Results obtained with the parameters space explorer with profiling option.")
            for i in range(self.npars):
                self.coll_all = np.append(self.coll_all, profile[i], axis=0)
                self.coll_all = self.coll_all[np.argsort(self.coll_all[:,-1])]
                mll = self.coll_all[0,-1]
                prof_tst = np.copy(self.profile[i])
                prof_tst[:,-1] = prof_tst[:,-1] - mll - chicrit_single
                if prof_tst.shape[0]>0:
                    # the likelihood of the first point of the profile is below the critical level for the 95% confidence level
                    if prof_tst[0,-1] < 0:
                        res_parspace[i,1] = (10**(self.model.parbound_lower[self.posfree[i]])*self.model.islog[self.posfree[i]] +
                                              (self.model.parbound_lower[self.posfree[i]])*(1-self.model.islog[self.posfree[i]]))
                        res_parspace[i,3] = 1 # flag for lower bound
                    else:
                        ind_low = np.argwhere(prof_tst[:,-1] < 0).flatten().min()
                        if ind_low.size > 0:
                            # need to interpolate to find the exact value. Interpolation done only on the 
                            # two points around the crossing
                            # IN NP.INTERP THE X VALUES NEED TO BE IN ASCENDING ORDER, SO I NEED TO FLIP THE ORDER OF THE POINTS
                            val=np.interp(0, np.array([prof_tst[ind_low,-1], prof_tst[ind_low-1,-1]]), 
                                             np.array([prof_tst[ind_low,i],prof_tst[ind_low-1,i]]))
                            res_parspace[i,1] = (10**val*self.model.islog[self.posfree[i]] + 
                                                 val*(1-self.model.islog[self.posfree[i]]))
                    if prof_tst[-1,-1] < 0:
                        # the likelihood of the last point of the profile is below the critical level for the 95% confidence level
                        res_parspace[i,2] = (10**(self.model.parbound_upper[self.posfree[i]])*self.model.islog[self.posfree[i]] + 
                                             (self.model.parbound_upper[self.posfree[i]])*(1-self.model.islog[self.posfree[i]]))
                        res_parspace[i,4] = 1 # flag for upper bound
                    else:
                        ind_up = np.argwhere(prof_tst[:,-1] < 0).flatten().max()
                        if ind_up.size > 0:
                            val = np.interp(0, np.array([prof_tst[ind_up,-1],prof_tst[ind_up+1,-1]]),
                                               np.array([prof_tst[ind_up,i],prof_tst[ind_up+1,i]]))
                            res_parspace[i,2] = (10**(val)*self.model.islog[self.posfree[i]] +
                                                     (val)*(1-self.model.islog[self.posfree[i]]))
                    num_x = np.sum(np.diff(np.sign(prof_tst[:,-1]))!=0)
                    num_x = num_x + np.sum(res_parspace[i,3:5]==1)
                    if num_x > 2:
                        res_parspace[i,5] = 1 # means we have broken CI
            print("Best fit likelihood: %.2f"%mll)
            print("Best fit AIC: %.2f"%(2*mll+2*self.npars))
            print("Parameter values:")
            print("{:<15} {:<10} {:<15} {:<15} {:<11} {:<11} {:<11}".format("Parameter",
                                                                            "Best", "Lower 95%", "Upper 95%",
                                                                            "l.bound hit","u.bound hit","broken CI"))
            for i in range(self.npars):
                print("{:<15} {:<10.4f} ({:<14.4f} {:<14.4f}) {:<11.0f} {:<11.0f} {:<11.0f}".format(
                       self.model.parnames[self.posfree[i]], res_parspace[i,0], res_parspace[i,1], 
                       res_parspace[i,2],res_parspace[i,3], res_parspace[i,4], res_parspace[i,5]))            

        else:
            coll_all = np.copy(self.coll_all)
            mll = coll_all[0,-1]
            # find the indices of the 95% confidence intervals
            ind_prop1 = np.argwhere(coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[0]).flatten().max()
            ind_prop2 = np.argwhere(coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[1]).flatten().max()
            res_parspace = np.zeros((self.npars,3))
            res_parspace[:,0] = (10**(coll_all[0,:-1])*self.model.islog[self.posfree] + coll_all[0,:-1]*(1-self.model.islog[self.posfree])).transpose()
            res_parspace[:,1] = (10**(coll_all[ind_prop1:ind_prop2+1,:-1].min(axis=0))*self.model.islog[self.posfree] + 
                                    (coll_all[ind_prop1:ind_prop2+1,:-1].min(axis=0))*(1-self.model.islog[self.posfree]))
            res_parspace[:,2] = (10**(coll_all[ind_prop1:ind_prop2+1,:-1].max(axis=0))*self.model.islog[self.posfree] + 
                                    (coll_all[ind_prop1:ind_prop2+1,:-1].max(axis=0))*(1-self.model.islog[self.posfree]))
            print("The results here are obtained from the paramter space explorer without profiling option.")
            print("Best fit values and CI estimates could be considered as estimates")
            print("Best fit likelihood: %.2f"%mll)
            print("Best fit AIC: %.2f"%(2*mll+2*self.npars))
            print("Parameter values:")
            print("{:<15} {:<10} {:<15} {:<15}".format("Parameter", "Best", "Lower 95%", "Upper 95%"))
            for i in range(self.npars):
                print("{:<15}: {:<10.4f} ({:<10.4f} {:<10.4f})".format(
                      self.model.parnames[self.posfree[i]], res_parspace[i,0], res_parspace[i,1], res_parspace[i,2]))

    # "private" method function that generates the figure with the sampling of 
    # the parameter space explorer includes arguments to save the resulting figure
    # There is a wrapper written to replot the results of the fit
    def _plot_samples(self,profile=None, batchmode= False, savefig=False, figbasename="", extension=".png"):
        '''
        Plot the results of the parameter space explorer as a corner plot
        In the diagonal there is the likelihood as a function of the parameter value
        and off the diagonal the correlation plots between parameter pairs
        Arguments:
        ----------
        profile : list
            list of np.arrays with parameter likelihood profiling
            length of the list is the number of free parameters
        savefig : bool
            flag to save the figure
        figbasename : str
            base name of the figure
        extension : str
            extension of the figure
        '''
        npars = self.coll_all.shape[1]-1
        chicrit_joint = 0.5 * self.opts.crit_table[npars-1]
        chicrit_single = 0.5 * self.opts.crit_table[0]
        best_set = self.coll_all[0]
        mll = self.coll_all[0,-1]
        ind_single = np.argwhere(self.coll_all[:,-1] < (mll + chicrit_single)).flatten().max()
        ind_fin95 = np.argwhere(self.coll_all[:,-1] < (mll + chicrit_joint)).flatten().max()
        coll_inner = self.coll_all[0:ind_single+1]
        coll_joint = self.coll_all[ind_single+1:ind_fin95+1]
        ind_prop1 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[0]).flatten().max()
        ind_prop2 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[1]).flatten().max()
        handles=[]
        labels=[]
        # plot the results
        plt.figure()
        ax = np.zeros((npars,npars),dtype=object)
        for i in range(npars):
            j=0
            while j<=i:
                ax[i,j]=plt.subplot(npars,npars,i*npars+j+1)
                if i==j:
                    # diagonal plots with the likelihood as a function of the parameter value
                    ax[i,j].plot(coll_joint[:,i], coll_joint[:,-1]-mll, ".", color="tab:blue",label="Joint CI")
                    ax[i,j].plot(coll_inner[:,i], coll_inner[:,-1]-mll, ".", color="tab:green", label="Inner CI")
                    ax[i,j].plot(best_set[i], 0, "o", color="tab:red", label="Best fit")
                    ax[i,j].hlines(0.5 * self.opts.crit_prop[0], 
                                   self.coll_all[ind_prop1:ind_prop2+1,i].min(axis=0),
                                   self.coll_all[ind_prop1:ind_prop2+1,i].max(axis=0),
                                   linestyle=':',color='k', label="95% CI for propagation")
                    ax[i,j].hlines(0.5 * self.opts.crit_prop[1],
                                   self.coll_all[ind_prop1:ind_prop2+1,i].min(axis=0),
                                   self.coll_all[ind_prop1:ind_prop2+1,i].max(axis=0),
                                   linestyle=':',color='k')
                    ax[i,j].hlines(chicrit_single,
                                   self.coll_all[ind_prop1:ind_prop2+1,i].min(axis=0),
                                   self.coll_all[ind_prop1:ind_prop2+1,i].max(axis=0),
                                   linestyle='-',color='k')
                    if self.opts.profile:
                        ax[i,j].plot(profile[j][:,j], profile[j][:,-1]-mll, ".-", color="r", label = "Profile")
                    if j==self.npars-1:
                        ax[i,j].set_xlabel(self.parlabels[i])
                    if i==0:
                        ax[i,j].set_ylabel("-$\\Delta$LL")
                else:
                    ax[i,j].plot(coll_joint[:,j], coll_joint[:,i], "o", color="tab:blue")
                    ax[i,j].plot(coll_inner[:,j], coll_inner[:,i], "o", color="tab:green")
                    ax[i,j].plot(best_set[j], best_set[i], "o", color="tab:red")
                    if ((i*npars+j+1)%npars==1):
                        ax[i,j].set_ylabel(self.parlabels[i])
                    if (i==npars-1):
                        ax[i,j].set_xlabel(self.parlabels[j])

                handles, labels = ax[i,j].get_legend_handles_labels()            
                j+=1
        plt.figlegend(handles, labels, loc='upper right')
        plt.tight_layout()
        if batchmode==False:
            plt.show()
            if savefig:
                plt.savefig(figbasename+"_"+self.model.variant+extension)
        else:
            if savefig:
                plt.savefig(figbasename+"_"+self.model.variant+extension)
            plt.close()

    # Run the paramter space explorer with the given options defined in the 
    # intialization of the class
    def run_parspace(self, batchmode=True, savefig=False,figbasename="fit",extension=".png"):
        """
        Run the parameter space explorer with the given options.
        Parameters:
        -----------
        batchmode : bool
            If True, the figure is not shown
        savefig : bool
            If True, the figure is saved to a file 
            (it works independently from batchmode).
        figbasename : str
            Base name of the figure file.
        extension : str
            Extension of the figure file.
        """
        # crit_add = self.opts.crit_add
        # n_tr = self.opts.n_tr
        # f_d = self.opts.f_d
        # n_ok = self.opts.n_ok[self.npars-1]
        # n_conf = self.opts.n_conf_all[self.npars-1]
        # # support for multiple hb values (because of multiple datasets)
        # tries_1 = np.concatenate((self.opts.tries, [self.opts.tries[-1]]*(self.model.ndatasets-1)))
        # tries_1 = tries_1[self.posfree]
        # n_max = self.opts.n_max
        crit_add = self.opts.crit_add
        n_tr = self.opts.n_tr
        f_d = self.opts.f_d
        n_ok = self.opts.n_ok[self.npars-1]
        n_conf = self.opts.n_conf_all[self.npars-1]
        tries_1 = self.opts.tries * np.ones((1,self.npars))
        n_max = self.opts.n_max


        # We work here with the log-likelihood itself, so the chi2 criterion needs to be divided by 2.
        chicrit_joint = 0.5*self.opts.crit_table[self.npars-1] #criterion for joint 95% CI or parameters
        chicrit_single= 0.5*self.opts.crit_table[0] #criterion forsingle parmeter
        chicrit_rnd = chicrit_joint+crit_add #criterion for random mutations
        chicrit_max = max(1.2*chicrit_joint, chicrit_single+1.5) # criterion for, roughly, a 97.5% joint CI for pruning the sample

        # If number of parameters is below 5, we use regular grid, otherwise we go with the latin hypercube
        # as in BYOM
        n_tries = int(np.prod(tries_1))
        l_bounds = self.model.parbound_lower[self.posfree]
        u_bounds = self.model.parbound_upper[self.posfree]
        if self.npars>=5:
            n_tries=10000
            sampler = qmc.LatinHypercube(d=self.npars) # dimension of the parameter space
            sample = sampler.random(n=n_tries) 
            sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        else:
            p_try = [np.linspace(np.array(lb),np.array(ub),tr) for (lb,ub,tr) in zip(l_bounds, u_bounds, tries_1.astype(int)[0])]
            # calculate all the possible combinations of the grid points
            sample_scaled = np.array(np.meshgrid(*(p_try))).T.reshape(-1, self.npars)
            # reshuffle all combinations (np.random.shuffle works in place)
            np.random.shuffle(sample_scaled)
        
        print(['Starting round ',1,' Initial run with ',n_tries,' parameter sets'])
        d_grid = (np.array(u_bounds)-np.array(l_bounds))/tries_1
        
        llog = np.zeros(sample_scaled.shape[0])
        llog = self._applylog(sample_scaled) 
        sortind = np.argsort(llog) # sort the likelihood values in ascending order
    
        # sorted list of paramters
        coll_all = sample_scaled[sortind]   # paramter vectors
        coll_allL = llog[sortind]           # corresponding likelihood values
        mll = coll_allL[0]                  # best likelihood
    
        ind_cont = np.argwhere(coll_allL < mll+chicrit_rnd[0]).flatten().max()
        ind_cont = max(ind_cont, n_ok) # take at least n_ok points for the next iteration
        ind_cont = min(ind_cont, len(coll_allL)-1) # but not more than we have!

        # parameter sets that are carried to the next iteration
        coll_ok = coll_all[0:ind_cont+1,:]
        coll_okL = coll_allL[0:ind_cont+1]
    
        # remove points above the maximum
        mask = coll_allL < mll+chicrit_max
        coll_all = coll_all[mask,:]
        coll_allL = coll_allL[mask]
    
        # Check if we found a lot of ok values (that can for example happen when
        # the ranges are set much tighter by the user).
        print("So far best likelihood value is: %.4f"%mll)
        n_rnd = 2  # counter, we start from 2 because the first step is the first setup
        n_tr_i = n_tr[n_rnd-1]
        f_d_i = f_d[n_rnd-1]
        chicrit_i = chicrit_rnd[n_rnd-1]
    
        # Also check after first round if it's not too many. If we have a lot of
        # parameter sets to continue with, we can use less tries in the next round.
        if ind_cont > 0.5*n_conf[0]:
            n_tr_i = np.floor(n_tr_i/2)
            if ind_cont >= 1*n_conf[0]:
                n_tr_i = np.floor(n_tr_i/2)
        n_tr_i = min(n_tr_i,max(2,np.floor(10*n_conf[0]/ind_cont)))

        flag_stop = False
        flag_inner = 0
        while flag_stop == False:
            print(['Starting round ',n_rnd,', refining a selection of ',len(coll_okL),' parameter sets, with ',n_tr_i,' tries each'])
    
            # perform random mutations and add to the selected sample
            coll_tries, coll_triesL = self._random_mutations(coll_ok, l_bounds, u_bounds,int(n_tr_i), f_d_i*d_grid)
    
            # perform a Nelder-Mead optimization on the selected sample
            pfit = coll_tries[0]
            bfit, bllog = self._NM_minimizer(pfit,self._startp,self.posfree, rough=1)
            coll_all = np.append([bfit], coll_tries, axis=0)
            coll_allL = np.append(bllog, coll_triesL)
            sortind2 = np.argsort(coll_allL)
            coll_all = coll_all[sortind2]
            coll_allL = coll_allL[sortind2]
    
            mll = coll_allL[0]
    
            # check numbers of sets in the joint and inner rims
            ind_final = np.argwhere(coll_allL < coll_allL[0]+chicrit_joint).flatten().max()   # index in <coll_all> for total joint CI
            ind_single = np.argwhere(coll_allL < coll_allL[0]+chicrit_single).flatten().max() # index in <coll_all> for inner rim
            ind_cont_t = np.argwhere(coll_triesL < coll_triesL[0]+chicrit_i).flatten().max()  # index in just-tried sets that qualify for continuation to next round
            ind_cont_a = np.argwhere(coll_allL < coll_allL[0]+chicrit_i).flatten().max()      # index in <coll_all> that qualify for continuation to next round
            ind_inner = np.argwhere(coll_allL < coll_allL[0]+chicrit_single+0.2).flatten().max()  # index for inner rim, with a little extra
            
            # From openGUTS code:
            # Two additional indices (roughly for a 97.5% joint CI) in case we run
            # into trouble (if we try too many new parameters with chicrit_i).
            # These are indices to the last entry in the two matrices that still
            # are within the total cloud that we like to calculate (slightly more
            # than <chicrit_joint>).
            ind_cont_t2 = np.argwhere(coll_triesL < coll_triesL[0]+chicrit_max).flatten().max()
            ind_cont_a2 = np.argwhere(coll_allL < coll_allL[0]+chicrit_max).flatten().max()
    
            print('  Status: ',ind_final,' sets within total CI and ',ind_single,' within inner. Best fit: %.4f'%coll_allL[0])   

            # ## GUTS only. Check for slow kinetic
            # if ((((self.model.isfree[0]==1) & (self.model.isfree[2]==1)) & ((self.model.islog[2]==0) & (self.model.islog[0]==1))) & ((self.model.parbound_upper[2]/self.model.parbound_lower[2]) > 10)):                

            #     # From openGUTS code:
            #     # Only perform this check when both <kd> and <mw> are fitted, <mw>
            #     # is still on normal scale, and <kd> is still on log-scale. If one
            #     # of these things is different, the user has tampered with the
            #     # parameter ranges and he/she is on his/her own! And, also don't do
            #     # this check when the range of <mw> is less than a factor of 10
            #     # (when the user has modified the range); for a small range, normal
            #     # scale will generally suffice.

            #     coll_tst = coll_all[:max(ind_final,n_ok),:]
            #     min_zs = min(coll_tst[:,2])  # threshold parameter (zs)
            #     min_kd = min(coll_tst[:,0])  # dynamic rate parameter
            #     # get the correlation coefficient between the threshold and kd
            #     check_corr = np.corrcoef(np.log10(coll_tst[:,2]),coll_tst[:,0]) 
            #     # distance from lower bound as fraction of range
            #     crit_zs = (min_zs - self.model.parbound_lower[2]) / (self.model.parbound_upper[2] - self.model.parbound_lower[2])
            #     # distance from lower bound as fraction of range
            #     crit_kd = (min_kd - self.model.parbound_lower[0]) / (self.model.parbound_upper[0] - self.model.parbound_lower[0])
            #     if (check_corr[0,1] > self.opts.slowkin_corr):
            #         if (crit_zs < self.opts.slowkin_pars) | (crit_kd < self.opts.slowkin_pars):
            #             # fix here
            #             # the new limits are taken as the minimum and maximum of the current cloud
            #             lowerv = np.min(coll_tst, axis=0)                        
            #             upperv = np.max(coll_tst, axis=0)
            #             slwkin = -1
            #             # return to the main code highlighting that slow kinetic was found
            #             # the main code will restart the profile changing the scale of
            #             # the threshold parameter (from linear to log)
            #             return (slwkin,lowerv,upperv)
    
            # This block is aiming to make sure
            # not to try too few or too many points in the next round, but select
            # an efficient set for continuation in <coll_ok>.
            if ind_final > n_conf[0]:      # checking if the outer rim has enough values (see options)
                if ind_single > n_conf[1]: # checking if the inner rim has enough values (see options)
                    print('  Stopping criterion met: ',ind_final,' sets within total CI and ',ind_single,' within inner')
                    flag_stop = True
                else:
                    print("Next round will be focussed on the inner rim (outer rim has enough values)")
                    coll_ok = coll_all[0:ind_inner+1]
                    coll_okL = coll_allL[0:ind_inner+1]
                    flag_inner = 1
                    if ind_inner < n_ok:  # if there are very few currently in the inner rim, take the <n_ok> best ones from <coll_all>
                        coll_ok = coll_all[0:n_ok]
                        coll_okL = coll_allL[0:n_ok]
                    elif (ind_inner>0.5*n_conf[1]): #if we already have quite some values in inner rim
                        limits = [max(0, ind_single-n_ok), ind_single+n_ok]
                        coll_ok = coll_all[limits[0]:limits[1]+1]
                        coll_okL = coll_allL[limits[0]:limits[1]+1]
                        # add the optimized values
                        coll_ok = np.append([coll_all[0]], coll_ok, axis=0)
                        coll_okL = np.append(coll_allL[0], coll_okL) 
            else:
                # not enough values in the joint dataset
                if ind_cont_t > n_ok:  
                # From openGUTS code: if there are enough in <coll_tries> to continue with ...
                # in principle, the next round will continue from sets in
                # <coll_tries> and not from <coll_all>. The reason is that it is
                # better to propagate new sets than to continue propagating
                # sets that have been propagated before already.
                    if ind_cont_t > 2*n_conf[0]: # if we will try more than 2 times of what we finally need
                        if ind_final > 0.5*n_conf[0]: # if we already have half the values we need
                            # focus on the edge of the inner rim as there more precision
                            # is needed
                            limits=[max(0,ind_single-n_ok),min(ind_cont_t,ind_single+n_ok)]
                            coll_ok = coll_tries[limits[0]:limits[1]]
                            coll_okL = coll_triesL[limits[0]:limits[1]]
                        else: # then we have a lot of values to try, but not so much accepted yet
                            if coll_triesL[2*n_conf[0]] > mll + chicrit_max:
                                ind_cont_t = 2*n_conf[0]
                            else:
                                # then it is not a good idea to limit continuation to 2 * <n_conf(0)>
                                ind_cont_t = ind_cont_t2
                            coll_ok  = coll_tries[0:ind_cont_t+1]
                            coll_okL = coll_triesL[0:ind_cont_t+1]
                    else:
                        coll_ok  = coll_tries[0:ind_cont_t]
                        coll_okL = coll_triesL[0:ind_cont_t]
                    coll_ok = np.append([coll_all[0]], coll_ok, axis=0)
                    coll_okL = np.append(coll_allL[0], coll_okL)
                else:
                    # there are NOT enough values in <coll_tries> to continu with ... then we need to look at <coll_all>
                    if ind_cont_a > n_ok:
                        if ind_cont_a > 2 * n_conf[0]:
                            if coll_allL[2*n_conf[0]] > mll + chicrit_max:
                                ind_cont_a = 2*n_conf[0]
                            else:
                                ind_cont_a = ind_cont_a2
                        coll_ok = coll_all[0:ind_cont_a+1]
                        coll_okL = coll_allL[0:ind_cont_a+1]
                    else:
                        coll_ok = coll_all[0:n_ok]
                        coll_okL = coll_allL[0:n_ok]
    
            # From openGUTS code: See if we need a new round, and if so, prepare for it.
            # Again, special care is taken to avoid taking too few or too many new
            # tries in the next round. Therefore, the tricky part is to come up
            # with an efficient value for <n_tr_i> (number of random mutations per
            # set in the next round).
            if (n_rnd == n_max) & (flag_stop == False):
                # the loop stops anyway here as maximum number of tries is reached
                flag_stop = True
                print('  Stopping criterion not met after ',n_rnd,' rounds')
            n_rnd = n_rnd + 1
            if n_rnd <= len(n_tr):
                n_tr_i = n_tr[n_rnd-1]
                f_d_i = f_d[n_rnd-1]
                chicrit_i = chicrit_rnd[n_rnd-1]
            else:
                n_tr_i = n_tr[-1]
                f_d_i = f_d[-1]
                chicrit_i = chicrit_rnd[-1]
            
            #  Modify number of tries if we have a lot or only few sets within the total cloud or inner rim.
            crit_ntry = [ ind_final/n_conf[0], ind_single/n_conf[1] ]
            if (crit_ntry[flag_inner] > 0.75) | (len(coll_okL) > 2000):
                # If we already have more than 75% of what we finally need, or more than 2000 sets to continue with
                # Note: when <flag_inner> = 1, we look at the status of the inner rim!
                n_tr_i = np.floor(n_tr_i/2) # decrease the number of tries per set for next round
            elif n_rnd > 3: # starting at round 4, we start to worry if we haven't found so many yet
                if (len(coll_okL) < 0.5*n_conf[flag_inner]) & (len(coll_okL) < 1000):
                    # If we have less than 50% of what we finally need, and less than 1000 sets to try next
                    n_tr_i = 2* n_tr_i  # double number of tries
                    if (n_rnd > 4) & (crit_ntry[flag_inner] < 0.25):
                        n_tr_i = 2*n_tr_i
                elif n_rnd > 7:
                    n_tr_i = 2*n_tr_i
            
            n_tr_i = min(n_tr_i, max(2, np.floor(10*n_conf[flag_inner]/len(coll_okL))))
            # From openGUTS code: 
            # This allows a max of 10x the target value to be tried in the next
            # round (scaling back <n_tr_i>). Well, sometimes more, as at least
            # 2 tries will be allowed for each set in <coll_ok>. I used to have
            # a check for whether we are in the final rounds or if we found
            # very little, but I don't think that is needed (and sometimes
            # leads to large numbers of new tries). If all fails, the extra
            # sampling rounds will come to the rescue.


            # TODO further testing, maybe make a separate function for it, like in BYOM
            # pruning option.
            mask=coll_allL < mll+chicrit_max
            coll_all = coll_all[mask,:]
            coll_allL = coll_allL[mask]
            # The following code works because coll_allL is a sorted array.
            # This means that if 2 parameter vectors have the same likelihood, they will
            # be next to each other. We want to avoid the eventuality that we remove
            # two parameter vectors that are different, but that have the same likelihood
            # so first we check if there are parameter duplicates and remove them
            # only if they are really the same (so if the sum of the mask is equal to the number of parameters)
            # This might require some more testing
            # in the openGUTS code the duplicate check is not present, but it is there in BYOM
            mask = np.sum(coll_all[1:]==coll_all[:-1], axis=1)
            coll_all = np.append([coll_all[0]], coll_all[1:,:][mask!=self.npars], axis=0)
            coll_allL = np.append(coll_allL[0], coll_allL[1:][mask!=self.npars])
            print("Removed ", np.sum(mask==self.npars), " duplicate values")
        print('Loop is over')
    
        # perform now a simplex optimization
        print("Now we proform a final simplex optimization")
        pfit = coll_all[0]  # take the best set
        bfit_final,llog_final = self._NM_minimizer(pfit,self._startp,self.posfree, rough=0)
        coll_all = np.append([bfit_final], coll_all, axis=0)
        coll_allL = np.append(llog_final, coll_allL)
        mll = coll_allL[0]
        ind_final = np.argwhere(coll_allL < coll_allL[0]+chicrit_joint).flatten().max()
        ind_single = np.argwhere(coll_allL < coll_allL[0]+chicrit_single).flatten().max()
        print('  Status: ',ind_final,' sets within total CI and ',ind_single,' within inner. Best fit: %.4f'%coll_allL[0])
        # merging points and likelihood values
        self.coll_all = np.append(coll_all, coll_allL[:,None], axis=1) # try with this
        if (self.opts.profile==0):
            ind_prop1 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[0]).flatten().max()
            ind_prop2 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[1]).flatten().max()
            self.pbest = self.coll_all[0,:]
            self.propagationset = self.coll_all[ind_prop1:ind_prop2,:-1]
            self.model.parvals[self.posfree] = self.pbest[:-1]
            self.fullset = np.copy(self.model.parvals)
            print("Final results:")
            self._print_results()
            self._plot_samples(batchmode=batchmode,savefig=savefig,
                               figbasename=figbasename,
                               extension=extension)    
        else:
            # perform a profile likelihood to determine the confidence intervals
            # if the profiling process finds a new minimum, we run further
            # optimization from there
            print("Starting round ",n_rnd," for profile likelihood of each parameter")
            parprofile=[0]*n_cores
            pbest = np.zeros((self.npars,self.npars+1))
            # TESTING PROFILING WITH REDUCED SAMPLE. REMOVE AFTER TESTING
            # self.coll_all = np.copy(self.coll_all[np.unique(np.sort(np.random.randint(0,len(coll_allL),500)))])
            # END TEST
            # sequential implementation replaced with parallel implementation
            # for i in range(self.npars):
            #     parprofile.append(self._parameter_profile_sub(i))
            # each parameter sent to a different core
            with mp.Pool(n_cores) as pool:
                results = pool.starmap(parameter_profile_sub_wrapper, [(i, self) for i in range(self.npars)])
                parprofile, pbest, coll_ok = zip(*results)
            # TODO proper pruning of the profiling results
            self.pbest = pbest[np.argmin([p[-1] for p in pbest])]
            #self.coll_ok = coll_ok[np.argmin([p[-1] for p in pbest])] # take the best value
            self.coll_ok = np.concatenate((coll_ok), axis=0) # rejoin all the coll_ok arrays from the profiling function
            if self.pbest[-1] < mll:
                print("The profiling found a new minimum. Optimizing from here")
                pfit,llog = self._NM_minimizer(self.pbest[:-1],self._startp,self.posfree, rough=0)
                self.coll_all = np.append([np.concatenate((pfit, [llog]))], self.coll_all, axis=0)
                if self.coll_all[1,-1] < self.coll_all[0,-1] - self.opts.real_better:
                    # This will trigger new round of optimisation
                    print("The new minimum is significantly better than the old one")
                    self.coll_ok = np.append([np.concatenate((pfit, [llog]))], self.coll_ok, axis=0)
                ind_final = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_joint).flatten().max()
                ind_single = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_single).flatten().max()
            mll = self.coll_all[0,-1] # store the new best likelihood value    
            if (self.coll_ok.size > 0) | (ind_single < n_conf[1]):
                # still not enough values in the inner rim or there are gaps between profile and samples
                # resampling will be needed
                if (self.coll_ok.size < 1) | ((ind_single < n_conf[1]) & (self.coll_ok.shape[0] < 10)):
                    self.coll_ok = np.append(self.coll_ok, self.coll_all[max(0,ind_single-n_ok):min(ind_single+n_ok,self.coll_all.shape[0])], axis=0)
                coll_tmp    = np.append(self.coll_ok[:,:-1] , self.coll_all[0:ind_final+1,:-1], axis=0)
                edges_cloud = np.array([coll_tmp.min(axis=0), coll_tmp.max(axis=0)])
                # TODO remove coll_tmp
                d_grid = self.opts.d_extra * (edges_cloud[1] - edges_cloud[0])
                f_d_i = self.opts.f_d_extra
                self.coll_ok = self.coll_ok[np.argsort(self.coll_ok[:,-1])]
                self.coll_ok = self.coll_ok[self.coll_ok[:,-1] < self.coll_ok[0,-1] + self.opts.crit_add_extra,:]
                n_rnd_x=0
                n_test =0
                if ind_single < n_conf[1]:
                    print("Profiling has led to a new optimum, which left insufficient sets in the inner rim: extra sampling rounds will be started.")
                elif (self.coll_ok.size>0):
                    print("Profiling has detected gaps between profile and sample, which requires extra sampling rounds.")
                while (self.coll_ok.shape[0]>0) & (n_rnd_x < 10):
                    n_cont = self.coll_ok.shape[0]
                    n_rnd_x = n_rnd_x+1

                    # set a reasonable number of new tries
                    n_tr_i = 40
                    n_tr_i = int(min(n_tr_i,max(2, np.floor(5*n_conf[1]/n_cont))))

                    print("Starting round ", n_rnd, " (extra ", n_rnd_x, ") refining ", self.coll_ok.shape[0], " parameter sets, with ", n_tr_i, " tries each")
                    samples = np.copy(self.coll_ok[:,:-1])
                    coll_tries, coll_triesL = self._random_mutations(samples, l_bounds, u_bounds, n_tr_i, f_d_i*d_grid)
                    coll_tries = np.append(coll_tries, coll_triesL[:,None], axis=1)
                    self.coll_all = np.append(self.coll_all, coll_tries, axis=0)
                    self.coll_all = self.coll_all[np.argsort(self.coll_all[:,-1])]
                    self.coll_all = self.coll_all[self.coll_all[:,-1] < (self.coll_all[0,-1] + chicrit_max),:]

                    ind_final = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_joint).flatten().max()
                    ind_single = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_single).flatten().max()
                    # test if the profile is very far from the sample
                    # Print debugging info
                    # print("n_tr_i ",n_tr_i)
                    # print("f_d_i*d_grid ", f_d_i*d_grid)
                    # print("size coll_tries ",coll_tries.shape)
                    # print("size coll_tries ",coll_tries.shape)
                    # print("size coll_all ",self.coll_all.shape)
                    # print("ind_final: ",ind_final)
                    # print("ind_single: ",ind_single)

                    # From openGUTS code: Run a test to see whether the sample now DOES match
                    # the profile likelihood. It checks where the profile is
                    # considerably lower than the sample (which means that more
                    # sampling is needed), and when the sample is lower than the
                    # profile (which means we may need a new profile).
                    flag_profile=self._test_profile(self.coll_all, parprofile, self.opts)
                    # print(self.coll_ok)
                    # print("flag_profile")
                    # print(flag_profile) # DEBUG
                    if ind_single < n_conf[1]:
                        # Are more samples needed? First check in the inner rim 

                        # In the openGUTS code there is a warining that theoretically this could lead
                        # to an infinite loop, but it is also written that in practice it would be
                        # highly unlikly that a continuous sampling would not fill the inner rim
                        ind_cont = np.argwhere(coll_tries[:,-1] < mll+self.opts.crit_add_extra).flatten().max() # only continue with new tries that are close to inner rim
                        if ind_cont < (0.5 * n_conf[1]):
                            self.coll_ok = np.append(self.coll_ok, coll_tries[0:ind_cont+1,:], axis=0)
                        else:
                            self.coll_ok = coll_tries[0:ind_cont+1,:]
                        f_d_i = max(0.1, f_d_i * 0.7) # reduce jump size so as to contract the extra sampling
                    elif (self.coll_ok.shape[0] > 0) & (n_rnd_x<5):
                        f_d_i = max(0.1,f_d_i * 0.7)
                    elif (flag_profile[1] > 0.05) & (n_test < 2):
                        n_rnd = n_rnd +1
                        print("Starting round ",n_rnd,", refining the profile likelihoods for each parameter again")
                        n_test = n_test + 1
                        parprofile = list() # overwrite the profile
                        # for i in range(self.npars):
                        #     parprofile.append(self._parameter_profile_sub(i))
                        with mp.Pool(n_cores) as pool:
                            results = pool.starmap(parameter_profile_sub_wrapper, [(i, self) for i in range(self.npars)])
                            parprofile, pbest, coll_ok = zip(*results)
                        self.pbest = pbest[np.argmin([p[-1] for p in pbest])]
                        #self.coll_ok = coll_ok[np.argmin([p[-1] for p in pbest])] # take the best value
                        self.coll_ok = np.concatenate((coll_ok), axis=0)
                        if self.pbest[-1] < mll:
                            self.coll_all = np.append([self.pbest], self.coll_all, axis=0)
                            ind_final = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_joint).flatten().max()
                            ind_single = np.argwhere(self.coll_all[:,-1] < self.coll_all[0,-1]+chicrit_single).flatten().max()
                        print("Status: ", ind_final, " sets within total CI and ", ind_single, " within inner. Best fit: ", self.coll_all[0,-1])
                        if self.coll_ok.size>0:
                            n_rnd_x = 0 # resetting extra counter
                            f_d_i = self.opts.f_d_extra
                            print("Now profiling has detected gaps, which require additional sampling rounds.")
                    else:
                        if self.coll_ok.size > 0:
                            print("Exiting parspace explorer, but still ", self.coll_ok.shape[0], " sets flagged (too much distance between profile and sample). Check plot.")
                        if flag_profile[1] > 0.05:
                            print("Exiting parameter space explorer, but there still seem to be parameter sets outside of the profile curves. Check plot.")
                        self.coll_ok=np.array([])
                if n_rnd_x > 10:
                    print("Exiting parameter space explorer, as mx number of extra rounds reached. Check plot carefully!")

            self.coll_all = self.coll_all[self.coll_all[:,-1]<self.coll_all[0,-1]+chicrit_max,:]
            self.pbest = self.coll_all[0,:]
            self.model.parvals[self.posfree] = self.pbest[:-1]
            self.fullset = np.copy(self.model.parvals)
            self.bestaic = 2*self.pbest[-1]+2*self.npars

            self._plot_samples(parprofile,batchmode=batchmode,
                               savefig=savefig,
                               figbasename=figbasename,
                               extension=extension)
            self.profile = parprofile
            self._print_results(self.profile)
            
            ind_prop1 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[0]).flatten().max()
            ind_prop2 = np.argwhere(self.coll_all[:,-1] < mll + 0.5 * self.opts.crit_prop[1]).flatten().max()
            self.propagationset = self.coll_all[ind_prop1:ind_prop2+1,:-1]
        return (0,0,0)
    
    def replot_results(self, batchmode=False, savefig=False, figbasename="", extension=".png"):
        """
        Function to reproduce the parameter
        space plot calling the function _plot_samples
        
        Arguments
        ---------
        bacthmode : bool
            flag to run the plotting in batch mode
            (no plotting shown on screen)
        savefig : bool
            flag to save the figure
        figbasename : string
            base name of the figure
        extension : string
            extension of the figure
        """
        if self.profile:
            self._plot_samples(profile=self.profile, batchmode=batchmode, savefig=savefig, figbasename=figbasename, extension=extension)
        else:
            self._plot_samples(savefig=savefig, batchmode=batchmode, figbasename=figbasename, extension=extension)
        
    def reprint_results(self):
        """
        Function to reprint the results of 
        the parameter space explorer
        """
        if self.profile:
            self._print_results(self.profile)
        else:
            self._print_results()

    def save_sample(self,filename):
        """
        Saves the class to file
        
        Argument
        --------
        filename : string
            path to save the class
        """
        import pickle
        with open(filename,'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load_class(cls, filename):
        """
        Reload a saved results

        Argument
        --------
        filename : string
            path of the file where the class
            has been strored
        """
        import pickle
        with open(filename,'rb') as f:
            return(pickle.load(f))

        





