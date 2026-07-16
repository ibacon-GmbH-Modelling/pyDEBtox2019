import numpy as np
from numba import jit
from scipy.integrate import odeint,solve_ivp

@jit(nopython=True)
def DEBtox2019_derivatives_solveivp(t, y, C, timextr, DEBpars, moa, feedb):
#def DEBtox2019_derivatives(y, t, C, timextr, DEBpars, moa, feedb):
    # y0 damage
    # y1 length
    # y2 reproduction
    # y3 survival
    dydt = np.zeros(4)

    #unpack parameters
    FBV, KRV, kap, yP, Lm_ref, L0, Lp, Lm, rB, Rm, f, hb, a, Lf, Lj, Tlag, kd, bb, zb, bs, zs = DEBpars
    hb = a * hb**a * t**(a-1)  # age-dependent background hazard
    Cval = np.interp(t, timextr,C)  #to be seen here how to deal with multiplication factors

    y[1] = max(1e-3*L0,y[1])  # to avoid numerical issues with length = 0
    if Lf>0:
        f = f/(1+(Lf*Lf*Lf)/(y[1]+y[1]+y[1]))
    if Lj>0:
        f = f * min(1, y[1]/Lj)

    stress = bb*max(y[0]-zb,0)
    hazard = bs*max(y[0]-zs,0)

    hazard = min(hazard, 111.) # avoid stiffness
    sMOA = moa * stress
    sMOA[0] = min(sMOA[0],1) 
    sA,sM,sG,sR,sH = sMOA
    dydt[1] = rB * ((1+sM)/(1+sG)) * (f*Lm*((1-sA)/(1+sM)) - y[1])  # ODE for body length
    # introduce starvation rules
    fR = f    # if there is no starvation, f for reproduction is the standard f
    if (dydt[1]<0):
        fR = (f - kap * (y[1]/Lm) * ((1+sM)/(1-sA)))/(1-kap)
        if (fR >= 0):
            dydt[1] = 0  # stop growth but no shrinking
        else:
            fR = 0
            dydt[1] = (rB*(1+sM)/yP) * ((f*Lm/kap)*((1-sA)/(1+sM)) - y[1]) # allow some shrinking
        
    R = 0  # reproduction is 0, unless...
    if (y[1]>=Lp):
        R = max(0.,(np.exp(-sH)*Rm/(1+sR)) * (fR*Lm*(y[1]*y[1])*(1-sA) - (Lp*Lp*Lp)*(1+sM))/(Lm*Lm*Lm - Lp*Lp*Lp))
    dydt[2] = R
    dydt[3]  = -(hazard + hb) * y[3]

    xu,xe,xG,xR = feedb * np.array([Lm_ref/y[1], Lm_ref/y[1], (3/y[1])*dydt[1], R*FBV*KRV])
    if xu==0:
        xu = 1
    if xe==0:
        xe = 1
    xG = max(xG,0)

    dydt[0] = kd * (xu * Cval - xe * y[0]) - (xG + xR) * y[0]

    if (y[1] <= 0.5 * L0): # if an animal has size less than half the start size ...
        dydt[1] = 0.  # don't let it grow or shrink any further (to avoid numerical issues)
            
    if (t<Tlag):
        # derivatives are non-zero only if time is greater than Tlag
        dydt[0] = 0
        dydt[1] = 0
        dydt[2] = 0
        dydt[3] = 0

    return(dydt)

@jit(nopython=True)
def DEBtox2019_derivatives_odeint(y, t, C, timextr, DEBpars, moa, feedb):
    # y0 damage
    # y1 length
    # y2 reproduction
    # y3 survival
    dydt = np.zeros(4)

    #unpack parameters
    FBV, KRV, kap, yP, Lm_ref, L0, Lp, Lm, rB, Rm, f, hb, a, Lf, Lj, Tlag, kd, bb, zb, bs, zs = DEBpars
    hb = a * hb**a * t**(a-1)  # age-dependent background hazard
    Cval = np.interp(t, timextr,C)  #to be seen here how to deal with multiplication factors

    y[1] = max(1e-3*L0,y[1])  # to avoid numerical issues with length = 0
    if Lf>0:
        f = f/(1+(Lf*Lf*Lf)/(y[1]+y[1]+y[1]))
    if Lj>0:
        f = f * min(1, y[1]/Lj)

    stress = bb*max(y[0]-zb,0)
    hazard = bs*max(y[0]-zs,0)

    hazard = min(hazard, 111.) # avoid stiffness
    sMOA = moa * stress
    sMOA[0] = min(sMOA[0],1) 
    sA,sM,sG,sR,sH = sMOA
    dydt[1] = rB * ((1+sM)/(1+sG)) * (f*Lm*((1-sA)/(1+sM)) - y[1])  # ODE for body length
    # introduce starvation rules
    fR = f    # if there is no starvation, f for reproduction is the standard f
    if (dydt[1]<0):
        fR = (f - kap * (y[1]/Lm) * ((1+sM)/(1-sA)))/(1-kap)
        if (fR >= 0):
            dydt[1] = 0  # stop growth but no shrinking
        else:
            fR = 0
            dydt[1] = (rB*(1+sM)/yP) * ((f*Lm/kap)*((1-sA)/(1+sM)) - y[1]) # allow some shrinking
        
    R = 0  # reproduction is 0, unless...
    if (y[1]>=Lp):
        R = max(0.,(np.exp(-sH)*Rm/(1+sR)) * (fR*Lm*(y[1]*y[1])*(1-sA) - (Lp*Lp*Lp)*(1+sM))/(Lm*Lm*Lm - Lp*Lp*Lp))
    dydt[2] = R
    dydt[3]  = -(hazard + hb) * y[3]

    xu,xe,xG,xR = feedb * np.array([Lm_ref/y[1], Lm_ref/y[1], (3/y[1])*dydt[1], R*FBV*KRV])
    if xu==0:
        xu = 1
    if xe==0:
        xe = 1
    xG = max(xG,0)

    dydt[0] = kd * (xu * Cval - xe * y[0]) - (xG + xR) * y[0]

    if (y[1] <= 0.5 * L0): # if an animal has size less than half the start size ...
        dydt[1] = 0.  # don't let it grow or shrink any further (to avoid numerical issues)
            
    if (t<Tlag):
        # derivatives are non-zero only if time is greater than Tlag
        dydt[0] = 0
        dydt[1] = 0
        dydt[2] = 0
        dydt[3] = 0

    return(dydt)



def calc_DEBresults(C, timextr, y0, DEBpars, moa, feedb,timeext,solver='RK45'):
    # L0 = DEBpars[5]
    # y0 = np.array([0., L0*1.0, 0., 1.])
    # TODO: test if we need a very extended time vector or it will be enough to use the union of
    #       the time points of the experimental data (for speed reasons, given that rk45 should not care)
    #       in byom it seems to be enough to take the intersection of all the points in the datasets.
    if solver=="RK45":
        # solve_ivp has the possibility to choose a different algorithm (default is RK45)
        sol = solve_ivp(fun=DEBtox2019_derivatives_solveivp, t_span=np.array([timeext[0], timeext[-1]]), y0=y0, 
                    args=(C, timextr, DEBpars, moa, feedb), t_eval=timeext, rtol=1e-9,atol=1e-9)#, dense_output=True)
        return(sol.y)
    elif solver=="LSODA":
    # odeint uses LSODA algorithm
        sol = odeint(DEBtox2019_derivatives_odeint, y0, timeext, args=(C, timextr, DEBpars, moa, feedb), rtol=1e-9, atol=1e-9)
        return(sol.T)
    else:
        raise ValueError("Solver not recognized. Use 'RK45' or 'LSODA'.")


@jit(nopython=True)
def survival_loglikelihood(modelvector, commontime, deathvector):
    # print(commontime)
    surviv_selected = modelvector[commontime]
    #print(surviv_selected)
    pdeath = np.append(-np.diff(surviv_selected),surviv_selected[-1])
    pdeath = np.maximum(pdeath,1e-50)
    #print(deathvector)
    llik=np.dot(deathvector,np.log(pdeath))
    return(llik)


@jit(nopython=True)
def scaled_loglikelihood(model,lengths,weights,transf):
    # print("model:", model)
    # print("lengths:", lengths)
    llk=0.0
    ind_fin = np.isfinite(lengths) & (weights>0)
    weights = weights[ind_fin]
    n = np.sum(ind_fin)
    Nv = np.sum(weights[weights!=0])
    if transf == 0:
        #log transformation
        caplengths = np.maximum(lengths,1e-10)
        capmodel = np.maximum(model,1e-10)
        res = np.log(caplengths) - np.log(capmodel)
        mn = np.mean(np.log(caplengths))
        res_tot = np.log(caplengths) - mn
    else:
        res = lengths**transf - model**transf
        mn = np.mean(lengths**transf)
        res_tot = lengths**transf - mn   
    wssq = np.dot(res*weights,res)
    wssq2= np.dot(res * weights**2,res)
    wssq_tot = np.dot(res_tot*weights,res_tot)
    llk = -0.5*n * np.log(wssq2) - Nv * wssq/(2*wssq2)
    return(llk)


# @jit(nopython=True)
# def repro_loglikelihood(modelvector, commontime, reproarray):
#     return(0)


class DEBtox2019models:
    '''
    Class that contains the functions that are used to calculate the likelihood
    of the GUTS model.

    Attributes:
    - variant: string that specifies the variant of the GUTS model (SD or IT)
    - ndatasets: number of datasets
    - concstruct: list of concentration data structures
    - datastruct: list of survival data structures
    - nbinsperday: number of bins per day
    - timeext: extended time vector
    - index_commontime: indices of the common time points between the concentration and survival data
    - parnames: list of parameter names
    - parvals: list of parameter values
    - islog: list of booleans that specify if the parameter is log-transformed
    - isfree: list of booleans that specify if the parameter is free
    - posfree: indices of the free parameters
    - parbound_lower: list of lower bounds for the parameters
    - parbound_upper: list of upper bounds for the parameters

    Methods:
    - calc_ext_time: calculate the extended time vector and the indices of the common time points with the 
                     original survival data	
    - calc_damage: calculate the damage variable
    - calc_survival: calculate the survival probability
    - log_likelihood: calculate the log-likelihood of the GUTS model
    '''
    def __init__(self, 
                 completedataset_list, 
                 debparameterclass,
                 moa, feedb,
                 Tbp = 0,
                 min_t=500,
                 solver ='LSODA',
                 breaktime = False):
        '''
        Constructor for the GUTSmodels class. This class contains the
        functions that are used to calculate the likelihood of the GUTS
        model. The class is initialized with the following arguments:

        Arguments:
          - completedataset_list: list of classes of complete datasets
          - concstruct: list of concentration data structures (length depends on the number of datasets)
          - variant: string that specifies the variant of the GUTS model (SD or IT)
            - parnames: list of parameter names
            - parvals: list of parameter values
            - islog: list of booleans that specify if the parameter is log-transformed
            - isfree: list of booleans that specify if the parameter is free
            - parbound_lower: list of lower bounds for the parameters
            - parbound_upper: list of upper bounds for the parameters
            - min_t: 
        '''
        self.debparameterclass = debparameterclass
        self.ndatasets = len(completedataset_list)  # number of datasets
        self.par_dataset_map = debparameterclass.par_dataset_map
        self.full_base_names = debparameterclass.full_base_names
        # attributes that deal with the model parameters
        self.parnames = np.array(debparameterclass.full_names,dtype=object)   # make sure these are numpy arrays
        self.parvals = np.array(debparameterclass.full_list)
        self.islog = np.array(debparameterclass.full_islog)                   # make sure these are numpy arrays	
        self.isfree = np.array(debparameterclass.full_isfree)                 # make sure these are numpy arrays
        self.posfree = np.argwhere(self.isfree == 1).flatten()  # positions of the free parameters in the parameter vector
        self.parbound_lower = np.array(debparameterclass.full_lowlim) # make sure these are numpy arrays
        self.parbound_upper = np.array(debparameterclass.full_uplim) # make sure these are numpy arrays
        # islogindex = np.argwhere(self.islog==True).flatten()
        # self.parbound_lower[islogindex] = np.log10(self.parbound_lower[islogindex])
        # self.parbound_upper[islogindex] = np.log10(self.parbound_upper[islogindex])
        #self.parvals[self.islog] = np.log10(self.parvals[self.islog]) leave it as is..in the input should have the right scaling aleardy
        self.moa = moa
        self.feedb = feedb
        self.Tbp = Tbp
        self.min_t = min_t
        self.solver = solver
        self.breaktime = breaktime
        # deal with the actual data and concentration
        self.timeext = []

        self.endpoints = np.array([0,1,2,3])

        # for now assumption that all the endpoints are there.
        # will deal with missing data later
        self.concstruct_list = [None]*self.ndatasets
        self.lengthstruct_list = [None]*self.ndatasets
        self.reprostruct_list = [None]*self.ndatasets
        self.survstruct_list = [None]*self.ndatasets
        self.active_endpoints = [] 
        self.indexcommon_surv = [None]*self.ndatasets
        self.indexcommon_length = [None]*self.ndatasets
        self.indexcommon_repro = [None]*self.ndatasets
        for i in range(self.ndatasets):
            self.active_endpoints.append([])
            self.timeext.append(completedataset_list[i].complete_timevec) # complete extended time vector for the dataset
            self.concstruct_list[i] = completedataset_list[i].concdata
            if hasattr(completedataset_list[i], 'lengthdata'):
                self.lengthstruct_list[i] = completedataset_list[i].lengthdata
                self.indexcommon_length[i] = completedataset_list[i].time_indices['lengthdata']
                self.active_endpoints[i].append(1)
            if hasattr(completedataset_list[i], 'reprodata'):
                self.reprostruct_list[i] = completedataset_list[i].reprodata
                self.indexcommon_repro[i] = completedataset_list[i].time_indices['reprodata']
                self.active_endpoints[i].append(2)
            if hasattr(completedataset_list[i], 'survdata'):
                self.survstruct_list[i] = completedataset_list[i].survdata
                self.indexcommon_surv[i] = completedataset_list[i].time_indices['survdata']
                self.active_endpoints[i].append(0)
            # this is because the more stuff are precalculated before likelihood evaluation, the better
            # makes the code faster
        #TODO: Add print statement to show which parameters are free, which are fixed, and their bounds for verification purposes.
        print("Initialized DEBtox2019models with the following parameters:")
        print("For easiness of reading, log-transformed parameters are shown in their original scale (10^value) if islog is True.")
        # Add header line for the printout
        print(f"{'Parameter':<10} {'Value':<8} {'Log-Tr.':<8} {'Free':<6} {'Lower Bound':<10} {'Upper Bound':<10}")
        for i in range(len(self.parnames)):
            if self.islog[i]:
                print(f"{self.parnames[i]:<10} {10**self.parvals[i]:<8.4f} {self.islog[i]:<8} {self.isfree[i]:<6} ({10**(self.parbound_lower[i]):<10.4f}, {10**(self.parbound_upper[i]):<10.4f})")
            else:
                print(f"{self.parnames[i]:<10} {self.parvals[i]:<8.4f} {self.islog[i]:<8} {self.isfree[i]:<6} ({self.parbound_lower[i]:<10.4f}, {self.parbound_upper[i]:<10.4f})")

    
    def build_dataset_parameters(self, expanded_parvals, nd):
        """
        Collapse expanded parameter vector into a dataset-specific
        DEB parameter vector compatible with the ODE solver.
        """
    
        # Canonical DEB parameter order (normalized)
        deb_order = [
            "fbv", "krv", "kap", "yp", "lm_ref",
            "l0", "lp", "lm", "rb", "rm",
            "f", "hb", "a", "lf", "lj", "tlag",
            "kd", "bb", "zb", "bs", "zs"
        ]
    
        compact = np.zeros(len(deb_order))
    
        for i, pname in enumerate(deb_order):
        
            # All expanded indices for this base parameter
            indices = np.where(self.full_base_names == pname)[0]
    
            if len(indices) == 0:
                raise RuntimeError(f"Parameter '{pname}' missing.")
    
            # 1️ Find grouped / dataset-specific match
            selected = []
            for idx in indices:
                owner = self.par_dataset_map[idx]
                if owner == -1:
                    continue
                if nd in owner:
                    selected.append(idx)
    
            if len(selected) == 1:
                compact[i] = expanded_parvals[selected[0]]
                continue
            
            if len(selected) > 1:
                raise RuntimeError(
                    f"Ambiguous grouped definition for '{pname}' in dataset {nd}"
                )
    
            # 2️ Fallback to shared parameter
            shared = [idx for idx in indices if self.par_dataset_map[idx] == -1]
    
            if len(shared) == 1:
                compact[i] = expanded_parvals[shared[0]]
                continue
            
            raise RuntimeError(
                f"Cannot resolve parameter '{pname}' for dataset {nd}"
            )
    
        return compact

    
    def calc_model(self, C, timextr, DEBpars, moa, feedb, timeext):
        """
        Compute DEB model with optional reproduction time delay (Tbp).
        Assumptions:
          - modelsol has shape (n_states, n_timepoints).
          - timeext is a 1D increasing array of time points.
          - calc_DEBresults returns solution aligned to the provided time grid.
        """
        L0 = DEBpars[5]
        y0 = np.array([0., L0*1.0, 0., 1.])
        # If no delay, shortcut
        if not (hasattr(self, "Tbp") and self.Tbp and self.Tbp > 0):
            return calc_DEBresults(C, timextr, y0, DEBpars, moa, feedb, timeext, solver=self.solver)

        Tbp = self.Tbp
        # compute delayed times (tbp) relative to original grid
        # only times strictly greater than Tbp contribute to a delayed output
        tbp = timeext[timeext > Tbp] - Tbp
        if tbp.size == 0:
            # Nothing to delay; just solve on the original grid
            return calc_DEBresults(C, timextr, y0, DEBpars, moa, feedb, timeext, solver=self.solver)
        # Merge original time grid with delayed grid so the solver "sees" the delayed trajectory
        # Keep order and uniqueness
        newtime = np.unique(np.concatenate((timeext, tbp, timextr))) # adding the original conc times for stability
        modelsol_union = np.zeros((4, len(newtime)))  # pre-allocate for union grid results
        if self.breaktime:
            # print("Newtime vector for breaktime approach: ", newtime)
            for t in range(len(C)-1):
                shifted_tvec = newtime[(newtime <= timextr[t+1]) & (newtime >= timextr[t])] - timextr[t]
                # print("Segment ", t, " with shifted time vector: ", shifted_tvec)
                # print("Concentration segment: ", C[t:t+1])
                modelsl = calc_DEBresults(C[t:t+2], timextr[t:t+2]-timextr[t], y0, DEBpars, moa, feedb, shifted_tvec, solver=self.solver)
                # print("Model solution for segment ", t, ": ", modelsl)
                y0 = modelsl[:, -1]  # update initial condition for next segment
                modelsol_union[:, (newtime >= timextr[t]) & (newtime <= timextr[t+1])] = modelsl
        else:
            # Solve once on the union grid
            modelsol_union = calc_DEBresults(C, timextr, y0, DEBpars, moa, feedb, newtime, solver=self.solver)
            # modelsol_union: shape (n_states, len(newtime))

        # We will rebuild the final solution strictly on the original timeext grid
        # Build indexes that map timeext → newtime and tbp → newtime
        # Because arrays are sorted, use searchsorted then verify exact equality to guard against FP mismatch
        idx_timeext_in_union = np.searchsorted(newtime, timeext)
        match_timeext = (newtime[idx_timeext_in_union] == timeext)
        if not np.all(match_timeext):
            # In case of floating-point differences, fall back to robust approach
            # (Still O(n log n), but safe)
            idx_timeext_in_union = np.array([np.where(newtime == t)[0][0] for t in timeext])
        idx_tbp_in_union = np.searchsorted(newtime, tbp)
        match_tbp = (newtime[idx_tbp_in_union] == tbp)
        if not np.all(match_tbp):
            idx_tbp_in_union = np.array([np.where(newtime == t)[0][0] for t in tbp])
        # Extract the delayed state from union grid at tbp positions
        REPRO_STATE_IDX = 2  # replace with a named constant or parameter if available
        delayed_values = modelsol_union[REPRO_STATE_IDX, idx_tbp_in_union]  # shape (len(tbp),)
        # Build the final model solution aligned to original timeext
        modelsol = modelsol_union[:, idx_timeext_in_union]  # shape (n_states, len(timeext))
        # Zero out reproduction before applying delayed contribution (as per your logic)
        # If you need additive behavior, change this to additive instead of overwrite.
        modelsol[REPRO_STATE_IDX, :] = 0.0
        # Now place delayed values at indices corresponding to (tbp + Tbp) on timeext
        # Because timeext is sorted, we can locate these quickly
        target_times = tbp + Tbp
        idx_targets = np.searchsorted(timeext, target_times)
        match_targets = (timeext[idx_targets] == target_times)
        if not np.any(match_targets):
            # No exact matches → likely due to floating-point spacing.
            # If acceptable, we can snap to nearest indices within a small tolerance.
            # Otherwise, skip with a warning.
            # Here, we use a tolerance approach:
            tol = np.finfo(float).eps * 10  # small tolerance
            # Compute closest indices by absolute difference
            # (This is O(n*m) if done naively; be cautious if arrays are huge.)
            # Optimized nearest neighbor via searchsorted with bounds check:
            idx_targets = np.clip(np.searchsorted(timeext, target_times), 0, len(timeext)-1)
            close_enough = np.abs(timeext[idx_targets] - target_times) <= tol
            # Assign only where close enough
            if np.any(close_enough):
                modelsol[REPRO_STATE_IDX, idx_targets[close_enough]] = delayed_values[close_enough]
        else:
            # Assign only for exact matches
            modelsol[REPRO_STATE_IDX, idx_targets[match_targets]] = delayed_values[match_targets]
        return(modelsol)

    # def worker_DEBresults(self,pars,parvals,posfree,concarray_i,
    #                       time,islog,moa,feedb,tevals):
    #     par95 = np.copy(parvals)
    #     par95[posfree] = pars
    #     transformed = np.where(islog, 10**par95, par95)
    #     #transformed = 10**(par95)*islog + par95*(~islog)
    #     return(self.calc_model(concarray_i,time,transformed,moa,feedb,tevals).T)

    def _calc_modelvalues(self):
        # calculate the model points at exactly the time
        # points of the experimental data
        basepars = self.parvals.copy()
        basepars[self.islog] = 10 ** basepars[self.islog]
        modelsolcontainer = [None]*self.ndatasets
        for nd in range(self.ndatasets):  # iterate over datasets
            modelpars = self.build_dataset_parameters(basepars, nd)
            fullmodelvector1 = np.array([])
            fullmodelvector2 = np.array([])
            fulllengthvector = np.array([])
            fullreprovector = np.array([])
            fullweightslengthvector = np.array([])
            fullweightsreprovector = np.array([])
            newtime = self.timeext[nd]
            modelsoltreatlevel = np.full((4,len(newtime)),np.nan) 
            # FIX this part for the brood pouch delay.
            # CHECK if anything can be pre-computed
            # tbp = 0 # make sure it is declared here
            newtimeext = np.unique(np.concatenate((np.linspace(newtime[0],newtime[-1],max(self.min_t,len(newtime))),newtime)))
            # print("newtimeext: ", newtimeext)
            for i in range(self.concstruct_list[nd].ntreats):  # iterate over treatments within the dataset
                try:
                    modelsol = self.calc_model(self.concstruct_list[nd].concarraytr[i], self.concstruct_list[nd].timetr,
                                               modelpars, self.moa, self.feedb,
                                               newtimeext)
                except:
                    # there was a problem with the ODE solver
                    return(np.inf)
                
                idx_targets = np.searchsorted(newtimeext, self.timeext[nd])
                # match_targets = (self.timeext[nd][idx_targets] == target_times)
                
                # mask = np.isin(newtimeext,self.timeext[nd])
                # indices = np.nonzero(mask)[0]
                # # print("indices", indices)
                #modelsol = modelsol[:,idx_targets]
                # modelsoltreatlevel = modelsol[:,idx_targets]
                # modelsolcontainer[nd] = modelsoltreatlevel
                # print("modelsol before substitution: ")
                # print(modelsol[2,:])

                for endpoint in self.active_endpoints[nd]:
                    if endpoint == 0:
                        # llsurv = survival_loglikelihood(modelsol[3, :], self.indexcommon_surv[nd][i],
                        #                                 self.survstruct_list[nd].deatharraytreat[i])
                        # # print("llsurv treatment ", i)
                        # # print(llsurv)
                        # llik += llsurv
                        modelsoltreatlevel[:, self.indexcommon_surv[nd][i]] = modelsol[:, self.indexcommon_surv[nd][i]] 
                    elif (endpoint == 1):  # length
                        lengthtreat = self.lengthstruct_list[nd].flatdataclean[i]
                        weights = self.lengthstruct_list[nd].flatweightsclean[i]
                        commontime =  np.array([self.indexcommon_length[nd][j] for j in range(len(self.indexcommon_length[nd])) if self.lengthstruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
                        modelvector = np.tile(modelsol[1, :][commontime[0]],len(commontime))[self.lengthstruct_list[nd].indfintable[i]]

                        fullmodelvector1 = np.concatenate((fullmodelvector1, modelvector))
                        fulllengthvector = np.concatenate((fulllengthvector, lengthtreat))
                        fullweightslengthvector = np.concatenate((fullweightslengthvector, weights))
                    elif (endpoint == 2):  # reproduction
                        reprotreat = self.reprostruct_list[nd].flatdataclean[i]
                        weights = self.reprostruct_list[nd].flatweightsclean[i]
                        commontime =  np.array([self.indexcommon_repro[nd][j] for j in range(len(self.indexcommon_repro[nd])) if self.reprostruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
                        modelvector = np.tile(modelsol[2, :][commontime[0]],len(commontime))[self.reprostruct_list[nd].indfintable[i]]
                        fullmodelvector2 = np.concatenate((fullmodelvector2, modelvector))
                        fullreprovector = np.concatenate((fullreprovector, reprotreat))
                        fullweightsreprovector = np.concatenate((fullweightsreprovector, weights))
            if self.lengthstruct_list[nd] is not None:
                transf = self.lengthstruct_list[nd].statstype
                lllength = scaled_loglikelihood(fullmodelvector1, fulllengthvector, fullweightslengthvector, transf)
                # print("lllength treatment ", i)
                # print(lllength)
                llik += lllength
            if self.reprostruct_list[nd] is not None:
                transf = self.reprostruct_list[nd].statstype
                llrepro = scaled_loglikelihood(fullmodelvector2, fullreprovector, fullweightsreprovector, transf)
                # print("llrepro treatment ", i)
                # print(llrepro)
                llik += llrepro
        # print("Total llk: ", -llik)
        return(modelsolcontainer)

    
    def worker_DEBresults(
        self,
        pars,
        parvals,
        posfree,
        concarray,
        time,
        islog,
        moa,
        feedb,
        tevals,
        nd):
        """
        Worker function for CI propagation.
        """

        # 1. Rebuild expanded parameter vector
        expanded = np.array(parvals, copy=True)
        expanded[posfree] = pars

        # 2. Apply log-transform in expanded space
        expanded = np.where(islog, 10 ** expanded, expanded)

        # 3. Collapse to dataset-specific solver parameters
        solver_pars = self.build_dataset_parameters(expanded, nd)

        # 4. Run model
        return self.calc_model(
            concarray,
            time,
            solver_pars,
            moa,
            feedb,
            tevals
        ).T


    # def calc_model(self, C, timextr, DEBpars, moa, feedb, timeext):
    #     # somewhere here need to implement also the Tbp part
    #     if self.Tbp > 0:
    #         # apply a time delay for the reproduction
    #         # tbp = timeext + self.Tbp
    #         tbp = timeext[timeext>self.Tbp] - self.Tbp
    #         newtime = np.unique(np.concatenate((timeext,tbp)))
    #         modelsol = calc_DEBresults(C, timextr, DEBpars, moa, feedb, newtime, solver=self.solver)
    #         mask = np.isin(newtime,tbp)
    #         indices = np.nonzero(mask)[0]
    #         modelsol_tbp = np.copy(modelsol[2,indices])
    #         # print("model_tbp before delay: ", modelsol_tbp)
    #         modelsol[2,:] = 0
    #         mask2 = np.isin(newtime,timeext)
    #         indices2 = np.nonzero(mask2)[0]
    #         modelsol = modelsol[:,indices2]
    #         mask3 = np.isin(timeext,tbp+self.Tbp)
    #         indices3 = np.nonzero(mask3)[0]
    #         # print("indices with Tbp: ", len(indices3))
    #         # print("modelsol_tbp at those indices: ", len(modelsol_tbp))
    #         modelsol[2,indices3] = modelsol_tbp
    #     else:
    #         modelsol = calc_DEBresults(C, timextr, DEBpars, moa, feedb, timeext, solver=self.solver)
    #     return(modelsol)

    # def log_likelihood_wrong(self, theta, DEBallpars, posfree):
    #     # fallback function that separates the likelihood calculation for each
    #     # treatment and summs the contriubtions
    #     # The reason this is wrong is that the treatments might not be
    #     # independent and therefore the loglikelihood cannot be summed
    #     '''
    #     Calculate the log-likelihood of the GUTS model.

    #     Arguments:
    #     - theta: vector of free parameter values
    #     - DEBallpars: vector of all parameter values
    #     - posfree: indices of the free parameters in the parameter vector
    #     '''
    #     DEBallpars[posfree] = theta
    #     # TODO: make sure that for each dataset the respective hb value is correctly passed
    #     modelpars = 10**DEBallpars*self.islog + DEBallpars*(1-self.islog)
    #     llik = 0
    #     for nd in range(self.ndatasets):  # iterate over datasets
    #         for i in range(self.concstruct_list[nd].ntreats):  # iterate over treatments within the dataset
    #             try:
    #                 modelsol = self.calc_model(self.concstruct_list[nd].concarray[i], self.concstruct_list[nd].time,
    #                        modelpars, self.moa, self.feedb,
    #                        self.timeext[nd])
    #             except:
    #                 # there was a problem with the ODE solver
    #                 return(np.inf)
    #             for endpoint in self.active_endpoints[nd]:
    #                 if endpoint == 0:
    #                     llsurv = survival_loglikelihood(modelsol[3, :], self.indexcommon_surv[nd][i],
    #                                                     self.survstruct_list[nd].deatharraytreat[i])
    #                     print("llsurv treatment ", i)
    #                     print(llsurv)
    #                     llik += llsurv
    #                 elif (endpoint == 1):  # length
    #                     lengthtreat = self.lengthstruct_list[nd].flatdataclean[i]
    #                     weights = self.lengthstruct_list[nd].flatweightsclean[i]
    #                     commontime =  np.array([self.indexcommon_length[nd][j] for j in range(len(self.indexcommon_length[nd])) if self.lengthstruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
    #                     modelvector = np.tile(modelsol[1, :][commontime[0]],len(commontime))[self.lengthstruct_list[nd].indfintable[i]]
    #                     transf = self.lengthstruct_list[nd].statstype
    #                     lllength = scaled_loglikelihood(modelvector, lengthtreat, weights, transf)
    #                     print("lllength treatment ", i)
    #                     print(lllength)
    #                     llik += lllength
    #                 elif (endpoint == 2):  # reproduction
    #                     reprotreat = self.reprostruct_list[nd].flatdataclean[i]
    #                     weights = self.reprostruct_list[nd].flatweightsclean[i]
    #                     commontime =  np.array([self.indexcommon_repro[nd][j] for j in range(len(self.indexcommon_repro[nd])) if self.reprostruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
    #                     modelvector = np.tile(modelsol[2, :][commontime[0]],len(commontime))[self.reprostruct_list[nd].indfintable[i]]
    #                     transf = self.reprostruct_list[nd].statstype
    #                     llrepro = scaled_loglikelihood(modelvector, reprotreat, weights, transf)
    #                     print("llrepro treatment ", i)
    #                     print(llrepro)
    #                     llik += llrepro
    #     print("Total llk: ", -llik)
    #     return(-llik)
    
    # new version with all the treatments together
    def log_likelihood(self, theta, DEBallpars, posfree):
        '''
        Calculate the log-likelihood of the GUTS model.

        Arguments:
        - theta: vector of free parameter values
        - DEBallpars: vector of all parameter values
        - posfree: indices of the free parameters in the parameter vector
        '''
        DEBallpars[posfree] = theta
        # TODO: make sure that for each dataset the respective hb value is correctly passed
        # TODO: modify in the future remove explicit naming of the endpoints
        #       and make it more general, so that arbitrary endpoints can be handled
        #       without knowing them in advance 
        
        basepars = DEBallpars.copy()
        basepars[self.islog] = 10 ** basepars[self.islog]

        llik = 0
        for nd in range(self.ndatasets):  # iterate over datasets
            modelpars = self.build_dataset_parameters(basepars, nd)
            fullmodelvector1 = np.array([])
            fullmodelvector2 = np.array([])
            fulllengthvector = np.array([])
            fullreprovector = np.array([])
            fullweightslengthvector = np.array([])
            fullweightsreprovector = np.array([])
            newtime = self.timeext[nd]
            # FIX this part for the brood pouch delay.
            # CHECK if anything can be pre-computed
            # tbp = 0 # make sure it is declared here
            newtimeext = np.unique(np.concatenate((np.linspace(newtime[0],newtime[-1],max(self.min_t,len(newtime))),newtime)))
            # print("newtimeext: ", newtimeext)
            for i in range(self.concstruct_list[nd].ntreats):  # iterate over treatments within the dataset
                try:
                    modelsol = self.calc_model(self.concstruct_list[nd].concarraytr[i], self.concstruct_list[nd].timetr,
                                               modelpars, self.moa, self.feedb,
                                               newtimeext)
                except:
                    # there was a problem with the ODE solver
                    return(np.inf)
                
                idx_targets = np.searchsorted(newtimeext, self.timeext[nd])
                # match_targets = (self.timeext[nd][idx_targets] == target_times)
                
                # mask = np.isin(newtimeext,self.timeext[nd])
                # indices = np.nonzero(mask)[0]
                # # print("indices", indices)
                modelsol = modelsol[:,idx_targets]
                # print("modelsol before substitution: ")
                # print(modelsol[2,:])

                for endpoint in self.active_endpoints[nd]:
                    if endpoint == 0:
                        llsurv = survival_loglikelihood(modelsol[3, :], self.indexcommon_surv[nd][i],
                                                        self.survstruct_list[nd].deatharraytreat[i])
                        # print("llsurv treatment ", i)
                        # print(llsurv)
                        llik += llsurv
                    elif (endpoint == 1):  # length
                        lengthtreat = self.lengthstruct_list[nd].flatdataclean[i]
                        weights = self.lengthstruct_list[nd].flatweightsclean[i]
                        commontime =  np.array([self.indexcommon_length[nd][j] for j in range(len(self.indexcommon_length[nd])) if self.lengthstruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
                        modelvector = np.tile(modelsol[1, :][commontime[0]],len(commontime))[self.lengthstruct_list[nd].indfintable[i]]
                        fullmodelvector1 = np.concatenate((fullmodelvector1, modelvector))
                        fulllengthvector = np.concatenate((fulllengthvector, lengthtreat))
                        fullweightslengthvector = np.concatenate((fullweightslengthvector, weights))
                    elif (endpoint == 2):  # reproduction
                        reprotreat = self.reprostruct_list[nd].flatdataclean[i]
                        weights = self.reprostruct_list[nd].flatweightsclean[i]
                        commontime =  np.array([self.indexcommon_repro[nd][j] for j in range(len(self.indexcommon_repro[nd])) if self.reprostruct_list[nd].treatmentsnames[j] == self.concstruct_list[nd].conctreatsnames[i]])
                        modelvector = np.tile(modelsol[2, :][commontime[0]],len(commontime))[self.reprostruct_list[nd].indfintable[i]]
                        fullmodelvector2 = np.concatenate((fullmodelvector2, modelvector))
                        fullreprovector = np.concatenate((fullreprovector, reprotreat))
                        fullweightsreprovector = np.concatenate((fullweightsreprovector, weights))
            if self.lengthstruct_list[nd] is not None:
                transf = self.lengthstruct_list[nd].statstype
                lllength = scaled_loglikelihood(fullmodelvector1, fulllengthvector, fullweightslengthvector, transf)
                # print("lllength treatment ", i)
                # print(lllength)
                llik += lllength
            if self.reprostruct_list[nd] is not None:
                transf = self.reprostruct_list[nd].statstype
                llrepro = scaled_loglikelihood(fullmodelvector2, fullreprovector, fullweightsreprovector, transf)
                # print("llrepro treatment ", i)
                # print(llrepro)
                llik += llrepro
        # print("Total llk: ", -llik)
        return(-llik)
