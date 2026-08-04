import numpy as np
from dataclasses import dataclass
from numba import jit
from scipy.integrate import odeint,solve_ivp
from scipy.optimize import brentq
import multiprocessing as mp
import psutil
n_cores = psutil.cpu_count(logical=False) # to have the number of physical cores only


@dataclass(frozen=True)
class EndpointSpec:
    """
    Everything the rest of the package needs to know about one observable
    endpoint, in one place, so it doesn't have to be re-declared (and kept
    in sync by hand) everywhere an endpoint is referenced by code.

    Attributes:
    - code: the integer endpoint code used throughout the package (in
      active_endpoints, calc_ecx/calc_epx's `endpoints=` argument, etc.)
    - name: display/lookup name, e.g. 'length'
    - state_idx: row in the array returned by calc_model that this
      endpoint reads. NOTE this is *not* the same numbering as `code` -
      e.g. survival is code 0 but state row 3 - which is exactly why this
      needs to be an explicit, named mapping rather than an assumption.
    - dataset_attr: attribute name on a `completedataset` instance that
      signals this endpoint is present (e.g. 'lengthdata'); also the key
      used in `completedataset.time_indices` for this endpoint.
    - struct_list_attr: attribute name on `DEBtox2019models` holding the
      per-dataset list of this endpoint's data structure (e.g.
      'lengthstruct_list').
    - indexcommon_attr: attribute name on `DEBtox2019models` holding the
      per-dataset list of this endpoint's common-time indices (e.g.
      'indexcommon_length').
    - is_survival: True only for the survival endpoint - used where
      survival needs different handling from the other (continuous)
      endpoints, e.g. LCx/LPx vs ECx/EPx labeling.
    """
    code: int
    name: str
    state_idx: int
    dataset_attr: str
    struct_list_attr: str
    indexcommon_attr: str
    is_survival: bool = False


# Declared in the same order the pre-registry code used to append to
# active_endpoints (length, reproduction, survival) so that ordering -
# e.g. the order endpoints are summed in log_likelihood, or printed in
# efsa_criteria - is unchanged by this being a loop over a dict now.
ENDPOINTS = {
    1: EndpointSpec(1, 'length', state_idx=1, dataset_attr='lengthdata',
                     struct_list_attr='lengthstruct_list', indexcommon_attr='indexcommon_length'),
    2: EndpointSpec(2, 'reproduction', state_idx=2, dataset_attr='reprodata',
                     struct_list_attr='reprostruct_list', indexcommon_attr='indexcommon_repro'),
    0: EndpointSpec(0, 'survival', state_idx=3, dataset_attr='survdata',
                     struct_list_attr='survstruct_list', indexcommon_attr='indexcommon_surv',
                     is_survival=True),
}
NAME_TO_CODE = {spec.name: code for code, spec in ENDPOINTS.items()}


class DEBSolverError(Exception):
    """
    Raised by calc_DEBresults when the ODE solver (solve_ivp/odeint) fails
    to produce a usable trajectory for the requested time grid - e.g. it
    did not converge, bailed out early leaving a truncated result, or
    produced non-finite values.

    This is deliberately the *only* failure this module signals as "the
    model could not be evaluated for this parameter point" (callers such
    as log_likelihood catch specifically this type and return an infinite
    penalty). Any other exception raised while solving indicates a real
    bug (bad arguments, a shape mismatch, a typo'd solver name, ...) and
    is left to propagate normally instead of being silently swallowed.
    """
    pass


@jit(nopython=True, cache=True)
def _DEBtox2019_derivatives_core(y, t, C, timextr, DEBpars, moa, feedb):
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


@jit(nopython=True, cache=True)
def DEBtox2019_derivatives_odeint(y, t, C, timextr, DEBpars, moa, feedb):
    # odeint calls its right-hand side as func(y, t, *args)
    return _DEBtox2019_derivatives_core(y, t, C, timextr, DEBpars, moa, feedb)


@jit(nopython=True, cache=True)
def DEBtox2019_derivatives_solveivp(t, y, C, timextr, DEBpars, moa, feedb):
    # solve_ivp calls its right-hand side as fun(t, y, *args) - the only
    # difference from odeint's convention above, hence the thin wrapper
    # instead of a second copy of the (identical) model physics.
    return _DEBtox2019_derivatives_core(y, t, C, timextr, DEBpars, moa, feedb)



def calc_DEBresults(C, timextr, y0, DEBpars, moa, feedb,timeext,solver='RK45'):
    # L0 = DEBpars[5]
    # y0 = np.array([0., L0*1.0, 0., 1.])
    # TODO: test if we need a very extended time vector or it will be enough to use the union of
    #       the time points of the experimental data (for speed reasons, given that rk45 should not care)
    #       in byom it seems to be enough to take the intersection of all the points in the datasets.
    # A non-finite input (e.g. a NaN rate constant from a bad optimizer
    # step) is not just an "eventually fails" case: a NaN right-hand side
    # can make solve_ivp's adaptive step-size control spin forever (a NaN
    # error estimate never satisfies its "step accepted" check, and never
    # triggers its "step too small, give up" check either, since any
    # comparison against NaN is False). Reject non-finite inputs up front,
    # before ever calling the solver, rather than relying on the
    # (occasionally non-terminating) solver to fail on its own.
    if (not np.all(np.isfinite(DEBpars))) or (not np.all(np.isfinite(y0))):
        raise DEBSolverError("calc_DEBresults received non-finite DEBpars or y0")
    if solver=="RK45":
        # solve_ivp has the possibility to choose a different algorithm (default is RK45)
        sol = solve_ivp(fun=DEBtox2019_derivatives_solveivp, t_span=np.array([timeext[0], timeext[-1]]), y0=y0,
                    args=(C, timextr, DEBpars, moa, feedb), t_eval=timeext, rtol=1e-9,atol=1e-9)#, dense_output=True)
        # solve_ivp does NOT raise on a failed integration: on failure it
        # just sets sol.success=False and often returns a truncated sol.y
        # (fewer columns than len(timeext)). Detect that explicitly here
        # instead of letting some downstream indexing operation stumble
        # into it by accident.
        if (not sol.success) or (sol.y.shape[1] != len(timeext)) or (not np.all(np.isfinite(sol.y))):
            raise DEBSolverError(
                "solve_ivp failed to integrate the DEB model over the requested "
                "time grid (message: %s)" % sol.message
            )
        return(sol.y)
    elif solver=="LSODA":
    # odeint uses LSODA algorithm
        sol = odeint(DEBtox2019_derivatives_odeint, y0, timeext, args=(C, timextr, DEBpars, moa, feedb), rtol=1e-9, atol=1e-9)
        # odeint also does not raise on a failed integration: it prints an
        # ODEPACK warning to stderr and still returns an array of the
        # expected shape, just filled with non-finite/garbage values past
        # the point of failure. Check explicitly for that too.
        if (sol.shape != (len(timeext), len(y0))) or (not np.all(np.isfinite(sol))):
            raise DEBSolverError(
                "odeint failed to integrate the DEB model over the requested "
                "time grid (non-finite or malformed result)"
            )
        return(sol.T)
    else:
        raise ValueError("Solver not recognized. Use 'RK45' or 'LSODA'.")


@jit(nopython=True, cache=True)
def survival_loglikelihood(modelvector, commontime, deathvector):
    # print(commontime)
    surviv_selected = modelvector[commontime]
    #print(surviv_selected)
    pdeath = np.append(-np.diff(surviv_selected),surviv_selected[-1])
    pdeath = np.maximum(pdeath,1e-50)
    #print(deathvector)
    llik=np.dot(deathvector,np.log(pdeath))
    return(llik)


@jit(nopython=True, cache=True)
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


class DEBtox2019models:
    '''
    Container for one DEBtox2019 physiological + toxicokinetic-toxicodynamic
    (TK-TD) model, set up against one or more experimental datasets. Provides
    the ODE-solving, log-likelihood, and ECx/EPx machinery used by the
    parameter-space explorer (parspace.PyParspace) and by the higher-level
    helpers in debtox2019api.py.

    Each dataset may contribute up to three observable endpoints - survival,
    body length, and reproduction - alongside its exposure (concentration)
    data. Which endpoints are present for a given dataset, and where their
    data lives, is determined at construction time from the
    `completedataset` instances passed in (see `active_endpoints` and the
    `ENDPOINTS` registry near the top of this module).

    Attributes:
    - debparameterclass: the DEBparameters instance the model was built
      from (kept around for later reuse, e.g. by validation()).
    - ndatasets: number of datasets.
    - par_dataset_map: per-(expanded)-parameter ownership, from
      debparameterclass.par_dataset_map (-1 for a shared parameter, or the
      list of dataset indices a grouped/dataset-specific value belongs
      to); used by build_dataset_parameters to resolve which value
      applies to a given dataset.
    - full_base_names: per-(expanded)-parameter base name (group/dataset
      suffixes stripped), from debparameterclass.full_base_names.
    - parnames: full expanded parameter names (object ndarray).
    - parvals: full expanded parameter values. Log-transformed parameters
      (see `islog`) are stored in log10-space.
    - islog: bool ndarray, per parameter, whether it is log-transformed.
    - isfree: bool ndarray, per parameter, whether it is free (vs fixed).
    - posfree: int ndarray, indices of the free parameters into the
      expanded arrays above.
    - parbound_lower / parbound_upper: per-parameter bounds (log10-space
      for log-transformed parameters).
    - moa: length-5 mode-of-action array (assimilation, maintenance,
      growth, reproduction, hazard).
    - feedb: length-4 feedback array (see _DEBtox2019_derivatives_core).
    - Tbp: brood-pouch/egg-development delay applied to reproduction (0 to
      disable).
    - min_t: minimum number of points for the dense internal solver time
      grid built per dataset (see `newtimeext`).
    - solver: 'RK45' (solve_ivp) or 'LSODA' (odeint).
    - breaktime: if True, solve the ODE separately on each segment between
      concentration time points - needed for some renewal/pulsed exposure
      designs - instead of over the whole time span at once.
    - timeext: list (one per dataset) of the union of that dataset's
      active endpoints' observation times.
    - newtimeext: list (one per dataset) of a denser time grid (built
      from `min_t`), used internally when solving and then subsampled
      back onto `timeext` - precomputed once here rather than on every
      log_likelihood call.
    - concstruct_list: list (one per dataset) of that dataset's
      `concclass` exposure data.
    - lengthstruct_list / reprostruct_list / survstruct_list: list (one
      per dataset) of that dataset's `lengthdataclass` / `reproclass` /
      `survdataclass` instance, or None where that endpoint isn't present.
    - active_endpoints: list (one per dataset) of the endpoint codes
      present for that dataset (see the ENDPOINTS registry).
    - indexcommon_length / indexcommon_repro / indexcommon_surv: list
      (one per dataset) of the indices locating that endpoint's own
      observation times within `timeext[nd]`.
    - endpoints: currently unused (set once, never read elsewhere in the
      package) - likely a leftover from an earlier version; kept as-is
      pending cleanup.

    Selected methods (see individual docstrings for details):
    - build_dataset_parameters: collapse the expanded parameter vector
      into the compact, dataset-specific vector calc_model expects.
    - calc_model: run the ODE model for one dataset/treatment, including
      the optional brood-pouch delay (Tbp) and breaktime handling.
    - log_likelihood: the (negative) log-likelihood used by the
      optimizer, combining survival, length and reproduction
      contributions across all datasets.
    - calc_ecx_core / calc_epx_core: numerical ECx/LCx and EPx/LPx
      calculation (wrapped by debtox2019api.calc_ecx / calc_epx).
    - worker_ecx / worker_epx / worker_DEBresults: picklable worker
      functions used for multiprocessing-based confidence-interval
      propagation.
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
        Build a DEBtox2019models instance for one or more datasets.

        Arguments:
        - completedataset_list: list of `completedataset` instances (one
          per dataset), each holding that dataset's exposure data and
          whichever of survival/length/reproduction data it has.
        - debparameterclass: a `DEBparameters` instance describing the
          (expanded) parameter vector - values, bounds, log/free flags,
          and the shared/grouped/dataset-specific ownership of each
          parameter - to be used across all datasets in
          completedataset_list.
        - moa: length-5 mode-of-action array (assimilation, maintenance,
          growth, reproduction, hazard).
        - feedb: length-4 feedback array (see
          _DEBtox2019_derivatives_core).
        - Tbp: brood-pouch/egg-development delay applied to reproduction
          (0 to disable). Default 0.
        - min_t: minimum number of points for the dense internal solver
          time grid built per dataset (see the `newtimeext` attribute).
          Default 500.
        - solver: 'RK45' (solve_ivp) or 'LSODA' (odeint). Default
          'LSODA'.
        - breaktime: if True, solve the ODE separately on each segment
          between concentration time points, instead of over the whole
          time span at once (needed for some renewal/pulsed exposure
          designs). Default False.
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
        self.newtimeext = []

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
            for code, spec in ENDPOINTS.items():
                if hasattr(completedataset_list[i], spec.dataset_attr):
                    getattr(self, spec.struct_list_attr)[i] = getattr(completedataset_list[i], spec.dataset_attr)
                    getattr(self, spec.indexcommon_attr)[i] = completedataset_list[i].time_indices[spec.dataset_attr]
                    self.active_endpoints[i].append(code)
            # this is because the more stuff are precalculated before likelihood evaluation, the better
            # makes the code faster
            newtime = self.timeext[i]
            self.newtimeext.append(np.unique(np.concatenate((np.linspace(newtime[0],newtime[-1],max(self.min_t,len(newtime))),newtime))))
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
        REPRO_STATE_IDX = ENDPOINTS[NAME_TO_CODE['reproduction']].state_idx
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
        if not np.all(match_targets):
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

    # State index (row in the array returned by calc_model) that each endpoint
    # code refers to. Derived from the single ENDPOINTS registry (see above)
    # rather than declared again here, so this and the registry can't drift.
    ENDPOINT_STATE_IDX = {code: spec.state_idx for code, spec in ENDPOINTS.items()}

    def _bisect_log(self, f, low, high, max_expand=60, xtol=1e-8, increasing=False,
                     plateau_tol=1e-6):
        """
        Find the value (returned in linear scale) at which `f` (a function
        of log10(value)) crosses zero, expanding the (low, high) bracket
        geometrically (in log space) until a sign change is found or
        `max_expand` expansions have been tried.

        `increasing` indicates the expected sign of df/d(value): False for
        a decreasing f (e.g. response falling with concentration, as in
        calc_ecx_core), True for an increasing f (e.g. effect growing with
        an exposure multiplication factor, as in calc_epx_core). This only
        affects which side of the bracket gets expanded first; brentq
        itself does not care about the direction.

        `plateau_tol`: some endpoints can plateau well before reaching a
        requested effect level - e.g. body length has a hard floor in the
        model (growth is not allowed to shrink below half the starting
        length), so at high enough concentration the effect on length stops
        increasing at all, and no concentration will ever reach, say, 99%
        effect. Without a check, each expansion step keeps multiplying the
        bracket by 10x anyway, burning through the full `max_expand` budget
        before giving up. Instead, after each expansion step, if `f` has
        changed by less than `plateau_tol` from the previous step (and no
        sign change has been found), the search concludes the underlying
        response has plateaued and stops early, returning NaN. Set to None
        to disable this check and always use the full `max_expand` budget.

        Returns np.nan if no bracket could be found or the search fails.
        """
        loglow, loghigh = np.log10(low), np.log10(high)
        flow, fhigh = f(loglow), f(loghigh)
        if abs(flow) < 1e-12:
            return 10 ** loglow
        if abs(fhigh) < 1e-12:
            return 10 ** loghigh
        expand = 0
        while flow * fhigh > 0 and expand < max_expand:
            need_higher = (fhigh < 0) if increasing else (fhigh > 0)
            if need_higher:
                loghigh += 1.0
                f_prev, fhigh = fhigh, f(loghigh)
                if plateau_tol is not None and abs(fhigh - f_prev) < plateau_tol:
                    return np.nan
            else:
                loglow -= 1.0
                f_prev, flow = flow, f(loglow)
                if plateau_tol is not None and abs(flow - f_prev) < plateau_tol:
                    return np.nan
            expand += 1
        if not (np.isfinite(flow) and np.isfinite(fhigh)) or flow * fhigh > 0:
            return np.nan
        try:
            root = brentq(f, loglow, loghigh, xtol=xtol)
        except (ValueError, RuntimeError):
            return np.nan
        return 10 ** root

    def calc_ecx_core(self, modelpars, Tend, X=(10, 50), endpoints=(0, 1, 2),
                       conc_bounds=(1e-6, 1e6), max_expand=60, xtol=1e-8,
                       plateau_tol=1e-6):
        """
        Core (numerical) ECx/LCx calculation for a single, fixed
        dataset-specific parameter vector (as produced by
        build_dataset_parameters). This is the pyDEBtox2019 equivalent of
        calc_ecx.m from the DEBtox2019/BYOM toolbox: for every evaluation
        time in `Tend` and every effect level in `X`, finds - by bisection
        over a constant exposure concentration applied from t=0 to that
        time - the concentration producing an X% effect relative to the
        untreated (C=0) control, using the same parameter set.

        Survival (endpoint 0) is treated as LCx (percentage additional
        mortality); length and reproduction (endpoints 1, 2) are treated as
        ECx (percentage reduction relative to the control).

        Arguments:
        - modelpars: dataset-specific DEB parameter vector (linear scale).
        - Tend: array-like of evaluation times.
        - X: iterable of effect levels in percent (0 <= x < 100).
        - endpoints: iterable of endpoint codes (0=survival, 1=length, 2=reproduction).
        - conc_bounds: initial (low, high) concentration bracket for the
          bisection search; automatically expanded if needed.
        - max_expand, xtol: passed to the bracket search / brentq.
        - plateau_tol: passed to the bracket search (see _bisect_log);
          stops expanding early (returning NaN) once the response has
          visibly plateaued (e.g. length hitting its shrink floor) rather
          than always exhausting max_expand. Set to None to disable.

        Returns:
        - dict: results[endpoint][x] -> np.ndarray aligned with Tend.
        """
        Tend_arr = np.atleast_1d(np.asarray(Tend, dtype=float))
        clow0, chigh0 = conc_bounds

        def response(C, t):
            Cprofile = np.array([C, C])
            tprofile = np.array([0.0, t])
            # NOTE: timeext must NOT be a bare 2-point [0, t] array here. With
            # solver='LSODA' (odeint/LSODA), the mxstep budget (default 500)
            # applies PER INTERVAL between consecutive requested time points,
            # not to the whole integration. A 2-point request forces the
            # entire (potentially stiff, high-concentration) integration into
            # a single 500-step budget; if exceeded, odeint silently returns
            # a corrupted result with only a non-fatal warning. A denser grid
            # gives each sub-interval its own step budget and avoids this.
            sol = self.calc_model(Cprofile, tprofile, modelpars, self.moa, self.feedb,
                                   timeext=np.linspace(0.0, t, 50))
            return sol[:, -1]

        results = {ep: {x: np.full(Tend_arr.shape, np.nan) for x in X} for ep in endpoints}

        for it, t in enumerate(Tend_arr):
            if t <= 0:
                continue
            control = response(0.0, t)
            for ep in endpoints:
                si = self.ENDPOINT_STATE_IDX[ep]
                target_base = control[si]
                for x in X:
                    target = target_base * (1.0 - x / 100.0)

                    def f(logC, si=si, target=target, t=t):
                        return response(10 ** logC, t)[si] - target

                    results[ep][x][it] = self._bisect_log(f, clow0, chigh0, max_expand, xtol,
                                                           increasing=False,
                                                           plateau_tol=plateau_tol)
        return results

    def worker_ecx(self, pars, parvals, posfree, islog, dataset, Tend, X, endpoints,
                    conc_bounds, max_expand, xtol, plateau_tol=1e-6):
        """
        Worker function for ECx/LCx CI propagation: rebuilds a dataset-specific
        parameter vector from a free-parameter sample and runs calc_ecx_core.
        """
        expanded = np.array(parvals, copy=True)
        expanded[posfree] = pars
        expanded = np.where(islog, 10 ** expanded, expanded)
        modelpars = self.build_dataset_parameters(expanded, dataset)
        return self.calc_ecx_core(modelpars, Tend, X, endpoints, conc_bounds, max_expand, xtol,
                                   plateau_tol=plateau_tol)

    @staticmethod
    def _window_profile(exposure_time, exposure_conc, t_start, Twin):
        """
        Build a (timextr, C) pair (as consumed by calc_model) representing
        the exposure profile as seen through a window of length `Twin`
        starting at `t_start`, shifted so that the window start maps to
        t=0. Whenever the window extends before the first recorded profile
        time point, or beyond the last one, the corresponding head/tail of
        the window is zero-padded (no exposure is assumed to have occurred
        before the profile starts, nor after it ends) - each transition is
        represented as a two-point step (flat zero, then an instantaneous
        jump to/from the profile's actual first/last value), rather than a
        gradual ramp.
        """
        prof_start = exposure_time[0]
        prof_end = exposure_time[-1]
        t_end = t_start + Twin

        if t_end <= prof_start or t_start >= prof_end:
            # window entirely outside the profile's time range: no exposure at all
            return np.array([0.0, Twin]), np.array([0.0, 0.0])

        lo = max(t_start, prof_start)
        hi = min(t_end, prof_end)
        mask = (exposure_time > lo) & (exposure_time < hi)
        inner_t = (exposure_time[mask] - t_start).tolist()
        inner_c = exposure_conc[mask].tolist()

        t_list = [0.0]
        c_list = [0.0]
        if t_start < prof_start:
            # exposure is zero until the profile actually starts, then
            # steps up to the profile's own initial value
            head = prof_start - t_start
            t_list += [head, head]
            c_list += [0.0, exposure_conc[0]]
        else:
            c_list[0] = np.interp(t_start, exposure_time, exposure_conc)

        t_list += inner_t
        c_list += inner_c

        if t_end > prof_end:
            # exposure holds its last known value up to the profile's end,
            # then steps down to zero for the remainder of the window
            tail = prof_end - t_start
            t_list += [tail, tail, Twin]
            c_list += [exposure_conc[-1], 0.0, 0.0]
        else:
            t_list.append(Twin)
            c_list.append(np.interp(t_end, exposure_time, exposure_conc))

        return np.asarray(t_list), np.asarray(c_list)

    @staticmethod
    def _prune_windows_mask(exposure_time, exposure_conc, window_starts, Twin):
        """
        Prune the set of candidate window start times to the ones that
        could possibly be the worst case. This is the pyDEBtox2019
        equivalent of prune_windows.m (BYOM, trick by Neil Sherborne):
        first find, across all candidate windows, the largest *minimum*
        concentration (the window whose floor is highest); any window
        whose *maximum* concentration is below that value cannot be the
        worst case, since there is always some other window where
        exposure is at least as high at every point in time within it.

        This relies on a monotonic dose-response relationship and, per the
        original MATLAB implementation, is not advised when there are
        feedbacks on the elimination rate combined with an
        assimilation/maintenance/growth mode of action - under those
        conditions the worst case may not be unique, and pruning could
        discard the true worst window.

        Returns a boolean array (True = keep, False = pruned), aligned
        with `window_starts`.
        """
        mins = np.empty(window_starts.shape)
        maxs = np.empty(window_starts.shape)
        for i, ts in enumerate(window_starts):
            _, c_list = DEBtox2019models._window_profile(exposure_time, exposure_conc, ts, Twin)
            mins[i] = c_list.min()
            maxs[i] = c_list.max()
        maxmin = mins.max()
        return maxs >= maxmin

    def _epx_window_task(self, exposure_time, exposure_conc, ts, tw, si, control_val, target,
                          modelpars, mflow0, mfhigh0, max_expand, xtol, plateau_tol=1e-6):
        """
        Compute the critical multiplication factor for a single moving-
        time-window position. This is the unit of work dispatched to
        worker processes when calc_epx_core is run with multicore=True
        (it is a plain, self-contained function of its arguments - no
        shared/global state - so, unlike BYOM's MATLAB implementation
        which stores the exposure scenario in a global and therefore
        cannot parallelize across windows, this can be run independently
        on any core).
        """
        def f(logMF):
            MF = 10 ** logMF
            t_list, c_list = self._window_profile(exposure_time, exposure_conc, ts, tw)
            # see the NOTE in calc_ecx_core.response: a bare 2-point timeext
            # can silently corrupt LSODA results for stiff/high-concentration
            # windows, since odeint's mxstep budget applies per interval.
            sol = self.calc_model(MF * c_list, t_list, modelpars, self.moa, self.feedb,
                                   timeext=np.linspace(0.0, tw, 50))
            effect = 1.0 - sol[si, -1] / control_val
            return effect - target

        return self._bisect_log(f, mflow0, mfhigh0, max_expand, xtol, increasing=True,
                                 plateau_tol=plateau_tol)

    def calc_epx_core(self, modelpars, exposure_time, exposure_conc, Twin,
                       X=(10, 50), endpoints=(0, 1, 2), Tstep=1.0,
                       MF_bounds=(1e-3, 1e3), max_expand=60, xtol=1e-8,
                       prune_win=False, multicore=False, plateau_tol=1e-6):
        """
        Core (numerical) EPx/LPx calculation for a single, fixed
        dataset-specific parameter vector. This is the pyDEBtox2019
        equivalent of calc_epx.m from the DEBtox2019/BYOM toolbox, using
        the moving time window (MTW) method: for every window length in
        `Twin`, a window of that length is slid in steps of `Tstep` across
        the whole exposure profile - including window starts *before* the
        profile's first time point (so the trailing part of the window
        probes the initial rise of the profile) and up to a start at the
        profile's very last time point. Whenever the window extends before
        the first, or beyond the last, recorded profile time point, the
        corresponding head/tail of the window is zero-padded (see
        _window_profile).

        For each window position, a per-window critical multiplication
        factor is found by bisection (see _epx_window_task): the value
        MF(t_start) that, applied to the whole profile, makes the endpoint
        at the end of that single window (run from a "fresh", undamaged
        organism) reach exactly X% effect relative to an unexposed control
        run over the same duration. Because effect increases monotonically
        with MF, the overall EPx/LPx is the *minimum* of this per-window
        curve (the window that is easiest to push to X% effect is the
        worst case), and the window start at that minimum is the
        worst-case window time. This is mathematically equivalent to (and
        no more expensive than) first finding, for a trial MF, the worst
        effect over all windows and then bisecting on MF - but it
        additionally yields the full per-window MF(t_start) curve as a
        natural by-product, useful for diagnostics/plotting.

        Survival (endpoint 0) yields LPx (lethal profile factor); length
        and reproduction (endpoints 1, 2) yield EPx (effect profile
        factor).

        Arguments:
        - modelpars: dataset-specific DEB parameter vector (linear scale).
        - exposure_time, exposure_conc: 1D arrays describing the (long)
          exposure profile to be scaled and scanned.
        - Twin: array-like of window lengths.
        - X: iterable of effect levels in percent (0 <= x < 100).
        - endpoints: iterable of endpoint codes (0=survival, 1=length, 2=reproduction).
        - Tstep: step by which the window start is advanced across the
          (head-to-tail extended) profile range.
        - MF_bounds: initial (low, high) bracket for the multiplication
          factor search; automatically expanded if needed.
        - max_expand, xtol: passed to the bracket search / brentq.
        - prune_win: if True, skip windows that provably cannot be the
          worst case before running any bisection on them (see
          _prune_windows_mask); their mf_curve entry stays NaN.
        - multicore: if True, distribute the per-window bisections (for
          each Twin/endpoint/X combination) across worker processes using
          a multiprocessing.Pool sized to the number of physical cores.
          Do NOT set this True when calc_epx_core is itself already being
          run inside a worker process (e.g. from worker_epx during CI
          propagation with multicore Pool) - nested pools are not
          supported by Python's multiprocessing.
        - plateau_tol: passed to the bracket search (see _bisect_log);
          stops expanding a window's MF search early (NaN) once the effect
          has visibly plateaued (e.g. length hitting its shrink floor)
          rather than always exhausting max_expand. Set to None to disable.

        Returns:
        - dict: results[endpoint][x] -> dict with:
            'value': np.ndarray aligned with Twin - the EPx/LPx (minimum
                     of the per-window critical MF curve).
            'worst_time': np.ndarray aligned with Twin - the window start
                     time at which that minimum was found (NaN if no
                     window could reach the target effect within MF_bounds).
            'window_starts': list (one array per Twin entry) of the window
                     start times that were scanned.
            'mf_curve': list (one array per Twin entry) of the per-window
                     critical MF at each of those window starts (NaN where
                     that window alone could not reach the target effect,
                     or was skipped by pruning).
        """
        exposure_time = np.asarray(exposure_time, dtype=float)
        exposure_conc = np.asarray(exposure_conc, dtype=float)
        Twin_arr = np.atleast_1d(np.asarray(Twin, dtype=float))
        prof_start = exposure_time[0]
        prof_end = exposure_time[-1]
        mflow0, mfhigh0 = MF_bounds

        results = {
            ep: {
                x: {
                    'value': np.full(Twin_arr.shape, np.nan),
                    'worst_time': np.full(Twin_arr.shape, np.nan),
                    'window_starts': [None] * len(Twin_arr),
                    'mf_curve': [None] * len(Twin_arr),
                }
                for x in X
            }
            for ep in endpoints
        }

        pool = mp.Pool(n_cores) if multicore else None
        try:
            for iw, tw in enumerate(Twin_arr):
                if tw <= 0:
                    continue
                # window starts: a regular Tstep grid spanning from a window that
                # just reaches the profile's first point (start = prof_start - tw)
                # to one that starts exactly at the profile's last point
                window_starts = np.arange(prof_start - tw, prof_end + 0.5 * Tstep, Tstep)
                if prune_win:
                    keep = self._prune_windows_mask(exposure_time, exposure_conc, window_starts, tw)
                else:
                    keep = np.ones(window_starts.shape, dtype=bool)
                active_idx = np.where(keep)[0]

                # dense timeext: see the NOTE in calc_ecx_core.response
                control = self.calc_model(np.zeros(2), np.array([0.0, tw]), modelpars,
                                           self.moa, self.feedb,
                                           timeext=np.linspace(0.0, tw, 50))[:, -1]
                for ep in endpoints:
                    si = self.ENDPOINT_STATE_IDX[ep]
                    control_val = control[si]
                    for x in X:
                        target = x / 100.0
                        mf_curve = np.full(window_starts.shape, np.nan)

                        if pool is not None:
                            args = [(exposure_time, exposure_conc, window_starts[i], tw, si,
                                      control_val, target, modelpars, mflow0, mfhigh0,
                                      max_expand, xtol, plateau_tol) for i in active_idx]
                            vals = pool.starmap(self._epx_window_task, args)
                            for i, v in zip(active_idx, vals):
                                mf_curve[i] = v
                        else:
                            for i in active_idx:
                                mf_curve[i] = self._epx_window_task(
                                    exposure_time, exposure_conc, window_starts[i], tw, si,
                                    control_val, target, modelpars, mflow0, mfhigh0,
                                    max_expand, xtol, plateau_tol)

                        results[ep][x]['window_starts'][iw] = window_starts
                        results[ep][x]['mf_curve'][iw] = mf_curve
                        if np.any(np.isfinite(mf_curve)):
                            worst_idx = np.nanargmin(mf_curve)
                            results[ep][x]['value'][iw] = mf_curve[worst_idx]
                            results[ep][x]['worst_time'][iw] = window_starts[worst_idx]
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        return results

    def worker_epx(self, pars, parvals, posfree, islog, dataset, exposure_time, exposure_conc,
                   Twin, X, endpoints, Tstep, MF_bounds, max_expand, xtol, prune_win=False,
                   plateau_tol=1e-6):
        """
        Worker function for EPx/LPx CI propagation: rebuilds a dataset-specific
        parameter vector from a free-parameter sample and runs calc_epx_core.
        Always runs calc_epx_core with multicore=False: when this worker is
        itself dispatched inside a multiprocessing.Pool (CI propagation
        across parameter sets), a nested pool for the per-window loop is
        not supported.
        """
        expanded = np.array(parvals, copy=True)
        expanded[posfree] = pars
        expanded = np.where(islog, 10 ** expanded, expanded)
        modelpars = self.build_dataset_parameters(expanded, dataset)
        return self.calc_epx_core(modelpars, exposure_time, exposure_conc, Twin, X, endpoints,
                                   Tstep, MF_bounds, max_expand, xtol,
                                   prune_win=prune_win, multicore=False,
                                   plateau_tol=plateau_tol)
    
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
            # newtime = self.timeext[nd]
            # # FIX this part for the brood pouch delay.
            # # CHECK if anything can be pre-computed
            # # tbp = 0 # make sure it is declared here
            # newtimeext = np.unique(np.concatenate((np.linspace(newtime[0],newtime[-1],max(self.min_t,len(newtime))),newtime)))
            # print("newtimeext: ", newtimeext)
            for i in range(self.concstruct_list[nd].ntreats):  # iterate over treatments within the dataset
                try:
                    modelsol = self.calc_model(self.concstruct_list[nd].concarraytr[i], self.concstruct_list[nd].timetr,
                                               modelpars, self.moa, self.feedb,
                                               self.newtimeext[nd])
                except DEBSolverError:
                    # the ODE solver could not integrate this parameter point:
                    # treat it as an infinitely bad fit rather than crashing.
                    # Any other exception (a real bug) is left to propagate.
                    return(np.inf)
                
                idx_targets = np.searchsorted(self.newtimeext[nd], self.timeext[nd])
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
