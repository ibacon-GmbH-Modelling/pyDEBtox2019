# read in the data and pack them in a class
# with methods for further elaboration that can
# also be expanded in case the data require
# additional processing (e.g. the proper 
# calculation of the reproduction)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# ENDPOINTS is the single source of truth for which observable endpoints
# (length/reproduction/survival) exist and what their completedataset
# attribute is named - see A3 in the code review: completedataset used to
# duplicate that list by hand, three different ways. endpoints.py has no
# dependency on this module or on models.py, so importing it here doesn't
# create a cycle either way.
from .endpoints import ENDPOINTS

def select_control_raw(dataframe, option='both'):
    '''
    Select only control treatments from a raw dataframe.
    Args:
        dataframe (pd.DataFrame): The input dataframe containing the data.
        option (str): Option to select controls. 'both' for negative and solvent controls,
                      'negcontrol' for negative control only, 'solventcontrol' for solvent control only.
    Returns:
        pd.DataFrame: A dataframe containing only the selected control treatments.
    '''
    if option=='both':
        checktreatments = [0, 0.1]
    elif option=='negcontrol':
        checktreatments = [0]
    elif option=='solventcontrol':
        checktreatments = [0.1]
    else:
        print("Option not recognized. Use 'both', 'negcontrol', or 'solventcontrol'.")
        return None
    control_cols = dataframe.iloc[0,1:].isin(checktreatments)
    controlframe = dataframe.iloc[:, np.append([True], control_cols.values)]
    return(controlframe)

class concclass:
    """
    A class to handle concentration data, pre-calculate quantities needed for 
    the GUTS model fits, and plotting of exposure data.
    Attributes:
        name (str): The name or origin of the data.
        concdata (numpy.ndarray): The input concentration data array. The first column represents 
            time, and the subsequent columns represent concentration values for different treatments.
        ntreats (int): The number of treatments.
        timetr (numpy.ndarray): Unmodified time vector extracted from the input data, used for plotting.
        time (numpy.ndarray): The unique time values extracted from the input data.
        concarraytr (numpy.ndarray): Array of concentrations at different time points for every treatment.
        concmax (numpy.ndarray): Array to store the maximum concentration for each treatment.
        concunits (str): The units of the concentration data.
        concarray (numpy.ndarray): Array of concentration data, reshaped to match the number of
            treatments and unique time points.
    Methods:
        __init__(concdata, name, concunits):
            Initializes the concclass object, processes the input data, interpolates missing values, 
            calculates slopes, time-weighted averages, and other attributes.
        plot_exposure(savefig=False, figname='', extension='.png'):
            Builds the exposure figure (concentration vs. time) for each treatment and
            returns (fig, ax). Optionally saves it to a file. Does not display it -
            call plt.show() yourself from a plain script.
            Args:
                savefig (bool): Whether to save the figure to a file. Default is False.
                figname (str): The base name of the file to save the figure. Default is an empty string.
                extension (str): The file extension for the saved figure. Default is '.png'.
    """
    def __init__(self,concdata,name,concunits,focus = False):
        self.name = name # to store the origin of the data
        if focus:
            self.concdata = concdata
        else:
            self.concdata = concdata[1:,:] # remove the first line as it is the names of the treatments
        self.ntreats = concdata.shape[1] - 1
        self.conctreatsnames = concdata[0,1:]
        self.timetr = self.concdata[:,0] # needed only for plotting
        self.time = self.concdata[:,0]
        self.concarraytr = np.transpose(self.concdata[:,1:])
        self.concmax = np.zeros(self.ntreats)
        self.concunits = concunits
        # all the following is to account for all the cases in which the data is not complete
        # in presence of NaNs the values areinterpolated beteween the
        # closest non-NaN values. If only one values is given, then the concentration is
        # assumed constant
        for i in range(self.ntreats):
            nans, x = np.isnan(self.concarraytr[i]), lambda z: z.nonzero()[0]
            if np.sum(~nans) == 1:
                self.concarraytr[i][nans] = self.concarraytr[i][~nans][0] # fill the nan with the first non-nan value
            else:
                for nan_idx in np.where(nans)[0]:
                    if nan_idx == 0:
                        # mirror of the trailing-NaN case below: no earlier
                        # value exists to interpolate from, so extend the
                        # concentration backwards from the first available
                        # (non-NaN) value instead of leaving it as NaN
                        next_idx = nan_idx + 1
                        while next_idx < len(self.time) and np.isnan(self.concarraytr[i][next_idx]):
                            next_idx += 1   # make sure to find the next non-NaN value
                        if next_idx < len(self.time):
                            self.concarraytr[i][nan_idx] = self.concarraytr[i][next_idx]  # use the first non-nan value
                        else:
                            self.concarraytr[i][nan_idx] = np.nan
                    elif nan_idx == len(self.time) - 1:
                        self.concarraytr[i][nan_idx] = self.concarraytr[i][nan_idx - 1]  # use the last non-nan value
                    else:
                        # interpoletion between the previous and next non-NaN values
                        prev_idx = nan_idx - 1
                        next_idx = nan_idx + 1
                        while next_idx < len(self.time) and np.isnan(self.concarraytr[i][next_idx]):
                            next_idx += 1   # make sure to find the next non-NaN value
                        if next_idx < len(self.time):
                            self.concarraytr[i][nan_idx] = np.interp(self.time[nan_idx], [self.time[prev_idx], self.time[next_idx]], [self.concarraytr[i][prev_idx], self.concarraytr[i][next_idx]])
                        else:
                            self.concarraytr[i][nan_idx] = np.nan
            self.concmax[i] = np.max(self.concarraytr[i])
        self.time = np.unique(self.time)
        # concarray: concarraytr (one column per *raw* row of the input file,
        # so a repeated time value - used to encode an instantaneous
        # concentration step, e.g. two rows both at t=5 with different
        # concentrations - has two columns) reduced to one column per
        # *unique* time in self.time. For a repeated time value we keep the
        # last matching raw column, i.e. the concentration that applies
        # going forward from that time - the same value calc_model's
        # breaktime segmenting uses for the segment that starts there -
        # rather than the value just before the step.
        # np.unique() on the *reversed* raw time vector reports, for each
        # sorted unique time, the index of its first occurrence when
        # scanning from the end; converting that back to an index into the
        # original (un-reversed) array gives that time's last occurrence.
        _, first_idx_in_reversed = np.unique(self.timetr[::-1], return_index=True)
        last_idx = (len(self.timetr) - 1) - first_idx_in_reversed
        self.concarray = self.concarraytr[:, last_idx]

    def plot_exposure(self, savefig=False, figname='', extension='.png'):
        '''
        Build the exposure figure, one panel per treatment, and return
        (fig, ax) with ax a 1-D array of Axes of length ntreats.

        Deliberately does not display anything: showing a figure is the
        caller's job. plt.show() blocks until the window is dismissed, and it
        shows every open figure rather than just this one. Under IPython or
        Jupyter with matplotlib integration the figure appears by itself;
        from a plain script, call plt.show() once at the end.
        '''
        fig = plt.figure()
        # squeeze=False keeps the result 2-D whatever ntreats is: with a
        # single treatment, subplots() would otherwise return a bare Axes
        # and every ax[i] below would raise.
        ax = fig.subplots(1,self.ntreats, squeeze=False)[0]
        cmax = np.max(self.concmax)
        for i in range(self.ntreats):
            ax[i].fill_between(self.timetr,self.concarraytr[i], label='Concentration', color='blue', alpha=0.2)
            ax[i].set_ylim([0, cmax*1.1])
            ax[i].set_xlabel("Time [d]")
            ax[i].set_title("T %s"%self.conctreatsnames[i])
        ax[0].set_ylabel("Concentration [%s]"%self.concunits)
        fig.tight_layout()
        if savefig:
            fig.savefig(figname+"_"+self.name+"_conc"+extension)
        return fig, ax


class dataclass:
    """
    A general class to handle data, to be inherited by more specific classes
    """
    def __init__(self,data):
        self.data = data[1:,:] # remove the first line as it is the names of the treatments
        self.ntreats = data.shape[1] - 1
        self.treatmentsnames = data[0,1:]  # it should be the same in all files
        self.uniquetreats = np.unique(self.treatmentsnames)
        self.statstype = data[0,0]
        self.time = self.data[:,0]
        self.timetreat = []
        self.dataarray = np.transpose(self.data[:,1:])
        self.weights = np.ones_like(self.dataarray) # to be used in the fitting
        for i in range(self.ntreats):
            tmptime = self.time #self.time[np.isnan(self.dataarray[i])==False] # remove this from here and do it in the surv subclass
            self.timetreat.append(tmptime)
    
    def flatten_and_clean(self,dataarray,weights):
        # flatten the data array and remove NaNs and zero weights
        flat_data_clean=list()
        flat_weights_clean=list()
        ind_fin_table = list()
        for i in self.uniquetreats:
            flatdata = dataarray[self.treatmentsnames == i].flatten()
            flatweights = weights[self.treatmentsnames == i].flatten()
            ind_fin = ((np.isfinite(flatdata)) & (flatweights>0))
            flat_data_clean.append(flatdata[ind_fin])
            flat_weights_clean.append(flatweights[ind_fin])
            ind_fin_table.append(ind_fin)
        return(flat_data_clean, flat_weights_clean, ind_fin_table)
    
    def _modify_treatmentsnames(self, dataset_id):
        # use here the same solution that Tjalling has in BYOM
        self.trrateatsnames = self.treatmentsnames + (dataset_id)*100

    def plot_data(self, dataarray=None, label="Data", wmeans=False):
        '''
        Build the data figure, one panel per unique treatment with that
        treatment's replicates overplotted, and return (fig, ax) with ax a
        1-D array of Axes of length len(uniquetreats).

        Deliberately does not display anything - see concclass.plot_exposure
        for why.
        '''
        if dataarray is None:
            dataarray = self.dataarray
        fig = plt.figure()
        # squeeze=False keeps the result 2-D whatever the number of unique
        # treatments is. The previous code sized the grid from uniquetreats
        # but picked the scalar-vs-array branch from ntreats, so a single
        # treatment measured in several replicates got a bare Axes and then
        # raised TypeError: 'Axes' object is not subscriptable.
        ax = fig.subplots(1,self.uniquetreats.shape[0], squeeze=False)[0]
        maxval = np.nanmax(dataarray)
        minval = min(0,np.nanmin(dataarray))
        for i in range(len(self.uniquetreats)):
            for j in range(len(dataarray)):
                if self.treatmentsnames[j] == self.uniquetreats[i]:
                    ax[i].plot(self.time, dataarray[j], 'o', color='blue')
            if wmeans:
                ax[i].errorbar(self.time,self.meanvalstransf[i,:],
                               yerr=[self.meanvalstransf[i,:] - self.lowlimtreat[i,:],
                                     self.upplimtreat[i,:] - self.meanvalstransf[i,:]],
                               fmt='s', label='Weighted mean', color='red')
            ax[i].set_xlabel("Time [d]")
            ax[i].set_title("T %s"%self.uniquetreats[i])
            ax[i].set_ylim([minval, maxval*1.1])
        ax[0].set_ylabel(label)
        fig.tight_layout()
        return fig, ax

    def add_plotdata(self, ax, ntreat, label="Data", wmeans=False):
        '''
        Add data points to an existing axis object.
        '''
        dataarray = self.dataarray[self.treatmentsnames==ntreat]
        maxval = np.nanmax(dataarray)
        minval = min(0,np.nanmin(dataarray))
        if wmeans:
                mask = self.uniquetreats==ntreat
                ax.errorbar(self.time,self.meanvalstransf[mask,:].flatten(), 
                            yerr=[self.meanvalstransf[mask,:].flatten() - self.lowlimtreat[mask,:].flatten(),
                                  self.upplimtreat[mask,:].flatten() - self.meanvalstransf[mask,:].flatten()],
                            fmt='s', label='Weighted mean', color='red')
        else:
            for i in range(len(dataarray)):
                ax.plot(self.time, dataarray[i], 'o', color='blue')
        # ax.set_ylim([minval, maxval*1.1])
        return ax
    
    def calc_mean_and_ci(self,dataarray=None):
        '''
        Calculate the weighted mean and the confidence interval
        for each tratment and each time point according
        to the transformation indicated in the data file

        The operation is done per treatment, but the 
        function will return the overall thing.
        '''
        self.meanvals = np.zeros((len(self.uniquetreats),len(self.time)))
        self.meanvalstransf = np.zeros((len(self.uniquetreats),len(self.time)))
        self.lowlimtreat = np.zeros((len(self.uniquetreats),len(self.time)))
        self.upplimtreat = np.zeros((len(self.uniquetreats),len(self.time)))
        for i in range(len(self.uniquetreats)):    
            if dataarray is None:
                datain = self.dataarray[self.treatmentsnames == self.uniquetreats[i]]
            else:
                # this is needed because the reproduction data can be given in different
                # formats, and the function should be able to handle all of them
                datain = dataarray[self.treatmentsnames == self.uniquetreats[i]]
            weightsin = self.weights[self.treatmentsnames == self.uniquetreats[i]]
            # make first sure that all the weights are Nan where the data is
            # not finite (NaN or +/-Inf - not just NaN, so a stray Inf can't
            # leak into the nansum/nanmean calls below)
            mask = ~np.isfinite(datain)
            weightsin[mask] = np.nan

            # 2. Row-wise sum ignoring NaNs
            row_sums = np.nansum(weightsin, axis=0)

            # 3. Indices where sum == 0
            ind_zero = np.where(row_sums == 0)[0]

            # 4. Indices where sum != 0
            ind_nonzero = np.where(row_sums != 0)[0]
            if self.statstype == 0:
                # logtransform
                data_in = np.log(np.maximum(datain, 1e-10)) # use with caution
            else:
                data_in = datain ** self.statstype

            # This is the SE of the weighted mean, calculated according to Madansky
            # and Alexander, following the WinCross/Quantum approach.
            # See: http://www.analyticalgroup.com/download/WEIGHTED_MEAN.pdf. This
            # also works when there are no weights, but then all w_in should be 1!
            # --- Initialize mean vector ---
            data_mn = np.full((data_in.shape[1],), np.nan)
            data_mnnt = np.full((datain.shape[1],), np.nan)
            # --- Means ---
            # For outliers (no weights → simple mean)
            data_mn[ind_zero] = np.nanmean(data_in[:, ind_zero], axis=0)
            data_mnnt[ind_zero] = np.nanmean(datain[:, ind_zero], axis=0)

            # For regular points → weighted mean
            num = np.nansum(data_in[:,ind_nonzero] * weightsin[:,ind_nonzero], axis=0)
            den = np.nansum(weightsin[:,ind_nonzero], axis=0)
            data_mn[ind_nonzero] = num / den
            data_mnnt[ind_nonzero] = np.nansum(datain[:,ind_nonzero] * weightsin[:,ind_nonzero], axis=0) / np.nansum(weightsin[:,ind_nonzero], axis=0)

            # --- Clean data based on weights ---
            data_in[weightsin == 0] = np.nan
            data_in[np.isnan(weightsin)] = np.nan

            # --- Standard deviation and variance ---
            data_sd = np.nanstd(data_in, axis=0, ddof=0)   # MATLAB std(...,0,...)
            data_var = data_sd ** 2

            # --- Effective sample size ("b") ---
            sum_w = np.nansum(weightsin, axis=0)
            sum_w_sq = np.nansum(weightsin**2, axis=0)
            b = (sum_w ** 2) / sum_w_sq

            # --- Variance of weighted mean ---
            var_mean = data_var / b

            # --- Standard error ---
            data_se = np.sqrt(var_mean)

            # --- Confidence interval (± 2 SE) ---
            data_ci = np.column_stack((data_mn - 2 * data_se,
                                       data_mn + 2 * data_se))

            # --- Back-transform ---
            if self.statstype == 0:
                # log-transform case
                data_mn = np.exp(data_mn)
                data_ci = np.exp(data_ci)
            else:
                # power transform
                data_mn = data_mn ** (1 / self.statstype)
                data_ci = data_ci ** (1 / self.statstype)

            self.meanvals[i,:] = data_mnnt # TODO: fix this here!!!
            self.meanvalstransf[i,:] = data_mn
            self.lowlimtreat[i,:] = data_ci[:, 0]
            self.upplimtreat[i,:] = data_ci[:, 1]



class survdataclass(dataclass):
    """
    """
    def __init__(self, survdata, missing=None):
        '''
        Arguments:
        - survdata: survival table (header row with treatment labels, first
          column with the observation times), numbers of survivors.
        - missing: optional matrix of missing/removed animals per interval,
          i.e. BYOM's W{i} for a data set with lam = -1. Note that for
          survival data this is NOT a statistical weight: it is the count of
          animals that were lost/removed rather than observed to die, and it
          defaults to zeros (as in prelim_checks.m), not to ones.
          Accepted either with the same layout as `survdata` (header row and
          time column included, which are then stripped) or as the bare
          (ntimes x ntreats) block.
        '''
        super().__init__(survdata)
        self.weights = self._parse_missing(missing, survdata)
        self._rebuild_derived()

    def _parse_missing(self, missing, survdata):
        '''
        Normalise the missing/removed-animals matrix to the (ntreats, ntimes)
        layout of self.dataarray. Mirrors the size handling of BYOM's
        prelim_checks.m, which accepts the weight matrix either with or
        without the time/scenario headers.
        '''
        ntimes, ntreats = self.dataarray.shape[1], self.dataarray.shape[0]
        if missing is None:
            return np.zeros_like(self.dataarray, dtype=float)
        missing = np.asarray(missing, dtype=float)
        if missing.shape == survdata.shape:
            # headers included: strip the label row and the time column
            missing = missing[1:, 1:]
        elif missing.shape != (ntimes, ntreats):
            raise ValueError(
                "missing/removed-animals matrix has shape %s; expected %s "
                "(bare block) or %s (with time column and header row), to "
                "match the survival data" % (
                    missing.shape, (ntimes, ntreats), survdata.shape))
        missing = np.transpose(missing)
        if np.any(missing < 0):
            raise ValueError("missing/removed animals cannot be negative")
        return missing

    def _pad_to_time(self, values, isfin):
        '''
        Lift a per-treatment vector that was computed on the NaN-stripped
        series back onto the full self.time grid, leaving NaN where the
        observation is missing.

        Everything that is compared or plotted against the shared time
        vector has to live on this grid: debtox2019api.get_survival_data
        builds the model values on self.time, and calc_survival_metrics
        then masks data and model alike with ~isnan(probs). Handing those
        a stripped vector silently pairs each observation with the wrong
        model time point. Padding also lets the three arrays below stack
        into a rectangular (ntreats x ntimes) array no matter where - or
        how many - NaNs each treatment has; matplotlib skips the NaNs.

        The likelihood triple (timetreat / deatharraytreat /
        missingarraytreat) is deliberately NOT padded: it stays stripped
        and self-consistent, which is exactly what transfer.m does per
        treatment, and what indexcommon_surv is built against.
        '''
        out = np.full(len(self.time), np.nan)
        out[isfin] = values
        return out

    def _rebuild_derived(self):
        '''
        Recompute every quantity derived from dataarray/data/time. Called from
        __init__ and from completedataset.subset() so both paths stay in sync
        instead of drifting apart (see review item B4).

        Two groups come out of here and they are indexed differently:
        - per REPLICATE (one entry per column of the data set): timetreat,
          deatharraytreat, missingarraytreat, survarrtreat, survprobstreat.
          The first three drive the likelihood, the last two are paired with
          the model output in debtox2019api.get_survival_data.
        - per UNIQUE TREATMENT (one row per exposure level): meanvalstransf,
          lowlimtreat, upplimtreat. These are the display arrays, and the
          replicates are pooled into them (see below).
        '''
        # additional elements specific to survival data
        self.timetreat = []
        self.survarrtreat = []
        self.deatharraytreat = []
        self.missingarraytreat = []
        self.survprobstreat = []
        self.lowlimtreat = []
        self.upplimtreat = []
        self.meanvalstransf = []
        z= 1.96
        for i in range(self.ntreats):
            isfin = np.isnan(self.dataarray[i])==False
            # BYOM prelim_checks.m: missing/removed animals may not be entered
            # at a time point where the observation itself is NaN - the two
            # would then be stripped inconsistently below.
            if np.any(self.weights[i][isfin==False] > 0):
                raise ValueError(
                    "treatment %s: missing/removed animals are entered at a "
                    "time point where the number of survivors is NaN"
                    % str(self.treatmentsnames[i]))
            tmpsurv = self.dataarray[i, isfin]
            tmptime = self.time[isfin]
            tmpmiss = self.weights[i, isfin].astype('float')
            # BYOM prelim_checks.m: no zombies. Checked on the NaN-stripped
            # series, so that a NaN in between two observations does not hide
            # an increase (in BYOM the NaN difference silently compares false).
            if np.any(np.diff(tmpsurv) > 0):
                raise ValueError(
                    "treatment %s: the number of survivors should never "
                    "increase in time" % str(self.treatmentsnames[i]))
            # survarrtreat is paired with the model output on the full time
            # grid (see _pad_to_time), so it is padded; timetreat stays
            # stripped because indexcommon_surv is built from it.
            self.survarrtreat.append(self._pad_to_time(tmpsurv, isfin))
            self.timetreat.append(tmptime)
            # deaths per interval, corrected for the animals that went missing
            # rather than died (transfer.m: Ndeaths = -diff([D_i;0]) - w_i)
            ndeaths = np.append( -(np.diff(tmpsurv[:]).astype('float')), tmpsurv[-1]) - tmpmiss
            if np.any(ndeaths < 0):
                raise ValueError(
                    "treatment %s: negative number of deaths after correcting "
                    "for missing/removed animals - check the missing-animals "
                    "matrix against the survival data"
                    % str(self.treatmentsnames[i]))
            self.deatharraytreat.append(ndeaths)
            self.missingarraytreat.append(tmpmiss)
            ninit = self.data[0,i+1] # time 0 in principle should never have a nan value
            tmpprob = tmpsurv/ninit
            self.survprobstreat.append(self._pad_to_time(tmpprob, isfin))

        # ------------------------------------------------------------------
        # Display arrays: one row per UNIQUE TREATMENT, not per replicate.
        #
        # A panel of the figure answers "what happened at this exposure
        # level?", so replicates of one level belong on the same axes - which
        # is what dataclass.plot_data already does when it draws the raw
        # points. The mean/CI arrays have to be indexed the same way, or
        # panel i gets the error bars of replicate i (a different treatment
        # entirely once the data set has replicates, e.g. test_Sgrp.txt).
        # This also matches lengthdataclass/reproclass, whose
        # calc_mean_and_ci already produces one row per unique treatment.
        #
        # The replicates are pooled, not averaged: survivors and initial
        # counts are summed across the replicates that have an observation at
        # that time, and the Wilson interval is computed on the pooled
        # proportion. Pooling is the correct binomial treatment (it is one
        # larger sample); averaging the per-replicate proportions would give
        # a narrower interval that is not a valid binomial CI.
        # ------------------------------------------------------------------
        for lab in self.uniquetreats:
            rows = np.where(self.treatmentsnames == lab)[0]
            obs = self.dataarray[rows, :]                    # (n_rep, n_time)
            isfin_t = np.isnan(obs)==False
            # initial numbers per replicate, broadcast over time, counted only
            # where that replicate actually has an observation
            ninit_rep = np.array([self.data[0, r+1] for r in rows], dtype=float)
            n_pooled = np.sum(np.where(isfin_t, ninit_rep[:, None], 0.0), axis=0)
            s_pooled = np.sum(np.where(isfin_t, obs, 0.0), axis=0)
            isfin = n_pooled > 0                             # some replicate reported
            n_i = n_pooled[isfin]
            tmpprob = s_pooled[isfin]/n_i
            # Wilson score interval on data probabilities. From openGUTS code
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            a = (tmpprob + z**2/(2*n_i))/(1+z**2/n_i)
            b = z/(1+z**2/n_i) * np.sqrt(tmpprob*(1-tmpprob)/n_i + z**2/(4*n_i**2))
            a[0]=1
            b[0]=0
            # The Wilson interval is defined to contain the sample proportion,
            # but a+b can land a few ULPs below it (e.g. 0.9999999999999999
            # against a pooled proportion of exactly 1). Enforce the bracket
            # explicitly: without it the error bars come out negative by ~1e-16
            # and matplotlib's errorbar refuses to draw them.
            tmplowlim = np.minimum(np.maximum(0,a-b), tmpprob)
            tmpupplim = np.maximum(np.minimum(1,a+b), tmpprob)
            self.lowlimtreat.append(self._pad_to_time(tmplowlim, isfin))
            self.upplimtreat.append(self._pad_to_time(tmpupplim, isfin))
            self.meanvalstransf.append(self._pad_to_time(tmpprob, isfin))
        # transform lists in arrays
        self.meanvalstransf = np.array(self.meanvalstransf)
        self.lowlimtreat = np.array(self.lowlimtreat)
        self.upplimtreat = np.array(self.upplimtreat)

    def plot_data(self, dataarray=None, label="numbers alive", scaleto1=False, wmeans=False):
        if dataarray is None:
            dataarray = self.dataarray
        if scaleto1:
            dataarray = np.array(dataarray, copy=True)  # avoid mutating the underlying data
            for i in range(dataarray.shape[0]):
                ninit =dataarray[i,0]
                dataarray[i,:] = dataarray[i,:]/ninit
        return super().plot_data(dataarray=dataarray, label=label, wmeans=wmeans)
    
    def add_plotdata(self, ax, ntreat, label="Data", scaleto1=False,wmeans=False):
        '''
        Add data points to an existing axis object.
        '''
        dataarray = self.dataarray[self.treatmentsnames==ntreat]
        if scaleto1:
            for i in range(dataarray.shape[0]):
                ninit = dataarray[i,0]
                dataarray[i,:] = dataarray[i,:]/ninit
        maxval = np.nanmax(dataarray)
        minval = min(0,np.nanmin(dataarray))
        if wmeans:
            mask = self.uniquetreats==ntreat
            # print("mask: ", mask)
            # print("meanvalstransf: ", self.meanvalstransf[mask])
            # print("lowlimtreat: ", self.lowlimtreat[mask])
            # print("upplimtreat: ", self.upplimtreat[mask])
            ax.errorbar(self.time,self.meanvalstransf[mask].flatten(), 
                            yerr=[self.meanvalstransf[mask].flatten() - self.lowlimtreat[mask].flatten(),
                                  self.upplimtreat[mask].flatten() - self.meanvalstransf[mask].flatten()],
                            fmt='s', label='Weighted mean', color='red')
        else:
            for i in range(len(dataarray)):
                ax.plot(self.time, dataarray[i], 'o', color='blue')
        # ax.set_ylim([minval, maxval*1.1])
        return ax



class lengthdataclass(dataclass):
    """
    """
    def __init__(self,lengthdata):
        super().__init__(lengthdata)
        self._rebuild_derived()

    def _rebuild_derived(self):
        '''
        Recompute every per-replicate quantity derived from dataarray/weights.
        Called from __init__ and from completedataset.subset() so both paths stay
        in sync instead of drifting apart (see review item B4).
        '''
        #self.lengthweights = np.ones_like(self.dataarray) # ported in parent class
        #self.lengthtreat = np.copy(self.dataarray)
        self.lengthtreat = []
        self.flatdataclean, self.flatweightsclean, self.indfintable = self.flatten_and_clean(self.dataarray, self.weights)
        for i in range(self.ntreats):
            tmplength = self.dataarray[i, np.isnan(self.dataarray[i])==False]
            self.lengthtreat.append(tmplength)
        self.calc_mean_and_ci()

    def plot_data(self, dataarray=None, label="Length [cm]", wmeans=False):
        return super().plot_data(dataarray=dataarray, label=label, wmeans=wmeans)
    
        

class reproclass(dataclass):
    # this class might need in the input also the
    # survival data and female data
    def __init__(self,reprodata, reprocase, optcase, survtable = None, femaletable = None, sexratio = 0.5):
        super().__init__(reprodata)
        self.reprocase = reprocase
        self.optcase = optcase
        self.survtable = survtable
        self.femaletable = femaletable
        self.sexratio = sexratio
        self.dataarray_cumulative = np.copy(self.dataarray)
        # insert here all the details of the repro handling
        # e.g. individual data, cumulative data, sex differentiation...
        # make methods for each of these cases
        if reprocase == "individual":
            self.makerepro_ind(optcase)
        elif reprocase == "group":
            self.makerepro_grp(self.survtable, self.femaletable, self.sexratio)
        # elif reprocase == "sex":
        #     self.makerepro_sex()
        self._rebuild_derived()

    def _rebuild_derived(self):
        '''
        Recompute the per-replicate aggregates (reprocumtreat, the flattened
        fit vectors, weighted means/CIs) from dataarray_cumulative. Does not
        redo the case-specific makerepro_ind/makerepro_grp transform - that
        depends on the raw construction inputs (e.g. survtable/femaletable)
        and only ever runs once, in __init__. completedataset.subset() slices
        dataarray_cumulative by row and then just needs this aggregation step
        redone (see review item B4).
        '''
        self.reprocumtreat = []
        for i in range(self.ntreats):
            tmprepro = self.dataarray_cumulative[i, np.isfinite(self.dataarray_cumulative[i])]
            self.reprocumtreat.append(tmprepro)
        self.flatdataclean, self.flatweightsclean, self.indfintable = self.flatten_and_clean(self.dataarray_cumulative, self.weights)
        # to be used in the fitting
        #self.reproweights = np.ones_like(self.dataarray) # ported in parent class
        if self.dataarray_cumulative is not None:
            self.calc_mean_and_ci(self.dataarray_cumulative)

    def makerepro_ind(self, optcase):
        # optcase can be 0, 1, 2, 3
        # 0: single intermoult period for the entire data set
        # 1: cumulative reproduction, but with the removal of time points with 0 reproduction. Use -1 in the dataset to indicate the appearence of the first egg
        if optcase not in [0, 1, 2]: 
            print("Values that are allowed for case are 0, 1, 2, 3")
            print("case 0 "
            "is for a single intermoult period for the entire data set." \
            "Screen output will show mean intermoult times and brood sizes " \
            "across the replicates, as function of brood number and treatment.")
            print("case 1 "
            "is for cumulative reproduction, " \
            "but with the removal of time points with 0 reproduction. " \
            "This is good for clutch-wise reproduction")
            print("case 2 "
            "is for cumulative reproduction, " \
            "but it does not remove the 0s. Good for continuous reproduction")
            print("case 3 "
            "The case 3 present in BYOM has not yet being implemented, " \
            "as it might be problematic already in the matlab implementation")            
        match optcase:
            case 0:
                # copilot did this. TODO: check if it works properly
                # Extract reproduction array (rows = individuals)
                R = np.copy(self.dataarray)
                t = self.time
                all_id = self.treatmentsnames
                c_u = self.uniquetreats

                max_broods = 0
                interm = [None] * R.shape[0]
                brdsz = [None] * R.shape[0]

                # -------------------------------------------------
                # 1) Loop over individuals: compute intermoult and brood size
                # -------------------------------------------------
                for i in range(R.shape[0]):
                    Rtmp = R[i, :].copy()

                    # indices of -1 (first egg)
                    ind_eggs = np.where(Rtmp == -1)[0]

                    # "moults": non-zero and non-nan observations
                    ind_moults = np.where((Rtmp != 0) & ~np.isnan(Rtmp))[0]

                    # replace -1 → 0 for brood-size calculations
                    if ind_eggs.size > 0:
                        Rtmp[ind_eggs] = 0

                    # intermoult periods
                    interm[i] = np.diff(t[ind_moults]) if ind_moults.size > 1 else np.array([])

                    # brood sizes at moults
                    if ind_moults.size > 0:
                        b = Rtmp[ind_moults]
                        # remove the first brood (which is zero)
                        brdsz[i] = b[1:] if b.size > 1 else np.array([])
                    else:
                        brdsz[i] = np.array([])

                    max_broods = max(max_broods, len(interm[i]))

                # -------------------------------------------------
                # 2) Build output matrices (interm_out, brdsz_out)
                # -------------------------------------------------
                interm_out = np.full((max_broods, R.shape[0]), np.nan)
                brdsz_out  = np.full((max_broods, R.shape[0]), np.nan)

                for i in range(R.shape[0]):
                    ilen = len(interm[i])
                    blen = len(brdsz[i])
                    interm_out[:ilen, i] = interm[i]
                    brdsz_out[:blen, i]  = brdsz[i]

                # -------------------------------------------------
                # 3) Print global summary
                # -------------------------------------------------
                print("\nChecking the data set regarding the intermoult period")
                print(f"Overall mean intermoult time  : {np.nanmean(interm_out):.4g}")
                print(f"Overall std of intermoult time: {np.nanstd(interm_out):.4g}")
                print(" ")

                # -------------------------------------------------
                # 4) Per-treatment means (Rmim = intermoults, Rbrdsz = brood size)
                # -------------------------------------------------
                # matrices have (max_broods+1) rows, first row for identifiers
                Rmim = np.full((max_broods + 1, len(c_u) + 1), np.nan)
                Rbrdsz = np.full_like(Rmim, np.nan)

                # first row: [-1, c_u]
                Rmim[0, 0] = -1
                Rmim[0, 1:] = c_u
                Rbrdsz[0, :] = Rmim[0, :]

                # first column = brood number
                Rmim[1:, 0] = np.arange(1, max_broods + 1)
                Rbrdsz[1:, 0] = np.arange(1, max_broods + 1)

                # fill per‑treatment means
                for j, cu in enumerate(c_u):
                    inds = np.where(all_id == cu)[0]   # replicates belonging to treatment cu
                    Rmim[1:, j + 1]  = np.nanmean(interm_out[:, inds], axis=1)
                    Rbrdsz[1:, j + 1] = np.nanmean(brdsz_out[:, inds], axis=1)

                # -------------------------------------------------
                # 5) Display treatment-level matrices
                # -------------------------------------------------
                print("Mean intermoult period across broods and across treatments")
                print("------------------------------------------------------------")
                print(Rmim)

                print("Mean brood size across broods and across treatments")
                print("------------------------------------------------------------")
                print(Rbrdsz)

                # MATLAB sets Rout = 0; here we simply do not assign dataarray_cumulative
                self.dataarray_cumulative = None
            case 1:
                # copilot did this. TODO: check if it works properly
                t = self.time
                R = np.copy(self.dataarray)          # reproduction array (individual rows)
                all_id = self.treatmentsnames        # treatment IDs for each individual

                for i in range(R.shape[0]):          # run through individuals
                    Rtmp = R[i, :].copy()

                    # --- 1. Detect time of death ---
                    ind_dth = np.where(np.isnan(Rtmp))[0]
                    if ind_dth.size > 0:
                        ind_dth = ind_dth[0]
                    else:
                        ind_dth = len(t) - 1

                    # --- 2. Intermoult periods from non-zero (non-nan) observations ---
                    ind_moults = np.where((Rtmp != 0) & ~np.isnan(Rtmp))[0]
                    if ind_moults.size > 1:
                        interm = np.diff(t[ind_moults])
                    else:
                        interm = np.array([])

                    # --- 3. Identify zeros and -1 ("first egg") ---
                    ind_zero = np.where(Rtmp == 0)[0]
                    ind_eggs = np.where(Rtmp == -1)[0]

                    if ind_eggs.size == 0:
                        # assume the first time point is true zero
                        ind_eggs = np.array([0])
                    else:
                        # convert -1 → 0 for cumsum
                        Rtmp[ind_eggs] = 0

                    # --- 4. Cumulative reproduction ---
                    cRtmp = np.cumsum(Rtmp)

                    # Remove zero points (set them back to NaN)
                    cRtmp[ind_zero] = np.nan

                    # Force all values up to first-egg index to 0
                    cRtmp[: ind_eggs[0] + 1] = 0

                    # Write back
                    R[i, :] = cRtmp

                    # --- 5. Check the last intermoult ---
                    ind_last = np.where(~np.isnan(cRtmp))[0]
                    if ind_last.size > 0:
                        ind_last = ind_last[-1]
                    else:
                        ind_last = 0

                    interm_last = t[ind_dth] - t[ind_last]

                    # Warning condition (same as MATLAB)
                    if (interm.size > 0) and ((interm > 5).any() or (interm_last > 5)):
                        print(
                            f"Warning: Some true zeros may need to be added "
                            f"for individual {i+1}, treatment {all_id[i]}"
                        )
                # Store result
                self.dataarray_cumulative = R
            case 2:
                for i in range(self.dataarray.shape[0]): # run through individuals
                    Rtmp     = self.dataarray[i,:].copy();      # extract one individual (copy: cases 0/1 also copy - see B11)
                    ind_eggs = np.where(Rtmp == -1)[0]  # index for time with first egg, or moults without neonates
                    if ind_eggs.size > 0:  # if there are such observations on first egg/moults ...
                        Rtmp[ind_eggs] = 0  # turn the -1 into a zero for cumsum
                    cRtmp = np.cumsum(Rtmp,0)  # cumulative reproduction
                    self.dataarray_cumulative[i,:] = cRtmp  # put the cumulated repro back into the data array in the correct column
            # case 3:
            #     pass


    def _parse_raw_table(self, table, name):
        '''
        Parse a BYOM-style raw table (first row = treatment/replicate ids,
        preceded by a type code; first column = time, preceded by the same
        code) into a time vector and a value array laid out as
        (nreplicates, ntime), i.e. the same orientation as self.dataarray.
        '''
        table = np.asarray(table, dtype=float)
        if table.shape[1] - 1 != self.ntreats:
            raise ValueError(
                f"The {name} table must have the same number of treatments/replicates "
                f"as the reproduction data")
        time = table[1:, 0]
        values = np.transpose(table[1:, 1:])
        return time, values

    def makerepro_grp(self, survtable=None, femaletable=None, sexratio=0.5):
        '''
        Translation of BYOM's makerepro_grp.m. Converts reproduction data for
        grouped animals (offspring summed over all mothers in a replicate)
        into the mean cumulative reproduction per female, weighted by the
        (estimated) number of females alive over each time interval.

        survtable: raw survivor table (same layout as reprodata), required.
        femaletable: raw table with the number of females observed at the
            point of sex determination (one time point per replicate). If
            None, the number of females is assumed to be the number of
            survivors times sexratio at every time point.
        sexratio: presumed female:male sex ratio, used when femaletable is
            None, and to account for the (unknown) sex of animals that died.
        '''
        if survtable is None:
            raise ValueError(
                "makerepro_grp requires a survtable (survivor counts) with the "
                "same time/treatment layout as the reproduction data")

        _, Sval = self._parse_raw_table(survtable, "survtable")
        Rval = self.dataarray
        if Sval.shape != Rval.shape:
            raise ValueError("The survtable and the reproduction data must be equally sized")
        ntreats, ntime = Rval.shape

        if femaletable is None:
            # no female counts given: assume number of females is a fixed
            # fraction of the survivors, at every time point
            Fnew = Sval * sexratio
        else:
            Ftime, Fval = self._parse_raw_table(femaletable, "femaletable")
            Fnew = np.full((ntreats, ntime), np.nan)
            for j, ft in enumerate(Ftime):
                idx = np.where(self.time == ft)[0]
                if idx.size == 0:
                    raise ValueError(
                        "Time points in femaletable must be a subset of the "
                        "reproduction data time points")
                Fnew[:, idx[0]] = Fval[:, j]

            for i in range(ntreats):
                ind_sex = np.where(~np.isnan(Fnew[i, :]))[0]
                if ind_sex.size != 1:
                    raise ValueError(
                        "Need 1 (and no more than 1) observation on the number "
                        "of females, per replicate")
                k = ind_sex[0]
                # propagate the observed number of females back to the earlier
                # time points (before sex could be determined)
                Fnew[i, :k] = Fnew[i, k]
                if np.isnan(Sval[i, :k]).any():
                    raise ValueError(
                        "There can (for now) not be NaNs in the survivor matrix "
                        "before the sex determination")

            # account for the (assumed) females among the animals that died:
            # add back, at each time point, the females estimated to have
            # died from that point onward
            F_dead = sexratio * (-np.diff(Sval, axis=1))
            F_dead = np.hstack([F_dead, np.zeros((ntreats, 1))])
            F_dead[np.isnan(F_dead)] = 0
            F_dead = np.cumsum(F_dead[:, ::-1], axis=1)[:, ::-1]
            Fnew = Fnew + F_dead

        # account for deaths of females within an interval: the observed
        # offspring have been produced by the average number of females
        # alive during that interval
        Fnew_avg = np.copy(Fnew)
        Fnew_avg[:, 1:] = (Fnew[:, 1:] + Fnew[:, :-1]) / 2
        Fnew = Fnew_avg

        self.weights = Fnew
        # Once the estimated number of females alive reaches zero (e.g. all
        # females in a replicate have died), the per-female reproduction
        # rate for that interval (and onward) is undefined. Rather than
        # dividing by zero and letting the outcome (NaN or Inf, depending
        # incidentally on whether the numerator is also zero) propagate
        # through the cumulative sum, mark those positions NaN explicitly
        # and deterministically - so downstream NaN-based filtering always
        # catches them, instead of only when the division happens to yield
        # NaN rather than Inf.
        ratio = np.full_like(Rval, np.nan)
        np.divide(Rval, Fnew, out=ratio, where=Fnew > 0)
        self.dataarray_cumulative = np.cumsum(ratio, axis=1)


    def plot_data(self, dataarray=None, label="Individual reproduction", wmeans=False):
        return super().plot_data(dataarray=dataarray, label=label,wmeans=wmeans)
    
    def plot_data_cumulative(self, label="Cumulative reproduction", wmeans=False):
        return super().plot_data(dataarray=self.dataarray_cumulative, label=label, wmeans=wmeans)
    
    def add_plotdata(self, ax, ntreat, label="Data",wmeans=False):
        '''
        Add data points to an existing axis object.
        '''
        dataarray = self.dataarray_cumulative[self.treatmentsnames==ntreat]
        maxval = np.nanmax(dataarray)
        minval = min(0,np.nanmin(dataarray))
        if wmeans:
            mask = self.uniquetreats==ntreat
            ax.errorbar(self.time,self.meanvalstransf[mask,:].flatten(), 
                        yerr=[self.meanvalstransf[mask,:].flatten() - self.lowlimtreat[mask,:].flatten(),
                              self.upplimtreat[mask,:].flatten() - self.meanvalstransf[mask,:].flatten()],
                        fmt='s', label='Weighted mean', color='red')
        else:
            for i in range(len(dataarray)):
                ax.plot(self.time, dataarray[i], 'o', color='blue')
        # ax.set_ylim([minval, maxval*1.1])
        return ax
    

# Maps each ENDPOINTS entry's dataset_attr to the class its data must be an
# instance of - used by completedataset.__init__'s isinstance checks below.
# concdata isn't in ENDPOINTS (it's exposure data, not an observable
# endpoint with a state_idx), so it's handled separately.
_ENDPOINT_DATA_CLASSES = {
    'lengthdata': lengthdataclass,
    'reprodata': reproclass,
    'survdata': survdataclass,
}


class completedataset:
    def __init__(self,
                 concdata=None,
                 lendata=None,
                 reprodata=None,
                 survdata=None):
        # None means "this endpoint isn't provided for this dataset" and is
        # silently skipped - that's a legitimate, common case (e.g. a
        # survival-only completedataset). Anything else that isn't the
        # expected class is not a legitimate case (e.g. a raw ndarray
        # passed where a lengthdataclass was expected) and must raise
        # instead of silently being dropped - see review item A4.
        if concdata is not None:
            if not isinstance(concdata, concclass):
                raise TypeError(
                    f"concdata must be a concclass instance or None, got {type(concdata).__name__}"
                )
            self.concdata = concdata

        # dataset_attr -> raw constructor input, for the ENDPOINTS loop
        # below (lendata is the one constructor parameter whose name
        # doesn't match its dataset_attr, 'lengthdata').
        raw_inputs = {'lengthdata': lendata, 'reprodata': reprodata, 'survdata': survdata}
        for spec in ENDPOINTS.values():
            candidate = raw_inputs[spec.dataset_attr]
            if candidate is None:
                continue
            expected = _ENDPOINT_DATA_CLASSES[spec.dataset_attr]
            if not isinstance(candidate, expected):
                raise TypeError(
                    f"{spec.dataset_attr} must be a {expected.__name__} instance or None, "
                    f"got {type(candidate).__name__}"
                )
            setattr(self, spec.dataset_attr, candidate)

        # create complete time vector
        self.complete_timevec = np.array([])
        if hasattr(self, 'concdata'):
            self.complete_timevec = np.concatenate((self.complete_timevec, self.concdata.time))
        for spec in ENDPOINTS.values():
            if hasattr(self, spec.dataset_attr):
                self.complete_timevec = np.concatenate(
                    (self.complete_timevec, getattr(self, spec.dataset_attr).time))
        self.complete_timevec = np.unique(np.sort(self.complete_timevec))
        self.calc_common_timeindices()

    def calc_common_timeindices(self):
        """
        Calculate the indices of the complete time vector in each dataset class.
        Returns:
            dict: A dictionary containing the indices for each dataset class.
        """
        self.time_indices = {}
        if hasattr(self, 'concdata'):
            xy, self.time_indices['concdata'], y_ind = np.intersect1d(self.complete_timevec, self.concdata.time, return_indices=True)
        for spec in ENDPOINTS.values():
            if hasattr(self, spec.dataset_attr):
                ep = getattr(self, spec.dataset_attr)
                self.time_indices[spec.dataset_attr] = []
                for i in range(ep.ntreats):
                    xy, tmpcommonindices, y_ind = np.intersect1d(
                        self.complete_timevec, ep.timetreat[i], return_indices=True)
                    self.time_indices[spec.dataset_attr].append(tmpcommonindices)


    def subset(self, selector):
        """
        Return a new completedataset that contains only selected treatments.
    
        Parameters
        ----------
        selector : array-like[bool] | callable | array-like of labels
            - If boolean mask: length must equal #treatments inferred from concentration data.
            - If callable: will be called with the list/array of treatment labels (from concdata)
              and must return a boolean mask of the same length.
            - If array-like of labels: will be converted to a mask by matching against labels.
    
        Returns
        -------
        completedataset
            A new dataset with sliced data arrays and recomputed time indices.
        """
        # --- 1) Determine reference labels from concentration data (authoritative) ---
        labels = self.concdata.conctreatsnames  # ndarray of treatment labels
        # Normalize to numpy array of objects/strings
        labels = np.array(labels)
    
        # --- 2) Build boolean mask from selector ---
        if callable(selector):
            mask = np.array(selector(labels), dtype=bool)
        else:
            sel = np.array(selector)
            if sel.dtype == bool:
                mask = sel
            else:
                # assume a list of label values
                mask = np.isin(labels, sel)
    
        if mask.shape[0] != labels.shape[0]:
            raise ValueError("Selector length does not match number of treatments.")
    
        # --- 3) Slice each endpoint if present ---
        new = completedataset.__new__(completedataset)  # allocate without __init__
    
        # Slice concentration
        if hasattr(self, 'concdata'):
            new.concdata = deepcopy(self.concdata)
            new.concdata.concarray = new.concdata.concarray[mask, :]
            new.concdata.ntreats = int(mask.sum())
            new.concdata.conctreatsnames = labels[mask]
            # also slice raw/tr arrays if you rely on them later (optional):
            new.concdata.concarraytr = new.concdata.concarraytr[mask, :]
            new.concdata.concmax = new.concdata.concmax[mask]
            # timetr/time stay unchanged (time dimension is not filtered here)
    
        # Helper to slice generic dataclass-like endpoints safely

        def _slice_endpoint(ep, ordered_unique_labels):
            """
            Slice a dataclass-like endpoint to selected treatments and recompute
            every attribute derived from the raw arrays by delegating to the same
            _rebuild_derived() that __init__ uses (flatdataclean/flatweightsclean/
            indfintable, plus whatever the subclass adds: lengthtreat, reprocumtreat,
            deatharraytreat, survprobstreat, lowlimtreat/upplimtreat, meanvalstransf,
            ...). Patching a hand-picked subset of these by name (the previous
            approach) silently went stale whenever a new derived attribute was
            added - see review item B4.

            Parameters
            ----------
            ep : dataclass | lengthdataclass | reproclass | survdataclass
                Endpoint instance to slice.
            ordered_unique_labels : array-like
                The selected treatment labels in the desired order (e.g., the same
                order as concdata.conctreatsnames for alignment with the exposure).
            """
            ep_new = deepcopy(ep)

            # --- 1) Slice rows (replicates) that belong to selected treatments ---
            keep_rows = np.isin(ep.treatmentsnames, np.array(ordered_unique_labels))
            ep_new.dataarray       = ep.dataarray[keep_rows, :]
            ep_new.weights         = ep.weights[keep_rows, :]
            ep_new.treatmentsnames = ep.treatmentsnames[keep_rows]
            ep_new.ntreats         = int(np.sum(keep_rows))
            # keep the raw (header-stripped) table column-aligned with dataarray:
            # survdataclass._rebuild_derived reads per-replicate initial counts off it
            ep_new.data = ep.data[:, np.concatenate(([True], keep_rows))]

            # Repro: also slice the cumulative matrix if present (the case-specific
            # makerepro_ind/makerepro_grp transform is not redone here, only sliced)
            if hasattr(ep_new, 'dataarray_cumulative') and ep.dataarray_cumulative is not None:
                ep_new.dataarray_cumulative = ep.dataarray_cumulative[keep_rows, :]

            # --- 2) Force unique-treatment order to match concentration order ---
            # This ensures indexing like indfintable[i] matches treatment i in conc.
            ep_new.uniquetreats = np.array(ordered_unique_labels, dtype=object)

            # --- 3) Recompute every derived attribute through the same path __init__ uses ---
            ep_new._rebuild_derived()

            return ep_new
    
        # labels = self.concdata.conctreatsnames (authoritative order)
        labels = np.array(self.concdata.conctreatsnames)
        
        # ... your mask construction ...        
        selected_labels = labels[mask]
        
        # Ensure they are unique but keep original order from conc:
        _, first_idx = np.unique(selected_labels, return_index=True)
        ordered_unique_labels = selected_labels[np.sort(first_idx)]
        
        # Now call the helper for each endpoint
        if hasattr(self, 'lengthdata'):
            new.lengthdata = _slice_endpoint(self.lengthdata, ordered_unique_labels)
        if hasattr(self, 'reprodata'):
            new.reprodata  = _slice_endpoint(self.reprodata,  ordered_unique_labels)
        if hasattr(self, 'survdata'):
            new.survdata   = _slice_endpoint(self.survdata,   ordered_unique_labels)
   
        # --- 4) Recompute complete_timevec and time indices on the new object ---
        # mirror what __init__ would do:
        new.complete_timevec = np.array([])
        if hasattr(new, 'concdata'):
            new.complete_timevec = np.concatenate((new.complete_timevec, new.concdata.time))
        if hasattr(new, 'lengthdata'):
            new.complete_timevec = np.concatenate((new.complete_timevec, new.lengthdata.time))
        if hasattr(new, 'reprodata'):
            new.complete_timevec = np.concatenate((new.complete_timevec, new.reprodata.time))
        if hasattr(new, 'survdata'):
            new.complete_timevec = np.concatenate((new.complete_timevec, new.survdata.time))
        new.complete_timevec = np.unique(np.sort(new.complete_timevec))
        new.calc_common_timeindices()  # reuse your method
    
        return new




