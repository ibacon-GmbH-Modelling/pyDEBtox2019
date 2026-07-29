# read in the data and pack them in a class
# with methods for further elaboration that can
# also be expanded in case the data require
# additional processing (e.g. the proper 
# calculation of the reproduction)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

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

def compile_dataset_dict(ccl, scl, lcl, rcl, ndataset):
    """
    Compile all dataset classes into a single dictionary for further processing.
    Args:
        ccl (concclass): Concentration data class instance.
        scl (survdataclass): Survival data class instance.
        lcl (lengthdataclass): Length data class instance.
        rcl (reproclass): Reproduction data class instance.
    Returns:
        dict: A dictionary containing all dataset classes and a complete time vector.
    """
    dataset_dict = {}
    dataset_dict['complete_timevec'] = np.array([])
    dataset_dict['endpoints']=np.array([])
    if ndataset < 1:
        print("ndataset must be at least 1")
        return None
    dataset_dict['ndatasets'] = ndataset
    if ccl is None:
        print("Concentration data class is required")
        return None
    dataset_dict['concdata'] = ccl
    dataset_dict['complete_timevec'] = np.concatenate((dataset_dict['complete_timevec'], ccl.time))
    if lcl is not None:
        dataset_dict['lengthdata'] = lcl
        dataset_dict['complete_timevec'] = np.concatenate((dataset_dict['complete_timevec'], lcl.time))
        dataset_dict['endpoints'] = np.append(dataset_dict['endpoints'],1) # length endpoint
    if rcl is not None:
        dataset_dict['reprodata'] = rcl
        dataset_dict['complete_timevec'] = np.concatenate((dataset_dict['complete_timevec'], rcl.time))
        dataset_dict['endpoints'] = np.append(dataset_dict['endpoints'],2) # reproduction endpoint
    if scl is not None:
        dataset_dict['survdata'] = scl
        dataset_dict['complete_timevec'] = np.concatenate((dataset_dict['complete_timevec'], scl.time))
        dataset_dict['endpoints'] = np.append(dataset_dict['endpoints'],3) # survival endpoint
    dataset_dict['complete_timevec'] = np.unique(np.sort(dataset_dict['complete_timevec']))
    return(dataset_dict)


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
        concslopestr (numpy.ndarray): Array to store the slopes of concentration changes over time 
            for each treatment.
        conctwa (numpy.ndarray): Array to store the time-weighted average concentration for each treatment.
        concconst (numpy.ndarray): Array to indicate if a treatment has constant concentration (1) or not (0).
        concmax (numpy.ndarray): Array to store the maximum concentration for each treatment.
        concunits (str): The units of the concentration data.
        concslopes (numpy.ndarray): Array of slopes of concentration changes over time, reshaped 
            to match the number of treatments and unique time points.
        concarray (numpy.ndarray): Array of concentration data, reshaped to match the number of 
            treatments and unique time points.
    Methods:
        __init__(concdata, name, concunits):
            Initializes the concclass object, processes the input data, interpolates missing values, 
            calculates slopes, time-weighted averages, and other attributes.
        plot_exposure(savefig=False, figname='', extension='.png'):
            Plots the exposure data (concentration vs. time) for each treatment. Optionally saves 
            the plot to a file.
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
        self.concslopestr = np.zeros_like(self.concarraytr)
        self.conctwa = np.zeros(self.ntreats)
        # array to store if a treatment has constant concentration or not
        self.concconst = np.zeros(self.ntreats) 
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
                        self.concarraytr[i][nan_idx] = np.nan # TO DO CHECK THIS
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
            self.concslopestr[i,:-1] = np.diff(self.concarraytr[i])/np.diff(self.time)   
            self.conctwa[i] = np.trapz(self.concarraytr[i],self.time)/self.time[-1]  # time weighted average
            self.concmax[i] = np.max(self.concarraytr[i])
            if (np.all(self.concslopestr[i])==0) & (len(np.unique(self.concarraytr[i]))<2):
                self.concconst[i] = 1
        self.time = np.unique(self.time)
        tmpslopes = self.concslopestr[np.isfinite(self.concslopestr)]
        tmparray = self.concarraytr[np.isfinite(self.concslopestr)]
        self.concslopes = tmpslopes.reshape((self.ntreats,len(self.time)))
        self.concarray = tmparray.reshape((self.ntreats,len(self.time)))

    def plot_exposure(self, savefig=False, figname='', extension='.png'):
        fig = plt.figure()
        ax = fig.subplots(1,self.ntreats)
        cmax = np.max(self.concmax)
        if self.ntreats==1:
            ax.fill_between(self.timetr,self.concarraytr[0], label='Concentration', color='blue', alpha=0.2)
            ax.set_ylim([0, cmax*1.1])
            ax.set_ylabel("Concentration [%s]"%self.concunits)
            ax.set_xlabel("Time [d]")
        else:
            for i in range(self.ntreats):
                ax[i].fill_between(self.timetr,self.concarraytr[i], label='Concentration', color='blue', alpha=0.2)
                ax[i].set_ylim([0, cmax*1.1])
                ax[i].set_xlabel("Time [d]")
                ax[i].set_title("T %s"%self.conctreatsnames[i])
            ax[0].set_ylabel("Concentration [%s]"%self.concunits)
        plt.tight_layout()
        plt.show()
        if savefig:
            fig.savefig(figname+"_"+self.name+"_conc"+extension)


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
        if dataarray is None:
            dataarray = self.dataarray
        fig = plt.figure()
        ax = fig.subplots(1,self.uniquetreats.shape[0])
        maxval = np.nanmax(dataarray)
        minval = min(0,np.nanmin(dataarray))
        if self.ntreats==1:
            ax.plot(self.time,dataarray[0], 'o', label='Data', color='blue')
            if wmeans:
                ax.errorbar(self.time,self.meanvalstransf, 
                            yerr=[self.meanvalstransf - self.lowlimtreat,
                                  self.upplimtreat - self.meanvalstransf],
                            fmt='s', label='Weighted mean', color='red')
            ax.set_ylabel("Data")
            ax.set_xlabel("Time [d]")
        else:
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
        plt.tight_layout()
        plt.show()

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
            # make first sure that all the weights are Nan where the data is NaN
            mask = np.isnan(datain)
            weightsin[mask] = np.nan

            # 2. Row-wise sum ignoring NaNs
            row_sums = np.nansum(weightsin, axis=0)

            # 3. Indices where sum == 0
            ind_zero = np.where(row_sums == 0)[0]

            # 4. Indices where sum != 0
            ind_nonzero = np.where(row_sums != 0)[0]
            if self.statstype == 0:
                # logtransform
                data_in = np.log(datain,1e-10) # use with caution
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
    def __init__(self,survdata):
        super().__init__(survdata)

        # additional elements specific to survival data
        self.timetreat = []
        self.survarrtreat = []
        self.deatharraytreat = []
        self.survprobstreat = []
        self.lowlimtreat = []
        self.upplimtreat = []
        self.meanvalstransf = []
        z= 1.96
        for i in range(self.ntreats):
            tmpsurv = self.dataarray[i, np.isnan(self.dataarray[i])==False]
            tmptime = self.time[np.isnan(self.dataarray[i])==False]
            self.survarrtreat.append(tmpsurv)
            self.timetreat.append(tmptime)
            self.deatharraytreat.append(np.append( -(np.diff(tmpsurv[:]).astype('float')), tmpsurv[-1]) )
            ninit = self.data[0,i+1] # time 0 in principle should never have a nan value
            tmpprob = tmpsurv/ninit
            self.survprobstreat.append(tmpprob)
            # Wilson score interval on data probabilities. From openGUTS code
            # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            a = (tmpprob + z**2/(2*ninit))/(1+z**2/ninit)
            b = z/(1+z**2/ninit) * np.sqrt(tmpprob*(1-tmpprob)/ninit + z**2/(4*ninit**2))
            a[0]=1
            b[0]=0
            tmplowlim = np.maximum(0,a-b)
            tmpupplim = np.minimum(1,a+b)
            self.lowlimtreat.append(tmplowlim)
            self.upplimtreat.append(tmpupplim)
            self.meanvalstransf.append(tmpprob) #?? check
        # transform lists in arrays
        self.meanvalstransf = np.array(self.meanvalstransf)
        self.lowlimtreat = np.array(self.lowlimtreat)
        self.upplimtreat = np.array(self.upplimtreat)

    def plot_data(self, dataarray=None, label="numbers alive", scaleto1=False, wmeans=False):
        if dataarray is None:
            dataarray = self.dataarray
        if scaleto1:
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
            print("mask: ", mask)
            print("meanvalstransf: ", self.meanvalstransf[mask])
            print("lowlimtreat: ", self.lowlimtreat[mask])
            print("upplimtreat: ", self.upplimtreat[mask])
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
        #self.lengthweights = np.ones_like(self.dataarray) # ported in parent class
        #self.lengthtreat = np.copy(self.dataarray)
        self.lengthtreat = []
        self.flatdataclean, self.flatweightsclean, self.indfintable = self.flatten_and_clean(self.dataarray, self.weights)
        for i in range(self.ntreats):
            tmplength = self.dataarray[i, np.isnan(self.dataarray[i])==False]
            self.lengthtreat.append(tmplength)

    def plot_data(self, dataarray=None, label="Length [cm]", wmeans=False):
        return super().plot_data(dataarray=dataarray, label=label, wmeans=wmeans)
    
        

class reproclass(dataclass):
    # this class might need in the input also the
    # survival data and female data
    def __init__(self,reprodata, reprocase, optcase, survtable = None, femaletable = None):
        super().__init__(reprodata)
        self.reprocase = reprocase
        self.optcase = optcase
        self.dataarray_cumulative = np.copy(self.dataarray)
        self.reprocumtreat = []
        # insert here all the details of the repro handling
        # e.g. individual data, cumulative data, sex differentiation...
        # make methods for each of these cases
        if reprocase == "individual":
            self.makerepro_ind(optcase)
        elif reprocase == "group":
            self.makerepro_grp()
        # elif reprocase == "sex":
        #     self.makerepro_sex()
        for i in range(self.ntreats):
            tmprepro = self.dataarray_cumulative[i, np.isnan(self.dataarray_cumulative[i])==False]
            self.reprocumtreat.append(tmprepro)
        self.flatdataclean, self.flatweightsclean, self.indfintable = self.flatten_and_clean(self.dataarray_cumulative, self.weights)
        # to be used in the fitting
        #self.reproweights = np.ones_like(self.dataarray) # ported in parent class

    def makerepro_ind(self, optcase):
        # optcase can be 0, 1, 2, 3
        # 0: single intermoult period for the entire data set
        # 1: cumulative reproduction, but with the removal of time points with 0 reproduction. Use -1 in the dataset to indicate the appearence of the first egg
        if optcase not in [0, 1, 2, 3]: 
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
            "is for shifting the neonate release back to previous moult. " \
            "When this option is used, don't shift the model predictions " \
            "with <glo.Tbp>: the data now represent egg production rather " \
            "than neonate release")            
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
                    Rtmp     = self.dataarray[i,:];      # extract one individual
                    ind_eggs = np.where(Rtmp == -1)[0]  # index for time with first egg, or moults without neonates
                    if ind_eggs.size > 0:  # if there are such observations on first egg/moults ...
                        Rtmp[ind_eggs] = 0  # turn the -1 into a zero for cumsum
                    cRtmp = np.cumsum(Rtmp,0)  # cumulative reproduction
                    self.dataarray_cumulative[i,:] = cRtmp  # put the cumulated repro back into the data array in the correct column
            case 3:
                pass


    def makerepro_grp(self):
        pass


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
    

class completedataset:
    def __init__(self,
                 concdata=None,
                 lendata=None,
                 reprodata=None,
                 survdata=None):
        if type(concdata) is concclass:
            self.concdata = concdata
        if type(lendata) is lengthdataclass:
            self.lengthdata = lendata
        if type(reprodata) is reproclass:
            self.reprodata = reprodata
        if type(survdata) is survdataclass:
            self.survdata = survdata
        # create complete time vector
        self.complete_timevec = np.array([])
        if hasattr(self, 'concdata'):
            self.complete_timevec = np.concatenate((self.complete_timevec, self.concdata.time))
        if hasattr(self, 'lengthdata'):
            self.complete_timevec = np.concatenate((self.complete_timevec, self.lengthdata.time))
        if hasattr(self, 'reprodata'):
            self.complete_timevec = np.concatenate((self.complete_timevec, self.reprodata.time))
        if hasattr(self, 'survdata'):
            self.complete_timevec = np.concatenate((self.complete_timevec, self.survdata.time))
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
        if hasattr(self, 'lengthdata'):
            self.time_indices['lengthdata'] = []
            for i in range(self.lengthdata.ntreats):
                xy, tmpcommonindices, y_ind = np.intersect1d(self.complete_timevec, self.lengthdata.timetreat[i], return_indices=True)
                self.time_indices['lengthdata'].append(tmpcommonindices)
        if hasattr(self, 'reprodata'):
            self.time_indices['reprodata'] = []
            for i in range(self.reprodata.ntreats):
                xy, tmpcommonindices, y_ind = np.intersect1d(self.complete_timevec, self.reprodata.timetreat[i], return_indices=True)
                self.time_indices['reprodata'].append(tmpcommonindices)
        if hasattr(self, 'survdata'):
            self.time_indices['survdata'] = []
            for i in range(self.survdata.ntreats):
                xy, tmpcommonindices, y_ind = np.intersect1d(self.complete_timevec, self.survdata.timetreat[i], return_indices=True)
                self.time_indices['survdata'].append(tmpcommonindices)


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
        import numpy as np
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
        from copy import deepcopy
        new = completedataset.__new__(completedataset)  # allocate without __init__
    
        # Slice concentration
        if hasattr(self, 'concdata'):
            new.concdata = deepcopy(self.concdata)
            new.concdata.concarray = new.concdata.concarray[mask, :]
            new.concdata.concslopes = new.concdata.concslopes[mask, :]
            new.concdata.ntreats = int(mask.sum())
            new.concdata.conctreatsnames = labels[mask]
            # also slice raw/tr arrays if you rely on them later (optional):
            new.concdata.concarraytr = new.concdata.concarraytr[mask, :]
            new.concdata.concmax = new.concdata.concmax[mask]
            new.concdata.concconst = new.concdata.concconst[mask]
            # timetr/time stay unchanged (time dimension is not filtered here)
    
        # Helper to slice generic dataclass-like endpoints safely

        def _slice_endpoint(ep, ordered_unique_labels):
            """
            Slice a dataclass-like endpoint to selected treatments and rebuild
            collapsed lists (flatdataclean, flatweightsclean, indfintable).
        
            Parameters
            ----------
            ep : dataclass | lengthdataclass | reproclass | survdataclass
                Endpoint instance to slice.
            ordered_unique_labels : array-like
                The selected treatment labels in the desired order (e.g., the same
                order as concdata.conctreatsnames for alignment with the exposure).
            """
            from copy import deepcopy
            import numpy as np
        
            ep_new = deepcopy(ep)
        
            # --- 1) Slice rows (replicates) that belong to selected treatments ---
            keep_rows = np.isin(ep.treatmentsnames, np.array(ordered_unique_labels))
            ep_new.dataarray       = ep.dataarray[keep_rows, :]
            ep_new.weights         = ep.weights[keep_rows, :]
            ep_new.treatmentsnames = ep.treatmentsnames[keep_rows]
            ep_new.ntreats         = int(np.sum(keep_rows))
        
            # Some endpoints carry additional per-row structures. Keep them aligned:
            if hasattr(ep_new, 'timetreat') and isinstance(ep_new.timetreat, list):
                ep_new.timetreat = [ep_new.timetreat[i] for i, k in enumerate(keep_rows) if k]
        
            # Repro: also slice the cumulative matrix if present (this is the base for flattening there)
            if hasattr(ep_new, 'dataarray_cumulative'):
                ep_new.dataarray_cumulative = ep.dataarray_cumulative[keep_rows, :]
        
            # Length: if 'lengthtreat' is stored per replicate, rebuild it
            if hasattr(ep_new, 'lengthtreat'):
                ep_new.lengthtreat = [
                    ep_new.dataarray[i, np.isnan(ep_new.dataarray[i]) == False]
                    for i in range(ep_new.ntreats)
                ]
        
            # Repro: if 'reprocumtreat' is stored per replicate, rebuild it from cumulative data
            if hasattr(ep_new, 'reprocumtreat'):
                arr = ep_new.dataarray_cumulative if hasattr(ep_new, 'dataarray_cumulative') else ep_new.dataarray
                ep_new.reprocumtreat = [
                    arr[i, np.isnan(arr[i]) == False]
                    for i in range(ep_new.ntreats)
                ]
        
            # --- 2) Force unique-treatment order to match concentration order ---
            # This ensures your model indexing like indfintable[i] matches treatment i in conc.
            ep_new.uniquetreats = np.array(ordered_unique_labels, dtype=object)
        
            # --- 3) Recompute the flattened lists using the correct base array ---
            # For reproduction use dataarray_cumulative; otherwise use dataarray.
            base_array = ep_new.dataarray_cumulative if hasattr(ep_new, 'dataarray_cumulative') else ep_new.dataarray
            ep_new.flatdataclean, ep_new.flatweightsclean, ep_new.indfintable = ep_new.flatten_and_clean(
                base_array,
                ep_new.weights
            )
        
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




