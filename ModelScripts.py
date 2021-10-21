# %% 
"""File to run analyses on various sleep model classes. Contains funtions to score models based on control values

@author: Zachary Spalding
"""

############### Imports and Constants ############

import random
import sys
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy import signal, stats
from tqdm import tqdm, trange

from ModelMapChanges import ModelMapChanges
from ModelWeber import ModelWeber
from ModelConcChanges import ModelConcChanges
from ModelOriginal import ModelOriginal
from ModelDunConcVar import ModelDunConcVar
from ModelMapChangesConcVar import ModelMapChangesConcVar
from ModelTest import ModelTest
from FileHandling import ParamCSVFile
from test import test

# CONTROL DATA VALUES (commented values are for data with MA)

# %
REM_AVG_PCT = 7.842727367596591 # 7.842727367596591
W_AVG_PCT = 24.69688725878789 # 27.01748276484954
S_AVG_PCT = 65.97726450653279 # 63.656669000471155

# seconds
REM_AVG_DUR = 55.0749551297144 # 55.0749551297144
W_AVG_DUR = 198.8703698539787 # 106.46357608588114
S_AVG_DUR = 173.70774099647616 # 138.55946450866577

# h^-1
REM_AVG_FREQ = 5.221150947082167 # 5.221150947082167
W_AVG_FREQ = 5.088634746407049 # 10.242111447948274
S_AVG_FREQ = 15.70906506878378 # 18.389295711477423


####################### Function Definitions ##################################

def laser_trig_percents(data_coll, hypno_coll, dt, pre_post, dur, ci = 95):
    """Plots effect of laser stimulation on brainstates over simulation period
    (Weber et al. 2018 - Figure 1c)

    Arguments:
        data_coll {list of numpy arrays} -- list containing simulation data for all simulated mice
        hypno_coll {list of numpy arrays} -- list containing simulation hypnograms for all simulated mice
        dt {float} -- timestep for mice simulations
        pre_post {float or int} -- duration of data retrieval before and after laser stimulation period
        dur {float or int} -- duration of laser stimulation

    Keyword Arguments:
            ci {int} -- confidence interval percentage (default: {95})
    """

    brainstates_df = pd.DataFrame()

    #iterate through all sleep simulations (separate mice)
    for k in range(len(data_coll)):

        #extract current sleep data and hypnogram data of input colleciton
        X = data_coll[k]
        H = hypno_coll[k]

        brainstates = []
        #iterate through laser stimulation data
        for i in range(1, len(X)):
            #find where laser onset begins
            if X[i][-1] > 0 and X[i-1][-1] == 0:
                #append sleep states in defined period before and after laser stimulation
                brainstates.append(H[0][int(i - (pre_post/dt)): int(i + (dur/dt) + (pre_post/dt) + 1)])

        #remove brainstate data if length is not a full defined period (i.e. recording of stimulation at end of sleep simulation)
        for i in range(len(brainstates)):
            if len(brainstates[i]) <= (2*pre_post + dur)/dt:
                del brainstates[i]

        #define lists for brainstate percentages
        sleep_state_percents = []
        timepoints = []
        states = []
        sleep_state_dict = {"REM": 1, "Wake": 2, "NREM": 3}
        #total value to calculate percentages
        total = len(brainstates)
        #iterate through the columns of brainstates (separate time points) in outer loop
        for i in range(len(brainstates[0])):
            #reset temporary list for sleep states corresponding to column/timestep
            temp = []
            #iterate through rows of brainstates (separate laser stimulation periods) in inner loop
            for j in range(len(brainstates)):
                #extract sleep state data for the current time point and laser stimulation trial
                temp.append(brainstates[j][i])
            #calculate sleep state percents REM sleep and add timepoint
            for key, value in sleep_state_dict.items():
                sleep_state_percents.append(temp.count(value) / total)
                states.append(key)
                timepoints.append(i*dt - pre_post)

        df = pd.DataFrame()
        #put data into pandas dataframe
        data_dict = {"Timepoints": timepoints, "Sleep State": states, "Percentages": sleep_state_percents}
        for key, value in data_dict.items():
            df[key] = value

        #append dataframe from single simulation to dataframe for entire collection
        brainstates_df = brainstates_df.append(df, ignore_index = True)

    #plot data with seaborn
    palette_colors = ['cyan', 'light purple', 'grey']
    cust_pal = sns.xkcd_palette(palette_colors)

    fig, ax = plt.subplots()
    sns.lineplot(x="Timepoints", y="Percentages", hue='Sleep State', data=brainstates_df, palette=cust_pal, ci = ci)
    ax.axvspan(0, dur, alpha = 0.1, color = 'blue')
    plt.ylim([-0.01, 1])
    plt.xlabel("Time (s, t = 0 is onset of laser stimulation)")
    plt.ylabel("Sleep State Percentages (%)")
    plt.title("Laser Triggered Sleep State Percentages")
    plt.show()

def pct_sq_diff(model_pcts, pr=0):
    """Calculates the squared difference between input sleep state percents
    and the average sleep state percents gathered from a control dataset of
    mice

    Args:
        model_pcts (list): sleep state percents ([REM, Wake, NREM])
        pr (int, optional): Printing variable: Prints scores if pr=1, does not plot otherwise. Defaults to 0.

    Returns:
        [list]: squared differences for each state in the same order as above
    """
    remWeight = 2.5 / REM_AVG_PCT
    wWeight = 1 / W_AVG_PCT
    sWeight = 1 / S_AVG_PCT
    weights = [remWeight, wWeight, sWeight]

    pct_diffs = [(model_pcts[0] - REM_AVG_PCT)**2, 
                (model_pcts[1] - W_AVG_PCT)**2, 
                (model_pcts[2] - S_AVG_PCT)**2]

    pct_scores = np.multiply(pct_diffs, weights)       

    if pr==1:
        print(f'    Percents: {pct_scores}')

    return pct_scores

def dur_sq_diff(model_durs, pr=0):
    """Calculates the squared difference between input sleep state durations
    and the average sleep state durations gathered from a control dataset of
    mice

    Args:
        model_durs (list): sleep state durations ([REM, Wake, NREM])
        pr (int, optional): Printing variable: Prints scores if pr=1, does not plot otherwise. Defaults to 0.

    Returns:
        [list]: squared differences for each state in the same order as above
    """
    remWeight = 1.3 / REM_AVG_DUR
    wWeight = 0.6 / W_AVG_DUR
    sWeight = 0.5 / S_AVG_DUR
    weights = [remWeight, wWeight, sWeight]
    
    dur_diffs = [(model_durs[0] - REM_AVG_DUR)**2, 
                (model_durs[1] - W_AVG_DUR)**2, 
                (model_durs[2] - S_AVG_DUR)**2]

    dur_scores = np.multiply(dur_diffs, weights)

    if pr==1:
        print(f'    Durations: {dur_scores}')

    return dur_scores

def freq_sq_diff(model_freqs, pr=0):
    """Calculates the squared difference between input sleep state frequencies
    and the average sleep state frequencies gathered from a control dataset of
    mice

    Args:
        model_freqs (list): sleep state frequencies ([REM, Wake, NREM])
        pr (int, optional): Printing variable: Prints scores if pr=1, does not plot otherwise. Defaults to 0.

    Returns:
        [list]: squared differences for each state in the same order as above
    """
    remWeight = 5.0 / REM_AVG_FREQ
    wWeight = 1 / W_AVG_FREQ
    sWeight = 1 / S_AVG_FREQ
    weights = [remWeight, wWeight, sWeight]

    freq_diffs = [(model_freqs[0] - REM_AVG_FREQ)**2, 
                (model_freqs[1] - W_AVG_FREQ)**2, 
                (model_freqs[2] - S_AVG_FREQ)**2]

    freq_scores = np.multiply(freq_diffs, weights)

    if pr==1:
        print(f'    Frequencies: {freq_scores}')

    return freq_scores

def plot_stats(model):
    """Plots model state percents, durations, and frequencies in comparison to control values

    Args:
        model (ModelMapchanges, ModelWeber, ModelOriginal): Model to get statistics from
    """
    labels = ['REM', 'Wake', 'NREM']
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(ncols=3, figsize=(12,5))
    
    #percents
    statePcts = model.get_state_pcts()
    bars = ax[0].bar(x, statePcts, width)
    ax[0].hlines(REM_AVG_PCT, x[0] - width/2, x[0] + width/2, color='red')
    ax[0].hlines(W_AVG_PCT, x[1] - width/2, x[1] + width/2, color='red')
    ax[0].hlines(S_AVG_PCT, x[2] - width/2, x[2] + width/2, color='red')
    ax[0].set_ylabel('Percentage of Total Sleep (%)')
    ax[0].set_title('Simulated Sleep State Percentages')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)

    for i, v in enumerate(statePcts):
                ax[0].text(x[i] - 0.15, v + 0.5, str(round(v, 2)), fontsize=9.0, fontweight='bold')

    #durations
    avgVals, seVals = model.get_state_durs()
    bars = ax[1].bar(x, avgVals, width, yerr=seVals)
    ax[1].hlines(REM_AVG_DUR, x[0] - width/2, x[0] + width/2, color='red')
    ax[1].hlines(W_AVG_DUR, x[1] - width/2, x[1] + width/2, color='red')
    ax[1].hlines(S_AVG_DUR, x[2] - width/2, x[2] + width/2, color='red')
    ax[1].set_ylabel('Average State Duration (s)')
    ax[1].set_title('Simulated Sleep State Average Durations')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)

    for i, v in enumerate(avgVals):
        ax[1].text(x[i] + 0.05, v + 1, str(round(v, 2)), fontsize=9.0, fontweight='bold')

    #frequencies
    stateFreqs = model.get_state_freqs()
    bars = ax[2].bar(x, stateFreqs, width)
    ax[2].hlines(REM_AVG_FREQ, x[0] - width/2, x[0] + width/2, color='red')
    ax[2].hlines(W_AVG_FREQ, x[1] - width/2, x[1] + width/2, color='red')
    ax[2].hlines(S_AVG_FREQ, x[2] - width/2, x[2] + width/2, color='red')
    ax[2].set_ylabel('Sleep State Frequency (1/h)')
    ax[2].set_title('Simulated Sleep State Frequencies')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels)

    for i, v in enumerate(stateFreqs):
        ax[2].text(x[i] - 0.15, v + 0.5, str(round(v, 2)), fontsize=9.0, fontweight='bold')

    plt.show()

def score_model(model, pr=0, p=0):
    """Scores model state values based on control values. Minimization of this score was key to parameter optimization.

    Args:
        model (ModelMapchanges, ModelWeber, ModelOriginal): Model to get statistics from
        pr (int, optional): Printing variable: Prints scores if pr=1, does not plot otherwise. Defaults to 0.
        p (int, optional): Plotting variable: Plots distribution if p=1, does not plot otherwise. Defaults to 0.

    Returns:
        [float]: model score
    """
    model.update_param_dict()

    if p==1:
        plot_stats(model)

    pcts = model.get_state_pcts()
    durs, _ = model.get_state_durs()
    freqs = model.get_state_freqs()

    if pr==1:
        print('Squared Differences:')

    dPcts = pct_sq_diff(pcts, pr=pr)
    dDurs = dur_sq_diff(durs, pr=pr)
    dFreqs = freq_sq_diff(freqs, pr=pr)

    pctWeight = 1.0 # 5.0
    durWeight = 1.0 # 0.3
    freqWeight = 1.0 # 5.0

    score = pctWeight*sum(dPcts) + durWeight*sum(dDurs) + freqWeight*sum(dFreqs)

    if pr==1:
        print(f'Score: {score} \n')
    
    return score

def show_winners(m_arr, s_arr, csvfile):
    """Plots statistics for 3 models with the lowest scores in the input model and score lists. Also writes winner parameters to input csv file

    Args:
        m_arr (list): list of models
        s_arr (list): list of scores for corresponding models
        csvfile (ParamCSVFile): csv file to write data to (class from FileHandling.py)

    Returns:
        [list]: score list of winning models
        [list]: model list of winning models
    """
    m_list = []
    s_list = []

    #add lowest 3 scores to m_list ([first, second, third])
    for i in range(3):
        ind = np.argmin(s_arr)
        m_list.append(m_arr[ind])
        s_list.append(s_arr[ind])
        m_arr = np.delete(m_arr, ind)
        s_arr = np.delete(s_arr, ind)

    #plot winners
    for i in [2, 1, 0]:
        if i == 2:
            print('----- Third -----')
        elif i == 1:
            print('----- Second -----')
        elif i == 0:
            print('----- First -----')

        m_list[i].hypnogram(p=1)
        _ = score_model(m_list[i], pr=1, p=1)
        csvfile.writeToFile(m_list[i], s_list[i])

    return s_list, m_list    

def brute_force(hrs, group, sigma, dur, delay, noise, csvfile):
    """Brute force method to calculate parameter values

    Args:
        *Same args as in run_mi_model for model classes excluding csv file*
        hrs {int or float (usually int)} -- simulation length in hours
        group {str} -- Neuron population to receive optogenetic activation 
        sigma {int or float (usually int)} -- optogenetic activation value from laser data 
        dur {int or float (usually int)} -- duration for laser stimulation 
        delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours 
        noise {bool} -- adds noise to simulation if true 
        csvfile (ParamCSVFile): csv file to write data to (class from FileHandling.py)

    Returns:
        [list]: score list of winning models
        [list]: model list of winning models
    """
    
    first_s = sys.maxsize
    second_s = sys.maxsize
    third_s = sys.maxsize

    first_m = None
    second_m = None
    third_m = None

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    dt = 0.05

    print('Running Parameter Iterations...')

    tot = len(np.arange(-6, -0.9, 0.5))*len(np.arange(-6, -0.9, 0.5))*\
          len(np.arange(600, 1501, 150))

    with tqdm(total=tot, file=sys.stdout) as pbar:
        for i in np.arange(-6, -0.9, 0.5): # g_Roff2R
            for j in np.arange(-6, -0.9, 0.5): # g_R2Roff
                for k in np.arange(600, 1501, 150): # tau_stpdown
                    #set parameters
                    model = ModelMapChanges(X0, dt)
                    model.g_Roff2R = i
                    model.g_R2Roff = j
                    model.tau_stpdown = k
                    # model.tau_stpup = x
                    # model.tau_stim = y
                    # model.delta_update = z

                    model.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
                    model.hypnogram(p=0)
                    s = score_model(model)
                    # print(f'g_Roff2R = {i}')
                    # print(f'g_R2Roff = {j}')
                    # # print(f'tau_stpdown = {k}')
                    # print(f'Score {s}')

                    #assign positions based on score
                    if s < first_s:
                        third_s = second_s
                        third_m = second_m
                        second_s = first_s
                        second_m = first_m
                    
                        first_s = s
                        first_m = model
                    elif s < second_s:
                        third_s = second_s
                        third_m = second_m

                        second_s = s
                        second_m = model
                    elif s < third_s:
                        third_s = s
                        third_m = model

                    pbar.update(1)
                    

    print('Done! \n')

    m_arr = np.array([first_m, second_m, third_m])
    s_arr = np.array([first_s, second_s, third_s])

    s_list, m_list = show_winners(m_arr, s_arr, csvfile)

    return s_list, m_list 

def brute_force_mod(model, mType, mod, div, hrs, group, sigma, dur, delay, noise):
    """Modification to brute force method to be used iteratively in funnel_solve function

    Args:
        model {} -- model to run simulations of
        mod {} -- modifier changing range around selected parameter value
        div {} -- number of parameter values to select in determined range 
        *Same args as in run_mi_model for model classes*
        hrs {int or float (usually int)} -- simulation length in hours
        group {str} -- Neuron population to receive optogenetic activation 
        sigma {int or float (usually int)} -- optogenetic activation value from laser data 
        dur {int or float (usually int)} -- duration for laser stimulation 
        delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours 
        noise {bool} -- adds noise to simulation if true 

    Returns:
        [list]: score list of winning models
        [list]: model list of winning models
    """
    first_s = sys.maxsize
    second_s = sys.maxsize
    third_s = sys.maxsize

    first_m = None
    second_m = None
    third_m = None

    sc = [first_s, second_s, third_s]
    mo = [first_m, second_m, third_m]

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    dt = 0.05

    tot = div #div**2

    inter = False
    try:
        with tqdm(total=tot, file=sys.stdout) as pbar:
            for i in np.linspace(model.g_W2Roff + (5/mod), model.g_W2Roff - (5/mod), div): # g_Roff2R
                # for j in np.linspace(model.g_W2S + (5/mod), model.g_W2S - (5/mod), div): # g_R2Roff
                    # for k in np.linspace(model.tau_stpdown + (1000/mod), model.tau_stpdown - (1000/mod), div): # tau_stpdown
                    #     for x in np.linspace(model.tau_stpup + (1000/mod), model.tau_stpup - (1000/mod), div): # tau_stpup
                            # for y in np.linspace(model.delta2W + (0.5/mod), model.delta2W - (0.5/mod), div):
                            #     for z in np.linspace(model.delta2Roff + (0.5/mod), model.delta2Roff - (0.5/mod), div):

                                    #set parameters
                                    if mType == 'MC':
                                        m = ModelMapChanges(X0, dt)
                                    elif mType == 'W':
                                        m = ModelWeber(X0, dt)
                                    m.g_W2Roff = i
                                    # m.g_R2Roff = j
                                    m.g_Roff2R = -4.0
                                    m.g_R2Roff = -3.6

                                    # m.g_S2W = i
                                    # m.g_W2S = j
                                    # m.tau_stpdown = k
                                    # m.tau_stpup = x
                                    # m.delta2W = y
                                    # m.delta2Roff = z

                                    m.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
                                    m.hypnogram(p=0)
                                    s = score_model(m)
                                    # print(f'g_Roff2R = {i}')
                                    # print(f'g_R2Roff = {j}')
                                    # print(f'tau_stpdown = {k}')
                                    # print(f'Score {s}')

                                    #assign positions based on score
                                    if s < first_s:
                                        third_s = second_s
                                        third_m = second_m
                                        second_s = first_s
                                        second_m = first_m
                                    
                                        first_s = s
                                        first_m = m
                                    elif s < second_s:
                                        third_s = second_s
                                        third_m = second_m

                                        second_s = s
                                        second_m = m
                                    elif s < third_s:
                                        third_s = s
                                        third_m = m

                                    sc = [first_s, second_s, third_s]
                                    mo = [first_m, second_m, third_m]

                                    pbar.update(1)

    except KeyboardInterrupt:
        inter = True
        return sc, mo, inter
        sys.exit()
        

    return sc, mo, inter

def funnel_solve(div, hrs, group, sigma, dur, delay, noise, csvfile, mType):
    """Determines regions of ideal parameter values by selecting value providing best score over multiple steps becoming finer and finer

    Args:
        div {} -- number of parameter values to select in determined range 
        hrs {int or float (usually int)} -- simulation length in hours
        group {str} -- Neuron population to receive optogenetic activation 
        sigma {int or float (usually int)} -- optogenetic activation value from laser data 
        dur {int or float (usually int)} -- duration for laser stimulation 
        delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours 
        noise {bool} -- adds noise to simulation if true 
        csvfile (ParamCSVFile): csv file to write data to (class from FileHandling.py)
        mType (string): 'MC' or 'W' - declares model type

    Returns:
        [list]: score list of winning models
        [list]: model list of winning models
    """
    #initialize standard model for first iteration
    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    dt = 0.05

    # select model type
    if mType == 'MC':
        model = ModelMapChangesConcVar(X0, dt)
    elif mType == 'W':
        model = ModelWeber(X0, dt)

    #create arrays to hold models and scores
    m_arr = np.array([])
    s_arr = np.array([])
    sc = []
    mo = []

    #funnel parameters
    numIter = 3
    mod = 1

    inter = False
    
    for i in range(numIter):  
        if not inter:
            print(f'Running Parameter Iteration ({i+1}/{numIter})...')
            #run brute-force, save models and scores to corresponding arrays  
            sc, mo, inter = brute_force_mod(model, mType, mod, div, hrs, group, sigma, dur, delay, noise)
            s_arr = np.append(s_arr, sc)
            m_arr = np.append(m_arr, mo)
            print('Done! \n')

            if not None in m_arr:
                #grab model with lowest score
                low_sc_ind = np.argmin(s_arr)
                model = m_arr[low_sc_ind]
                
                print('Winner:')
                print(f'    g_W2Roff: {model.g_W2Roff}')
                # print(f'    g_W2S: {model.tau_stpup}')
                # print(f'    tau_stpdown: {model.tau_stpdown}')
                # print(f'    tau_stpup: {model.tau_stpup}')
                print(f'    Score: {s_arr[low_sc_ind]} \n')
                #update parameter to reduce range for next step of funnel
                mod *= 2
        
        #save current data to csv file upon keyboard exit
        else:
            print('     -----------------------')
            print('     ----- Interrupted -----')
            print('     ----------------------- \n')
            if None in m_arr:
                print('Not enough models to write to CSV file! \n')

                return s_arr, m_arr

            else:
                s_list, m_list = show_winners(m_arr, s_arr, csvfile)

                return s_list, m_list

            
    
        
    s_list, m_list = show_winners(m_arr, s_arr, csvfile)

    return s_list, m_list
    
def avgRegressionSlope(hrs, group, sigma, dur, delay, noise):
    """Calculates and prints average inter-REM->REM correlation regression slope

    Args:
        hrs {int or float (usually int)} -- simulation length in hours
        group {str} -- Neuron population to receive optogenetic activation 
        sigma {int or float (usually int)} -- optogenetic activation value from laser data 
        dur {int or float (usually int)} -- duration for laser stimulation 
        delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours 
        noise {bool} -- adds noise to simulation if true 

    Returns:
        [float]: average regression slope
        [list]: all regression slopes
    """
    slopes = []

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    dt = 0.05

    print('Averaging REM/Inter-REM Regression Slopes...')
    for i in trange(50, file=sys.stdout):
        m = ModelMapChanges(X0, dt)
        m.g_Roff2R = -7.0
        m.g_R2Roff = -5.0
        m.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
        m.hypnogram(p=0)
        _, _, m = m.inter_REM()
        slopes.append(m)

    avgM = np.average(slopes)
    print(f'Average Regression Slope: {avgM}')

    return avgM, slopes


############################### Figure Generation #######################################

# parameters
dur = 5*60 # seconds
gap = 15*60 # seconds
# sigma = 0.5
sigma = 0
delay = 2 # hrs
hrs = 8 # hrs
dt = 0.05 # seconds (timestep)
# X0 = [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, delta, omega, sigma]
X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
#IC_conc_var = [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, zeta_Ron, zeta_Roff, zeta_S, zeta_W, delta, omega, sigma]
IC_conc_var = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0]
group = 'Roff'
noise = True
refractory_activation = True

div = 10

# create file for MC object (instantiation changes whether or not file is already in directory)
filename = 'paramSetsMC.csv'
paramSetsMC = None
for _, _, files in os.walk('.', topdown=False):
    if filename in files:
        paramSetsMC = ParamCSVFile(filename, False)
    else:
        paramSetsMC = ParamCSVFile(filename, True)

# create file for W object
filename = 'paramSetsW.csv'
paramSetsW = None
for _, _, files in os.walk('.', topdown=False):
    if filename in files:
        paramSetsW = ParamCSVFile(filename, False)
    else:
        paramSetsW = ParamCSVFile(filename, True)


######################### Testing of Various Model Classes (most code commented out) ############################

if sigma > 0:
    print(f"OPTO STIM ENABLED - Group: {group}")
    if refractory_activation:
        refract_dur = dur / 2
        refract_gap = gap / 3
        print('Applying optogenetic activation at refractory periods')
else:
    print("NO OPTO STIM")

if noise:
    print("NOISE ON")
else:
    print("NOISE OFF")

# %%

# # create model objects, simulate data and hypnograms

# scores, models = brute_force(hrs, group, sigma, dur, delay, noise, paramSets)

# print('----- Funnel-Solve -----')
# scores2, models2 = funnel_solve(div, 4, group, sigma, dur, delay, noise, paramSetsW, 'MC')

# mMC = ModelMapChanges(X0, dt)
# mMC.g_Roff2R = -7.0
# mMC.g_R2Roff = -5.0
# mMC.tau_stpdown = 1650
# mMC.tau_stpup = 1650

# mMC.run_mi_model(72, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mMC.hypnogram(p=1)
# s2 = score_model(mMC, pr=1, p=1)
# # mMC.avg_Ron_and_Roff_by_state()
# _,_,_ = mMC.inter_REM(p=1, nremOnly=False, log=False)
# # _,_,_ = mMC.inter_REM(p=1, nremOnly=False, log=True)
# # laser_trig_percents([mMC.X], [mMC.H], dt, 600, 300, ci='None')

##### STANDARD MODEL #####

mMCCV = ModelMapChangesConcVar(IC_conc_var, dt)
# mMCCV.g_Roff2R = -7.0
# mMCCV.g_R2Roff = -5.0
# mMCCV.tau_stpdown = 1650
# mMCCV.tau_stpup = 1650
mMCCV.g_W2stp = 0.2

# mMCCV.run_mi_model(8, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mMCCV.hypnogram(p=1)
# sCV = score_model(mMCCV, pr=1, p=1)
# # mMCCV.avg_Ron_and_Roff_by_state()
# _,_,_ = mMCCV.inter_REM(p=1, nremOnly=False, log=False)

# mMCCV.g_W2Roff = 5.0
mMCCV.run_mi_model(80 + 2, group=group, sigma=sigma, dur=dur, delay=delay, gap=gap, noise=noise, refractory_activation=False)
mMCCV.hypnogram_fig1(p=1, save=True, filename='fig1_hypno_%.1f_w2stp' % mMCCV.g_W2stp)
# mMCCV.hypnogram_fig1(p=1, p_zoom=1, save=True, filename='fig3_optoHypno')
sCV = score_model(mMCCV, pr=1, p=1)
# ron_rem, ron_wake, ron_nrem, roff_rem, roff_wake, roff_nrem = mMCCV.avg_Ron_and_Roff_by_state()
_,_,_ = mMCCV.inter_REM(p=1, nremOnly=True, log=True, save=True, filename='fig1_remPre_%.1f_w2stp' % mMCCV.g_W2stp)
# mbRon, mbRoff, mbstp, mbDelta, mlRon, mlRoff, mlstp, mlDelta = mMCCV.avg_Ron_Roff_seq_REM()
# mbRon, mbRoff, mbstp, mbDelta, mlRon, mlRoff, mlstp, mlDelta = mMCCV.avg_Ron_Roff_seq_REM_norm()
# mMCCV.avg_Ron_Roff_seq_REM_norm_REM_pre_grad(bin_size=40)
# wake_chunks, nrem_chunks = mMCCV.weber_fig_5b(num_chunks=4, save_fig=True)
# mMCCV.hysteresis_loop(save_fig=True)
# laser_df = mMCCV.laser_trig_percents(pre_post=gap, dur=dur, multiple=True, ci=95, group=group, refractory_activation=False, save_fig=False)

# getting average fW during sleep
# data_W = mMCCV.X[:,3]
# data_H = mMCCV.H
# sleep_inds = np.where(data_H[0] != 1)[0]
# sleep_W = data_W[sleep_inds]
# avg_sleep_W = np.mean(sleep_W)
# print(f'Average fW during sleep: {avg_sleep_W}')

# %% ##### RUN WITH REFRACTORY ACTIVATION #####

mMCCV.run_mi_model(8 + 2, group=group, sigma=sigma, dur=dur, delay=delay, gap=gap, noise=noise, refractory_activation=refractory_activation)
mMCCV.hypnogram_fig1(p=1, save=False)
# mMCCV.hypnogram_fig1(p=1, p_zoom=1, save=True, filename='fig3_optoHypno')
# sCV = score_model(mMCCV, pr=1, p=1)
# ron_rem, ron_wake, ron_nrem, roff_rem, roff_wake, roff_nrem = mMCCV.avg_Ron_and_Roff_by_state()
# _,_,_ = mMCCV.inter_REM(p=1, nremOnly=True, log=True)
# mbRon, mbRoff, mbstp, mbDelta, mlRon, mlRoff, mlstp, mlDelta = mMCCV.avg_Ron_Roff_seq_REM()
# mbRon, mbRoff, mbstp, mbDelta, mlRon, mlRoff, mlstp, mlDelta = mMCCV.avg_Ron_Roff_seq_REM_norm()
# mMCCV.avg_Ron_Roff_seq_REM_norm_REM_pre_grad(bin_size=40)
# wake_chunks, nrem_chunks = mMCCV.weber_fig_5b(num_chunks=4, save_fig=True)
# mMCCV.hysteresis_loop(save_fig=True)
laser_df_refract = mMCCV.laser_trig_percents(pre_post=refract_gap, dur=refract_dur, multiple=True, ci=95, group=group, refractory_activation=refractory_activation, save_fig=True)

# %%

##### Testing New Model Parameters #####

mTest = ModelTest(IC_conc_var, dt)
mTest.g_Roff2R = -7.0
mTest.g_R2Roff = -5.0
mTest.tau_stpdown = 1650
mTest.tau_stpup = 1650
mTest.g_W2Roff = 5.0

mTest.run_mi_model(40 + 2, group=group, sigma=sigma, dur=dur, delay=delay, gap=gap, noise=noise, refractory_activation=False)
mTest.hypnogram_fig1(p=1, save=False)
# sTest = score_model(mTest, pr=1, p=1)
# _,_,_ = mTest.inter_REM(p=1, nremOnly=True, log=True)
# mTest.avg_Ron_and_Roff_by_state()
laser_df_test = mTest.laser_trig_percents(dur=dur, multiple=True, ci=95, group=group, refractory_activation=False, save_fig=True)


# mWeb = ModelWeber(X0, dt)
# mWeb.g_Roff2R = -4.0
# mWeb.g_R2Roff = -3.6
# mWeb.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mWeb.hypnogram(p=1)
# s5 = score_model(mWeb, pr=1, p=1)
# _,_,_ = mWeb.inter_REM(p=1, nremOnly=True, log=False)
# _,_,_ = mWeb.inter_REM(p=1, nremOnly=True, log=True)

# mWeb.stp_max = 1.3
# mWeb.run_mi_model(8, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mWeb.hypnogram(p=1)
# s5 = score_model(mWeb, pr=1, p=1)



# mMC.tau_stpdown = 1650
# mMC.tau_stpup = 1650

# # mMC.g_Roff2R = -9.0
# # mMC.g_R2Roff = -6.0

# mMC.run_mi_model(72, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mMC.hypnogram(p=1)
# s3 = score_model(mMC, pr=1, p=1)




# locs, durs, _ = mMC.inter_REM(p=1, zoom_out=0)
# remDurs = mMC.REM_dur_dist(p=1)
# mMC.rem_zoom(locs, durs)
# _, _ = mMC.avg_first_rem_dur()
# mMC.rem_cycles()


# avgM, slopes = avgRegressionSlope(hrs, group, sigma, dur, delay, noise)


# # mNew.g_Roff2R = -5.0
# # mNew.g_R2Roff = -3.5
# # mNew.tau_stpdown = 1600.0
# # mNew.tau_hup = 400.0

# # mNew.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# # mNew.hypnogram(p=1)
# # s3 = score_model(mNew, pr=1, p=1)

# mNew.tau_stpdown = 1400.0
# mNew.tau_stpup = 300.0

# mNew.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mNew.hypnogram(p=1)
# s4 = score_model(mNew, pr=1, p=1)

# mNew.delta_update = 3.0
# mNew.tau_stim = 10.0

# mNew.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mNew.hypnogram(p=1)
# s5 = score_model(mNew, pr=1, p=1)

# mNew.g_R2Roff = -6.0

# mNew.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mNew.hypnogram(p=1)
# s6 = score_model(mNew, pr=1, p=1)

# mNew.tau_stpdown = 200.0
# mNew.tau_stpup = 600.0

# mNew.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mNew.hypnogram(p=1)
# s7 = score_model(mNew, pr=1, p=1)


# mWeb = ModelWeber(X0, dt)
# mWeb.run_mi_model(hrs, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mWeb.hypnogram(p=1)
# s8 = score_model(mWeb, pr=1, p=1)
# mWeb.avg_Ron_and_Roff_by_state()
# mWeb.inter_REM()


#IC = [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, delta, omega, sigma]
# IC = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]

# mOld = ModelOriginal(X0, dt)
# mOld.run_mi_model(8, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mOld.hypnogram(p=1)
# s9 = score_model(mOld, pr=1, p=1)
# mOld.avg_Ron_and_Roff_by_state()
# # mOld.inter_REM()

# mConcVar = ModelDunConcVar(IC_conc_var, dt)
# mConcVar.run_mi_model(72, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# # mConcVar.hypnogram(p=1)
# mConcVar.hypnogram(p=0)
# # s10 = score_model(mConcVar, pr=1, p=1)
# # mConcVar.avg_Ron_and_Roff_by_state()
# mConcVar.inter_REM(p=1, nremOnly=True, log=True)

# mTest = test(IC_conc_var, dt)
# mTest.run_mi_model(8, group=group, sigma=sigma, dur=dur, delay=delay, noise=noise)
# mTest.hypnogram(p=1)
# s10 = score_model(mTest, pr=1, p=1)
# mTest.avg_Ron_and_Roff_by_state()
# mTest.inter_REM()

# %%


