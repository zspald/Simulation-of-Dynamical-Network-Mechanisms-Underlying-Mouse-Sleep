# %%
import random

import matplotlib.pylab as plt
from matplotlib import patches
import numpy as np
import seaborn as sns
from scipy import signal, stats
import pandas as pd


def mi_model(X, t):
    """
    Implementation of MI model as described in Dunmyre et al., 2014 

    [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h]
    """
    [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X

    # Parameters (comment denotes original parameters given)
    R_max = 5.0 #5.0
    Roff_max = 5.0 #5.0
    W_max = 5.50 #5.0
    S_max = 5.0 #5.0

    tau_Roff = 2.0 #2.0
    tau_R = 1.0 #2.0
    tau_W = 25.0 #2.0
    #tau_W = 0.5
    tau_S = 10.0 #2.0

    alpha_Roff = 2 #2
    alpha_R = 0.5 #0.5
    alpha_W = 0.5 #0.5
    beta_R = -0.5 #-0.5
    beta_W = -0.3 #-0.3
    alpha_S = 0.25 #0.25

    gamma_R = 4.0 #4.0
    gamma_Roff = 5.0 #5.0
    gamma_W = 5.0 #5.0
    gamma_S = 4.0 #4.0

    k1_Roff = 0.8 #0.8
    k2_Roff = 7.0 #7.0
    k1_S = 0 #0
    k2_S = -1.5 #-1.5

    stp_max = 1.2 #1.2
    stp_min = -0.8 #-0.8
    stp_r = 0.0 #0.0
    tau_stpW = 30.0 #30.0
    h_max = 0.8 #0.8
    h_min = 0.0 #0.0

    theta_R = 1.5 #1.5
    theta_W = 1.5 #1.5

    tau_stpup = 400.0 #400.0
    tau_stpdown = 400.0 #400.0
    tau_hup = 600.0 #600.0
    tau_hdown = 700.0 #700.0

    g_Roff2R = -2.0 #-2.0
    g_R2Roff = -5.0 #-5.0
    g_S2W = -2.0 #-2.0
    g_W2S = -2.0 #-2.0
    g_R2W = 0.1 #0.1
    g_W2Roff = 0 #0
    g_Roff2W = 0 #0

    tau_CR = 10.0 #10.0
    tau_CRoff = 1.0 #1.0
    tau_CW = 10.0 #10.0
    tau_CS = 10.0 #10.0

    def X_inf(c, X_max, beta, alpha): return (
        0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))

    def CX_inf(f, gamma): return np.tanh(f/gamma)
    def beta_X(y, k1_X, k2_X): return k2_X * (y - k1_X)
    # heavyside function
    def H(x): return 1 if x > 0 else 0

    # steady-state function for REM-ON popluation
    def R_inf(c): return X_inf(c, R_max, beta_R, alpha_R)
    # firing rate of REM (R) population
    dF_R = (R_inf(C_Roff * g_Roff2R) - F_R) / tau_R
    # steady state for neurotransmitter concentration:
    def CR_inf(x): return CX_inf(x, gamma_R)
    # dynamics for neurotransmitter
    dC_R = (CR_inf(F_R) - C_R) / tau_CR

    # homeostatic REM pressure
    if F_W > theta_W:
        dstp = (stp_r - stp) / tau_stpW
    else:
        dstp = (H(theta_R - F_R) * (stp_max - stp)) / tau_stpup + \
            (H(F_R - theta_R) * (stp_min - stp)) / tau_stpdown

    # REM-OFF population
    def beta_Roff(y): return beta_X(y, k1_Roff, k2_Roff)
    def Roff_inf(c): return X_inf(c, Roff_max, beta_Roff(stp), alpha_Roff)
    dF_Roff = (Roff_inf(C_R * g_R2Roff + C_W * g_W2Roff) - F_Roff) / tau_Roff
    def CRoff_inf(x): return CX_inf(x, gamma_Roff)
    dC_Roff = (CRoff_inf(F_Roff) - C_Roff) / tau_CRoff

    # Wake population
    def W_inf(c): return X_inf(c, W_max, beta_W, alpha_W)
    # firing rate of wake (W) population
    dF_W = (W_inf(C_S * g_S2W + C_R * g_R2W + C_Roff*g_Roff2W) - F_W) / tau_W
    # steady state for neurotransmitter concentration:
    def CW_inf(x): return CX_inf(x, gamma_W)
    # dynamics for neurotransmitter
    dC_W = (CW_inf(F_W) - C_W) / tau_CW

    # homeostatic sleep drive
    dh = (H(F_W - theta_W) * (h_max - h)) / tau_hup + \
        (H(theta_W - F_W) * (h_min - h)) / tau_hdown

    # Sleep population
    def beta_S(y): return beta_X(y, k1_S, k2_S)
    def S_inf(c): return X_inf(c, S_max, beta_S(h), alpha_S)
    # firing rate of non-REM (S) population
    dF_S = (S_inf(C_W * g_W2S) - F_S) / tau_S
    # steady state for neurotransmitter concentration:
    def CS_inf(x): return CX_inf(x, gamma_S)
    # dynamics for neurotransmitter
    dC_S = (CS_inf(F_S) - C_S) / tau_CS

    # [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X
    Y = [dF_R, dF_Roff, dF_S, dF_W, dC_R, dC_Roff, dC_S, dC_W, dstp, dh]

    return np.array(Y)


def mi_model_noise(X, group, t = 0):
    """Full deterministic MI model

    Arguments:
        X {numpy array} -- initial conditions of model parameters
        group {str} -- group to be optogenetically activated (see run_mi_model)

    Keyword Arguments:
        t {int} -- time in model (default: {0})

    Returns:
        {numpy array} -- time derivatives of model parameters
    """

    [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff,
        C_S, C_W, stp, h, delta, omega, sigma] = X

    #stimulation-group-dependent parameters
    sigma_R = 0
    sigma_Roff = 0
    sigma_W = 0
    sigma_S = 0

    if group == 'Ron':
        sigma_R = sigma
    elif group == 'Roff':
        sigma_Roff = sigma
    elif group =='W':
        sigma_W = sigma
    elif group == 'S':
        sigma_S = sigma

    # Parameters
    R_max = 5.0 # 5.0
    Roff_max = 5.0 # 5.0
    W_max = 5.50 # 5.50
    S_max = 5.0 # 5.0

    tau_Roff = 1.0 # 1.0
    tau_R = 1.0 # 1.0
    tau_W = 25.0 # 25.0
    tau_S = 10.0 # 10.0

    alpha_Roff = 1.5  # 1.5, 2
    alpha_R = 0.5 # 0.5
    alpha_W = 0.5 # 0.5
    beta_R = -0.5 # -0.5
    beta_W = -0.3 # -0.3
    alpha_S = 0.25 # 0.25

    gamma_R = 4.0 # 4.0
    gamma_Roff = 5.0 # 5.0
    gamma_W = 5.0 # 5.0
    gamma_S = 4.0 # 5.0

    k1_Roff = 0.8 # 0.8
    k2_Roff = 7.0 # 7.0
    k1_S = 0 # 0
    k2_S = -1.5 # -1.5

    stp_max = 1.2 # 1.2
    stp_min = -0.8 # -0.8
    stp_r = 0.0 # 0.0
    tau_stpW = 1000.0 # 1000.0
    h_max = 0.8 # 0.8
    h_min = 0.0 # 0.0
    omega_max = 0.1  # 0.02, 0.1
    omega_min = 0.00 # 0.00

    theta_R = 1.5 # 1.5
    theta_W = 1.5 # 1.5

    tau_stpup = 1000.0  # 400.0, 1000.0
    tau_stpdown = 1000.0  # 400.0, 1000.0
    tau_hup = 600.0 # 600.0
    tau_hdown = 2000.0 # 2000.0
    tau_omega = 20.0  # 10.0, 20.0
    tau_stim = 5.0  # 10.0, 5.0

    g_Roff2R = -2.0  # -2.0
    g_R2Roff = -5.0 #-5.0
    g_S2W = -2.0 #-2.0
    g_W2S = -2.0 # -2.0
    g_W2R = 0.0 # 0.0
    g_R2W = 0.0 # 0.0
    g_W2Roff = 0 # 0
    g_Roff2W = 0 # 0
    g_Roff2S = 0 # 0

    tau_CR = 10.0 # 10.0
    tau_CRf = 1 # 1
    tau_CRoff = 10.0  # 1.0, 10.0
    tau_CW = 10.0 # 10.0
    tau_CS = 10.0 # 10.0

    def X_inf(c, X_max, beta, alpha): return (
        0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))

    def CX_inf(f, gamma): return np.tanh(f/gamma)
    def beta_X(y, k1_X, k2_X): return k2_X * (y - k1_X)
    # heavyside function
    def H(x): return 1 if x > 0 else 0

    # steady-state function for REM-ON popluation
    def R_inf(c): return X_inf(c, R_max, beta_R, alpha_R)
    # firing rate of REM (R) population
    dF_R = (R_inf(C_Roff * g_Roff2R + C_W * g_W2R + sigma_R) - F_R) / tau_R
    # steady state for neurotransmitter concentration:
    def CR_inf(x): return CX_inf(x, gamma_R)
    # dynamics for neurotransmitter
    dC_R = (CR_inf(F_R) - C_R) / tau_CR
    dC_Rf = (CR_inf(F_R) - C_Rf) / tau_CRf

    # homeostatic REM pressure
    if F_W > theta_W:
        dstp = (stp_r - stp) / tau_stpW
        #dstp = 0
    else:
        dstp = (H(theta_R - F_R) * (stp_max - stp)) / tau_stpup + \
            (H(F_R - theta_R) * (stp_min - stp)) / tau_stpdown

    # update omega
    # parameter determining, how likely it is that a excitatory stimulus will happen during REM sleep
    if F_R > theta_R:
        domega = (omega_max - omega) / tau_omega
    else:
        domega = (omega_min - omega) / tau_omega

    # update delta
    ddelta = -delta / tau_stim

    # REM-OFF population
    def beta_Roff(y): return beta_X(y, k1_Roff, k2_Roff)
    def Roff_inf(c): return X_inf(c, Roff_max, beta_Roff(stp), alpha_Roff)
    dF_Roff = (Roff_inf(C_R * g_R2Roff + C_W * g_W2Roff + sigma_Roff) - F_Roff) / tau_Roff

    def CRoff_inf(x): return CX_inf(x, gamma_Roff)
    dC_Roff = (CRoff_inf(F_Roff) - C_Roff) / tau_CRoff

    # Wake population
    def W_inf(c): return X_inf(c, W_max, beta_W, alpha_W)
    # firing rate of REM (R) population
    dF_W = (W_inf(C_S * g_S2W + C_Rf * g_R2W +
                  C_Roff*g_Roff2W + delta + sigma_W) - F_W) / tau_W
    # steady state for neurotransmitter concentration:
    def CW_inf(x): return CX_inf(x, gamma_W)
    # dynamics for neurotransmitter
    dC_W = (CW_inf(F_W) - C_W) / tau_CW

    # homeostatic sleep drive
    dh = (H(F_W - theta_W) * (h_max - h)) / tau_hup + \
        (H(theta_W - F_W) * (h_min - h)) / tau_hdown

    # Sleep population
    def beta_S(y): return beta_X(y, k1_S, k2_S)
    def S_inf(c): return X_inf(c, S_max, beta_S(h), alpha_S)
    # firing rate of REM (R) population
    dF_S = (S_inf(C_W * g_W2S + C_Roff * g_Roff2S + sigma_S) - F_S) / tau_S
    # steady state for neurotransmitter concentration:
    def CS_inf(x): return CX_inf(x, gamma_S)
    # dynamics for neurotransmitter
    dC_S = (CS_inf(F_S) - C_S) / tau_CS

    dsigma = 0

    # [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X
    Y = [dF_R, dF_Roff, dF_S, dF_W, dC_R, dC_Rf, dC_Roff,
         dC_S, dC_W, dstp, dh, ddelta, domega, dsigma]
    return np.array(Y)


# %%

#simulates a population of neurons with optional optogenetic activation
def run_mi_model(X0, dt, hrs, group = 'None', sigma = 0, dur = 5*60, delay = 0, gap=15*60, gap_rand = False, gap_range = [1, 25], noise = False):
    """Simulates sleep from a model neuron population over time using the MI model with given initial conditions and optional optogenetic activation

    Arguments:
        X0 {numpy array} -- initial conditions of model parameters
        group {str} -- group to be optogenetically activated
        dt {float} -- timestep in seconds
        hrs {int or float (usually int)} -- simulation length in hours

    Keyword Arguments:
        sigma {int or float (usually int)} -- optogenetic activation value from laser data (default: 0)
        dur {int or float (usually int)} -- duration for laser stimulation (default: {5*60})
        delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours (default: {0})
        gap {int or float (usually int)} -- time between laser stimulation pulses, in seconds (default: {15*60})
        gap_rand {bool} -- randomizes gap duration if true (default: {False})
        gap_range {list} -- range for randomized gap duration to draw from, in minutes (default: {[1, 25]})
        noise {bool} -- adds noise to simulation if true (default: {False})

    Returns:
        {numpy array} -- simulated sleep data over full time interval with data at each timestep
    """
    #convert hrs and delay to seconds
    hrsInSec = hrs * 3600
    delayInSec = delay * 3600

    n = int(np.round(hrsInSec/dt))
    X = np.zeros((n, len(X0)))
    X[0, :] = np.array(X0)

    if gap_rand:
        gap = random.randint(gap_range[0], gap_range[-1])*60

    j = 0
    gap_time = 0
    for i in range(1, n):
        if i > ((delayInSec + j*dur + gap_time)/dt) and i <= ((delayInSec + (j+1)*dur + gap_time)/dt):
            X[i-1, -1] = sigma
        else:            
            if i == ((delayInSec + (j+1)*dur + gap_time + gap)/dt):
                j += 1
                gap_time += gap
                if gap_rand:
                    gap = random.randint(gap_range[0], gap_range[-1])*60
            X[i-1, -1] = 0        
        
        grad = mi_model_noise(X[i-1, :], group)
        X[i, :] = X[i-1, :] + grad * dt

        if noise:
            omega = X[i-1, -2]
            p_stim = 1 - np.exp(-omega * dt)
            p = np.random.binomial(1, p_stim)

            if p > 0:
                # print "motor noise"
                X[i, -3] += 3  # 10

    return X
# %%

# create hypnogram of data from the model showing mouse sleep states across the recording period
# plots showing firing rates, sleep pressures, and optogenetic activation period are included for analysis
def hypnogram(X, theta_R, theta_W, dt=0.05, p=0):
    """Converts simulated neuron data from a simulation of the MI model to an array of sleep states over time (to be plotted as a hypnogram)

    Arguments:
        X {numpy array} -- simulated neuron population data from MI model simulation
        theta_R {float} -- Firing rate threshold for Ron population to trigger REM state
        theta_W {float} -- Firing rate threshold for wake population to trigger wake state

    Keyword Arguments:
        dt {float} -- time step using in generation of simulation data from MI model (default: {0.05})
        p {int} -- plots hypnogram if and corresponding sleep data if equal to 1 (default: {0})

    Returns:
        {numpy array} -- array of sleep states over time
    """
    R = X[:, 0]
    W = X[:, 3]
    H = np.zeros((1, len(R)))

    idx_r = np.where(R > theta_R)[0]
    idx_w = np.where(W > theta_W)[0]
    H[0, :] = 3
    H[0, idx_r] = 1
    H[0, idx_w] = 2

    sns.set(font_scale=0.6)

    # make plot
    if p == 1:
        plt.figure()
        axes1 = plt.axes([0.1, 1.0, 0.8, 0.1])
        plt.imshow(H)
        plt.axis('tight')
        cmap = plt.cm.jet
        my_map = cmap.from_list(
            'brstate', [[0, 1, 1], [1, 0, 1], [0.8, 0.8, 0.8]], 3)
        tmp = axes1.imshow(H)
        tmp.set_cmap(my_map)
        axes1.axis('tight')
        tmp.axes.get_xaxis().set_visible(False)
        tmp.axes.get_yaxis().set_visible(False)

        t = np.arange(0, X.shape[0]*dt, dt)
        axes2 = plt.axes([0.1, 0.8, 0.8, 0.2])
        axes2.plot(t, X[:, [0, 1]])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('REM-on vs REM-off')

        axes3 = plt.axes([0.1, 0.6, 0.8, 0.2])
        axes3.plot(t, X[:, [2, 3]])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('Sleep vs Wake')

        axes4 = plt.axes([0.1, 0.4, 0.8, 0.2])
        axes4.plot(t, X[:, 9])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('REM pressure')

        axes5 = plt.axes([0.1, 0.2, 0.8, 0.2])
        axes5.plot(t, X[:, 10]*0.01)
        plt.xlim([t[0], t[-1]])
        plt.ylabel('Sleep pressure')

        axes6 = plt.axes([0.1, 0.0, 0.8, 0.2])
        axes6.plot(t, X[:, -1])
        plt.xlim(t[0], t[-1])
        plt.ylim(0, max(X[:, -1]) + 1)
        plt.ylabel('Sigma')

    return H

# %%
def avg_Ron_and_Roff_by_state(X, H):
    """Calculates and plots average firing rate (+/- one standard deviation) of REM-on and REM-off neurons during REM, wake, and NREM sleep states

    Arguments:
        X {numpy array} -- simulated neuron population data from MI model simulation
        H {numpy array} -- sleep state data from simulated data above

    Returns:
        [list] -- list of REM-on (indices 0-2) and REM-off (indices 3-5) average firing rates during each sleep state
    """

    #extract firing rates for Ron and Roff pop at each sleep states
    Ron_rem = X[(H==1)[0], 0]
    Ron_wake = X[(H==2)[0], 0]
    Ron_nrem = X[(H==3)[0], 0]
    Roff_rem = X[(H==1)[0], 1]
    Roff_wake = X[(H==2)[0], 1]
    Roff_nrem = X[(H==3)[0], 1]

    #put average values into array
    avg_by_state = np.zeros(6)
    avg_by_state[0] = np.mean(Ron_rem)
    avg_by_state[1] = np.mean(Ron_wake)
    avg_by_state[2] = np.mean(Ron_nrem)
    avg_by_state[3] = np.mean(Roff_rem)
    avg_by_state[4] = np.mean(Roff_wake)
    avg_by_state[5] = np.mean(Roff_nrem)

    #save standard deviation for error in plot
    Ron_std = np.zeros(3)
    Roff_std = np.zeros(3)
    Ron_std[0] = stats.sem(Ron_rem)
    Ron_std[1] = stats.sem(Ron_wake)
    Ron_std[2] = stats.sem(Ron_nrem)
    Roff_std[0] = stats.sem(Roff_rem)
    Roff_std[1] = stats.sem(Roff_wake)
    Roff_std[2] = stats.sem(Roff_nrem)

    #plot data together to graph
    Ron_pop = [avg_by_state[0], avg_by_state[1], avg_by_state[2]]
    Roff_pop = [avg_by_state[3], avg_by_state[4], avg_by_state[5]]

    #plot data in bar plot
    labels = ['REM', 'Wake', 'NREM']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    Ron_bars = ax.bar(x - width/2, Ron_pop, width, yerr = Ron_std, label = 'REM-On Activity')
    Roff_bars = ax.bar(x + width/2, Roff_pop, width, yerr = Roff_std, label = 'REM-Off Activity')
    ax.set_ylabel('Average Activity/Firing Rate (Hz)')
    ax.set_title('Neuron Activity During Each Sleep State')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

    #print results as mean +/- std
    print(f'Average REM-on activity during REM is {avg_by_state[0]} +/- {Ron_std[0]}')
    print(f'Average REM-on activity during Wake is {avg_by_state[1]} +/- {Ron_std[1]}')
    print(f'Average REM-on activity during NREM is {avg_by_state[2]} +/- {Ron_std[2]}')
    print(f'Average REM-off activity during REM is {avg_by_state[3]} +/- {Roff_std[0]}')
    print(f'Average REM-off activity during Wake is {avg_by_state[4]} +/- {Roff_std[1]}')
    print(f'Average REM-off activity during NREM is {avg_by_state[5]} +/- {Roff_std[2]}')


# %%
def inter_REM(H):
    """Plots association between REM durations and following inter-REM durations (NREM only)

    Arguments:
        H {numpy array} -- sleep state data from simulated data

    Returns:
        [lists] -- x (REM durations) and y (subsequent inter-REM durations) data of plot
    """

    #define variables for loop
    REM_counter = 0
    inter_counter = 0
    first_marker = 0
    REM_durations = []
    inter_durations = []

    #loop through data: determine current state, save length of period if state changes 
    # from REM or from inter-REM
    for i in range(len(H[0]) - 1):
        if (H[0][i] == 1):
            first_marker = 1
            REM_counter += 1
            if (H[0][i+1] != 1):
                REM_durations.append(REM_counter * 0.05)
                REM_counter = 0
        elif (H[0][i] != 1 & first_marker != 0):
            if (H[0][i] == 3):
                inter_counter += 1
            if (H[0][i+1] == 1):
                inter_durations.append(inter_counter * 0.05)
                inter_counter = 0

    #if last position of H is REM, marking the end of an unsaved inter-REM period
    if (H[0][-1] == 1 & inter_counter != 0):
        inter_durations.append(inter_counter * 0.05)
    
    #if the end of the last inter-REM period was not saved due to the full period 
    # not being saved, delete the last REM duration (corresponding to the incomplete)
    #inter-REM
    if (len(REM_durations) > len(inter_durations)):
        del REM_durations[-1]

    #trendline and r^2
    m, b, r, p, std = stats.linregress(REM_durations, inter_durations)

    #plot data in scatter plot
    plt.figure()
    plt.scatter(REM_durations, inter_durations)
    plt.plot(REM_durations, np.multiply(m,REM_durations) + b, color = 'red')
    plt.xlabel('REM Duration (s)')
    plt.ylabel('NREM During Inter-REM Duration (s)')
    plt.title('REM Duration vs Inter-REM Duration')
    plt.text(max(REM_durations) - 25, m * max(REM_durations) + (b + 50), f'R^2: {round(r**2, 2)}', fontsize = 12)
    plt.show()


# %%

def REM_pressure_laser_onset(X, H, dur, theta_R):
    """Plots REM pressure association with the delay between the end of a REM period and the next laser onset

    Arguments:
        X {numpy array} -- simulated neuron population data from MI model simulation (with optogenetic activation)
        H {numpy array} -- sleep state data from simulated data above
        theta_R {float} -- Firing rate threshold for Ron population to trigger REM

    Returns:
        [lists] -- x (time between end of REM and laser stimulation) and y (REM pressure) data of plot
    """

    #find delay between end of REM period and next laser onset, grab REM pressure at laser onset
    onset = []
    pressure = []
    REM_induced_onset = []
    REM_induced_dur = []
    record_time = False
    delay = 0
    for i in range(1, len(H[0])):
        #start recording time if REM period ends
        if H[0][i-1] == 1 and H[0][i] != 1:
            record_time = True
        #if time is being recorded, add timestep for each iteration
        if record_time:
            delay += 0.05
            #stop recording time and grab data upon laser onset
            if X[i-1, -1] == 0 and X[i, -1] != 0:
                record_time = False
                onset.append(delay)
                delay = 0
                pressure.append(X[i, -5])
                if H[0][i] == 1:
                    REM_induced_onset.append(1)
                else:
                    REM_induced_onset.append(0)
                REM_in_dur = False
                for j in range(int(dur / 0.05)):
                    if (i + j) >= len(H[0]):
                        continue
                    if H[0][i + j] == 1:
                        REM_in_dur = True
                if REM_in_dur:
                    REM_induced_dur.append(1)
                else:
                    REM_induced_dur.append(0)

    for i in range(len(onset)):
        if i >= len(onset):
            continue
        if onset[i] >= 4000:
            del onset[i]
            del pressure[i]
            del REM_induced_onset[i]
            del REM_induced_dur[i]
                    

    #trendline and r^2
    m, b, r, p, std = stats.linregress(onset, pressure)
                
    #plot data in scatter plot
    plt.figure()
    plt.scatter(onset, pressure)
    plt.plot(onset, np.multiply(m,onset) + b, color = 'red')
    plt.xlabel('Time Between End of REM Period and Next Laser Onset (s)')
    plt.ylabel('REM Pressure')
    plt.title('REM Pressure Upon Laser Stimulation After REM Period')
    plt.text(max(onset), m * max(onset) + b, f'R^2: {round(r**2, 2)}', fontsize = 12)
    plt.show()

    plt.figure()
    plt.scatter(onset, REM_induced_onset)
    plt.ylim([-0.5, 1.5])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.xlabel('Time Between End of REM Period and Next Laser Onset (s)')
    plt.ylabel('REM Successfully Induced?')
    plt.title('Ability to Laser-Induce REM After End of Previous REM Period on Laser Onset')
    plt.show()     

    plt.figure()
    plt.scatter(onset, REM_induced_dur)
    plt.ylim([-0.5, 1.5])
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.xlabel('Time Between End of REM Period and Next Laser Onset (s)')
    plt.ylabel('REM Successfully Induced?')
    plt.title('Ability to Laser-Induce REM After End of Previous REM Period Over Laser Duration')
    plt.show()      


# %%

def laser_trig_percents(data_coll, hypno_coll, dt, pre_post, dur, ci = 95):
    """Plots 

    Arguments:
        data_coll {list of numpy arrays} -- list containing
        hypno_coll {[type]} -- [description]
        dt {[type]} -- [description]
        pre_post {[type]} -- [description]
        dur {[type]} -- [description]

    Returns:
        [type] -- [description]
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

    return brainstates_df, df
# %% 

def Roff_FR_before_REM (X, H, dt):

    pre_REM = []
    period = int(60 / dt)
    time_vec = np.linspace(-period*dt, (period*dt)/2, int(1.5*period + 1))

    for i in range(len(H[0]) - 1):
        if H[0][i] != 1 and H[0][i+1] == 1:
            if i - period < 0 or (i + (period/2) + 1) > len(H[0]):
                continue
            pre_REM.append(X[i - period: i + int(period/2) + 1, 1])

    Roff_avg_FRs = []
    for i in range(len(pre_REM[0])):
        temp = []
        for j in range(len(pre_REM)):
            temp.append(pre_REM[j][i])
        Roff_avg_FRs.append(np.mean(temp))

    plt.figure()
    plt.plot(time_vec, Roff_avg_FRs)
    plt.vlines(0, min(Roff_avg_FRs), max(Roff_avg_FRs), linestyles='dashed', color='gray')
    plt.xlabel('Time (t=0 corresponds to timepoint just before REM) (s)')
    plt.ylabel('Average REM-Off Firing Rate (Hz)')
    plt.title('REM-off Firing Rate Before and During REM Sleep')
    plt.show()
    
    return Roff_avg_FRs, time_vec
# %%

def microarousal_count_helper(H, ma_length, dt, pos):
    for i in range(int(ma_length / dt) + 1):
        if H[0][i] != 2:
            return 1
        return 0
    

def microarousal_count(H, ma_length, dt):
    ma_counter = 0
    for i in range(len(H[0]) - 1):
        if H[0][i] == 2:
            ma_counter += microarousal_count_helper(H, ma_length, dt, i)
    return ma_counter

# %%

#From Dr. Weber - https://github.com/tortugar/Lab/blob/master/Photometry/pyphi.py

def downsample_vec(x, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]

    return x_down / nbin

def downsample_mx(X, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive
    bin by their mean
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down * nbin, :]
    X_down = np.zeros((n_down, X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin):
        idx = list(range(i, int(n_down * nbin), int(nbin)))
        X_down += X[idx, :]

    return X_down / nbin

def upsample_mx(x, nbin):
    """
    if x is a vector:
        upsample the given vector $x by duplicating each element $nbin times
    if x is a 2d array:
        upsample each matrix by duplication each row $nbin times
    """
    if nbin == 1:
        return x

    nelem = x.shape[0]
    if x.ndim == 1:
        y = np.zeros((nelem * nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    else:
        y = np.zeros((nelem * nbin, x.shape[1]))
        for k in range(nbin):
            y[k::nbin, :] = x

    return y

def time_morph(X, nstates):
    """
    upsample vector or matrix X to nstates states
    :param X, vector or matrix; if matrix, the rows are upsampled.
    :param nstates, number of elements or rows of returned vector or matrix
    I want to upsample m by a factor of x such that
    x*m % nstates == 0,
    a simple soluation is to set x = nstates
    then nstates * m / nstates = m.
    so upsampling X by a factor of nstates and then downsampling by a factor
    of m is a simple solution...
    """
    m = X.shape[0]
    A = upsample_mx(X, nstates)
    # now we have m * nstates rows
    if X.ndim == 1:
        Y = downsample_vec(A, int((m * nstates) / nstates))
    else:
        Y = downsample_mx(A, int((m * nstates) / nstates))
    # now we have m rows as requested
    return Y

# %%

def Roff_FR_inter_REM_norm(X, H, dt):

    record_inter = False
    record_rem = False
    inter_rem_periods = []
    rem_periods = []
    curr_inter = []
    curr_rem = []
    for i in range(1, len(H[0])):
        #store values for ended REM period and begin recording
        if H[0][i-1] == 1 and H[0][i] != 1 and record_rem:
            record_rem = False
            rem_periods.append(curr_rem)
            curr_rem = []
        #stop recording values and save data to list on REM transition
        elif H[0][i-1] != 1 and H[0][i] == 1 and record_inter:
            record_inter = False
            inter_rem_periods.append(curr_inter)
            curr_inter = []

        #begin recording desired states on desired transition
        if H[0][i-1] != 1 and H[0][i] == 1:
            record_rem = True
        elif H[0][i-1] == 1 and H[0][i] != 1:
            record_inter = True
        
        #record REM-off FR for current state
        if record_inter:
            curr_inter.append(X[i,1])
        elif record_rem:
            curr_rem.append(X[i,1])

    #if same amoung of inter-rem and rem recorded, delete last inter-rem recording to preserve
    #rem->inter-rem->rem pattern (keep length of rem_periods longer than inter-rem periods by 1)
    if len(inter_rem_periods) == len(rem_periods):
        del inter_rem_periods[-1]
    
    #normalize inter-rem periods
    nstates_inter = 20
    norm_FRs_inter = []
    for entry in inter_rem_periods:
        to_np = np.array(entry)
        # print(to_np.shape[0])
        norm_FRs_inter.append(time_morph(to_np, nstates_inter))
    norm_FRs_inter = np.array(norm_FRs_inter)

    #normalize rem periods
    nstates_rem = 10
    norm_FRs_rem = []
    for entry in rem_periods:
        to_np = np.array(entry)
        norm_FRs_rem.append(time_morph(to_np, nstates_rem))
    norm_FRs_rem = np.array(norm_FRs_rem)

    #combine rem and inter-REM into single slice of FR data
    FR_slices = []
    for i in range(len(norm_FRs_inter)):
        curr = []
        curr.extend(norm_FRs_rem[i]) #rem before inter-rem period
        curr.extend(norm_FRs_inter[i]) #inter-rem period
        curr.extend(norm_FRs_rem[i + 1]) #rem after inter-rem period (same as rem 
        #before next inter-rem period)
        FR_slices.append(curr)


    #average Roff firing rates at each timepoint durin inter-REM
    avg_FRs = []
    for i in range(len(FR_slices[0])):
        FR_sum = 0
        count = 0
        #gather firing rates by column
        for j in range(len(FR_slices)):
            FR_sum += FR_slices[j][i]
            count += 1
        #compute average column firing rate
        avg_FRs.append(FR_sum / count)


    plt.figure()
    plt.plot(avg_FRs)
    plt.vlines(9, min(avg_FRs), max(avg_FRs), linestyles='dashed', color='gray')
    plt.vlines(29, min(avg_FRs), max(avg_FRs), linestyles='dashed', color='gray')
    plt.xlabel('Norm. Time')
    plt.ylabel('Average REM-Off Firing Rate (Hz)')
    plt.title('REM-off Firing Rate During Normalized Inter-REM Periods')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.show()

    return avg_FRs
    


# %%
# create hypnogram for mouse with initial parameters X0 specified below

if __name__ == "__main__":
    
    #[F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, delta, omega, sigma] = X
    #parameters for model simulation
    dur = 5*60 #seconds
    gap = 15*60 #seconds
    sigma = 0
    delay = 2 #hrs
    hrs = 8 #hrs
    dt = 0.05 #seconds (timestep)
    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    group = 'Roff'
    noise = True
    #simulate mouse with uniform gaps between laser stimulation
    X = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, noise = noise)
    #simulate mouse with random laser stimulation gaps (drawn from input range)
    # Y = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, gap_rand = True, noise = noise)

    #parameters
    theta_R = 1.5
    theta_W = 1.5

    # print('##### Uniform activation gaps #####')

    #create hypnogram from data
    H_x = hypnogram(X, theta_R, theta_W, p=1)
    #calculate average Ron and Roff firing rates during each sleep state
    avg_Ron_and_Roff_by_state(X, H_x)
    #plot time relationship between REM period length and subsequent inter-REM length
    # inter_REM(H_x)
    #plot activity of REM-off neurons before REM period
    # a, b = Roff_FR_before_REM(X, H_x, dt)
    # ma_length = 20.0
    # num_ma = microarousal_count(H_x, ma_length, dt)
    # print(f'Number of microarousals in {hrs} hours of sleep with noise is {num_ma}')
    # FR_slices = Roff_FR_inter_REM_norm(X, H_x, dt)

    #as above but gaps between laser stimulations are randomized
    # print('##### Randomized activation gaps #####')

    # H_y = hypnogram(Y, theta_R, theta_W, p=1)
    # # avg_Ron_and_Roff_by_state(Y, H_y)
    # # inter_REM(H_y)
    # #plot REM pressures and ability to induce a REM period as a function of time after the end of the previous REM period
    # REM_pressure_laser_onset(Y, H_y, dur, theta_R)

    # print('##### Laser Triggered Brainstates #####')

    # #parameters
    # pre_post = 5*60
    # gap_range = [5, 25]
    # noise = True
    # ci = None

    # #simulate sleep for 4 mice
    # Z = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, gap_rand = True, gap_range = gap_range, noise = noise)
    # A = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, gap_rand = True, gap_range = gap_range, noise = noise)
    # B = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, gap_rand = True, gap_range = gap_range, noise = noise)
    # C = run_mi_model(X0, dt, hrs, group = group, sigma = sigma, dur = dur, delay = delay, gap_rand = True, gap_range = gap_range, noise = noise)

    # #create hypnograms for the 4 mice
    # H_z = hypnogram(Z, theta_R, theta_W, p=0)
    # H_a = hypnogram(A, theta_R, theta_W, p=0)
    # H_b = hypnogram(B, theta_R, theta_W, p=0)
    # H_c = hypnogram(C, theta_R, theta_W, p=0)

    # #combine sleep data and hypnograms into larger lists
    # data_coll = [Z, A, B, C]
    # hypno_coll = [H_z, H_a, H_b, H_c]

    # #calculate laser triggered sleep state percentages and show figure
    # # total_df, iter_df = laser_trig_percents(data_coll, hypno_coll, dt, pre_post, dur, ci)
    # brainstates = laser_trig_percents(data_coll, hypno_coll, dt, pre_post, dur, ci)

# %%
