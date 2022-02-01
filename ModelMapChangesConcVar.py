"""
*** Most recent and best performing model ***
Flip-Flop Model with architectural changes implemented to F_W -> stp inhibition (stp constant during wake, removal of inhibition)

@author: Zachary Spalding
"""

import random
import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib import cm
from scipy import signal, stats
from sklearn.mixture import GaussianMixture
from pingouin import compute_bootci
from random import randint

import sleepy

FONT_SIZE=14


class ModelMapChangesConcVar():
    """Object for Flip-Flop Model with architectural changes implemented to F_W -> stp inhibition (stp constant during wake, removal of inhibition)
    """
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

    tau_stpup = 1650.0  # 400.0, 1000.0
    tau_stpdown = 1650.0  # 400.0, 1000.0
    tau_hup = 600.0 # 600.0
    tau_hdown = 2000.0 # 2000.0
    tau_omega = 20.0  # 10.0, 20.0
    tau_stim = 5.0  # 10.0, 5.0

    g_Roff2R = -7.0  # -2.0
    g_R2Roff = -5.0 #-5.0
    g_S2W = -2.0 #-2.0
    g_W2S = -2.0 # -2.0
    g_W2R = 0.0 # 0.0
    g_R2W = 0.0 # 0.0
    g_W2Roff = 0 # 5.0
    g_Roff2W = 0 # 0
    g_Roff2S = 0 # 0
    g_W2stp = 0.15 # 0.15

    tau_CR = 10.0 # 10.0
    tau_CRf = 1.0 # 1.0
    tau_CRoff = 10.0  # 1.0, 10.0
    tau_CW = 10.0 # 10.0
    tau_CS = 10.0 # 10.0

    delta_update = 3.0 # 10.0, 3.0
    delta2W = 0.6
    delta2Roff = 1

    paramNameList = ['R_max', 'Roff_max', 'W_max', 'S_max', 'tau_Roff', 'tau_R', 
                    'tau_W', 'tau_S', 'alpha_Roff', 'alpha_R', 'alpha_W', 'beta_R',
                    'beta_W', 'alpha_S', 'gamma_R', 'gamma_Roff', 'gamma_W', 'gamma_S',
                    'k1_Roff', 'k2_Roff', 'k1_S', 'k2_S', 'stp_max', 'stp_min', 'stp_r',
                    'tau_stpW', 'h_max', 'h_min', 'omega_max', 'omega_min', 'theta_R',
                    'theta_W', 'tau_stpup', 'tau_stpdown', 'tau_hup', 'tau_hdown', 'tau_omega',
                    'tau_stim', 'g_Roff2R', 'g_R2Roff', 'g_S2W', 'g_W2S', 'g_W2R', 'g_R2W', 
                    'g_W2Roff', 'g_Roff2W', 'g_Roff2S', 'tau_CR', 'tau_CRf', 'tau_CRoff',
                    'tau_CW', 'tau_CS', 'delta_update']

    paramValList = [R_max, Roff_max, W_max, S_max, tau_Roff, tau_R, 
                    tau_W, tau_S, alpha_Roff, alpha_R, alpha_W, beta_R,
                    beta_W, alpha_S, gamma_R, gamma_Roff, gamma_W, gamma_S,
                    k1_Roff, k2_Roff, k1_S, k2_S, stp_max, stp_min, stp_r,
                    tau_stpW, h_max, h_min, omega_max, omega_min, theta_R,
                    theta_W, tau_stpup, tau_stpdown, tau_hup, tau_hdown, tau_omega,
                    tau_stim, g_Roff2R, g_R2Roff, g_S2W, g_W2S, g_W2R, g_R2W, 
                    g_W2Roff, g_Roff2W, g_Roff2S, tau_CR, tau_CRf, tau_CRoff,
                    tau_CW, tau_CS, delta_update]

    paramDict = {}
    for i in range(len(paramNameList)):
        paramDict[paramNameList[i]] = paramValList[i]
         
    def __init__(self, X0, dt):
        """Initialization of ModelMapChanges object

        Args:
            X0 (list): initial conditions for model
            self.dt (float): timestep for simulation
        """
        self.X0 = np.array(X0)
        self.dt = dt
        self.X = []
        self.H = []

        sns.set(font_scale=0.15)

    def update_param_dict(self):
        """Updates parameter dictionary, mostly used to fill csv file correctly in ModelScripts.py and FileHandling.py
        """
        self.paramValList = [self.R_max, self.Roff_max, self.W_max, self.S_max, self.tau_Roff, self.tau_R, 
                    self.tau_W, self.tau_S, self.alpha_Roff, self.alpha_R, self.alpha_W, self.beta_R,
                    self.beta_W, self.alpha_S, self.gamma_R, self.gamma_Roff, self.gamma_W, self.gamma_S,
                    self.k1_Roff, self.k2_Roff, self.k1_S, self.k2_S, self.stp_max, self.stp_min, self.stp_r,
                    self.tau_stpW, self.h_max, self.h_min, self.omega_max, self.omega_min, self.theta_R,
                    self.theta_W, self.tau_stpup, self.tau_stpdown, self.tau_hup, self.tau_hdown, self.tau_omega,
                    self.tau_stim, self.g_Roff2R, self.g_R2Roff, self.g_S2W, self.g_W2S, self.g_W2R, self.g_R2W, 
                    self.g_W2Roff, self.g_Roff2W, self.g_Roff2S, self.tau_CR, self.tau_CRf, self.tau_CRoff,
                    self.tau_CW, self.tau_CS, self.delta_update]

        self.paramDict = {}
        for i in range(len(self.paramNameList)):
            self.paramDict[self.paramNameList[i]] = self.paramValList[i]

    def run_mi_model(self, hrs, group='None', sigma=0, dur=5*60, delay=0, gap=15*60, gap_rand=False, gap_range=[1, 25], noise=False, refractory_activation=False):
        """Simulates sleep from a model neuron population over time using the MI model with given initial conditions and optional optogenetic activation

        Arguments:
            hrs {int or float (usually int)} -- simulation length in hours

        Keyword Arguments:
            group {str} -- Neuron population to receive optogenetic activation (default: 'None')
            sigma {int or float (usually int)} -- optogenetic activation value from laser data (default: 0)
            dur {int or float (usually int)} -- duration for laser stimulation (default: {5*60})
            delay {int or float (usually int)} -- delay from beginning for which laser stimulation should not occur, in hours (default: {0})
            gap {int or float (usually int)} -- time between laser stimulation pulses, in seconds (default: {15*60})
            gap_rand {bool} -- randomizes gap duration if true (default: {False})
            gap_range {list} -- range for randomized gap duration to draw from, in minutes (default: {[1, 25]})
            noise {bool} -- adds noise to simulation if true (default: {False})

        Returns:
            None - updates simulation data of model object
        """

        def mi_model_noise(simX, group, t = 0):
            """Full deterministic MI model

            Arguments:
                group {str} -- group to be optogenetically activated (see run_mi_model)

            Keyword Arguments:
                t {int} -- time in model (default: {0})

            Returns:
                {numpy array} -- time derivatives of model parameters
            """

            [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff,
                C_S, C_W, stp, h, zeta_Ron, zeta_Roff, zeta_S, zeta_W, delta, omega, sigma] = simX

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

            def X_inf(c, X_max, beta, alpha): return (
                0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))

            def CX_inf(f, gamma): return np.tanh(f/gamma)
            def beta_X(y, k1_X, k2_X): return k2_X * (y - k1_X)
            # heavyside function
            def H(x): return 1 if x > 0 else 0

            # steady-state function for REM-ON popluation
            def R_inf(c): return X_inf(c, self.R_max, self.beta_R, self.alpha_R)
            # firing rate of REM (R) population
            dF_R = (R_inf(C_Roff * self.g_Roff2R + C_W * self.g_W2R + sigma_R) - F_R) / self.tau_R
            # steady state for neurotransmitter concentration:
            def CR_inf(x): return CX_inf(x, self.gamma_R)
            # dynamics for neurotransmitter
            dC_R = (zeta_Ron * CR_inf(F_R) - C_R) / self.tau_CR
            dC_Rf = (CR_inf(F_R) - C_Rf) / self.tau_CRf

            # homeostatic REM pressure
            if F_W > self.theta_W:
                dstp = self.g_W2stp * (self.stp_r - stp) / self.tau_stpW # stp decreases during wake
                # dstp = 0 # stp constant during wake
            else:
                dstp = (H(self.theta_R - F_R) * (self.stp_max - stp)) / self.tau_stpup + \
                    (H(F_R - self.theta_R) * (self.stp_min - stp)) / self.tau_stpdown

            # update omega
            # parameter determining, how likely it is that a excitatory stimulus will happen during REM sleep
            if F_R > self.theta_R:
                domega = (self.omega_max - omega) / self.tau_omega
            else:
                domega = (self.omega_min - omega) / self.tau_omega

            # update delta
            ddelta = -delta / self.tau_stim

            # REM-OFF population
            def beta_Roff(y): return beta_X(y, self.k1_Roff, self.k2_Roff)
            def Roff_inf(c): return X_inf(c, self.Roff_max, beta_Roff(stp), self.alpha_Roff)
            dF_Roff = (Roff_inf(C_R * self.g_R2Roff + C_W * self.g_W2Roff + \
                self.delta2Roff * delta + sigma_Roff) - F_Roff) / self.tau_Roff

            def CRoff_inf(x): return CX_inf(x, self.gamma_Roff)
            dC_Roff = (zeta_Roff * CRoff_inf(F_Roff) - C_Roff) / self.tau_CRoff

            # Wake population
            def W_inf(c): return X_inf(c, self.W_max, self.beta_W, self.alpha_W)
            # firing rate of REM (R) population
            dF_W = (W_inf(C_S * self.g_S2W + C_Rf * self.g_R2W +
                        C_Roff * self.g_Roff2W + self.delta2W * delta + sigma_W) - F_W) / self.tau_W
            # steady state for neurotransmitter concentration:
            def CW_inf(x): return CX_inf(x, self.gamma_W)
            # dynamics for neurotransmitter
            dC_W = (zeta_W * CW_inf(F_W) - C_W) / self.tau_CW

            # homeostatic sleep drive
            dh = (H(F_W - self.theta_W) * (self.h_max - h)) / self.tau_hup + \
                (H(self.theta_W - F_W) * (self.h_min - h)) / self.tau_hdown

            # Sleep population
            def beta_S(y): return beta_X(y, self.k1_S,self. k2_S)
            def S_inf(c): return X_inf(c, self.S_max, beta_S(h), self.alpha_S)
            # firing rate of REM (R) population
            dF_S = (S_inf(C_W * self.g_W2S + C_Roff * self.g_Roff2S + sigma_S) - F_S) / self.tau_S
            # steady state for neurotransmitter concentration:
            def CS_inf(x): return CX_inf(x, self.gamma_S)
            # dynamics for neurotransmitter
            dC_S = (zeta_S * CS_inf(F_S) - C_S) / self.tau_CS

            dsigma = 0
            dzeta_Ron = 0
            dzeta_Roff = 0
            dzeta_S = 0
            dzeta_W = 0

            # [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X
            Y = [dF_R, dF_Roff, dF_S, dF_W, dC_R, dC_Rf, dC_Roff,
                dC_S, dC_W, dstp, dh, dzeta_Ron, dzeta_Roff, dzeta_S, dzeta_W, ddelta, domega, dsigma]
            return np.array(Y)

        #convert hrs and delay to seconds
        hrsInSec = hrs * 3600
        delayInSec = delay * 3600

        n = int(np.round(hrsInSec/self.dt))
        simX = np.zeros((n, len(self.X0)))
        simX[0, :] = np.array(self.X0)

        if gap_rand:
            gap = random.randint(gap_range[0], gap_range[-1])*60

        j = 0
        gap_time = 0
        refract_dur_counter = 0
        refract_cooldown = 0
        for i in range(1, n):
            # optogenetic activation at refractory periods
            if refractory_activation:
                if i > 1: # for proper bounds
                    if refract_cooldown > 0:
                        # set opto activation to 0
                        simX[i-1, -1] = 0

                        # decrement gap/cooldown
                        refract_cooldown -= 1
                    # maintain optogenetic activation if onset of activation was previously determined
                    elif refract_dur_counter > 0:
                        # apply optogenetic activation
                        simX[i-1, -1] = sigma

                        # decrement dur counter
                        refract_dur_counter -= 1

                        if refract_dur_counter == 0:
                            refract_cooldown = gap / self.dt
                    else:
                        # check for transition out of REM
                        if simX[i-2, 0] > self.theta_R and simX[i-1, 0] < self.theta_R:
                            # apply optogenetic activation
                            simX[i-1, -1] = sigma
                            # set counter to maintain duration for future periods (decremented by 1 for current period)
                            refract_dur_counter = dur / self.dt - 1
                        else:
                            simX[i-1, -1] = 0

            else: # regular optogenetic activation
                if i > ((delayInSec + j*dur + gap_time)/self.dt) and i <= ((delayInSec + (j+1)*dur + gap_time)/self.dt):
                    simX[i-1, -1] = sigma
                else:            
                    if i == ((delayInSec + (j+1)*dur + gap_time + gap)/self.dt):
                        j += 1
                        gap_time += gap
                        if gap_rand:
                            gap = random.randint(gap_range[0], gap_range[-1])*60
                    simX[i-1, -1] = 0        
            
            grad = mi_model_noise(simX[i-1, :], group)
            simX[i, :] = simX[i-1, :] + grad * self.dt

            if noise:
                #delta updates
                omega = simX[i-1, -2]

                # binomial delta
                p_stim = 1 - np.exp(-omega * self.dt)
                p = np.random.binomial(1, p_stim)

                # # poisson delta
                # p = np.random.poisson(lam = omega * self.dt)
                
                if p > 0:
                    # print "motor noise"
                    simX[i, -3] += self.delta_update  # 10, 3

                #zeta updates
                lambdaR = 0.01
                lambdaWS = 0.02

                pZetaRon = np.random.poisson(lam = lambdaR)
                pZetaRoff = np.random.poisson(lam = lambdaR)
                pZetaS = np.random.poisson(lam = lambdaWS)
                pZetaW = np.random.poisson(lam = lambdaWS)
                if pZetaRon > 0:
                    simX[i, -7] = np.random.uniform(low=0.5, high=1.5)
                if pZetaRoff > 0:
                    simX[i, -6] = np.random.uniform(low=0.5, high=1.5)
                if pZetaS > 0:
                    simX[i, -5] = np.random.uniform(low=0.5, high=1.5)
                if pZetaW > 0:
                    simX[i, -4] = np.random.uniform(low=0.5, high=1.5)

        self.X = simX
        # print("----- Simulated New Data (New Model) -----")

    def hypnogram(self, p=0):
        """Converts simulated neuron data from a simulation of the MI model to an array of sleep states over time (to be plotted as a hypnogram)

        Keyword Arguments:
            p {int} -- plots hypnogram and corresponding sleep data if equal to 1 (default: {0})

        Returns:
            None - updates hypnogram (H) of model object
        """
        R = self.X[:, 0]
        W = self.X[:, 3]
        simH = np.zeros((1, len(R)))

        idx_r = np.where(R > self.theta_R)[0]
        idx_w = np.where(W > self.theta_W)[0]
        simH[0, :] = 3
        simH[0, idx_r] = 1
        simH[0, idx_w] = 2

        self.H = simH

        sns.set(font_scale=0.6)

        # make plot
        if p == 1:
            plt.figure()
            axes1 = plt.axes([0.1, 1.0, 0.8, 0.1])
            plt.imshow(simH)
            plt.axis('tight')
            cmap = plt.cm.jet
            my_map = cmap.from_list(
                'brstate', [[0, 1, 1], [1, 0, 1], [0.8, 0.8, 0.8]], 3)
            tmp = axes1.imshow(simH)
            tmp.set_cmap(my_map)
            axes1.axis('tight')
            tmp.axes.get_xaxis().set_visible(False)
            tmp.axes.get_yaxis().set_visible(False)

            t = np.arange(0, self.X.shape[0]*self.dt, self.dt)
            axes2 = plt.axes([0.1, 0.8, 0.8, 0.2])
            axes2.plot(t, self.X[:, [0, 1]])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('REM-on vs REM-off')

            axes3 = plt.axes([0.1, 0.6, 0.8, 0.2])
            axes3.plot(t, self.X[:, [2, 3]])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('Sleep vs Wake')

            axes4 = plt.axes([0.1, 0.4, 0.8, 0.2])
            axes4.plot(t, self.X[:, 9])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('REM pressure')

            axes5 = plt.axes([0.1, 0.2, 0.8, 0.2])
            axes5.plot(t, self.X[:, 10]*0.01)
            plt.xlim([t[0], t[-1]])
            plt.ylabel('Sleep pressure')

            axes6 = plt.axes([0.1, 0.0, 0.8, 0.2])
            axes6.plot(t, self.X[:, -3])
            plt.xlim(t[0], t[-1])
            plt.ylim(0, max(self.X[:, -3]) + 1)
            plt.ylabel('Delta')
            plt.savefig('figures/MCCV_hypno.pdf', bbox_inches = "tight", dpi = 100)

            plt.show()

    def hypnogram_fig1(self, p=0, p_zoom=0, save=False, filename='fig1_hypno'):
        """Converts simulated neuron data from a simulation of the MI model to an array of sleep states over time (to be plotted as a hypnogram)
        Plotting is modified here, in comparison to the regular hypnogram function, to show the desired aspects for fig 1

        Keyword Arguments:
            p {int} -- plots hypnogram and corresponding sleep data if equal to 1 (default: {0})

        Returns:
            None - updates hypnogram (H) of model object
        """
        R = self.X[:, 0]
        W = self.X[:, 3]
        simH = np.zeros((1, len(R)))

        idx_r = np.where(R > self.theta_R)[0]
        idx_w = np.where(W > self.theta_W)[0]
        simH[0, :] = 3
        simH[0, idx_r] = 1
        simH[0, idx_w] = 2

        #cut off first 2 hours of sleep to visualize model in its stable regime
        start = int(2 * 3600 / self.dt)
        simH = np.expand_dims(simH[0, start:], axis=0)
        self.X = self.X[start:, :]
        self.H = simH

        # make plot
        if p == 1:
            plt.figure()
            axes1 = plt.subplot(411)
            # axes1 = plt.axes([0.1, 1.0, 0.8, 0.125])
            plt.imshow(simH)
            plt.axis('tight')
            cmap = plt.cm.jet
            my_map = cmap.from_list(
                'brstate', [[0, 1, 1], [1, 0, 1], [0.8, 0.8, 0.8]], 3)
            tmp = axes1.imshow(simH)
            tmp.set_cmap(my_map)
            axes1.axis('tight')
            axes1.get_xaxis().set_visible(False)
            axes1.get_yaxis().set_visible(False)
            if p_zoom == 1:
                start_zoom = int(0.5*self.X.shape[0])
                end_zoom = int(0.6*self.X.shape[0])
                plt.axvline(start_zoom, color='k', linestyle='--')
                plt.axvline(end_zoom, color='k', linestyle='--')
                plt.axhline(-0.02, start_zoom, end_zoom, color='k', linestyle='--')
                plt.axhline(0.02, start_zoom, end_zoom, color='k', linestyle='--')

            t = np.arange(0, self.X.shape[0]*self.dt, self.dt)
            t_ticks = np.linspace(t[0], t[-1], 7)
            tick_labels = t_ticks / 3600
            tick_labels = np.around(tick_labels, 2)
            empty_labels = ['','','','','','','']
            
            sns.set_context('paper')
            sns.set_style('white')
            axes2 = plt.subplot(412)
            # axes2 = plt.axes([0.1, 0.825, 0.8, 0.12])
            axes2.plot(t, self.X[:, 0])
            plt.setp(axes2.get_xticklabels(), visible=False)
            plt.ylabel('REM-On \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            # plt.ylabel('Sleep \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            sns.despine(ax=axes2)
            
            sns.set_context('paper')
            sns.set_style('white')
            axes3 = plt.subplot(413, sharex=axes2)
            # axes3 = plt.axes([0.1, 0.625, 0.8, 0.12])
            axes3.plot(t, self.X[:, 1])
            plt.setp(axes3.get_xticklabels(), visible=False)
            plt.ylabel('REM-Off \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            # plt.ylabel('Wake \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            sns.despine(ax=axes3)

            sns.set_context('paper')
            sns.set_style('white')
            axes4 = plt.subplot(414, sharex=axes2)
            # axes4 = plt.axes([0.1, 0.425, 0.8, 0.12])
            axes4.plot(t, self.X[:, 9])
            # axes4.plot(t, self.X[:, -1])
            plt.xlim([t[0], t[-1]])
            plt.xticks(t_ticks, tick_labels)
            plt.ylabel('REM \nPressure', rotation=0, ha='right', va='center')
            # plt.ylabel('Opto', rotation=0, ha='right', va='center')
            # plt.ylabel('Sleep \nPressure', rotation=0, ha='right', va='center')plt.ylabel('Sleep \nPressure', rotation=0, ha='right', va='center')
            plt.xlabel('Time (hr)')
            sns.despine(ax=axes4)

            if save:
                plt.savefig('figures/' + filename + '.pdf', bbox_inches = "tight", dpi = 100)

            plt.show()

        if p_zoom == 1:
            X_zoom = self.X[start_zoom:end_zoom, :]

            H_zoom = np.expand_dims(simH[0, start_zoom:end_zoom], axis=0)

            plt.figure()
            axes5 = plt.subplot(411)
            # axes5 = plt.axes([0.1, 1.0, 0.8, 0.125])
            plt.imshow(H_zoom)
            plt.axis('tight')
            cmap = plt.cm.jet
            tmp = axes5.imshow(H_zoom)
            tmp.set_cmap(my_map)
            axes5.axis('tight')
            axes5.get_xaxis().set_visible(False)
            axes5.get_yaxis().set_visible(False)

            t = np.arange(0, X_zoom.shape[0]*self.dt, self.dt)
            t_ticks = np.linspace(t[0], t[-1], 7)
            tick_labels = t_ticks / 3600
            tick_labels = np.around(tick_labels, 2)
            
            sns.set_context('paper')
            sns.set_style('white')
            axes6 = plt.subplot(412, sharex=axes1)
            # axes6 = plt.axes([0.1, 0.825, 0.8, 0.12])
            axes6.plot(t, X_zoom[:, 0])
            plt.setp(axes6.get_xticklabels(), visible=False)
            # plt.ylabel('REM-On \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            sns.despine(ax=axes6)
            
            sns.set_context('paper')
            sns.set_style('white')
            axes7 = plt.subplot(413, sharex=axes1)
            # axes7 = plt.axes([0.1, 0.625, 0.8, 0.12])
            axes7.plot(t, X_zoom[:, 1])
            plt.setp(axes7.get_xticklabels(), visible=False)
            # plt.ylabel('REM-Off \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center')
            sns.despine(ax=axes7)

            sns.set_context('paper')
            sns.set_style('white')
            axes8 = plt.subplot(414, sharex=axes1)
            # axes8 = plt.axes([0.1, 0.425, 0.8, 0.12])
            # axes8.plot(t, X_zoom[:, 9])
            axes8.plot(t, X_zoom[:, -1])
            plt.xlim([t[0], t[-1]])
            plt.xticks(t_ticks, tick_labels)
            # plt.ylabel('REM \nPressure', rotation=0, ha='right', va='center')
            plt.xlabel('Time (hr)')
            sns.despine(ax=axes8)

            if save:
                plt.savefig('figures/' + filename + '_zoom.pdf', bbox_inches = "tight", dpi = 100)

            plt.show()

    def get_state_pcts(self, p=0):
        """Getter method for sleep state percents

        Args:
            p (int, optional): Plotting variable: Plots distribution if p=1, does not plot otherwise. Defaults to 0.

        Returns:
            [list]: percents of each sleep state in hypnogram
        """
        wCounter = 0
        sCounter = 0
        remCounter = 0
        for entry in self.H[0]:
            if entry == 1:
                remCounter += 1
            elif entry == 2:
                wCounter += 1
            elif entry == 3:
                sCounter += 1
        total = wCounter + sCounter + remCounter
        statePcts = [remCounter/total*100, wCounter/total*100, sCounter/total*100]

        if p == 1:
            labels = ['REM', 'Wake', 'NREM']
            x = np.arange(len(labels))
            width = 0.5

            fig, ax = plt.subplots()
            bars = ax.bar(x, statePcts, width)
            ax.set_ylabel('Percentage of Total Sleep (%)')
            ax.set_title('Simulated Sleep State Percentages')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            for i, v in enumerate(statePcts):
                ax.text(x[i] - 0.1, v + 0.5, str(round(v, 2)), fontsize=9.0, fontweight='bold')

            fig.tight_layout()
            plt.show()
        
        return statePcts

    def get_state_durs(self, p=0):
        """Getter method for sleep state durations

        Args:
            p (int, optional): Plotting variable: Plots distribution if p=1, does not plot otherwise. Defaults to 0.

        Returns:
            [list]: durations of each sleep state in hypnogram
        """
        remDurs = []
        wDurs = []
        sDurs = []
        remTime = 0
        wTime = 0
        sTime = 0

        for entry in self.H[0]:
            #increment time of detected state (if other time counter is nonzero duing increment
            # i.e. states have just switched, append duration to corresponding list)
            if entry == 1:
                remTime += self.dt
                if wTime != 0:
                    wDurs.append(wTime)
                    wTime = 0
                elif sTime != 0:
                    sDurs.append(sTime)
                    sTime = 0
            if entry == 2:
                wTime += self.dt
                if remTime != 0:
                    remDurs.append(remTime)
                    remTime = 0
                elif sTime != 0:
                    sDurs.append(sTime)
                    sTime = 0
            if entry == 3:
                sTime += self.dt
                if remTime != 0:
                    remDurs.append(remTime)
                    remTime = 0
                elif wTime != 0:
                    wDurs.append(wTime)
                    wTime = 0
        
        avgVals = [np.average(remDurs), np.average(wDurs), np.average(sDurs)]
        seVals = [stats.sem(remDurs), stats.sem(wDurs), stats.sem(sDurs)]

        if p == 1:
            labels = ['REM', 'Wake', 'NREM']
            x = np.arange(len(labels))
            width = 0.5

            fig, ax = plt.subplots()
            bars = ax.bar(x, avgVals, width, yerr=seVals)
            ax.set_ylabel('Average State Duration (s)')
            ax.set_title('Simulated Sleep State Average Durations')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            for i, v in enumerate(avgVals):
                ax.text(x[i] + 0.05, v + 1, str(round(v, 2)), fontsize=9.0, fontweight='bold')

            fig.tight_layout()
            plt.show()

        return avgVals, seVals

    def get_state_freqs(self, p=0):
        """Getter method for sleep state frequencies

        Args:
            p (int, optional): Plotting variable: Plots distribution if p=1, does not plot otherwise. Defaults to 0.

        Returns:
            [list]: frequencies of each sleep state in hypnogram
        """
        remCounter = 0
        wCounter = 0
        sCounter = 0

        #check first value and update corresponding counter
        prev = self.H[0][0]
        if prev == 1:
            remCounter += 1
        elif prev == 2:
            wCounter += 1
        elif prev == 3:
            sCounter += 1

        #iterate through hypnogram and update counter on new occurences of sleep states (current state != prev state)
        for i in range(1, len(self.H[0])):
            if self.H[0][i] != prev:
                if self.H[0][i] == 1:
                    remCounter += 1
                elif self.H[0][i] == 2:
                    wCounter += 1
                elif self.H[0][i] == 3:
                    sCounter += 1
            prev = self.H[0][i]
        
        totTime = (self.dt * len(self.H[0])) / 3600 # simulation time in hours
        stateFreqs = [remCounter/totTime, wCounter/totTime, sCounter/totTime]

        if p == 1:
            labels = ['REM', 'Wake', 'NREM']
            x = np.arange(len(labels))
            width = 0.5

            fig, ax = plt.subplots()
            bars = ax.bar(x, stateFreqs, width)
            ax.set_ylabel('Sleep State Frequency (1/h)')
            ax.set_title('Simulated Sleep State Frequencies')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)

            for i, v in enumerate(stateFreqs):
                ax.text(x[i] - 0.1, v + 0.5, str(round(v, 2)), fontsize=9.0, fontweight='bold')

            fig.tight_layout()
            plt.show()
        
        return stateFreqs

    def nrem_before(self, pos):
        """Determines the amount of nrem sleep before the input position on the hypnogram until a rem period is encountered

        Args:
            pos (int): position for backwards tracing through hypnogram to begin at

        Returns:
            [float]: time in nrem prior to pos before encountering rem period (counting backwards)
        """
        nrem_counter = 0
        #update i to position just prior to onset of rem period
        i = int(pos - 1)
        #continue iteration while pos is > 0 or rem has not been encountered
        while i >= 0 and self.H[0][i] != 1:
            if self.H[0][i] == 3:
                nrem_counter += 1
            i -= 1

        return nrem_counter * self.dt

    def nrem_after(self, pos):
        """Determines the amount of nrem sleep after the input position on the hypnogram until a rem period is encountered

        Args:
            pos (int): position for forwards tracing through hypnogram to begin at

        Returns:
            [float]: time in nrem prior to pos before encountering rem period (counting backwards)
        """
        nrem_counter = 0
        #update i to position just after end of rem period
        i = int(pos + 1)
        #continue iteration while pos is > 0 or rem has not been encountered
        while i < len(self.H[0]) and self.H[0][i] != 1:
            if self.H[0][i] == 3:
                nrem_counter += 1
            i += 1

        return nrem_counter * self.dt

    def avg_Ron_and_Roff_by_state(self):
        """Calculates and plots average firing rate (+/- one standard deviation) of REM-on and REM-off neurons during REM, wake, and NREM sleep states

        Arguments:
            X {numpy array} -- simulated neuron population data from MI model simulation
            H {numpy array} -- sleep state data from simulated data above

        Returns:
            [list] -- list of REM-on (indices 0-2) and REM-off (indices 3-5) average firing rates during each sleep state
        """

        # df = pd.DataFrame(columns=['Type', 'State', 'Firing Rate'])

        # #extract firing rates for Ron and Roff pop at each sleep states and add to dataframe for plotting
        ron_rem = self.X[(np.where(self.H[0]==1))[0], 0]
        # for fr in ron_rem:
        #     to_add = {'Type': 'REM-On', 'State': 'REM', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)

        ron_wake = self.X[(np.where(self.H[0]==2))[0], 0]
        # for fr in ron_wake:
        #     to_add = {'Type': 'REM-On', 'State': 'Wake', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)

        ron_nrem = self.X[(np.where(self.H[0]==3))[0], 0]
        # for fr in ron_nrem:
        #     to_add = {'Type': 'REM-On', 'State': 'NREM', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)

        roff_rem = self.X[(np.where(self.H[0]==1))[0], 1]
        # for fr in roff_rem:
        #     to_add = {'Type': 'REM-Off', 'State': 'REM', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)

        roff_wake = self.X[(np.where(self.H[0]==2))[0], 1]
        # for fr in roff_wake:
        #     to_add = {'Type': 'REM-Off', 'State': 'Wake', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)

        roff_nrem = self.X[(np.where(self.H[0]==3))[0], 1]
        # for fr in roff_nrem:
        #     to_add = {'Type': 'REM-Off', 'State': 'NREM', 'Firing Rate': fr}
        #     df = df.append(to_add, ignore_index=True)


        #put average values into array
        avg_by_state = np.zeros(6)
        avg_by_state[0] = np.mean(ron_rem)
        avg_by_state[1] = np.mean(ron_wake)
        avg_by_state[2] = np.mean(ron_nrem)
        avg_by_state[3] = np.mean(roff_rem)
        avg_by_state[4] = np.mean(roff_wake)
        avg_by_state[5] = np.mean(roff_nrem)

        #get 95 CIs
        ron_ci = np.zeros((2,3))
        roff_ci = np.zeros((2,3))
        ron_ci[:,0] = compute_bootci(ron_rem, func='mean') - avg_by_state[0]
        ron_ci[:,1] = compute_bootci(ron_wake, func='mean') - avg_by_state[1]
        ron_ci[:,2] = compute_bootci(ron_nrem, func='mean') - avg_by_state[2]
        roff_ci[:,0] = compute_bootci(roff_rem, func='mean') - avg_by_state[3]
        roff_ci[:,1] = compute_bootci(roff_wake, func='mean') - avg_by_state[4]
        roff_ci[:,2] = compute_bootci(roff_nrem, func='mean') - avg_by_state[5]

        # #save standard deviation for error in plot
        # ron_std = np.zeros(3)
        # roff_std = np.zeros(3)
        # ron_std[0] = np.std(ron_rem, ddof=1)
        # ron_std[1] = np.std(ron_wake, ddof=1)
        # ron_std[2] = np.std(ron_nrem, ddof=1)
        # roff_std[0] = np.std(roff_rem, ddof=1)
        # roff_std[1] = np.std(roff_wake, ddof=1)
        # roff_std[2] = np.std(roff_nrem, ddof=1)

        #plot data together to graph
        ron_pop = [avg_by_state[0], avg_by_state[1], avg_by_state[2]]
        roff_pop = [avg_by_state[3], avg_by_state[4], avg_by_state[5]]

        #plot data in bar plot
        labels = ['REM', 'Wake', 'NREM']
        x = np.arange(len(labels))
        width = 0.35

        sns.set_context('paper')
        sns.set_style('white')

        fig, ax = plt.subplots()
        # sns.barplot(data = df, x='State', y='Firing Rate', Hue='Type', ci=95, ax=ax)
        ron_bars = ax.bar(x - width/2, ron_pop, width, yerr = ron_ci, label = 'REM-On Activity')
        roff_bars = ax.bar(x + width/2, roff_pop, width, yerr = roff_ci, label = 'REM-Off Activity')
        ax.set_ylabel('Average Activity/Firing Rate (Hz)')
        ax.set_title('Neuron Activity During Each Sleep State')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        sns.despine()
        ax.legend()

        fig.tight_layout()
        plt.savefig('figures/fig2_avgRonRoffByState.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        return ron_rem, ron_wake, ron_nrem, roff_rem, roff_wake, roff_nrem

        #print results as mean +/- std
        # print(f'Average REM-on activity during REM is {avg_by_state[0]} +/- {ron_std[0]}')
        # print(f'Average REM-on activity during Wake is {avg_by_state[1]} +/- {ron_std[1]}')
        # print(f'Average REM-on activity during NREM is {avg_by_state[2]} +/- {ron_std[2]}')
        # print(f'Average REM-off activity during REM is {avg_by_state[3]} +/- {roff_std[0]}')
        # print(f'Average REM-off activity during Wake is {avg_by_state[4]} +/- {roff_std[1]}')
        # print(f'Average REM-off activity during NREM is {avg_by_state[5]} +/- {roff_std[2]}')

    def inter_REM(self, seq_thresh=100, p=0, zoom_out=0, nremOnly=False, log=False, rem_pre_split=False, save=False, filename='fig1_remPre'):
        """Plots association between REM durations and following inter-REM durations (NREM only)
        """

        #define variables for loop
        REM_counter = 0
        inter_counter = 0
        first_marker = 0
        REM_durations = []
        inter_durations = []
        inter_locs = []

        #loop through data: determine current state, save length of period if state changes 
        # from REM or from inter-REM
        for i in range(len(self.H[0]) - 1):
            if (self.H[0][i] == 1):
                first_marker = 1
                REM_counter += 1
                if (self.H[0][i+1] != 1):
                    REM_durations.append(REM_counter * self.dt)
                    REM_counter = 0
            elif (self.H[0][i] != 1 & first_marker != 0):
                if nremOnly:
                    if (self.H[0][i] == 3):
                        inter_counter += 1
                else:
                    inter_counter += 1
                if (self.H[0][i+1] == 1):
                    inter_locs.append(i)
                    inter_durations.append(inter_counter * self.dt)
                    inter_counter = 0

        #if last position of H is REM, marking the end of an unsaved inter-REM period
        if (self.H[0][-1] == 1 & inter_counter != 0):
            inter_durations.append(inter_counter * self.dt)
        
        #if the end of the last inter-REM period was not saved due to the full period 
        # not being saved, delete the last REM duration (corresponding to the incomplete)
        #inter-REM
        if (len(REM_durations) > len(inter_durations)):
            del REM_durations[-1]

        #delete inter-REM durations of length zero (result of REM-wake-REM) and
        #corresponding REM duration
        for i in range(len(inter_durations)):
            if i >= len(inter_durations):
                continue
            if inter_durations[i] == 0:
                del inter_durations[i]
                del REM_durations[i]
                del inter_locs[i]

        #convert to np arrays for easier manipulation
        REM_durations = np.array(REM_durations)
        inter_locs = np.array(inter_locs)
        inter_durations = np.array(inter_durations)

        #separate by sequential and non-sequential
        seq_rem = REM_durations[inter_durations < seq_thresh]
        nonseq_rem = REM_durations[inter_durations >= seq_thresh]
        seq_inter = inter_durations[inter_durations < seq_thresh]
        nonseq_inter = inter_durations[inter_durations >= seq_thresh]

        if log:
            log_seq_inter = np.log(seq_inter+1)
            log_nonseq_inter = np.log(nonseq_inter+1)
            #regression line and r^2
            logM, logB, logR, _, _ = stats.linregress(nonseq_rem, log_nonseq_inter)
            # gmm = GaussianMixture(n_components=3).fit(log_inter.reshape(-1,1))
            # means_hat = gmm.means_.flatten()
            # weights_hat = gmm.weights_.flatten()
            # sds_hat = np.sqrt(gmm.covariances_).flatten()

        #regression line and r^2
        m, b, r, _, _ = stats.linregress(nonseq_rem, nonseq_inter)

        #plot data in scatter plot
        if not rem_pre_split:
            if p == 1:
                sns.set(font_scale=1)

                plt.figure()
                sns.set_context('paper')
                sns.set_style('white')
                plt.scatter(seq_rem, seq_inter, color='gray')
                plt.scatter(nonseq_rem, nonseq_inter, color='blue')
                plt.plot(REM_durations, np.multiply(m,REM_durations) + b, color = 'red')
                plt.xlabel('REM_pre (s)')
                if nremOnly:
                    plt.ylabel('|NREM| \n(s)', rotation=0, ha='center', va='center', labelpad=20)
                else:
                    plt.ylabel('Inter-REM Duration (s)')
                # plt.title('REM Duration vs Inter-REM Duration')
                # plt.text(max(REM_durations) - 25, m * max(REM_durations) + (b + 50), f'R^2: {round(r**2, 2)}', fontsize = 12)
                sns.despine()
                if save:
                    plt.savefig('figures/' + filename + '.pdf', bbox_inches = "tight", dpi = 100)
                plt.show()

                print(f'Regression Line: Inter = {np.round(m, 2)}(REM_pre) + {np.round(b, 2)}')
            
            if log and p == 1:
                # plt.figure()
                # plt.scatter(REM_durations, log_inter)
                # plt.plot(REM_durations, np.multiply(logM,REM_durations) + logB, color = 'red')
                # plt.xlabel('REM Duration (s)')
                # if nremOnly:
                #     plt.ylabel('NREM During Inter-REM Duration (s)')
                # else:
                #     plt.ylabel('Inter-REM Duration (s)')
                # plt.title('REM Duration vs Inter-REM Duration')
                # plt.text(max(REM_durations) - 1, logM * max(REM_durations) + (logB + 1.2), f'R^2: {round(logR**2, 2)}', fontsize = 12)
                # plt.show()

                # print(f'Regression Line: Inter = {np.round(logM, 2)}(REM_pre) + {np.round(logB, 2)}')

                _, (ax1, ax2) = plt.subplots(1,2)

                sns.histplot(REM_durations, bins=30, ax=ax1)
                ax1.set_ylabel('Count', rotation=0, ha='right', va='center')
                ax1.set_xlabel('REM_pre (s)')
                ax1.set_title('Distribution of REM_pre Length')

                sns.histplot(log_nonseq_inter, bins=15, color='blue', ax=ax2)
                sns.histplot(log_seq_inter, bins=30, color='gray', ax=ax2)
                ax2.set_ylabel('Count', rotation=0, ha='right', va='center')
                ax2.set_xlabel('Log(|NREM|)')
                ax2.set_title('Log Distribution of |NREM| Length')
                # mu1_h, sd1_h, w1_h = means_hat[0], sds_hat[0], weights_hat[0]
                # x1 = np.linspace(mu1_h-3*sd1_h, mu1_h+3*sd1_h, 1000)
                # plt.plot(w1_h*stats.norm.pdf(x1, mu1_h, sd1_h), x1)

                # mu2_h, sd2_h, w2_h = means_hat[1], sds_hat[1], weights_hat[1]
                # x2 = np.linspace(mu2_h-3*sd2_h, mu2_h+3*sd2_h, 1000)
                # plt.plot(w2_h*stats.norm.pdf(x2, mu2_h, sd2_h), x2)

                # mu3_h, sd3_h, w3_h = means_hat[2], sds_hat[2], weights_hat[2]
                # x3 = np.linspace(mu3_h-3*sd3_h, mu3_h+3*sd3_h, 1000)
                # plt.plot(w3_h*stats.norm.pdf(x3, mu3_h, sd3_h), x3)
                # plt.ylabel('')
                sns.despine()
                if save:
                    plt.savefig('figures/' + filename + '_log.pdf', bbox_inches = "tight", dpi = 100)
                plt.show()

            #plot data as above but with axes matching control dataset for better comparison
            if zoom_out == 1:
                plt.figure()
                plt.scatter(REM_durations, inter_durations)
                plt.plot(REM_durations, np.multiply(m,REM_durations) + b, color = 'red')
                plt.xlabel('REM Duration (s)')
                plt.ylabel('NREM During Inter-REM Duration (s)')
                plt.title('REM Duration vs Inter-REM Duration - Zoomed Out')
                plt.text(max(REM_durations) - 25, m * max(REM_durations) + (b + 50), f'R^2: {round(r**2, 2)}', fontsize = 12)
                plt.xlim([-10, 250])
                plt.ylim([-10, 2500])
                plt.show()
        
        else:
            rem_bounds = np.arange(0, 181, 30)
            seq_rem_pre_splits = []
            nonseq_rem_pre_splits = []
            seq_inter_splits = []
            nonseq_inter_splits = []

            for i in range(rem_bounds.shape[0] - 1):
                range_min = rem_bounds[i]
                range_max = rem_bounds[i + 1]

                temp_seq_rem_pre = seq_rem[np.logical_and(seq_rem >= range_min, seq_rem < range_max)]
                temp_nonseq_rem_pre = nonseq_rem[np.logical_and(nonseq_rem >= range_min, nonseq_rem < range_max)]
                temp_seq_inter = seq_inter[np.logical_and(seq_rem >= range_min, seq_rem < range_max)]
                temp_nonseq_inter = nonseq_inter[np.logical_and(nonseq_rem >= range_min, nonseq_rem < range_max)]
                
                seq_rem_pre_splits.append(temp_seq_rem_pre)
                nonseq_rem_pre_splits.append(temp_nonseq_rem_pre)
                seq_inter_splits.append(temp_seq_inter)
                nonseq_inter_splits.append(temp_nonseq_inter)

            seq_rem_pre_splits = np.array(seq_rem_pre_splits)
            nonseq_rem_pre_splits = np.array(nonseq_rem_pre_splits)
            seq_inter_splits = np.array(seq_inter_splits)
            nonseq_inter_splits = np.array(nonseq_inter_splits)

            if p==1:
                _, axs = plt.subplots(math.ceil(seq_rem_pre_splits.shape[0] / 3), 3)
                curr_split = 0
                # print(seq_inter_splits)
                # print(nonseq_inter_splits)
                for row in axs:
                    for ax in row:
                        if seq_inter_splits[curr_split].size == 0 or nonseq_inter_splits[curr_split].size == 0:
                            curr_split += 1
                            continue
                        else:
                            curr_log_seq_split_inter = np.log(seq_inter_splits[curr_split]+1)
                            curr_log_nonseq_split_inter = np.log(nonseq_inter_splits[curr_split]+1)
                            sns.histplot(curr_log_nonseq_split_inter[curr_split], color='blue', ax=ax)
                            sns.histplot(curr_log_seq_split_inter[curr_split], color='gray', ax=ax)
                            ax.set_ylabel('Count', rotation=0, ha='right', va='center')
                            ax.set_xlabel('Log(|NREM|)')
                            ax.set_title('Log Distribution of |NREM| Length for REM_pre %d <= %d' % (rem_bounds[curr_split], rem_bounds[curr_split+1]))
                            curr_split += 1
                
                sns.despine()
                if save:
                    plt.savefig('figures/' + filename + '_log_split.pdf', bbox_inches = "tight", dpi = 100)
                plt.show()
                    


        #get all inter-REM period lengths for sequential REM periods (inter-REM < 100 seconds), 
        #the number of inter-REM periods under 100s corresponds to the number of seq REM periods
        seqs = inter_durations[np.where(inter_durations <= seq_thresh)]

        #calculate percentage of sequential and single REM periods
        perc_seq = (len(seqs) / len(inter_durations))*100
        perc_sing = 100 - perc_seq

        print(f'Sequential REM: {perc_seq}%, Single REM: {perc_sing}%')

        return inter_locs, inter_durations, m

    def REM_dur_dist(self, p=0):
        """Calculates and optionally plots REM duration distribution

        Args:
            p (int, optional): Plotting variable: Plots distribution if p=1, does not plot otherwise. Defaults to 0.

        Returns:
            [list]: list of REM durations
        """
        
        REM_durations = []
        REM_counter = 0
        for i in range(len(self.H[0]) - 1):
            if (self.H[0][i] == 1):
                REM_counter += 1
                if (self.H[0][i+1] != 1):
                    REM_durations.append(REM_counter * self.dt)
                    REM_counter = 0

        if self.H[0][-1] == 1 and REM_counter != 0:
            REM_counter += 1
            REM_durations.append(REM_counter * self.dt)
            REM_counter = 0

        if p==1:
            plt.figure()
            plt.hist(REM_durations, bins=15)
            plt.ylabel('Frequency')
            plt.xlabel('REM Episode Duration (s)')
            plt.title('REM Episode Duration Distribution')
            plt.xlim([-10, 250])
            # plt.ylim([0, 120])
            plt.show()
        
        return REM_durations

    def REM_pressure_laser_onset(self, dur = 5*60):
        """Plots REM pressure association with the delay between the end of a REM period and the next laser onset

        Keyword Arguments:
            dur {int or float (usually int)} -- duration of laser stimulation used in simulation of model above (run_mi_model)

        """

        #find delay between end of REM period and next laser onset, grab REM pressure at laser onset
        onset = []
        pressure = []
        REM_induced_onset = []
        REM_induced_dur = []
        record_time = False
        delay = 0
        for i in range(1, len(self.H[0])):
            #start recording time if REM period ends
            if self.H[0][i-1] == 1 and self.H[0][i] != 1:
                record_time = True
            #if time is being recorded, add timestep for each iteration
            if record_time:
                delay += 0.05
                #stop recording time and grab data upon laser onset
                if self.X[i-1, -1] == 0 and self.X[i, -1] != 0:
                    record_time = False
                    onset.append(delay)
                    delay = 0
                    pressure.append(self.X[i, -5])
                    if self.H[0][i] == 1:
                        REM_induced_onset.append(1)
                    else:
                        REM_induced_onset.append(0)
                    REM_in_dur = False
                    for j in range(int(dur / 0.05)):
                        if (i + j) >= len(self.H[0]):
                            continue
                        if self.H[0][i + j] == 1:
                            REM_in_dur = True
                    if REM_in_dur:
                        REM_induced_dur.append(1)
                    else:
                        REM_induced_dur.append(0)

        for i in range(len(onset)):
            if i >= len(onset):
                continue
            #remove outliers
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

    def Roff_FR_before_REM (self):
        """Plots REM-off firing rate prior to REM sleep
        """

        pre_REM = []
        period = int(60 / self.dt)
        time_vec = np.linspace(-period*self.dt, (period*self.dt)/2, int(1.5*period + 1))

        for i in range(len(self.H[0]) - 1):
            if self.H[0][i] != 1 and self.H[0][i+1] == 1:
                if i - period < 0 or (i + (period/2) + 1) > len(self.H[0]):
                    continue
                pre_REM.append(self.X[i - period: i + int(period/2) + 1, 1])

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

    def microarousal_count(self, ma_length):
        """Calculates number of microarousals in simulated sleep data

        Args:
            ma_length (float): threshold for microarousal determination (seconds)
        """

        def microarousal_count_helper(H, ma_length, dt, pos):
            for i in range(int(ma_length / dt) + 1):
                if H[0][i] != 2:
                    return 1
            return 0

        ma_counter = 0
        for i in range(len(self.H[0]) - 1):
            if self.H[0][i] == 2:
                ma_counter += microarousal_count_helper(self.H, ma_length, self.dt, i)
        return ma_counter

    def Roff_FR_inter_REM_norm(self):
        """Plots average REM-off firing rate during inter-REM with all inter-REM periods normalized to a common length

        Returns:
            [list]: average REM-off firing rates over course of inter-REM period
        """
        ############################################################################################

        record_inter = False
        record_rem = False
        inter_rem_periods = []
        rem_periods = []
        curr_inter = []
        curr_rem = []
        for i in range(1, len(self.H[0])):
            #store values for ended REM period and begin recording
            if self.H[0][i-1] == 1 and self.H[0][i] != 1 and record_rem:
                record_rem = False
                rem_periods.append(curr_rem)
                curr_rem = []
            #stop recording values and save data to list on REM transition
            elif self.H[0][i-1] != 1 and self.H[0][i] == 1 and record_inter:
                record_inter = False
                inter_rem_periods.append(curr_inter)
                curr_inter = []

            #begin recording desired states on desired transition
            if self.H[0][i-1] != 1 and self.H[0][i] == 1:
                record_rem = True
            elif self.H[0][i-1] == 1 and self.H[0][i] != 1:
                record_inter = True
            
            #record REM-off FR for current state
            if record_inter:
                curr_inter.append(self.X[i,1])
            elif record_rem:
                curr_rem.append(self.X[i,1])

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
            norm_FRs_inter.append(self.time_morph(to_np, nstates_inter))
        norm_FRs_inter = np.array(norm_FRs_inter)

        #normalize rem periods
        nstates_rem = 10
        norm_FRs_rem = []
        for entry in rem_periods:
            to_np = np.array(entry)
            norm_FRs_rem.append(self.time_morph(to_np, nstates_rem))
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

    def rem_zoom(self, inter_locs, inter_durations):
        """Creates figures of the hypnogram zoomed in at periods of sequential REM

        Args:
            inter_locs (list): location of inter-REM periods in hypnogram
            inter_durations (list): durations of corresponding inter-REM periods
        """
        inter_thresh = 150
        seq_locs = inter_locs[np.where(inter_durations <= inter_thresh)]

        pre_post = 200
        for loc in seq_locs:
            simH = np.zeros((1, 2*int(pre_post/self.dt)))
            start = loc - int(pre_post/self.dt)
            end = loc + int(pre_post/self.dt)
            if start < 0 or end >= len(self.X):
                continue
            simH[0] = self.H[0][start:end]

            plt.figure()
            axes1 = plt.axes([0.1, 1.0, 0.8, 0.1])
            plt.imshow(simH)
            plt.axis('tight')
            cmap = plt.cm.jet
            my_map = cmap.from_list(
                'brstate', [[0, 1, 1], [1, 0, 1], [0.8, 0.8, 0.8]], 3)
            tmp = axes1.imshow(simH)
            tmp.set_cmap(my_map)
            axes1.axis('tight')
            tmp.axes.get_xaxis().set_visible(False)
            tmp.axes.get_yaxis().set_visible(False)

            t = np.arange(start*self.dt, end*self.dt, self.dt)
            axes2 = plt.axes([0.1, 0.8, 0.8, 0.2])
            axes2.plot(t, self.X[start:end, [0, 1]])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('REM-on vs REM-off')

            axes3 = plt.axes([0.1, 0.6, 0.8, 0.2])
            axes3.plot(t, self.X[start:end, [2, 3]])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('Sleep vs Wake')

            axes4 = plt.axes([0.1, 0.4, 0.8, 0.2])
            axes4.plot(t, self.X[start:end, 9])
            plt.xlim([t[0], t[-1]])
            plt.ylabel('REM pressure')

            axes5 = plt.axes([0.1, 0.2, 0.8, 0.2])
            axes5.plot(t, self.X[start:end, 10]*0.01)
            plt.xlim([t[0], t[-1]])
            plt.ylabel('Sleep pressure')

            axes6 = plt.axes([0.1, 0.0, 0.8, 0.2])
            axes6.plot(t, self.X[start:end, -3])
            plt.xlim(t[0], t[-1])
            # plt.ylim(0, max(self.X[:, -2]) + 1)
            plt.ylabel('Delta')

            plt.show()

    def avg_first_rem_dur(self):
        """Determines the average REM duration of the first REM period in a sequential REM episode and 
        the average REM duration of a single REM period

        Returns:
            [float]: average duration of first REM period in sequential REM episdoe (seconds)
            [float]: average duration of single REM period (seconds)
        """
        #extract all rem sequences from hypnogram
        remSeqs = sleepy.get_sequences(np.where(self.H[0] == 1)[0])

        #list to hold durations of first rem periods of sequential episodes
        firstDurs = []
        singleDurs = []

        #check nrem times before and after each rem period: first of sequential episodes will have
        #a time in nrem > 100s prior to rem period and time <= 100s after REM period
        for seq in remSeqs:
            if (self.nrem_before(self, seq[0])) > 100 and (self.nrem_after(self, seq[-1])) <= 100:
                firstDurs.append(len(seq) * self.dt)
            elif (self.nrem_before(self, seq[0])) > 100 and (self.nrem_after(self, seq[-1]) > 100):
                singleDurs.append(len(seq) * self.dt)

        avgFirstOfSeq = np.average(firstDurs)
        avgSingles = np.average(singleDurs)

        print(f'Average Duration of First REM Episode in Sequence: {avgFirstOfSeq}s')
        print(f'Average Duration of Single REM Episode: {avgSingles}s')

        return avgFirstOfSeq, avgSingles 

    def rem_cycles(self):
        """Calculates and plots distribution of REM cycles in sequential REM periods
        """
        remEps = sleepy.get_sequences(np.where(self.H[0] == 1)[0])

        seqRemLen = []
        currSeq = []
        for seq in remEps:
            if (self.nrem_before(seq[0])) <= 100 or (self.nrem_after(seq[-1])) <= 100:
                #save REM episode if proximity of another REM episodes indicates it is sequential
                currSeq.append(seq)
                #if NREM threshold is reached after sequence, save seq
                if (self.nrem_after(seq[-1])) > 100:
                    seqRemLen.append(len(currSeq))
                    currSeq = []

        seqRemLen = np.array(seqRemLen)
        seqRemLen -= 1

        plt.hist(seqRemLen, rwidth=20)
        plt.title('Sequential REM Cycle Distribution')
        plt.xlabel('REM Cycle Length')
        plt.ylabel('Frequency')
        plt.xlim([0.8, 5.2])
        plt.xticks(range(1,6))

    def get_inter_seq_starts(self):
        """Returns the start indices of all inter-REM periods in the current model, split into sequential
        REM (burst) inter-periods and 

        Returns:
            [np.arrays]: start indices of inter-REM transitions, split into 2 arrays by whether the
            transition is from sequential REM or single REM
        """
        #get sequences of all states excluding REM (extracts inter-REM periods)
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        #delete first and last sequences since they will not be between 2 REM periods
        interSeqs = np.delete(interSeqs, 0)
        interSeqs = np.delete(interSeqs, -1)

        #remove wake from sequences and save ones with < 150s of NREM (sequential REM)
        burstInterStarts = np.array([])
        longInterStarts = np.array([])
        for seq in interSeqs:
            #### Do we want only inter-REM with solely < 150 s (i.e. no wake at all)? If not, should only NREM in these periods
            #### be considered for analysis, or should all wake and nrem activity be analyzed, just on inter-REM periods with
            ### < 150 s of NREM sleep
            ### -> Take full period - look 30 seconds post and pre

            #if sequences contains < 150 s NREM, append
            if (len(seq) * self.dt < 150):
                burstInterStarts = np.append(burstInterStarts, seq[0])
            else:
                longInterStarts = np.append(longInterStarts, seq[0])

        return burstInterStarts, longInterStarts

    def get_inter_seq_ends(self):
        """Returns the end indices of all inter-REM periods in the current model, split into sequential
        REM (burst) inter-periods and 

        Returns:
            [np.arrays]: start indices of inter-REM transitions, split into 2 arrays by whether the
            transition is from sequential REM or single REM
        """
        #get sequences of all states excluding REM (extracts inter-REM periods)
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        #delete first and last sequences since they will not be between 2 REM periods
        interSeqs = np.delete(interSeqs, 0)
        interSeqs = np.delete(interSeqs, -1)

        #remove wake from sequences and save ones with < 150s of NREM (sequential REM)
        burstInterEnds = np.array([])
        longInterEnds = np.array([])
        for seq in interSeqs:
            #### Do we want only inter-REM with solely < 150 s (i.e. no wake at all)? If not, should only NREM in these periods
            #### be considered for analysis, or should all wake and nrem activity be analyzed, just on inter-REM periods with
            ### < 150 s of NREM sleep
            ### -> Take full period - look 30 seconds post and pre

            #if sequences contains < 150 s NREM, append
            if (len(seq) * self.dt < 150):
                burstInterEnds = np.append(burstInterEnds, seq[-1])
            else:
                longInterEnds = np.append(longInterEnds, seq[-1])

        return burstInterEnds, longInterEnds

    def avg_Ron_Roff_seq_REM(self):
        """Plots average REM-on and REM-off activity 30 seconds before and after the start of sequential
        REM -> inter-REM transitions and single REM -> inter-REM transitions. REM-On activities between the
        2 categories are plotted together and REM-off activity between the 2 categories are plotted together.

        Returns:
            [np.arrays]: mean REM-on and mean REM-off activity for sequential and single transitions (4 arrays total)
        """
        #get starting points for inter-REM, divided into sequential and single categories
        burstStarts, longStarts = self.get_inter_seq_starts()

        #define time to record prior to and after inter-REM start points
        prePostTime = 30
        prePostPoints = int(prePostTime / self.dt)

        #anonymous function to get timepoint ranges pre and post of start
        timePoints = lambda start: np.arange(start - prePostPoints, start + prePostPoints + 1, dtype=int) 

        #save REM-on and REM-off activity over pre-post period for each sequential REM -> inter transition
        burstPrePostRon = np.empty((0, 2*prePostPoints + 1), float)
        burstPrePostRoff = np.empty((0, 2*prePostPoints + 1), float)
        burstPrePostStp = np.empty((0, 2*prePostPoints + 1), float)
        burstPrePostDelta = np.empty((0, 2*prePostPoints + 1), float)
        for start in burstStarts:
            burstPrePostRon = np.append(burstPrePostRon, [self.X[timePoints(start), 0]], axis=0)
            burstPrePostRoff = np.append(burstPrePostRoff, [self.X[timePoints(start), 1]], axis=0)
            burstPrePostStp = np.append(burstPrePostStp, [self.X[timePoints(start), 9]], axis=0)
            burstPrePostDelta = np.append(burstPrePostDelta, [self.X[timePoints(start), -3]], axis=0)

        #save REM-on and REM-off activity over pre-post period for each single REM -> inter transition
        longPrePostRon = np.empty((0, 2*prePostPoints + 1), float)
        longPrePostRoff = np.empty((0, 2*prePostPoints + 1), float)
        longPrePostStp = np.empty((0, 2*prePostPoints + 1), float)
        longPrePostDelta = np.empty((0, 2*prePostPoints + 1), float)
        for start in longStarts:
            longPrePostRon = np.append(longPrePostRon, [self.X[timePoints(start), 0]], axis=0)
            longPrePostRoff = np.append(longPrePostRoff, [self.X[timePoints(start), 1]], axis=0)
            longPrePostStp = np.append(longPrePostStp, [self.X[timePoints(start), 9]], axis=0)
            longPrePostDelta = np.append(longPrePostDelta, [self.X[timePoints(start), -3]], axis=0)

        #get actvity means
        meanBurstRon = np.mean(burstPrePostRon, axis=0).flatten()
        meanBurstRoff = np.mean(burstPrePostRoff, axis=0).flatten()
        meanBurstStp = np.mean(burstPrePostStp, axis=0).flatten()
        meanBurstDelta = np.mean(burstPrePostDelta, axis=0).flatten()

        meanLongRon = np.mean(longPrePostRon, axis=0).flatten()
        meanLongRoff = np.mean(longPrePostRoff, axis=0).flatten()
        meanLongStp = np.mean(longPrePostStp, axis=0).flatten()
        meanLongDelta = np.mean(longPrePostDelta, axis=0).flatten()

        #get activity error (SD)
        sdBurstRon = np.std(burstPrePostRon, axis=0, ddof=1).flatten()
        sdBurstRoff = np.std(burstPrePostRoff, axis=0, ddof=1).flatten()
        sdBurstStp = np.std(burstPrePostStp, axis=0, ddof=1).flatten()
        sdBurstDelta = np.std(burstPrePostDelta, axis=0, ddof=1).flatten()

        sdLongRon = np.std(longPrePostRon, axis=0, ddof=1).flatten()
        sdLongRoff = np.std(longPrePostRoff, axis=0, ddof=1).flatten()
        sdLongStp = np.std(longPrePostStp, axis=0, ddof=1).flatten()
        sdLongDelta = np.std(longPrePostDelta, axis=0, ddof=1).flatten()

        #define x data for plots (time window from 30 seconds pre and post start point)
        xData = np.arange(-1*prePostTime, prePostTime + self.dt, self.dt)

        #plot REM-on activities by category
        plt.figure()
        plt.errorbar(xData, meanBurstRon, yerr=sdBurstRon, alpha=0.25, label = 'Sequential Transition')
        plt.errorbar(xData, meanLongRon, yerr=sdLongRon, alpha=0.1, label = 'Single Transition')
        plt.axvline(0, color='red', label='REM -> NREM')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('REM-On Firing Rate at REM->Inter-REM Transition')
        plt.legend()
        plt.savefig('figures/MCCV_Ron_seq_REM.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        #plot REM-off activities by category
        plt.figure()
        plt.errorbar(xData, meanBurstRoff, yerr=sdBurstRoff, alpha=0.25, label = 'Sequential Transition')
        plt.errorbar(xData, meanLongRoff, yerr=sdLongRoff, alpha=0.1, label = 'Single Transition')
        plt.axvline(0, color='red', label='REM -> NREM')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('REM-off Firing Rate at REM->Inter-REM Transition')
        plt.legend()
        plt.savefig('figures/MCCV_Roff_seq_REM.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        plt.figure()
        plt.errorbar(xData, meanBurstStp, yerr=sdBurstStp, alpha=0.25, label = 'Sequential Transition')
        plt.errorbar(xData, meanLongStp, yerr=sdLongStp, alpha=0.1, label = 'Single Transition')
        plt.axvline(0, color='red', label='REM -> NREM')
        plt.xlabel('Time (s)')
        plt.ylabel('Stp')
        plt.title('Stp at REM->Inter-REM Transition')
        plt.legend()
        plt.savefig('figures/MCCV_stp_seq_REM.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        plt.figure()
        plt.errorbar(xData, meanBurstDelta, yerr=sdBurstDelta, alpha=0.25, label = 'Sequential Transition')
        plt.errorbar(xData, meanLongDelta, yerr=sdLongDelta, alpha=0.1, label = 'Single Transition')
        plt.axvline(0, color='red', label='REM -> NREM')
        plt.xlabel('Time (s)')
        plt.ylabel('Delta')
        plt.title('Delta at REM->Inter-REM Transition')
        plt.legend()
        plt.savefig('MCCV_delta_seq_REM.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        return meanBurstRon, meanBurstRoff, meanBurstStp, meanBurstDelta, meanLongRon, meanLongRoff, meanLongStp, meanLongDelta

    def avg_Ron_Roff_seq_REM_norm(self):
        tot_points = 500
        nstates_inter = int(tot_points / 2)
        nstates_rem = int(tot_points / 4)

        #get inter-rem sequences with sleepy.get_sequences
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        #delete first and last inter period so all periods are REM->inter->REM
        interSeqs = np.delete(interSeqs, 0)
        interSeqs = np.delete(interSeqs, -1)

        #identify burst vs single rem periods (used later)
        burst_inds = []
        single_inds = []
        for i in range(len(interSeqs)):
            seq = interSeqs[i]
            if len(seq) * self.dt < 150:
                burst_inds.append(i)
            else:
                single_inds.append(i)

        #save FRon, FRoff, and stp during inter rem
        inter_fRon = []
        inter_fRoff = []
        inter_stp = []
        inter_delta = []
        for seq in interSeqs:
            inter_fRon.append(self.X[seq, 0])
            inter_fRoff.append(self.X[seq, 1])
            inter_stp.append(self.X[seq, 9])
            inter_delta.append(self.X[seq, -3])

        #get REM sequences with sleepy.get_sequences (number of REM periods should be number of inter + 1)
        remSeqs = sleepy.get_sequences(np.where(self.H[0] == 1)[0])

        #save FRoff, Fron, and stp during REM
        rem_fRon = []
        rem_fRoff = []
        rem_stp = []
        rem_delta = []
        for seq in remSeqs:
            rem_fRon.append(self.X[seq, 0])
            rem_fRoff.append(self.X[seq, 1])
            rem_stp.append(self.X[seq, 9])
            rem_delta.append(self.X[seq, -3])

        #normalize inter periods
        norm_inter_fRon = []
        norm_inter_fRoff = []
        norm_inter_stp = []
        norm_inter_delta = []
        for i in range(len(interSeqs)):
            #convert firing rate/stp list to np array for time_morph
            to_np_fRon = np.array(inter_fRon[i])
            to_np_fRoff = np.array(inter_fRoff[i])
            to_np_stp = np.array(inter_stp[i])
            to_np_delta = np.array(inter_delta[i])

            #time normalize inter periods
            norm_inter_fRon.append(self.time_morph(to_np_fRon, nstates_inter))
            norm_inter_fRoff.append(self.time_morph(to_np_fRoff, nstates_inter))
            norm_inter_stp.append(self.time_morph(to_np_stp, nstates_inter))
            norm_inter_delta.append(self.time_morph(to_np_delta, nstates_inter))

        #normalize REM periods
        norm_rem_fRon = []
        norm_rem_fRoff = []
        norm_rem_stp = []
        norm_rem_delta = []
        for i in range(len(remSeqs)):
            #convert firing rate/stp list to np array for time_morph
            to_np_fRon = np.array(rem_fRon[i])
            to_np_fRoff = np.array(rem_fRoff[i])
            to_np_stp = np.array(rem_stp[i])
            to_np_delta = np.array(rem_delta[i])

            #time normalize inter periods
            norm_rem_fRon.append(self.time_morph(to_np_fRon, nstates_rem))
            norm_rem_fRoff.append(self.time_morph(to_np_fRoff, nstates_rem))
            norm_rem_stp.append(self.time_morph(to_np_stp, nstates_rem))
            norm_rem_delta.append(self.time_morph(to_np_delta, nstates_rem))

        #attach normalized inter to normalized pre and post REM
        fRon_slices = []
        fRoff_slices = []
        stp_slices = []
        delta_slices = []
        for i in range(len(norm_inter_fRon)):
            curr_fRon = []
            curr_fRon.extend(norm_rem_fRon[i]) #rem before inter-rem period
            curr_fRon.extend(norm_inter_fRon[i]) #inter-rem period
            curr_fRon.extend(norm_rem_fRon[i + 1]) #rem after inter-rem period (same as rem 
            #before next inter-rem period)

            curr_fRoff = []
            curr_fRoff.extend(norm_rem_fRoff[i])
            curr_fRoff.extend(norm_inter_fRoff[i])
            curr_fRoff.extend(norm_rem_fRoff[i + 1])

            curr_stp = []
            curr_stp.extend(norm_rem_stp[i])
            curr_stp.extend(norm_inter_stp[i])
            curr_stp.extend(norm_rem_stp[i + 1])

            curr_delta = []
            curr_delta.extend(norm_rem_delta[i])
            curr_delta.extend(norm_inter_delta[i])
            curr_delta.extend(norm_rem_delta[i + 1])
            
            fRon_slices.append(curr_fRon)
            fRoff_slices.append(curr_fRoff)
            stp_slices.append(curr_stp)
            delta_slices.append(curr_delta)

        #separate slices by burst rem vs single rem
        burst_fRon_slices = [fRon_slices[i] for i in burst_inds]
        burst_fRoff_slices = [fRoff_slices[i] for i in burst_inds]
        burst_stp_slices = [stp_slices[i] for i in burst_inds]
        burst_delta_slices = [delta_slices[i] for i in burst_inds]

        single_fRon_slices = [fRon_slices[i] for i in single_inds]
        single_fRoff_slices = [fRoff_slices[i] for i in single_inds]
        single_stp_slices = [stp_slices[i] for i in single_inds]
        single_delta_slices = [delta_slices[i] for i in single_inds]

        #get actvity means
        mean_burst_fRon = np.mean(burst_fRon_slices, axis=0).flatten()
        mean_burst_fRoff = np.mean(burst_fRoff_slices, axis=0).flatten()
        mean_burst_stp = np.mean(burst_stp_slices, axis=0).flatten()
        mean_burst_delta = np.mean(burst_delta_slices, axis=0).flatten()

        mean_single_fRon = np.mean(single_fRon_slices, axis=0).flatten()
        mean_single_fRoff = np.mean(single_fRoff_slices, axis=0).flatten()
        mean_single_stp = np.mean(single_stp_slices, axis=0).flatten()
        mean_single_delta = np.mean(single_delta_slices, axis=0).flatten()


        #get 95 CIs of slices
        ci_burst_fRon = 1.96 * stats.sem(burst_fRon_slices, axis=0).flatten()
        ci_burst_fRoff = 1.96 * stats.sem(burst_fRoff_slices, axis=0).flatten()
        ci_burst_stp = 1.96 * stats.sem(burst_stp_slices, axis=0).flatten()
        ci_burst_delta = 1.96 * stats.sem(burst_delta_slices, axis=0).flatten()

        ci_single_fRon = 1.96 * stats.sem(single_fRon_slices, axis=0).flatten()
        ci_single_fRoff = 1.96 * stats.sem(single_fRoff_slices, axis=0).flatten()
        ci_single_stp = 1.96 * stats.sem(single_stp_slices, axis=0).flatten()
        ci_single_delta = 1.96 * stats.sem(single_delta_slices, axis=0).flatten()

        # #get activity error (SD)
        # sd_burst_fRon = np.std(burst_fRon_slices, axis=0, ddof=1).flatten()
        # sd_burst_fRoff = np.std(burst_fRoff_slices, axis=0, ddof=1).flatten()
        # sd_burst_stp = np.std(burst_stp_slices, axis=0, ddof=1).flatten()
        # sd_burst_delta = np.std(burst_delta_slices, axis=0, ddof=1).flatten()

        # sd_single_fRon = np.std(single_fRon_slices, axis=0, ddof=1).flatten()
        # sd_single_fRoff = np.std(single_fRoff_slices, axis=0, ddof=1).flatten()
        # sd_single_stp = np.std(single_stp_slices, axis=0, ddof=1).flatten()
        # sd_single_delta = np.std(single_delta_slices, axis=0, ddof=1).flatten()

        alpha=0.25

        #plot fRon data
        plt.figure()
        plt.errorbar(range(tot_points), mean_burst_fRon, yerr=ci_burst_fRon, alpha=alpha, label = 'Sequential Transition')
        plt.errorbar(range(tot_points), mean_single_fRon, yerr=ci_single_fRon, alpha=alpha, label = 'Single Transition')
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('Average REM-On Firing Rate (Hz)')
        plt.title('REM-on Firing Rate During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.savefig('figures/fig2_fRon_seq_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        #plot fRoff data
        plt.figure()
        plt.errorbar(range(tot_points), mean_burst_fRoff, yerr=ci_burst_fRoff, alpha=alpha, label = 'Sequential Transition')
        plt.errorbar(range(tot_points), mean_single_fRoff, yerr=ci_single_fRoff, alpha=alpha, label = 'Single Transition')
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('Average REM-Off Firing Rate (Hz)')
        plt.title('REM-off Firing Rate During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.savefig('figures/fig2_fRoff_seq_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        mean_fRon = np.mean(fRon_slices, axis=0).flatten()
        mean_fRoff = np.mean(fRoff_slices, axis=0).flatten()

        ci_fRon = 1.96 * stats.sem(fRon_slices, axis=0).flatten()
        ci_fRoff = 1.96 * stats.sem(fRoff_slices, axis=0).flatten()

        #plot combined data
        plt.figure()
        plt.errorbar(range(tot_points), mean_fRon, yerr=ci_fRon, alpha=alpha, label = 'REM-On Population')
        plt.errorbar(range(tot_points), mean_fRoff, yerr=ci_fRoff, alpha=alpha, label = 'REM-Off Population')
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('Average Firing Rate (Hz)')
        plt.title('Firing Rate During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis 
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.savefig('figures/fig2_fRon_fRoff_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        # #plot stp data
        # plt.figure()
        # plt.errorbar(range(tot_points), mean_burst_stp, yerr=ci_burst_stp, alpha=alpha, label = 'Sequential Transition')
        # plt.errorbar(range(tot_points), mean_single_stp, yerr=ci_single_stp, alpha=alpha, label = 'Single Transition')
        # plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        # plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        # plt.xlabel('Norm. Time')
        # plt.ylabel('Stp')
        # plt.title('stp During Normalized REM->Inter-REM->REM Periods')
        # plt.legend()
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        # plt.savefig('figures/MCCV_stp_seq_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        # plt.show()

        # plt.figure()
        # plt.errorbar(range(tot_points), mean_burst_delta, yerr=ci_burst_delta, alpha=alpha, label = 'Sequential Transition')
        # plt.errorbar(range(tot_points), mean_single_delta, yerr=ci_single_delta, alpha=alpha, label = 'Single Transition')
        # plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        # plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        # plt.xlabel('Norm. Time')
        # plt.ylabel('Delta')
        # plt.title('Delta During Normalized REM->Inter-REM->REM Periods')
        # plt.legend()
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        # plt.savefig('figures/MCCV_delta_seq_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        # plt.show()

        return mean_burst_fRon, mean_burst_fRoff, mean_burst_stp, mean_burst_delta, mean_single_fRon, mean_single_fRoff, mean_single_stp, mean_single_delta

    def avg_Ron_Roff_seq_REM_norm_REM_pre_grad(self, bin_size=60):
        tot_points = 500
        nstates_inter = int(tot_points / 2)
        nstates_rem = int(tot_points / 4)

        #get inter-rem sequences with sleepy.get_sequences
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        # delete first and last inter period so all periods are REM->inter->REM
        interSeqs = np.delete(interSeqs, 0)
        interSeqs = np.delete(interSeqs, -1)

        #get REM sequences with sleepy.get_sequences (number of REM periods should be number of inter + 1)
        remSeqs = sleepy.get_sequences(np.where(self.H[0] == 1)[0])

        #identify periods based on REM_pre
        pre_bin1_inds = []
        pre_bin2_inds = []
        pre_bin3_inds = []
        pre_bin4_inds = []
        for i in range(len(remSeqs)):
            seq = remSeqs[i]
            if len(seq) * self.dt <= bin_size:
                pre_bin1_inds.append(i)
            elif len(seq) * self.dt <= 2*bin_size:
                pre_bin2_inds.append(i)
            # elif len(seq) * self.dt <= 3*bin_size:
            #     pre_bin3_inds.append(i)
            else:
                pre_bin4_inds.append(i)
            # elif len(seq) * self.dt <= 120:
            #     pre_90_120_inds.append(i)
            # elif len(seq) * self.dt <= 150:
            #     pre_120_150_inds.append(i)
            # elif len(seq) * self.dt <= 180:
            #     pre_150_180_inds.append(i)
            # elif len(seq) * self.dt <= 210:
            #     pre_180_210_inds.append(i)
            # elif len(seq) * self.dt <= 240:
            #     pre_210_240_inds.append(i)
        print(len(pre_bin1_inds), len(pre_bin2_inds), len(pre_bin4_inds))

        #save FRon, FRoff, and stp during inter rem
        inter_fRon = []
        inter_fRoff = []
        inter_stp = []
        inter_delta = []
        for seq in interSeqs:
            inter_fRon.append(self.X[seq, 0])
            inter_fRoff.append(self.X[seq, 1])
            inter_stp.append(self.X[seq, 9])
            inter_delta.append(self.X[seq, -3])

        #save FRoff, Fron, and stp during REM
        rem_fRon = []
        rem_fRoff = []
        rem_stp = []
        rem_delta = []
        for seq in remSeqs:
            rem_fRon.append(self.X[seq, 0])
            rem_fRoff.append(self.X[seq, 1])
            rem_stp.append(self.X[seq, 9])
            rem_delta.append(self.X[seq, -3])

        #normalize inter periods
        norm_inter_fRon = []
        norm_inter_fRoff = []
        norm_inter_stp = []
        norm_inter_delta = []
        for i in range(len(interSeqs)):
            #convert firing rate/stp list to np array for time_morph
            to_np_fRon = np.array(inter_fRon[i])
            to_np_fRoff = np.array(inter_fRoff[i])
            to_np_stp = np.array(inter_stp[i])
            to_np_delta = np.array(inter_delta[i])

            #time normalize inter periods
            norm_inter_fRon.append(self.time_morph(to_np_fRon, nstates_inter))
            norm_inter_fRoff.append(self.time_morph(to_np_fRoff, nstates_inter))
            norm_inter_stp.append(self.time_morph(to_np_stp, nstates_inter))
            norm_inter_delta.append(self.time_morph(to_np_delta, nstates_inter))

        #normalize REM periods
        norm_rem_fRon = []
        norm_rem_fRoff = []
        norm_rem_stp = []
        norm_rem_delta = []
        for i in range(len(remSeqs)):
            #convert firing rate/stp list to np array for time_morph
            to_np_fRon = np.array(rem_fRon[i])
            to_np_fRoff = np.array(rem_fRoff[i])
            to_np_stp = np.array(rem_stp[i])
            to_np_delta = np.array(rem_delta[i])

            #time normalize inter periods
            norm_rem_fRon.append(self.time_morph(to_np_fRon, nstates_rem))
            norm_rem_fRoff.append(self.time_morph(to_np_fRoff, nstates_rem))
            norm_rem_stp.append(self.time_morph(to_np_stp, nstates_rem))
            norm_rem_delta.append(self.time_morph(to_np_delta, nstates_rem))

        #attach normalized inter to normalized pre and post REM
        fRon_slices = []
        fRoff_slices = []
        stp_slices = []
        delta_slices = []
        for i in range(len(norm_inter_fRon)):
            curr_fRon = []
            curr_fRon.extend(norm_rem_fRon[i]) #rem before inter-rem period
            curr_fRon.extend(norm_inter_fRon[i]) #inter-rem period
            curr_fRon.extend(norm_rem_fRon[i + 1]) #rem after inter-rem period (same as rem 
            #before next inter-rem period)

            curr_fRoff = []
            curr_fRoff.extend(norm_rem_fRoff[i])
            curr_fRoff.extend(norm_inter_fRoff[i])
            curr_fRoff.extend(norm_rem_fRoff[i + 1])

            curr_stp = []
            curr_stp.extend(norm_rem_stp[i])
            curr_stp.extend(norm_inter_stp[i])
            curr_stp.extend(norm_rem_stp[i + 1])

            curr_delta = []
            curr_delta.extend(norm_rem_delta[i])
            curr_delta.extend(norm_inter_delta[i])
            curr_delta.extend(norm_rem_delta[i + 1])
            
            fRon_slices.append(curr_fRon)
            fRoff_slices.append(curr_fRoff)
            stp_slices.append(curr_stp)
            delta_slices.append(curr_delta)

        #separate slices by REM_pre interval
        pre_bin1_fRon_slices = [fRon_slices[i] for i in pre_bin1_inds if i < len(fRon_slices)]
        pre_bin1_fRoff_slices = [fRoff_slices[i] for i in pre_bin1_inds if i < len(fRoff_slices)]
        pre_bin1_stp_slices = [stp_slices[i] for i in pre_bin1_inds if i < len(stp_slices)]
        pre_bin1_delta_slices = [delta_slices[i] for i in pre_bin1_inds if i < len(delta_slices)]

        pre_bin2_fRon_slices = [fRon_slices[i] for i in pre_bin2_inds if i < len(fRon_slices)]
        pre_bin2_fRoff_slices = [fRoff_slices[i] for i in pre_bin2_inds if i < len(fRoff_slices)]
        pre_bin2_stp_slices = [stp_slices[i] for i in pre_bin2_inds if i < len(stp_slices)]
        pre_bin2_delta_slices = [delta_slices[i] for i in pre_bin2_inds if i < len(delta_slices)]

        # pre_bin3_fRon_slices = [fRon_slices[i] for i in pre_bin3_inds if i < len(fRon_slices)]
        # pre_bin3_fRoff_slices = [fRoff_slices[i] for i in pre_bin3_inds if i < len(fRoff_slices)]
        # pre_bin3_stp_slices = [stp_slices[i] for i in pre_bin3_inds if i < len(stp_slices)]
        # pre_bin3_delta_slices = [delta_slices[i] for i in pre_bin3_inds if i < len(delta_slices)]

        pre_bin4_fRon_slices = [fRon_slices[i] for i in pre_bin4_inds if i < len(fRon_slices)]
        pre_bin4_fRoff_slices = [fRoff_slices[i] for i in pre_bin4_inds if i < len(fRoff_slices)]
        pre_bin4_stp_slices = [stp_slices[i] for i in pre_bin4_inds if i < len(stp_slices)]
        pre_bin4_delta_slices = [delta_slices[i] for i in pre_bin4_inds if i < len(delta_slices)]


        #get actvity means
        mean_bin1_fRon = np.mean(pre_bin1_fRon_slices, axis=0).flatten()
        mean_bin1_fRoff = np.mean(pre_bin1_fRoff_slices, axis=0).flatten()
        mean_bin1_stp = np.mean(pre_bin1_stp_slices, axis=0).flatten()
        mean_bin1_delta = np.mean(pre_bin1_delta_slices, axis=0).flatten()

        mean_bin2_fRon = np.mean(pre_bin2_fRon_slices, axis=0).flatten()
        mean_bin2_fRoff = np.mean(pre_bin2_fRoff_slices, axis=0).flatten()
        mean_bin2_stp = np.mean(pre_bin2_stp_slices, axis=0).flatten()
        mean_bin2_delta = np.mean(pre_bin2_delta_slices, axis=0).flatten()

        # mean_bin3_fRon = np.mean(pre_bin3_fRon_slices, axis=0).flatten()
        # mean_bin3_fRoff = np.mean(pre_bin3_fRoff_slices, axis=0).flatten()
        # mean_bin3_stp = np.mean(pre_bin3_stp_slices, axis=0).flatten()
        # mean_bin3_delta = np.mean(pre_bin3_delta_slices, axis=0).flatten()

        mean_bin4_fRon = np.mean(pre_bin4_fRon_slices, axis=0).flatten()
        mean_bin4_fRoff = np.mean(pre_bin4_fRoff_slices, axis=0).flatten()
        mean_bin4_stp = np.mean(pre_bin4_stp_slices, axis=0).flatten()
        mean_bin4_delta = np.mean(pre_bin4_delta_slices, axis=0).flatten()

        #get 95 CIs of slices
        ci_bin1_fRon = 1.96 * stats.sem(pre_bin1_fRon_slices, axis=0).flatten()
        ci_bin1_fRoff = 1.96 * stats.sem(pre_bin1_fRoff_slices, axis=0).flatten()
        ci_bin1_stp = 1.96 * stats.sem(pre_bin1_stp_slices, axis=0).flatten()
        ci_bin1_delta = 1.96 * stats.sem(pre_bin1_delta_slices, axis=0).flatten()

        ci_bin2_fRon = 1.96 * stats.sem(pre_bin2_fRon_slices, axis=0).flatten()
        ci_bin2_fRoff = 1.96 * stats.sem(pre_bin2_fRoff_slices, axis=0).flatten()
        ci_bin2_stp = 1.96 * stats.sem(pre_bin2_stp_slices, axis=0).flatten()
        ci_bin2_delta = 1.96 * stats.sem(pre_bin2_delta_slices, axis=0).flatten()

        # ci_bin3_fRon = 1.96 * stats.sem(pre_bin3_fRon_slices, axis=0).flatten()
        # ci_bin3_fRoff = 1.96 * stats.sem(pre_bin3_fRoff_slices, axis=0).flatten()
        # ci_bin3_stp = 1.96 * stats.sem(pre_bin3_stp_slices, axis=0).flatten()
        # ci_bin3_delta = 1.96 * stats.sem(pre_bin3_delta_slices, axis=0).flatten()

        ci_bin4_fRon = 1.96 * stats.sem(pre_bin4_fRon_slices, axis=0).flatten()
        ci_bin4_fRoff = 1.96 * stats.sem(pre_bin4_fRoff_slices, axis=0).flatten()
        ci_bin4_stp = 1.96 * stats.sem(pre_bin4_stp_slices, axis=0).flatten()
        ci_bin4_delta = 1.96 * stats.sem(pre_bin4_delta_slices, axis=0).flatten()


        # #get activity error (SD)
        # sd_burst_fRon = np.std(burst_fRon_slices, axis=0, ddof=1).flatten()
        # sd_burst_fRoff = np.std(burst_fRoff_slices, axis=0, ddof=1).flatten()
        # sd_burst_stp = np.std(burst_stp_slices, axis=0, ddof=1).flatten()
        # sd_burst_delta = np.std(burst_delta_slices, axis=0, ddof=1).flatten()

        # sd_single_fRon = np.std(single_fRon_slices, axis=0, ddof=1).flatten()
        # sd_single_fRoff = np.std(single_fRoff_slices, axis=0, ddof=1).flatten()
        # sd_single_stp = np.std(single_stp_slices, axis=0, ddof=1).flatten()
        # sd_single_delta = np.std(single_delta_slices, axis=0, ddof=1).flatten()

        num_lines = 4
        colors = [cm.jet(x) for x in np.linspace(0.73, 1, num_lines)]
        alpha = 0.27

        sns.set_context('paper')
        sns.set_style('white')

        #plot fRon data
        plt.figure()
        if len(pre_bin1_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin1_fRon, yerr=ci_bin1_fRon, alpha=alpha, label = f'REM_pre 0-{bin_size}',
            color=colors[0])
        if len(pre_bin2_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin2_fRon, yerr=ci_bin2_fRon, alpha=alpha, label = f'REM_pre {bin_size}-{2*bin_size}', 
            color=colors[1])
        # if len(pre_bin3_inds) > 3:
        #     plt.errorbar(range(tot_points), mean_bin3_fRon, yerr=ci_bin3_fRon, alpha=alpha, label = 'REM_pre 120-180', color=colors[2])
        if len(pre_bin4_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin4_fRon, yerr=ci_bin4_fRon, alpha=alpha, label = f'REM_pre > {2*bin_size}', 
            color=colors[3])
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('REM-On Firing Rate (Hz)')
        plt.title('REM-on Firing Rate During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        sns.despine()
        plt.savefig('figures/fig2_fRon_seq_REM_norm_REM_pre_grad_%d_bin.pdf' % bin_size, bbox_inches = "tight", dpi = 100)
        plt.show()

        #plot fRoff data
        plt.figure()
        if len(pre_bin1_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin1_fRoff, yerr=ci_bin1_fRoff, alpha=alpha, label = f'REM_pre 0-{bin_size}', color=colors[0])
        if len(pre_bin2_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin2_fRoff, yerr=ci_bin2_fRoff, alpha=alpha, label = f'REM_pre {bin_size}-{2*bin_size}', color=colors[1])
        # if len(pre_bin3_inds) > 3:
        #     plt.errorbar(range(tot_points), mean_bin3_fRoff, yerr=ci_bin3_fRoff, alpha=alpha, label = 'REM_pre 120-180', color=colors[2])
        if len(pre_bin4_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin4_fRoff, yerr=ci_bin4_fRoff, alpha=alpha, label = f'REM_pre > {2*bin_size}', color=colors[3])
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('REM-Off Firing Rate (Hz)')
        plt.title('REM-off Firing Rate During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        sns.despine()
        plt.savefig('figures/fig2_fRoff_seq_REM_norm_REM_pre_grad_%d_bin.pdf' % bin_size, bbox_inches = "tight", dpi = 100)
        plt.show()

        # mean_fRon = np.mean(fRon_slices, axis=0).flatten()
        # mean_fRoff = np.mean(fRoff_slices, axis=0).flatten()

        # ci_fRon = 1.96 * stats.sem(fRon_slices, axis=0).flatten()
        # ci_fRoff = 1.96 * stats.sem(fRoff_slices, axis=0).flatten()

        # #plot combined data
        # plt.figure()
        # plt.errorbar(range(tot_points), mean_fRon, yerr=ci_fRon, alpha=alpha, label = 'REM-On Population')
        # plt.errorbar(range(tot_points), mean_fRoff, yerr=ci_fRoff, alpha=alpha, label = 'REM-Off Population')
        # plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        # plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        # plt.xlabel('Norm. Time')
        # plt.ylabel('Average Firing Rate (Hz)')
        # plt.title('Firing Rate During Normalized REM->Inter-REM->REM Periods')
        # plt.legend()
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis 
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        # plt.savefig('figures/fig2_fRon_fRoff_REM_norm.pdf', bbox_inches = "tight", dpi = 100)
        # plt.show()

        #plot stp data
        plt.figure()
        if len(pre_bin1_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin1_stp, yerr=ci_bin1_stp, alpha=alpha, label = f'REM_pre 0-{bin_size}', 
            color=colors[0])
        if len(pre_bin2_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin2_stp, yerr=ci_bin2_stp, alpha=alpha, label = f'REM_pre {bin_size}-{2*bin_size}', 
            color=colors[1])
        # if len(pre_bin3_inds) > 3:
        #     plt.errorbar(range(tot_points), mean_bin3_stp, yerr=ci_bin3_stp, alpha=alpha, label = 'REM_pre 120-180', color=colors[2])
        if len(pre_bin4_inds) > 3:
            plt.errorbar(range(tot_points), mean_bin4_stp, yerr=ci_bin4_stp, alpha=alpha, label = f'REM_pre > {2*bin_size}', 
            color=colors[3])
        plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        plt.xlabel('Norm. Time')
        plt.ylabel('stp')
        plt.title('REM-Pressure During Normalized REM->Inter-REM->REM Periods')
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        sns.despine()
        plt.savefig('figures/fig2_stp_seq_REM_norm_REM_pre_grad_%d_bin.pdf' % bin_size, bbox_inches = "tight", dpi = 100)
        plt.show()

        # plt.figure()
        # plt.errorbar(range(tot_points), mean_pre_0_30_fRon, yerr=ci_pre_0_30_fRon, alpha=alpha, label = 'REM_pre 0-30')
        # plt.errorbar(range(tot_points), mean_pre_30_60_fRon, yerr=ci_pre_30_60_fRon, alpha=alpha, label = 'REM_pre 30-60')
        # plt.errorbar(range(tot_points), mean_pre_60_90_fRon, yerr=ci_pre_60_90_fRon, alpha=alpha, label = 'REM_pre 60-90')
        # plt.errorbar(range(tot_points), mean_pre_90_120_fRon, yerr=ci_pre_90_120_fRon, alpha=alpha, label = 'REM_pre 90-120')
        # plt.errorbar(range(tot_points), mean_pre_120_150_fRon, yerr=ci_pre_120_150_fRon, alpha=alpha, label = 'REM_pre 120-150')
        # plt.errorbar(range(tot_points), mean_pre_150_180_fRon, yerr=ci_pre_150_180_fRon, alpha=alpha, label = 'REM_pre 150-180')
        # plt.errorbar(range(tot_points), mean_pre_180_210_fRon, yerr=ci_pre_180_210_fRon, alpha=alpha, label = 'REM_pre 180-210')
        # plt.errorbar(range(tot_points), mean_pre_210_240_fRon, yerr=ci_pre_210_240_fRon, alpha=alpha, label = 'REM_pre 210-240')
        # plt.axvline(nstates_rem - 1, linestyle='--', color='gray')
        # plt.axvline(nstates_rem + nstates_inter - 1, linestyle='--', color='gray')
        # plt.xlabel('Norm. Time')
        # plt.ylabel('Average REM-On Firing Rate (Hz)')
        # plt.title('REM-on Firing Rate During Normalized REM->Inter-REM->REM Periods')
        # plt.legend()
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        # plt.savefig('figures/fig2_delta_seq_REM_norm_REM_pre_grad.pdf', bbox_inches = "tight", dpi = 100)
        # plt.show()

    def time_morph(self, X, nstates):
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

        ##### From Dr. Weber - https://github.com/tortugar/Lab/blob/master/Photometry/pyphi.py #####

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

        m = X.shape[0]
        A = upsample_mx(X, nstates)
        # now we have m * nstates rows
        if X.ndim == 1:
            Y = downsample_vec(A, int((m * nstates) / nstates))
        else:
            Y = downsample_mx(A, int((m * nstates) / nstates))
        # now we have m rows as requested
        return Y

    def laser_trig_percents(self, pre_post=300, dur=300, ci = 95, multiple=False, chunk_length=8, downsample_factor=5, group='', refractory_activation=False, save_fig=False):
        
        data_coll = []
        hypno_coll = []

        # calculate number of points to downsample hyno vectors to
        curr_points = (pre_post * 2 + dur) / self.dt
        new_points = int(curr_points / downsample_factor)
        new_dt = (pre_post * 2 + dur) / new_points

        #divide data into chunks
        if multiple:
            hrs = len(self.H[0]) *  self.dt / 3600 #length of simulation in hours
            num_chunks = int(hrs / chunk_length)
            num_samp = int(chunk_length / self.dt * 3600)
            print(f"Number of data chunks: {num_chunks}")
            for i in range(0, len(self.H[0]), num_samp):
                data_coll.append(self.X[i:i + num_samp, :])
                hypno_coll.append(self.H[0][i:i + num_samp])
        else:
            data_coll = [self.X]
            hypno_coll = [self.H[0]]

        # print(f"Data collection shape: ({len(data_coll)}, {len(data_coll[0])})")
        # print(f"Hypno collection shape: ({len(hypno_coll)}, {len(hypno_coll[0])})")

        laser_trig_df = pd.DataFrame(columns=['Timepoints', 'Sleep State', 'Percentage'])
        for k in range(len(data_coll)):
            curr_data = data_coll[k]
            curr_hypno = hypno_coll[k]

            brainstates = []
            #iterate through laser stimulation data
            for i in range(1, len(curr_data)):
                #find where laser onset begins
                if curr_data[i][-1] > 0 and curr_data[i-1][-1] == 0:
                    #append sleep states in defined period before and after laser stimulation
                    try:
                        downsampled_hypno = self.time_morph(curr_hypno[int(i - (pre_post/self.dt)): int(i + (dur/self.dt) + (pre_post/self.dt) + 1)], new_points)
                        brainstates.append(downsampled_hypno)
                    except ZeroDivisionError:
                        continue

            brainstates = np.array(brainstates)
            print(brainstates.shape)

            #remove brainstate data if length is not a full defined period (i.e. recording of stimulation at end of sleep simulation)
            for i in range(len(brainstates)):
                if len(brainstates[i]) < new_points:
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
                #calculate sleep state percents and add timepoint
                for key, value in sleep_state_dict.items():
                    sleep_state_percents.append(temp.count(value) / total)
                    states.append(key)
                    timepoints.append(i*new_dt - pre_post)

            #put data into pandas dataframe
            timepoints = np.array(timepoints)
            states = np.array(states, dtype=object)
            sleep_state_percents = np.array(sleep_state_percents)
            curr_data = np.column_stack([timepoints, states, sleep_state_percents])
            temp_df = pd.DataFrame(curr_data, columns=['Timepoints', 'Sleep State', 'Percentage'])          

            laser_trig_df = pd.concat([laser_trig_df, temp_df], ignore_index=True)

        laser_trig_df['Timepoints'] = pd.to_numeric(laser_trig_df['Timepoints'])
        laser_trig_df['Percentage'] = pd.to_numeric(laser_trig_df['Percentage'])

        #plot data with seaborn
        palette_colors = ['cyan', 'light purple', 'grey']
        cust_pal = sns.xkcd_palette(palette_colors)

        sns.set_context('paper')
        sns.set_style('white')

        fig, ax = plt.subplots()
        sns.lineplot(x="Timepoints", y="Percentage", hue='Sleep State', data=laser_trig_df, ci=ci, palette=cust_pal)
        ax.axvspan(0, dur, alpha = 0.1, color = 'blue')
        plt.ylim([-0.01, 1])
        plt.xlabel("Time (s, t = 0 is onset of laser stimulation)")
        plt.ylabel("Sleep State Percentages (%)")
        plt.title("Laser Triggered Sleep State Percentages")
        sns.despine()

        if save_fig:
            if refractory_activation:
                plt.savefig('figures/fig3_laserTrigPercents_refractory.pdf', bbox_inches = "tight", dpi = 100)
            else:
                plt.savefig('figures/fig3_laserTrigPercents_%s.pdf' % group, bbox_inches = "tight", dpi = 100)

        plt.show()

        return laser_trig_df

    def weber_fig_5b(self, seq_thresh=150, num_chunks=5, save_fig=False):

        # get inter-REM sequences
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        # split each inter-REM sequence into num_chunks equal chunks
        split_data = []
        for seq in interSeqs:
            # remove sequential inter-REM preiods
            if len(seq) * self.dt < seq_thresh:
                continue
            split_seq = np.array_split(seq, num_chunks)
            split_data.append(split_seq)
        split_data = np.array(split_data)

        num_seqs = split_data.shape[0]

        # create arrays to hold fRoff data for wake and NREM states
        wake_fRoff = np.zeros((num_seqs, num_chunks))
        nrem_fRoff = np.zeros((num_seqs, num_chunks))

        # iterate through each inter-REM sequence
        for i in range(split_data.shape[0]):
            # iterate through each chunk
            for j in range(num_chunks):
                # get sleep states for first chunk
                chunk = split_data[i,j]
                states = self.H[0][chunk]

                # get indices of the wake and NREM sections of chunk
                w_states = np.where(states == 2)
                nrem_states = np.where(states == 3)

                # split chunk into wake and NREM sections
                w_in_chunk = chunk[w_states]
                nrem_in_chunk = chunk[nrem_states]

                # get mean fRoff by sleep state and save to array
                fRoff_w_chunk = self.X[w_in_chunk, 1]
                fRoff_nrem_chunk = self.X[nrem_in_chunk, 1]
                wake_fRoff[i,j] = np.mean(fRoff_w_chunk)
                nrem_fRoff[i,j] = np.mean(fRoff_nrem_chunk)

        # average fRoff state data by chunk to have a single data point for each chunk
        wake_fRoff_by_chunk = np.nanmean(wake_fRoff, axis=0)
        nrem_fRoff_by_chunk = np.nanmean(nrem_fRoff, axis=0)


        # create arrays to make pandas conversion for stripplot
        strip_data_w = np.zeros((num_chunks*num_seqs, 2))
        strip_data_nrem = np.zeros((num_chunks*num_seqs, 2))

        # create Nx2 arrays for strip data where N = number of sequences x number of chunks
        # (i.e. flattening fRoff data, grouped by chunk). Column 0 specifies the chunk for a
        # corresponding fRoff value and column 2 holds the flattened fRoff data
        for i in range(num_chunks):
            # fill array with flattened data and chunk labels -> wake
            strip_data_w[i*num_seqs:(i+1)*num_seqs, 0] = i
            strip_data_w[i*num_seqs:(i+1)*num_seqs, 1] = wake_fRoff[:, i]

            # as above for nrem
            strip_data_nrem[i*num_seqs:(i+1)*num_seqs, 0] = i
            strip_data_nrem[i*num_seqs:(i+1)*num_seqs, 1] = nrem_fRoff[:, i]

        # convert chunk labeled data to pandas dataframe for strip plots
        strip_df_w = pd.DataFrame(strip_data_w, columns=['Chunk', 'fRoff'])
        strip_df_nrem = pd.DataFrame(strip_data_nrem, columns=['Chunk', 'fRoff'])
        
        # figure settings
        sns.set_context('paper')
        sns.set_style('white')
        linewidth = 3.5
        marker_size = 75
        strip_alpha = 0.25
        strip_size = 1.5
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,5))
        x_pos = range(num_chunks)

        # plot NREM data
        sns.stripplot(x='Chunk', y='fRoff', data=strip_df_nrem, color='gray', edgecolor='gray', 
                      linewidth=1, alpha=strip_alpha, size=strip_size, ax=ax1)
        ax1.scatter(x_pos, nrem_fRoff_by_chunk, color='k', s=marker_size)
        sns.lineplot(x=x_pos, y=nrem_fRoff_by_chunk, ax=ax1, color='k', linewidth=linewidth)
        ax1.set_xlim(left=-0.5, right=num_chunks + 0.5)
        ax1.set_title('NREM', fontsize=FONT_SIZE)
        ax1.set_ylabel('REM-Off \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center', fontsize=FONT_SIZE)
        ax1.set_xlabel('Inter-REM (norm. time)', fontsize=FONT_SIZE)
        ax1.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        ax1.get_legend().remove()

        # plot wake data
        
        s2 = sns.stripplot(x='Chunk', y='fRoff', data=strip_df_w, color='purple', edgecolor='purple', 
                      linewidth=1, alpha=strip_alpha, size=strip_size,  ax=ax2)
        s2.set(ylabel=None)
        ax2.scatter(x_pos, wake_fRoff_by_chunk, color='k', s=marker_size)
        sns.lineplot(x=x_pos, y=wake_fRoff_by_chunk, ax=ax2, markers=True, color='k', linewidth=linewidth)
        ax2.set_xlim(left=-0.5, right=num_chunks + 0.5)
        ax2.set_title('Wake', fontsize=FONT_SIZE)
        # ax2.set_ylabel('REM-Off \nFiring Rate \n(Hz)', rotation=0, ha='right', va='center', fontsize=FONT_SIZE)
        ax2.set_xlabel('Inter-REM (norm. time)', fontsize=FONT_SIZE)
        ax2.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        ax2.get_legend().remove()

        sns.despine()
        if save_fig:
            plt.savefig('figures/weber_fig_5b.pdf', bbox_inches = "tight", dpi = 100)
        plt.show()

        return wake_fRoff_by_chunk, nrem_fRoff_by_chunk

    def hysteresis_loop(self, seq_thresh=150, save_fig=False, filename='hysteresis_loop'):
        # load steady state hysteresis data
        stab_high = np.load('stab_high_fr.npy', allow_pickle=True)
        stab_low = np.load('stab_low_fr.npy', allow_pickle=True)
        instab = np.load('instab_fr.npy', allow_pickle=True)

        # extract fRon and stp data
        fRon_data = self.X[:,0]
        stp_data = self.X[:,9]

        # get inter-REM sequences
        interSeqs = sleepy.get_sequences(np.where(self.H[0] != 1)[0])

        # delete first and last inter period so all periods are REM->inter->REM
        interSeqs = np.delete(interSeqs, 0)
        interSeqs = np.delete(interSeqs, -1)

        # get REM sequences
        remSeqs = sleepy.get_sequences(np.where(self.H[0] == 1)[0])

        # stitch together indices of REM->burst inter->REM periods
        burst_inds = []
        scatter_burst = []
        for i in range(len(interSeqs)):
            temp = []
            if len(interSeqs[i]) * self.dt < seq_thresh:
                temp.extend(remSeqs[i])
                temp.extend(interSeqs[i])
                temp.extend(remSeqs[i+1])
                burst_inds.append(temp)

                # append first burst nrem index
                scatter_burst.append(interSeqs[i][0])

        # # identify indices of burst inter-REM periods
        # burst_inds = []
        # for seq in interSeqs:
        #     if len(seq) * self.dt < seq_thresh:
        #         burst_inds.extend(seq)
        # burst_inds = np.array(burst_inds)

        # get paired fRon and stp data for burst REM->inter->REM periods
        burst_fRon_data = []
        burst_stp_data = []
        for seq in burst_inds:
            burst_fRon_data.append(fRon_data[seq])
            burst_stp_data.append(stp_data[seq])
        # burst_fRon_data = fRon_data[burst_inds]
        # burst_stp_data = stp_data[burst_inds]

        scatter_fRon_data = []
        scatter_stp_data = []
        for ind in scatter_burst:
            scatter_fRon_data.append(fRon_data[ind])
            scatter_stp_data.append(stp_data[ind])

        # create colors for sequences
        color = []
        num_lines = 10
        for i in range(num_lines):
            color.append('#%06X' % randint(0, 0xFFFFFF))

        # plot scattered hysteresis data
        loop_lw = 1

        plt.figure()
        plt.plot(stp_data, fRon_data, color='b', linewidth=loop_lw, alpha=0.7)
        # for i in range(num_lines):
        # # for i in range(len(burst_fRon_data)):
        #     plt.plot(burst_stp_data[i], burst_fRon_data[i], color=color[i], linewidth=loop_lw+0.35)
        
        plt.scatter(scatter_stp_data, scatter_fRon_data, color='red')
        plt.plot(stab_low[:,0], stab_low[:,1], color='black', lw=3)
        plt.plot(stab_high[:,0], stab_high[:,1], color='black', lw=3)            
        plt.plot(instab[:,0], instab[:,1], '--', color='gray', lw=3)
        plt.axhline(y=self.theta_R, color='k', linestyle='dashed')
        plt.xlabel('stp')
        plt.ylabel('fRon')
        plt.xlim([0.6, 1.1])
        sns.despine()

        if save_fig:
            plt.savefig('figures/' + filename + '.pdf', bbox_inches = "tight", dpi = 100)

        plt.show()
    
    def end_of_state_stp_hist(self, state_name, save_fig=False, filename='endOfState_stp_%s'):

        state_map = {'rem': 1, 'wake': 2, 'nrem': 3}

        state_name = state_name.lower()

        # convert state name to number from input state
        try:
            state = state_map[state_name]
        except KeyError:
            print('State must be rem, wake, or nrem')
            return

        # get all sequences of that state in the hypnogram
        state_seqs = sleepy.get_sequences(np.where(self.H[0] == state)[0])

        # save stp value at the end of all of those states
        stp_vals = np.zeros((len(state_seqs),))
        for i in range(len(state_seqs)):
            seq = state_seqs[i]
            end_of_state_ind = seq[-1]
            end_of_state_stp = self.X[end_of_state_ind, 9]
            stp_vals[i] = end_of_state_stp

        # plot stp histogram
        plt.figure()
        sns.histplot(stp_vals, color='blue')
        plt.ylabel('Count', rotation=0, ha='right', va='center')
        plt.xlabel('stp')
        plt.title('Stp at the end of %s' % (state_name.upper() if state_name != 'wake' else state_name.title()))

        if save_fig:
            plt.savefig('figures/' + (filename % state_name) + '.pdf', bbox_inches = "tight", dpi = 100)

        return stp_vals

    def stp_nrem_after_rem(self, p=0, save_fig=False, filename='stp_nrem_after_rem'):

        rem_seqs = sleepy.get_sequences(np.where(self.H[0] == 1)[0])
        nrem_seqs = sleepy.get_sequences(np.where(self.H[0] == 3)[0])
        
        # get all non-wake sequences
        rem_seqs.extend(nrem_seqs)
        rem_nrem_seqs = sorted(rem_seqs, key=lambda x: x[0])
        # print(len(rem_nrem_seqs))

        # classify each sequence as rem or nrem from hypnogram
        seq_labels = np.array([self.H[0][seq[0]] for seq in rem_nrem_seqs])
        # print(seq_labels)
        

        # use diff to determine where transitions from rem to nrem occur (avoiding nrem -> wake -> nrem, rem -> wake -> rem)
        seq_diffs = np.diff(seq_labels)
        # print(seq_diffs)

        # get stp value of all transition from rem to nrem
        stp_rem_to_nrem_direct = []
        stp_rem_to_nrem_indirect = []
        rem_durs_direct = []
        rem_durs_indirect = []
        for i, transition in enumerate(seq_diffs):
            if transition > 0:
                nrem_ind = (rem_nrem_seqs[i+1])[0]
                rem_seq = rem_nrem_seqs[i]
                
                # transition from rem directly to nrem
                if nrem_ind - rem_seq[-1] == 1:
                    stp_rem_to_nrem_direct.append(self.X[nrem_ind,9])
                    rem_durs_direct.append(len(rem_seq) * self.dt)
                # transition from rem to wake to nrem
                else:
                    stp_rem_to_nrem_indirect.append(self.X[nrem_ind,9])
                    rem_durs_indirect.append(len(rem_seq) * self.dt)


        stp_rem_to_nrem_direct = np.array(stp_rem_to_nrem_direct)
        stp_rem_to_nrem_indirect = np.array(stp_rem_to_nrem_indirect)
        rem_durs_direct = np.array(rem_durs_direct)
        rem_durs_indirect = np.array(rem_durs_indirect)
        print(f'Direct: {stp_rem_to_nrem_direct.shape, rem_durs_direct.shape}')
        print(f'Indirect: {stp_rem_to_nrem_indirect.shape, rem_durs_indirect.shape}')

        #plot data in scatter plot
        if p == 1:
            sns.set(font_scale=1)

            plt.figure()
            sns.set_context('paper')
            sns.set_style('white')
            plt.scatter(rem_durs_direct, stp_rem_to_nrem_direct, color='blue', label='REM->NREM')
            plt.scatter(rem_durs_indirect, stp_rem_to_nrem_indirect, color='red', label='REM->Wake->NREM')
            plt.xlabel('REM_pre (s)')
            plt.ylabel('STP', rotation=0, ha='center', va='center', labelpad=20)
            plt.title('STP From First NREM State Following REM')
            plt.legend()
            # plt.text(max(REM_durations) - 25, m * max(REM_durations) + (b + 50), f'R^2: {round(r**2, 2)}', fontsize = 12)
            sns.despine()
            if save_fig:
                plt.savefig('figures/' + filename + '.pdf', bbox_inches = "tight", dpi = 100)
            plt.show()

        return rem_nrem_seqs, seq_labels, seq_diffs, stp_rem_to_nrem_direct, stp_rem_to_nrem_indirect


    #TODO
    def delta_stp_inter(self):
        #get starting points for inter-REM, divided into sequential and single categories
        burstStarts, longStarts = self.get_inter_seq_starts()

        #define time to record prior to and after inter-REM start points
        prePostTime = 30
        prePostPoints = int(prePostTime / self.dt)

        #anonymous function to get timepoint ranges pre and post of start
        timePoints = lambda start: np.arange(start - prePostPoints, start + prePostPoints + 1, dtype=int) 

        #save REM-on and REM-off activity over pre-post period for each sequential REM -> inter transition
        burstPrePostRon = np.empty((0, 2*prePostPoints + 1), float)
        burstPrePostRoff = np.empty((0, 2*prePostPoints + 1), float)
        burstPrePostStp = np.empty((0, 2*prePostPoints + 1), float)
        for start in burstStarts:
            burstPrePostRon = np.append(burstPrePostRon, [self.X[timePoints(start), 0]], axis=0)
            burstPrePostRoff = np.append(burstPrePostRoff, [self.X[timePoints(start), 1]], axis=0)
            burstPrePostStp = np.append(burstPrePostStp, [self.X[timePoints(start), 9]], axis=0)

        #save REM-on and REM-off activity over pre-post period for each single REM -> inter transition
        longPrePostRon = np.empty((0, 2*prePostPoints + 1), float)
        longPrePostRoff = np.empty((0, 2*prePostPoints + 1), float)
        longPrePostStp = np.empty((0, 2*prePostPoints + 1), float)
        for start in longStarts:
            longPrePostRon = np.append(longPrePostRon, [self.X[timePoints(start), 0]], axis=0)
            longPrePostRoff = np.append(longPrePostRoff, [self.X[timePoints(start), 1]], axis=0)
            longPrePostStp = np.append(longPrePostStp, [self.X[timePoints(start), 9]], axis=0)
        

        return None



