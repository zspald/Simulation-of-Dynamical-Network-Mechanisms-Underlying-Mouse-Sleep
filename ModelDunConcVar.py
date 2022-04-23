"""Flip-Flop Model from original Dunmyre 2014 paper

@author: Zachary Spalding
"""

import random
import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy import signal, stats
import sleepy


class ModelDunConcVar():
    """Object for Flip-Flop Model from original Dunmyre 2014 paper with concentration noise. 
    Original parameter values and dynamics from Dunmyre 2014
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

    alpha_Roff = 0.5  # 1.5, 2
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
    tau_stpW = 30.0 # 1000.0
    h_max = 0.6 # 0.8
    h_min = 0.2 # 0.0
    omega_max = 0.01  # 0.02, 0.1
    omega_min = 0.003 # 0.00

    theta_R = 1.5 # 1.5
    theta_W = 1.5 # 1.5

    tau_stpup = 400.0  # 400.0, 1000.0
    tau_stpdown = 400.0  # 400.0, 1000.0
    tau_hup = 600.0 # 600.0, 400.0
    tau_hdown = 700.0 # 2000.0
    tau_omega = 5.0  # 10.0, 20.0
    tau_stim = 10.0  # 10.0, 5.0

    g_Roff2R = -2.0  # -2.0
    g_R2Roff = -5.0 #-5.0
    g_S2W = -2.0 #-2.0, -5.0
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

    delta_update = 10 # 10.0, 3.0

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
        """Initialization of ModelOriginal object

        Args:
            X0 (list): initial conditions for model
            self.dt (float): timestep for simulation
        """
        self.X0 = np.array(X0)
        self.dt = dt
        self.X = []
        self.H = []

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

    def run_mi_model(self, hrs, group = 'None', sigma = 0, dur = 5*60, delay = 0, gap = 15*60, gap_rand = False, gap_range = [1, 25], noise = False):
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
                dstp = (self.stp_r - stp) / self.tau_stpW # stp decreases during wake
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
            dF_Roff = (Roff_inf(C_R * self.g_R2Roff + C_W * self.g_W2Roff + sigma_Roff) - F_Roff) / self.tau_Roff

            def CRoff_inf(x): return CX_inf(x, self.gamma_Roff)
            dC_Roff = (zeta_Roff * CRoff_inf(F_Roff) - C_Roff) / self.tau_CRoff

            # Wake population
            def W_inf(c): return X_inf(c, self.W_max, self.beta_W, self.alpha_W)
            # firing rate of REM (R) population
            dF_W = (W_inf(C_S * self.g_S2W + C_Rf * self.g_R2W +
                        C_Roff * self.g_Roff2W + delta + sigma_W) - F_W) / self.tau_W
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
        for i in range(1, n):
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
                #delta update
                omega = simX[i-1, -2]
                p_stim = 1 - np.exp(-omega * self.dt)
                pDelt = np.random.binomial(1, p_stim)

                if pDelt > 0:
                    # print "motor noise"
                    simX[i, -3] += self.delta_update  # 10

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

    def hypnogram(self, p = 0):
        """Converts simulated neuron data from a simulation of the MI model to an array of sleep states over time (to be plotted as a hypnogram)

        Keyword Arguments:
            p {int} -- plots hypnogram if and corresponding sleep data if equal to 1 (default: {0})

        Returns:
            None - updates hypnogram of model object
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
            axes6.plot(t, self.X[:, -1])
            plt.xlim(t[0], t[-1])
            plt.ylim(0, max(self.X[:, -1]) + 1)
            plt.ylabel('Sigma')

            plt.show()

    def hypnogram_fig1(self, p=0, p_zoom=0, save=False, filename='fig1_dun_hypno'):
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

    def avg_Ron_and_Roff_by_state(self):
        """Calculates and plots average firing rate (+/- one standard deviation) of REM-on and REM-off neurons during REM, wake, and NREM sleep states

        Arguments:
            X {numpy array} -- simulated neuron population data from MI model simulation
            H {numpy array} -- sleep state data from simulated data above

        Returns:
            [list] -- list of REM-on (indices 0-2) and REM-off (indices 3-5) average firing rates during each sleep state
        """

        #extract firing rates for Ron and Roff pop at each sleep staets
        Ron_rem = self.X[(np.where(self.H[0]==1)), 0]
        Ron_wake = self.X[(np.where(self.H[0]==2)), 0]
        Ron_nrem = self.X[(np.where(self.H[0]==3)), 0]
        Roff_rem = self.X[(np.where(self.H[0]==1)), 1]
        Roff_wake = self.X[(np.where(self.H[0]==2)), 1]
        Roff_nrem = self.X[(np.where(self.H[0]==3)), 1]

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
        Ron_std[0] = np.std(Ron_rem, ddof=1)
        Ron_std[1] = np.std(Ron_wake, ddof=1)
        Ron_std[2] = np.std(Ron_nrem, ddof=1)
        Roff_std[0] = np.std(Roff_rem, ddof=1)
        Roff_std[1] = np.std(Roff_wake, ddof=1)
        Roff_std[2] = np.std(Roff_nrem, ddof=1)

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

    def inter_REM(self, seq_thresh=100, p=0, zoom_out=0, nremOnly=False, log=False, rem_pre_split=False, save=False, filename='fig1_remPre_dun'):
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
                
                plt.figure(figsize=(10,6))
                sns.set_context('paper')
                sns.set_style('white')
                plt.scatter(seq_rem, seq_inter, color='gray', alpha=0.25, label='Sequential REM')
                plt.scatter(nonseq_rem, nonseq_inter, color='blue', alpha=0.65, label='Singlular REM')
                plt.plot(REM_durations, np.multiply(m,REM_durations) + b, color = 'blue')
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

                _, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

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

    def end_of_state_stp_hist(self, state_name, save_fig=False, filename='endOfState_stp_%s_dun'):

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

    def stp_nrem_after_rem(self, p=0, save_fig=False, filename='stp_nrem_after_rem_dun'):

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
                # transition from REM->inter or vice versa
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

            #trendlines
            # TODO make trendline for separate cloud
            dirM, dirB, dirR, dirP, _ = stats.linregress(rem_durs_direct, stp_rem_to_nrem_direct)
            indirM, indirB, indirR, indirP, _ = stats.linregress(rem_durs_indirect, stp_rem_to_nrem_indirect)

            #fucntion for decimal rounding in plot text
            def round_decimals_up(number:float, decimals:int=2):
                """
                Returns a value rounded up to a specific number of decimal places. From https://kodify.net/python/math/round-decimals/
                """
                if not isinstance(decimals, int):
                    raise TypeError("decimal places must be an integer")
                elif decimals < 0:
                    raise ValueError("decimal places has to be 0 or more")
                elif decimals == 0:
                    return math.ceil(number)

                factor = 10 ** decimals
                return math.ceil(number * factor) / factor

            #plot data in scatter plot
            if p == 1:
                sns.set(font_scale=1)

                plt.figure(figsize=(8,5))
                sns.set_context('paper')
                sns.set_style('white')
                plt.scatter(rem_durs_direct, stp_rem_to_nrem_direct, color='blue', alpha=0.35, label='REM->NREM' + \
                    ':' + f'R^2={round_decimals_up(dirR**2, 2)}' + ', ' + f'P={round_decimals_up(dirP, 3)}')
                plt.plot(rem_durs_direct, np.multiply(dirM, rem_durs_direct) + dirB, + dirB, color = 'blue')
                plt.scatter(rem_durs_indirect, stp_rem_to_nrem_indirect, color='red', alpha=0.35, label='REM->Wake->NREM' + \
                    ':' + f'R^2={round_decimals_up(indirR**2, 2)}' + ', ' + f'P={round_decimals_up(indirP**2, 3)}')
                plt.plot(rem_durs_indirect, np.multiply(indirM, rem_durs_indirect) + indirB, + indirB, color = 'red')
                plt.xlabel('REM_pre (s)', fontsize=12)
                plt.ylabel('STP', rotation=0, ha='center', va='center', labelpad=20, fontsize=12)
                plt.title('STP From First NREM State Following REM', fontsize=12)
                plt.legend(fontsize=12)
                
                print(f'Regression Line REM->NREM: Inter = {np.round(dirM, 5)}(REM_pre) + {np.round(dirB, 2)}, R^2={round(dirR**2, 2)}, P={round_decimals_up(dirP**2, 3)}')
                print(f'Regression Line REM->Wake->REM: Inter = {np.round(indirM, 5)}(REM_pre) + {np.round(indirB, 2)}, R^2={round(indirR**2, 2)}, P={round_decimals_up(indirP**2, 3)}')

                # plt.text(max(rem_durs_direct) - 35, dirM * (max(rem_durs_direct) - 35) + (dirB + 0.05), f'R^2: {round(dirR**2, 2)}' + ', ' + f'P: {round(dirP, 4)}', fontsize = 12)
                # plt.text(min(rem_durs_indirect) + 25, indirM * (min(rem_durs_indirect) + 25) + (indirB + 0.05), f'R^2: {round(indirR**2, 2)}' + ', ' + f'P: {round(indirP**2, 4)}', fontsize = 12)

                sns.despine()
                if save_fig:
                    plt.savefig('figures/' + filename + '.pdf', bbox_inches = "tight", dpi = 100)
                plt.show()

            return rem_nrem_seqs, seq_labels, seq_diffs, stp_rem_to_nrem_direct, stp_rem_to_nrem_indirect
