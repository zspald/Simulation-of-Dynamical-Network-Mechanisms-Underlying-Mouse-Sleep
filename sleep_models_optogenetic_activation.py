# %%
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy import signal


def mi_model(X, t):
    """
    Implementation of MI model as described in Dunmyre et al., 2014 

    [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h]
    """
    [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X

    # Parameters
    R_max = 5.0
    Roff_max = 5.0
    W_max = 5.50
    S_max = 5.0

    tau_Roff = 2.0
    tau_R = 1.0
    tau_W = 25.0
    #tau_W = 0.5
    tau_S = 10.0

    alpha_Roff = 2
    alpha_R = 0.5
    alpha_W = 0.5
    beta_R = -0.5
    beta_W = -0.3
    alpha_S = 0.25

    gamma_R = 4.0
    gamma_Roff = 5.0
    gamma_W = 5.0
    gamma_S = 4.0

    k1_Roff = 0.8
    k2_Roff = 7.0
    k1_S = 0
    k2_S = -1.5

    stp_max = 1.2
    stp_min = -0.8
    stp_r = 0.0
    tau_stpW = 30.0
    h_max = 0.8
    h_min = 0.0

    theta_R = 1.5
    theta_W = 1.5

    tau_stpup = 400.0
    tau_stpdown = 400.0
    tau_hup = 600.0
    tau_hdown = 700.0

    g_Roff2R = -2.0
    g_R2Roff = -5.0
    g_S2W = -2.0
    g_W2S = -2.0
    g_R2W = 0.1
    g_W2Roff = 0
    g_Roff2W = 0

    tau_CR = 10.0
    tau_CRoff = 1.0
    tau_CW = 10.0
    tau_CS = 10.0

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


def mi_model_noise(X, t):
    """
    Full deterministic MI model 
    """
    [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff,
        C_S, C_W, stp, h, delta, omega, sigma] = X

    # Parameters
    R_max = 5.0
    Roff_max = 5.0
    W_max = 5.50
    S_max = 5.0

    tau_Roff = 1.0
    tau_R = 1.0
    tau_W = 25.0
    tau_S = 10.0

    alpha_Roff = 1.5  # 1.5, 2
    alpha_R = 0.5
    alpha_W = 0.5
    beta_R = -0.5
    beta_W = -0.3
    alpha_S = 0.25

    gamma_R = 4.0
    gamma_Roff = 5.0
    gamma_W = 5.0
    gamma_S = 4.0

    k1_Roff = 0.8
    k2_Roff = 7.0
    k1_S = 0
    k2_S = -1.5

    stp_max = 1.2
    stp_min = -0.8
    stp_r = 0.0
    tau_stpW = 1000.0
    h_max = 0.8
    h_min = 0.0
    omega_max = 0.1  # 0.02
    omega_min = 0.00

    theta_R = 1.5
    theta_W = 1.5

    tau_stpup = 1000.0  # 400.0
    tau_stpdown = 1000.0  # 400.0
    tau_hup = 600.0
    tau_hdown = 2000.0
    tau_omega = 20.0  # 10.0
    tau_stim = 5.0  # 10.0

    g_Roff2R = -2.0  # -2.0
    g_R2Roff = -5.0
    g_S2W = -2.0
    g_W2S = -2.0
    g_W2R = 0.0
    g_R2W = 0.0
    g_W2Roff = 0
    g_Roff2W = 0
    g_Roff2S = 0

    tau_CR = 10.0
    tau_CRf = 1
    tau_CRoff = 10.0  # 1.0
    tau_CW = 10.0
    tau_CS = 10.0

    def X_inf(c, X_max, beta, alpha): return (
        0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))

    def CX_inf(f, gamma): return np.tanh(f/gamma)
    def beta_X(y, k1_X, k2_X): return k2_X * (y - k1_X)
    # heavyside function
    def H(x): return 1 if x > 0 else 0

    # steady-state function for REM-ON popluation
    def R_inf(c): return X_inf(c, R_max, beta_R, alpha_R)
    # firing rate of REM (R) population
    dF_R = (R_inf(C_Roff * g_Roff2R + C_W * g_W2R) - F_R) / tau_R
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
    dF_Roff = (Roff_inf(C_R * g_R2Roff + C_W * g_W2Roff +
                        delta + sigma) - F_Roff) / tau_Roff

    def CRoff_inf(x): return CX_inf(x, gamma_Roff)
    dC_Roff = (CRoff_inf(F_Roff) - C_Roff) / tau_CRoff

    # Wake population
    def W_inf(c): return X_inf(c, W_max, beta_W, alpha_W)
    # firing rate of REM (R) population
    dF_W = (W_inf(C_S * g_S2W + C_Rf * g_R2W +
                  C_Roff*g_Roff2W + delta) - F_W) / tau_W
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
    dF_S = (S_inf(C_W * g_W2S + C_Roff * g_Roff2S) - F_S) / tau_S
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
# run mi_model_noise to determine sleep states at all timesteps specified below
# function below takes in optogenetic activation period T with laser information (Hz, pulse, train) to apply stimulation to specified neuronal populations
def run_mi_model_noise(X0, dt, T, Hz, pulse, train):
    a = int(T[0]/dt)
    b = int(T[1]/dt)

    sigma = Hz * pulse * train

    n = int(np.round(5*3600/dt))
    X = np.zeros((n, len(X0)))
    X[0, :] = np.array(X0)
    for i in range(1, n):
        if i>=a and i<=b:
            X[i-1, -1] = sigma
        else:
            X[i-1, -1] = 0
        grad = mi_model_noise(X[i-1, :], 0)
        omega = X[i-1, -2]
        p_stim = 1 - np.exp(-omega * dt)
        p = np.random.binomial(1, p_stim)

        X[i, :] = X[i-1, :] + grad * dt

        if p > 0:
            # print "motor noise"
            X[i, -3] += 3  # 10

    return X


# %%

# create hypnogram of data from the model showing mouse sleep states across the recording period
# plots showing firing rates, sleep pressures, and optogenetic activation period are included for analysis
def hypnogram(X, theta_R, theta_W, dt=0.05, p=0):
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
        plt.ylim(0, 4)
        plt.ylabel('Sigma')

    return H

# %%
# create hypnogram for mouse with initial parameters X0 specified below


if __name__ == "__main__":
    #[F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, delta, omega, sigma] = X
    T = [0, 0]
    X0 = [5.0, 0.0, 4.0, 5.0, 0.2, 1.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0, 1.0, 0]
    H = hypnogram(run_mi_model_noise(X0, 0.05, T, 2.0, 1.0, 1.0), 1.5, 1.5, p=1)


# %%
T=300
D=900
N=24
shift = np.pi / 2  # number of cycles to shift (1/4 cycle in your example)
x = np.linspace(0, T*N, 10000, endpoint=False)
y=(signal.square(np.pi * (1/T) * x + 2*shift*np.pi))+1
plt.plot(x,y)
plt.ylim(-1, 3)
plt.xlim(0, T*N)


# %%
