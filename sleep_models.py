import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def ri_model(X):
    """
    Reciprocal interaction model
    :param X: X = [F_R, F_Roff]
    :return: [dF_R, dF_Roff]
    """
    [F_R, F_Roff] = X

    g_R2Roff = 7.
    g_Roff2R = -7.0
    
    # self-excitation of REM-ON population
    g_R2R = 6.0
    # self-inhibtion of REM-OFF population
    g_Roff2Roff = -1.0
    
    gamma_R = 5.0
    gamma_Roff = 5.0
    beta_R = 0.0
    beta_Roff = 1.5
    alpha_R = 0.5
    alpha_Roff = 0.5
    R_max = 5.0
    Roff_max = 5.0
    tau_R = 240.0      # changed from 60
    tau_Roff = 240.0   # changed from 60

    # steady-state function of firing rate
    X_inf  = lambda c, X_max, beta, alpha: (0.5 * X_max * (1 + np.tanh((c - beta) / alpha)))
    # steady-state function for neurotransmitter concentration:
    CX_inf = lambda f, gamma: np.tanh(f / gamma)
    # Note: firing rates are always >=0, therefore CX_inf only takes values between 0 and 1

    R_inf     = lambda x:  X_inf(x, R_max, beta_R, alpha_R)
    Roff_inf  = lambda x:  X_inf(x, Roff_max, beta_Roff, alpha_Roff)
    CR_inf    = lambda x: CX_inf(x, gamma_R)
    CRoff_inf = lambda x: CX_inf(x, gamma_Roff)

    # The synaptic input is expressed as wheighted sum of neurotransmitter concentrations 
    # which depend on the pre-synaptic firing rate.
    dF_R      = (R_inf(g_Roff2R * CRoff_inf(F_Roff) + g_R2R * CR_inf(F_R)) - F_R) / tau_R
    dF_Roff   = (Roff_inf(g_R2Roff * CR_inf(F_R) + g_Roff2Roff * CRoff_inf(F_Roff)) - F_Roff) / tau_Roff

    Y = np.array([dF_R, dF_Roff])
    return Y



def mi_model(X,t):
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

    tau_stpup   = 400.0
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

    tau_CR    = 10.0
    tau_CRoff = 1.0
    tau_CW = 10.0
    tau_CS = 10.0

    X_inf  = lambda c, X_max, beta, alpha : (0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))
    CX_inf = lambda f, gamma : np.tanh(f/gamma)
    beta_X = lambda y, k1_X, k2_X : k2_X * (y - k1_X)
    # heavyside function
    H = lambda x: 1 if x > 0 else 0


    # steady-state function for REM-ON popluation
    R_inf = lambda c : X_inf(c, R_max, beta_R, alpha_R)
    # firing rate of REM (R) population
    dF_R = (R_inf(C_Roff * g_Roff2R) - F_R) / tau_R
    # steady state for neurotransmitter concentration:
    CR_inf = lambda x : CX_inf(x, gamma_R)
    # dynamics for neurotransmitter
    dC_R = (CR_inf(F_R) - C_R) / tau_CR

    # homeostatic REM pressure
    if F_W > theta_W:
        dstp = (stp_r - stp) / tau_stpW
    else:
        dstp = (H(theta_R - F_R) * (stp_max - stp)) / tau_stpup + (H(F_R - theta_R) * (stp_min - stp)) / tau_stpdown

    # REM-OFF population
    beta_Roff = lambda y : beta_X(y, k1_Roff, k2_Roff)
    Roff_inf = lambda c : X_inf(c, Roff_max, beta_Roff(stp), alpha_Roff)
    dF_Roff = (Roff_inf(C_R * g_R2Roff + C_W * g_W2Roff) - F_Roff) / tau_Roff
    CRoff_inf = lambda x : CX_inf(x, gamma_Roff)
    dC_Roff = (CRoff_inf(F_Roff) - C_Roff) / tau_CRoff

    # Wake population
    W_inf = lambda c : X_inf(c, W_max, beta_W, alpha_W)
    # firing rate of REM (R) population
    dF_W = (W_inf(C_S * g_S2W + C_R * g_R2W + C_Roff*g_Roff2W) - F_W) / tau_W
    # steady state for neurotransmitter concentration:
    CW_inf = lambda x : CX_inf(x, gamma_W)
    # dynamics for neurotransmitter
    dC_W = (CW_inf(F_W) - C_W) / tau_CW

    # homeostatic sleep drive
    dh = (H(F_W - theta_W) * (h_max - h)) / tau_hup + (H(theta_W - F_W) * (h_min - h)) / tau_hdown

    # Sleep population
    beta_S = lambda y: beta_X(y, k1_S, k2_S)
    S_inf = lambda c : X_inf(c, S_max, beta_S(h), alpha_S)
    # firing rate of REM (R) population
    dF_S = (S_inf(C_W * g_W2S) - F_S) / tau_S
    # steady state for neurotransmitter concentration:
    CS_inf = lambda x : CX_inf(x, gamma_S)
    # dynamics for neurotransmitter
    dC_S = (CS_inf(F_S) - C_S) / tau_CS

    # [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X
    Y = [dF_R, dF_Roff, dF_S, dF_W, dC_R, dC_Roff, dC_S, dC_W, dstp, dh]
    
    return np.array(Y)






def mi_model_noise(X, t):
    """
    Full deterministic MI model 
    """
    [F_R, F_Roff, F_S, F_W, C_R, C_Rf, C_Roff, C_S, C_W, stp, h, delta, omega] = X

    # Parameters
    R_max = 5.0
    Roff_max = 5.0
    W_max = 5.50
    S_max = 5.0

    tau_Roff = 1.0
    tau_R = 1.0
    tau_W = 25.0
    tau_S = 10.0


    alpha_Roff = 1.5 #1.5, 2
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
    omega_max = 0.1 #0.02
    omega_min = 0.00


    theta_R = 1.5
    theta_W = 1.5

    tau_stpup   = 1000.0 #400.0
    tau_stpdown = 1000.0 #400.0
    tau_hup   = 600.0
    tau_hdown = 2000.0
    tau_omega = 20.0 #10.0
    tau_stim  = 5.0 #10.0

    g_Roff2R = -2.0#-2.0
    g_R2Roff = -5.0
    g_S2W = -2.0
    g_W2S = -2.0
    g_W2R = 0.0
    g_R2W = 0.0
    g_W2Roff = 0
    g_Roff2W = 0
    g_Roff2S = 0

    tau_CR    = 10.0
    tau_CRf = 1
    tau_CRoff = 10.0#1.0
    tau_CW = 10.0
    tau_CS = 10.0

    X_inf  = lambda c, X_max, beta, alpha : (0.5 * X_max * (1 + np.tanh((c-beta)/alpha)))
    CX_inf = lambda f, gamma : np.tanh(f/gamma)
    beta_X = lambda y, k1_X, k2_X : k2_X * (y - k1_X)
    # heavyside function
    H = lambda x: 1 if x > 0 else 0


    # steady-state function for REM-ON popluation
    R_inf = lambda c : X_inf(c, R_max, beta_R, alpha_R)
    # firing rate of REM (R) population
    dF_R = (R_inf(C_Roff * g_Roff2R + C_W * g_W2R) - F_R) / tau_R
    # steady state for neurotransmitter concentration:
    CR_inf = lambda x : CX_inf(x, gamma_R)
    # dynamics for neurotransmitter
    dC_R = (CR_inf(F_R) - C_R) / tau_CR
    dC_Rf = (CR_inf(F_R) - C_Rf) / tau_CRf

    # homeostatic REM pressure
    if F_W > theta_W:
        dstp = (stp_r - stp) / tau_stpW
        #dstp = 0
    else:
        dstp = (H(theta_R - F_R) * (stp_max - stp)) / tau_stpup + (H(F_R - theta_R) * (stp_min - stp)) / tau_stpdown


    # update omega
    # parameter determining, how likely it is that a excitatory stimulus will happen during REM sleep
    if F_R > theta_R:
        domega = (omega_max - omega) / tau_omega
    else :
        domega = (omega_min - omega) / tau_omega
            

    # update delta
    ddelta = -delta / tau_stim


    # REM-OFF population
    beta_Roff = lambda y : beta_X(y, k1_Roff, k2_Roff)
    Roff_inf = lambda c : X_inf(c, Roff_max, beta_Roff(stp), alpha_Roff)
    dF_Roff = (Roff_inf(C_R * g_R2Roff + C_W * g_W2Roff + delta) - F_Roff) / tau_Roff
    CRoff_inf = lambda x : CX_inf(x, gamma_Roff)
    dC_Roff = (CRoff_inf(F_Roff) - C_Roff) / tau_CRoff

    # Wake population
    W_inf = lambda c : X_inf(c, W_max, beta_W, alpha_W)
    # firing rate of REM (R) population
    dF_W = (W_inf(C_S * g_S2W + C_Rf * g_R2W + C_Roff*g_Roff2W + delta) - F_W) / tau_W
    # steady state for neurotransmitter concentration:
    CW_inf = lambda x : CX_inf(x, gamma_W)
    # dynamics for neurotransmitter
    dC_W = (CW_inf(F_W) - C_W) / tau_CW

    # homeostatic sleep drive
    dh = (H(F_W - theta_W) * (h_max - h)) / tau_hup + (H(theta_W - F_W) * (h_min - h)) / tau_hdown

    # Sleep population
    beta_S = lambda y: beta_X(y, k1_S, k2_S)
    S_inf = lambda c : X_inf(c, S_max, beta_S(h), alpha_S)
    # firing rate of REM (R) population
    dF_S = (S_inf(C_W * g_W2S + C_Roff * g_Roff2S) - F_S) / tau_S
    # steady state for neurotransmitter concentration:
    CS_inf = lambda x : CX_inf(x, gamma_S)
    # dynamics for neurotransmitter
    dC_S = (CS_inf(F_S) - C_S) / tau_CS

    # [F_R, F_Roff, F_S, F_W, C_R, C_Roff, C_S, C_W, stp, h] = X
    Y = [dF_R, dF_Roff, dF_S, dF_W, dC_R, dC_Rf, dC_Roff, dC_S, dC_W, dstp, dh, ddelta, domega]
    return np.array(Y)



def run_mi_model_noise(X0, dt):
    
    n = int(np.round(5*3600/dt))
    X = np.zeros((n, len(X0)))
    X[0,:] = np.array(X0)
    for i in range(1,n):
        grad = mi_model_noise(X[i-1,:], 0)
        omega = X[i-1,-1]
        p_stim = 1 - np.exp(-omega * dt)
        p = np.random.binomial(1, p_stim)
        
        X[i,:] = X[i-1,:] + grad * dt

        if p>0:
            #print "motor noise"
            X[i,-2] += 3 #10            

    return X
    

#%%
def hypnogram(X, theta_R, theta_W, dt=0.05, p=0):
    R = X[:,0]
    W = X[:,3]
    H = np.zeros((1,len(R)))
    
    idx_r = np.where(R>theta_R)[0]
    idx_w = np.where(W>theta_W)[0]
    H[0,:] = 3
    H[0,idx_r] = 1
    H[0,idx_w] = 2
    
    sns.set(font_scale=0.6)
    
    # make plot
    if p==1:    
        plt.figure()
        axes1=plt.axes([0.1, 0.8, 0.8, 0.1])
        plt.imshow(H)
        plt.axis('tight')
        cmap = plt.cm.jet
        my_map = cmap.from_list('brstate', [[0,1,1],[1,0,1], [0.8, 0.8, 0.8]], 3)
        tmp = axes1.imshow(H)
        tmp.set_cmap(my_map)
        axes1.axis('tight')
        tmp.axes.get_xaxis().set_visible(False)
        tmp.axes.get_yaxis().set_visible(False)

        t = np.arange(0,X.shape[0]*dt, dt)
        axes2=plt.axes([0.1, 0.6, 0.8, 0.2])
        axes2.plot(t,X[:,[0,1]])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('REM-on vs REM-off')
        
        axes3=plt.axes([0.1, 0.4, 0.8, 0.2])
        axes3.plot(t,X[:,[2,3]])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('Sleep vs Wake')

        
        axes4=plt.axes([0.1, 0.2, 0.8, 0.2])
        axes4.plot(t,X[:,8])
        plt.xlim([t[0], t[-1]])
        plt.ylabel('REM pressure')

        axes5=plt.axes([0.1, 0.0, 0.8, 0.2])
        axes5.plot(t,X[:,9]*0.01)
        plt.xlim([t[0], t[-1]])
        plt.ylabel('Sleep pressure')
        
    return H
    


#%% Runge Kutta 0.05s Timestep
if __name__ == '__main__' :

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2,  1.0,  0.0, 0.4, 0, 0]
    dt = 0.05

    n = int(np.round(5*3600/dt))
    X = np.zeros((n, len(X0)))
    X[0, :]=np.array(X0)
        
    # solve diff. equation using Runge-Kutta
    for i in range(1,n):
           k1 = dt*mi_model(X[i-1,:],0)
           k2 = dt*mi_model(X[i-1,:]+0.5*k1,dt/2)
           k3 = dt*mi_model(X[i-1,:]+0.5*k2,dt/2)
           k4 = dt*mi_model(X[i-1,:]+k3,dt)
           X[i,:] = X[i-1,:] + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
         
    H = hypnogram(X, 1.5, 1.5, p=1)
    # ~80s run time

#%% Runge-Kutta 0.2s Timestep
if __name__ == '__main__' :

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2,  1.0,  0.0, 0.4, 0, 0]
    dt = 0.2

    n = int(np.round(5*3600/dt))
    X1 = np.zeros((n, len(X0)))
    X1[0, :]=np.array(X0)
        
    # solve diff. equation using Runge-Kutta
    for i in range(1,n):
           k1 = dt*mi_model(X1[i-1,:],0)
           k2 = dt*mi_model(X1[i-1,:]+0.5*k1,dt/2)
           k3 = dt*mi_model(X1[i-1,:]+0.5*k2,dt/2)
           k4 = dt*mi_model(X1[i-1,:]+k3,dt)
           X1[i,:] = X1[i-1,:] + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
         
    H = hypnogram(X1, 1.5, 1.5, p=1)
    # ~20s run time, much lower error than euler approximation
    
#%% Runge-Kutta 0.5s Timestep
if __name__ == '__main__' :

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2,  1.0,  0.0, 0.4, 0, 0]
    dt = 0.5

    n = int(np.round(5*3600/dt))
    X2 = np.zeros((n, len(X0)))
    X2[0, :]=np.array(X0)
        
    # solve diff. equation using Runge-Kutta
    for i in range(1,n):
           k1 = dt*mi_model(X2[i-1,:],0)
           k2 = dt*mi_model(X2[i-1,:]+0.5*k1,dt/2)
           k3 = dt*mi_model(X2[i-1,:]+0.5*k2,dt/2)
           k4 = dt*mi_model(X2[i-1,:]+k3,dt)
           X2[i,:] = X2[i-1,:] + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
         
    H = hypnogram(X2, 1.5, 1.5, p=1)
    # ~7s run time, some areas with lower error than 0.05s euler, though also some areas with more error
#%% Euler Method 0.05s Timestep
if __name__ == '__main__' :

    X0 = [5.0, 0.0, 4.0, 5.0, 0.2,  1.0,  0.0, 0.4, 0, 0]
    dt = 0.05

    n = int(np.round(5*3600/dt))
    Xe=np.zeros((n, len(X0)))
    Xe[0, :]=np.array(X0)
        
    # solve diff. equation using foward Euler
    for i in range(1,n):
        Xe[i,:] = Xe[i-1,:] + mi_model(Xe[i-1,:], 0) * dt

    H = hypnogram(Xe, 1.5, 1.5, p=1)
    # ~20s run time
    
#%%
    X_rk=X[::4,:]
    R=abs(X_rk-X1)
    X_ee=Xe[::4,:]
    P=abs(X_rk-X_ee)
    X_rk1=X[::10,:]
    X_ee1=Xe[::10,:]
    Q=abs(X_rk1-X2)
    N=abs(X_rk1-X_ee1)
    for i in range(10):
        plt.figure(i+1)
        plt.plot(R[:,i], label='RKT1 vs RKT2')
        plt.plot(P[:,i], label='RKT1 vs Euler')
                
    for j in range(10):
        plt.figure(j+11)
        plt.plot(Q[:,j], label='RKT1 vs RKT3')
        plt.plot(N[:,j], label='RKT1 vs Euler')
        
        
        
        
        

    
        
    

    
