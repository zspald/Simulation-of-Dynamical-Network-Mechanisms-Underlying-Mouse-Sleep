# %% Imports and function definintions

import numpy as np
import os
import scipy.optimize
import numdifftools as nd
import matplotlib.pyplot as plt

AVG_SLEEP_FW = 0.774338
psave = 1
out_path = 'figures/hysteresis'

def mi_const(X, t=0):
    """Reduction of flip-flop model to 3 dimensional NREM/REM model driven by homeostatic drive

    Args:
        X (array-like): [fRon, fRoff, fS, fW, stp, h]
        t (int, optional): Dummy time parameter. Defaults to 0.
    """

    [fRon, fRoff, fW, stp] = X

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
    g_W2Roff = 5.0 # 0
    g_Roff2W = 0 # 0
    g_Roff2S = 0 # 0
    g_W2stp = 0.15 # 0.15

    # general function definitions
    X_inf = lambda c, X_max, beta, alpha: (0.5 * X_max * (1 + np.tanh((c-beta) / alpha)))
    CX_inf = lambda f, gamma: np.tanh(f/gamma)
    beta_X = lambda y, k1_X, k2_X: k2_X * (y - k1_X)
    H = lambda x: 1 if x > 0 else 0

    # neurotransmitters - set as the steady state value for the given x, gamma
    CR_inf = lambda x: CX_inf(x, gamma_R)
    CRoff_inf = lambda x: CX_inf(x, gamma_Roff)
    CW_inf = lambda x: CX_inf(x, gamma_W)
    # CS_inf = lambda x: CX_inf(x, gamma_S)
    C_Ron = CR_inf(fRon)
    C_Roff = CRoff_inf(fRoff)
    C_W = CW_inf(fW)
    # C_S = CS_inf(fS)

    # REM-on
    R_inf = lambda c: X_inf(c, R_max, beta_R, alpha_R)
    dfRon = (R_inf(C_Roff * g_Roff2R) - fRon) / tau_R

    # homeostatic REM pressure (stp)
    dstp = (H(theta_R - fRon) * (stp_max - stp)) / tau_stpup + (H(fRon - theta_R) * (stp_min - stp)) / tau_stpdown

    # REM-off
    beta_Roff = lambda y: beta_X(y, k1_Roff, k2_Roff)
    Roff_inf = lambda c: X_inf(c, Roff_max, beta_Roff(stp), alpha_Roff)
    dfRoff = (Roff_inf(C_Ron * g_R2Roff + C_W * g_W2Roff) - fRoff) / tau_Roff

    dW = 0

    Y = [dfRon, dfRoff, dW, dstp]
    return np.array(Y)

def my_unique2d(X, Y, TOL = 0.01):

    G = np.array([X[0]])
    H = np.array([Y[0]])
    
    for (x,y) in zip(X[1:], Y[1:]):
        if np.min(np.abs(G - x)>TOL) or np.min(np.abs(H - y)>TOL):
            G = np.concatenate((G,[x]))
            H = np.concatenate((H,[y]))
    
    return G,H


# %% Run model 

# X0 = [0.1, 4.2, AVG_SLEEP_FW, 0.63]
X0 = [3.8, .1, AVG_SLEEP_FW, 0.63]

dt = 0.1
n = int(np.round(0.5*3600/dt))
X = np.zeros((n, len(X0)))
X[0,:] = np.array(X0)
for i in range(1,n):
    X[i,:] = X[i-1,:] + mi_const(X[i-1,:], 0) * dt

model_output = X

# matplotlib.rcParams.update({'font.size': 22})
plt.figure()
ax1 = plt.axes([0.1, 0.4, 0.8, 0.25]) 
ax1.spines["top"].set_visible(False)    
ax1.spines["right"].set_visible(False)
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left()  
ax1.set_yticks(np.array([0, 2, 4]))
ax1.set_xticks(np.array([]))

t = np.arange(0,dt*n,dt)/60
plt.plot(t, X[:,0], linewidth=3, color='black')
plt.plot(t, X[:,1], linewidth=3, color=[0.6, 0.6, 0.6])
plt.ylabel('Activity (a.u.)')
plt.ylim([-0.1, 5.1])

t1 = 9.55
t2 = 12.06
t3 = 16.04

dt = np.mean(np.diff(t))

it1 = np.int((t1 / dt))
it2 = np.int((t2 / dt))
it3 = np.int((t3 / dt))

h1 = model_output[it1, 2]
h2 = model_output[it2, 2]
h3 = model_output[it3, 2]
#plt.plot(t[it1], model_output[it1,1], 'o', color='blue')
#plt.plot(t[it2], model_output[it2,1], 'o', color='red')
#plt.plot(t[it3], model_output[it3,1], 'o', color='green')

s = np.arange(0, 5, .1)
plt.plot(t[it1]*np.ones((len(s),)), s, '--', color=[0.8, 0.8, 1], lw=4)
plt.plot(t[it2]*np.ones((len(s),)), s, '--', color=[0.5, 0.5, 1], lw=4)
plt.plot(t[it3]*np.ones((len(s),)), s, '--', color=[1, 0.4, 1], lw=4)


ax2 = plt.axes([0.1, 0.1, 0.8, 0.15]) 
ax2.spines["top"].set_visible(False)    
ax2.spines["right"].set_visible(False)
ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left()  
ax2.set_yticks(np.array([0, 1]))
ax2.set_xticks(np.array([0, 10, 20, 30]))

plt.plot(t, X[:,-1], linewidth=3, color='black')
plt.ylim([0, 1])
plt.ylabel('REM pressure (a.u.)')
plt.xlabel('Time (min)')
plt.show(block=False)

# determine brain state
B = np.zeros((1, X.shape[0]))
B[0,:] = np.array([np.argmax(X[i,:]) for i in range(X.shape[0])])
B[np.where(B==2)] = 1

ax = plt.axes([0.1, 0.7, 0.8, 0.08])
tmp = ax.imshow(B)
plt.axis('tight')
plt.axis('off')

cmap = plt.cm.jet
my_map = cmap.from_list('ha', [[0,0,0],[1,0,1], [0.6, 0.6, 0.6]], 3)
tmp.set_cmap(my_map)


if psave==1:
    fig = os.path.join(out_path, 'fig_mimodel.pdf')
    plt.savefig(fig, bbox_inches="tight"); 
# %% Stable and Unstable Points

# STP = np.arange(-0.8, 1.2, 0.1)
# STP = np.concatenate((np.arange(0.3, .48, 0.01), np.arange(0.48, 0.5, 0.001), np.arange(0.5, 0.79, 0.01), np.arange(0.79,0.8, 0.001), np.arange(0.8, 0.901, 0.01)))
STP = np.concatenate((np.arange(0.3, 0.7, 0.01), np.arange(0.7,0.75,0.001), np.arange(0.75,0.9,0.01), np.arange(0.9,0.95,0.001), np.arange(0.95,1.2,0.01)))
FR_opt = np.zeros((len(STP),))
FRoff_opt = np.zeros((len(STP),))
FR = np.arange(0, 5, 0.1)
FROFF = np.arange(0, 5, 0.1)
TOL = 0.01

# M = []
FP = {}
FP_stab = {}
# i = 0
for stp in STP:
    M = []
    for fr in FR:

        for froff in FROFF:
            F = lambda x: mi_const([x[0], x[1], AVG_SLEEP_FW, stp], 1.0)[0:2]
            soln = scipy.optimize.root(F, [fr, froff])
            y = F(soln.x)
            if np.linalg.norm(y,2) < 0.00000001:
                M.append(soln.x)
    X = np.array(M)

    x,y = my_unique2d(X[:,0], X[:,1], TOL=0.1)

    FP[stp] = [x,y]

    dF = nd.Jacobian(F)
    stab = []
    for i in range(len(x)):
        eig = np.linalg.eig(dF((x[i], y[i])))[0]
        if (eig[1] < 0 and eig[0] < 0):
            stab.append(-1)
        else:
            stab.append(1)
    FP_stab[stp] = stab


################################################################
#################           Plotting          ##################
################################################################
# %% 

plt.figure()

for stp in STP:
    fr = FP[stp][0]
    fn = FP[stp][1]
    
    for i in range(len(fn)):
        if FP_stab[stp][i] == -1:
            plt.plot(stp, fn[i], 'b.')
        else:
            plt.plot(stp, fn[i], 'r.')

plt.plot(model_output[:,2], model_output[:,1])    
plt.xlim((min(STP), max(STP)))
plt.ylim((-0.1, 5.1))
plt.xlabel('stp')
plt.ylabel('FR - REM-off')
# %% 

plt.figure()

for stp in STP:
    fr = FP[stp][0]
    fn = FP[stp][1]
    
    for i in range(len(fn)):
        if FP_stab[stp][i] == -1:
            plt.plot(stp, fn[i], 'b.')
        else:
            plt.plot(stp, fn[i], 'r.')

plt.plot(model_output[:,2], model_output[:,1])    
plt.xlim((min(STP), max(STP)))
plt.ylim((-0.1, 5.1))
plt.xlabel('stp')
plt.ylabel('FR - REM-off')



# another figure showing time evolution of fr
plt.figure()
for stp in STP:
    fr = FP[stp][0]
    fn = FP[stp][1]
    
    for i in range(len(fr)):
        if FP_stab[stp][i] == -1:
            plt.plot(stp, fr[i], 'b.')
        else:
            plt.plot(stp, fr[i], 'r.')
    
plt.plot(model_output[:,2], model_output[:,1])
plt.xlim((min(STP), max(STP)))
plt.ylim((-0.1, 5.1))
plt.xlabel('stp')
plt.ylabel('FR - REM-on')

#%%
stab_low = []
stab_high = []
instab = []
thr = 1.

for stp in STP:
    fr = FP[stp][0]
    fn = FP[stp][1]
    stab = FP_stab[stp]
    
    if len(stab) == 1:
        if fn > thr:
            stab_high.append((stp,fn))
        else:
            stab_low.append((stp,fn))
    else:
        for (s,r) in zip(stab,fn):
            if s == -1:
                if r > thr:
                    stab_high.append((stp,r))
                else:
                    stab_low.append((stp,r))
            else:
                instab.append((stp,r))
                    
stab_low = np.array(stab_low)
stab_high = np.array(stab_high)
instab   = np.array(instab)

plt.figure();
ax = plt.subplot(111)
plt.plot(stab_low[:,0], stab_low[:,1], color='black', lw=3)
plt.plot(stab_high[:,0], stab_high[:,1], color='black', lw=3)            
plt.plot(instab[:,0], instab[:,1], '--', color='gray', lw=3)

noffset = 600
plt.plot(model_output[noffset:,2], model_output[noffset:,1], '--', color=[0.6,0.6,1], lw=2)
        
plt.xlim([min(STP), max(STP)])    
plt.ylim([-0.1, 5.1])
ax.xaxis.set_ticks([0.4, 0.4, 0.6, 0.8])
ax.yaxis.set_ticks(range(0, 6))


plt.ylabel('FN')
plt.xlabel('stp')
# box_off(ax)

if psave == 1:
    fig_file = os.path.join(out_path, 'fig_bifurcation_fn.pdf')
    plt.savefig(fig_file, bbox_inches="tight"); 

##################################
#%%

stab_low = []
stab_high = []
instab = []
thr = 2.

for stp in STP:
    fr = FP[stp][0]
    fn = FP[stp][1]
    stab = FP_stab[stp]
    
    if len(stab) == 1:
        if fr > thr:
            stab_high.append((stp,fr))
        else:
            stab_low.append((stp,fr))
    else:
        for (s,r) in zip(stab,fr):
            if s == -1:
                if r > thr:
                    stab_high.append((stp,r))
                else:
                    stab_low.append((stp,r))
            else:
                instab.append((stp,r))
                    
stab_low = np.array(stab_low)
stab_high = np.array(stab_high)
instab   = np.array(instab)

plt.figure();
ax = plt.subplot(111)
plt.plot(stab_low[:,0], stab_low[:,1], color='black', lw=3)
plt.plot(stab_high[:,0], stab_high[:,1], color='black', lw=3)            
plt.plot(instab[:,0], instab[:,1], '--', color='gray', lw=3)
plt.hlines(1.5, min(STP), max(STP), linestyle='--', color='red')

# noffset = 600
# plt.plot(model_output[noffset:,2], model_output[noffset:,0], '--', color=[0.6,0.6,1], lw=2)
# plt.plot(model_output[noffset:,2], model_output[noffset:,0], '--', color=[0.6,0.6,1], lw=2)

        
s = np.arange(0, 5, .1)
#plt.plot(model_output[it1,2], model_output[it1,0], 'o', color=[0.8, 0.8, 1], lw=4)
#plt.plot(model_output[it2,2], model_output[it2,0], 'o', color=[0.5, 0.5, 1], lw=4)
#plt.plot(model_output[it3,2], model_output[it3,0], 'o', color=[1, 0.4, 1], lw=4)


plt.xlim([min(STP), max(STP)])    
plt.ylim([-0.1, 4.5])
ax.xaxis.set_ticks(np.arange(min(STP),max(STP),))
ax.yaxis.set_ticks(range(0, 5))


plt.ylabel('FR')
plt.xlabel('stp')
# box_off(ax)

if psave == 1:
    fig_file = os.path.join(out_path, 'fig_bifurcation_fr.pdf')
    plt.savefig(fig_file, bbox_inches="tight"); 

# %% Save fron bifurcation data

np.save('stab_high_fr.npy', stab_high)
np.save('stab_low_fr.npy', stab_low)
np.save('instab_fr.npy', instab)


#%%
fr = np.arange(0, 4.5, 0.01)
fn = np.arange(-2, 7, 0.01)

optr = []
optn1 = []
optn2 = []
optn3 = []
for n in fn:
    
    # FR nullcline
    F = lambda x: mi_const([x,n,AVG_SLEEP_FW, 0],1.0)[0]
    optr.append(scipy.optimize.root(F, 1).x[0])
    
    # FN nullcline; depends on h; bifurcation at about 0.7215
    F = lambda x: mi_const([n,x,AVG_SLEEP_FW,h1],1.0)[1]
    optn1.append(scipy.optimize.root(F, 0).x[0])
    
    F = lambda x: mi_const([n,x,AVG_SLEEP_FW,h2],1.0)[1]
    optn2.append(scipy.optimize.root(F, 0).x[0])
    
    F = lambda x: mi_const([n,x,AVG_SLEEP_FW,h3],1.0)[1]
    optn3.append(scipy.optimize.root(F, 5).x[0])


# FR_REM-on is on x-xaxis
plt.figure()
ax = plt.subplot(111)
plt.plot(optr, fn, '-', color='black', lw=3)

plt.plot(fn, optn1, '-', color=[0.8, 0.8, 1], lw=3)
plt.plot(fn, optn2, '-', color=[0.5, 0.5, 1], lw=3)
# "middle" nullcline
plt.plot(fn, optn3, '-', color=[1,0.4,1], lw=3)
plt.xlabel('F_R')
plt.ylabel('F_N')
ax.yaxis.set_ticks([0,5])
ax.xaxis.set_ticks([0,5])
# box_off(ax)

if psave == 1:
    fig_file = os.path.join(out_path, 'fig_mi_nullclines.pdf')
    plt.savefig(fig_file, bbox_inches="tight"); 
    
    
# plot the same with FR_REM-on on y-axis
# FR_REM-on is on x-xaxis
plt.figure()
ax = plt.subplot(111)
plt.plot(fn, optr, '-', color='black', lw=3)
plt.plot(optn1, fn, '-', color=[0.8, 0.8, 1], lw=3)
plt.plot(optn2, fn, '-', color=[0.5, 0.5, 1], lw=3)
# "middle" nullcline
plt.plot(optn3, fn, '-', color=[1,0.4,1], lw=3)
plt.ylabel('F_R')
plt.xlabel('F_N')
ax.yaxis.set_ticks([0,5])
ax.xaxis.set_ticks([0,5])
# box_off(ax)

if psave == 1:
    fig_file = os.path.join(out_path, 'fig_mi_nullclines2.pdf')
    plt.savefig(fig_file, bbox_inches="tight"); 

#%%
# draw vector field for given h
X, Y = np.meshgrid(np.arange(-2, 7, .5), np.arange(-2, 7, .5))
[n,m] = X.shape
U = np.zeros(X.shape)
V = np.zeros(X.shape)

for i in range(n):
    for j in range(m):
        a = mi_const([X[i,j],Y[i,j], AVG_SLEEP_FW, h3], 0)
        U[i,j] = a[0]
        V[i,j] = a[1]


plt.figure()
plt.plot(optr, fn, '-', color='black', lw=3)
plt.plot(fn, optn3, '-', color=[0.8, 0.8, 0.8], lw=3)
plt.quiver(X,Y,U,V, zorder=10)

if psave == 1:
    fig_file = os.path.join(out_path, 'fig_nullclines_vector_field.pdf')
    plt.savefig(fig_file, bbox_inches="tight"); 


# %%
