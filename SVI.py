# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import math


# %%  General Utilities

#Normal cumulative density function
def N(x):
    return 0.5*(1+math.erf(x/np.sqrt(2)))

#Call option BS price
def BS_call(S,K,T,sigma,r):
    if T <= 0:
        return max(S-K, 0.0)
    if sigma <= 0:
        return max(S - K*np.exp(-r*T), 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*N(d1) - K*np.exp(-r*T)*N(d2)

#Put option BS price (derived by call/put parity)
def BS_put(S,K,T,sigma,r):
    #Uses input call parity principle
    C = BS_call(S, K, T, sigma, r)
    return C-S+K*np.exp(-r*T)

def inverse(x):
    x = np.asarray(x)
    if x.ndim == 0:      # scalare
        return 1.0 / x
    elif x.ndim == 2:    # matrice
        return np.linalg.inv(x)
    else:
        raise ValueError("Solo scalari o matrici quadrate")
   

# %%  Data

#-----------------------------------------------------------------------------
#----------------------Utilities----------------------------------------------
#-----------------------------------------------------------------------------
#Checks if the prices are monotone withthe strikes. If not --> incoherent data
def monotoneCheck(C, tol = 1e-5):
    s = np.shape(C)
    monotone = True
    for i in range(s[0]):
        for j in range(1,s[1]):
            if C[i,j]-C[i,j-1]>tol:
                monotone = False
                break
    return monotone

#-----------------------------------------------------------------------------
#----------------------Code---------------------------------------------------
#-----------------------------------------------------------------------------

r = 0.02    #Risk free interest rate
S = 100     #Spot price (now)

# Strike grid (20 valori)
K = np.array([80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118], dtype=float)

# Maturities (4 valori)
T = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

# Prezzi call Black–Scholes coerenti (monotoni decrescenti in strike per riga)
C = np.array([
[20.738076,18.858798,17.015743,15.219468,13.482517,11.819164,10.244801, 8.774951, 7.424014, 6.203704, 5.121805, 4.181135, 3.379248, 2.708740, 2.158225, 1.713659, 1.359776, 1.081414, 0.864468, 0.696490],
[22.555196,20.792823,19.078224,17.421818,15.834787,14.328599,12.914217,11.601092,10.396317, 9.304093, 8.325232, 7.457190, 6.694520, 6.029441, 5.452753, 4.954645, 4.525389, 4.155707, 3.837212, 3.562403],
[26.593646,24.837819,23.155923,21.561837,20.068982,18.689920,17.435471,16.313698,15.329012,14.481915,13.768883,13.183116,12.715252,12.353899,12.086547,11.900124,11.781681,11.700118,11.600118,11.500118],
[33.684751,31.763276,29.953389,28.275879,26.751351,25.399531,24.238237,23.282748,22.544484,22.029525,21.737564,21.661601,21.561601,21.421601,21.241601,21.041601,20.821601,20.591601,20.391601,20.191601]
])
print('Check of prices monotonicity with the strikes: ' + str(monotoneCheck(C)))
print('Check of prices monotonicity with the maturities: ' + str(monotoneCheck(-C.T)))

# %% Compute the implied vol at nodes

#-----------------------------------------------------------------------------
#----------------------Utilities----------------------------------------------
#-----------------------------------------------------------------------------

def ImpliedVol(S,K,T,r,C):
    
    #Check no arbitrage in the price for the given parameters
    lb = max(S - K*np.exp(-r*T), 0.0)
    ub = S
    if C < lb - 1e-12 or C > ub + 1e-12:
        return np.nan
    
    #Compute the initial bracket
    a, b = 1e-8, 5.0  # [~0%, 500%]
    fa = BS_call(S,K,T,a,r) - C
    fb = BS_call(S,K,T,b,r) - C
    
    #Make the initial bracket larger if necessary
    it = 0
    while fa*fb > 0 and b < 10.0 and it < 30:
        b *= 2.0
        fb = BS_call(S,K,T,b,r) - C
        it += 1
    if fa*fb > 0:
        return np.nan  # Initial bracketing was not possible

    #Halving process taking the midpoint and excluding the point with the same sign
    #Halving 50 times times makes the bracket 1.126e15 times smaller. 
    #An early exit condition is set at 1e-8. 
    for _ in range(50):
        m = 0.5*(a+b)
        fm = BS_call(S,K,T,m,r) - C
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a < 1e-8:
            break
    return 0.5*(a+b)

#-----------------------------------------------------------------------------
#----------------------Code---------------------------------------------------
#-----------------------------------------------------------------------------

#Compute the implied volatility at the nodes        
ImpliedVol_data = np.zeros_like(C)
c_shape = C.shape
for i in range(c_shape[0]):
    for j in range(c_shape[1]):
        ImpliedVol_data[i,j] = ImpliedVol(S, K[j], T[i], r, C[i,j])

# Compute total variance w = sigma_implied^2*T
total_variance = np.zeros_like(C)
for i in range(c_shape[0]):
    total_variance[i,:] = (ImpliedVol_data[i,:]**2)*T[i]
   
# Change of variable log-moneyness x = log(K/F)
x = np.zeros_like(C)
F = S*np.exp(r*T)
for i in range(c_shape[0]):
    for j in range(c_shape[1]):
        x[i,j] = np.log(K[j]/F[i])

#Plot w(0) and total variance
for i in range(len(T)):
    plt.plot(x[i, :], total_variance[i, :], 'o-')
    plt.axvline(x = 0)
    plt.grid()

# %% Compute the fit of theta

#To be more precise, a fit can be employed
theta = np.zeros(len(T))
for i in range(len(T)):
    # 5 punti più vicini a k=0
    idx = np.argsort(np.abs(x[i, :]))[:5]          
    # Fit w(k) ≈ a*k^2 + b*k + c  -> theta_i = w(0) = c
    coeff = np.polyfit(x[i, idx], total_variance[i, idx], 2)
    theta[i] = coeff[-1]

#After the points w(0,T) = theta(T) are identified, they are fit.
theta = np.maximum.accumulate(theta)

def theta_interp_param(theta, T):
    theta = np.asarray(theta)          # shape (n,)
    T = np.asarray(T)                  # shape (n,)
    phi = np.array([np.exp(T) - 1,T,T**2])                # shape (n,2)
    print(np.shape(phi))
    # alpha = (phi^T theta) / (phi^T phi)
    param = np.linalg.inv(phi@phi.T)@(phi)@theta
    return param

def theta_interp(T,param):
    return param[0]*(np.exp(T)-1)+param[1]*T+param[2]*T**2

alpha = theta_interp_param(theta,T)
Tmax = np.max(T)
N_points = 1000
theta_function_vals = np.zeros(N_points)
for i in range(N_points):
    theta_function_vals[i] = theta_interp(Tmax*(i/N_points),alpha)
   
plt.figure(3)
plt.plot(np.linspace(0,Tmax,len(theta)),theta,'o')
plt.plot(np.linspace(0,Tmax,len(theta_function_vals)),theta_function_vals)

# %% SVI
#-----------------------------------------------------------------------------
#----------------------Utilities----------------------------------------------
#-----------------------------------------------------------------------------

def wSVI(k, a, b, rho, m, sigma):
    k = np.asarray(k)
    s = np.sqrt((k - m)**2 + sigma)
    return a + b * (rho * (k - m) + s)

def jacobian_terms(k, b, rho, m, sigma):
    k = np.asarray(k)
    s = np.sqrt((k - m)**2 + sigma)
    # Derivate parziali di w(k) per parametro:
    dw_da   = np.ones_like(k)
    dw_db   = rho*(k - m) + s
    dw_drho = b*(k - m)
    dw_dm   = b * (-rho + (m - k)/s)
    dw_dsigma = b * (1.0 / (2.0 * s))
    return dw_da, dw_db, dw_drho, dw_dm, dw_dsigma

def SVI_fit(k, w_data, iters=150):
   
    # inizializzazione
    a, b, rho, m, sigma = 0.01, 0.1, -0.1, 0.0, 0.05  # valori ragionevoli
    loss_hist = []

    for _ in range(iters):
        w_model = wSVI(k, a, b, rho, m, sigma)
        r = w_model - w_data  # residui
        # MSE
        loss = 0.5 * np.dot(r, r)
        loss_hist.append(loss)
        if np.abs(loss) < 0.5e-5:
            print('Optimization converged.')
            break

        # J^T r
        dw_da, dw_db, dw_drho, dw_dm, dw_dsigma = jacobian_terms(k, b, rho, m, sigma)
        J = np.column_stack([dw_da, dw_db, dw_drho, dw_dm, dw_dsigma]) #n x 5 matrix
        lam = 0.005
        param = np.array([a,b,rho,m,sigma])
        param = param - np.linalg.pinv(J.T@J+lam * np.eye(J.shape[1]))@(J.T)@r
        
        a = param[0]
        b = param[1]
        rho = param[2]
        m = param[3]
        sigma = param[4]

        # proiezioni / clip
        b   = max(b, 1e-6)
        sigma = max(sigma, 1e-8)
        rho = np.clip(rho, -0.999, 0.999)
        # Lee bound: b(1+|rho|) ≤ 2
        cap = 2.0/(1.0+abs(rho))
        if b > cap: b = cap

    return a, b, rho, m, sigma, loss_hist

plt.figure()
params = np.zeros((len(T),5))
for i in range(len(T)):
   
    a,b,rho,m,sigma,loss = SVI_fit(x[i,:],total_variance[i,:])
    params[i,0] = a
    params[i,1] = b
    params[i,2] = rho
    params[i,3] = m
    params[i,4] = sigma
    loss = np.array(loss)
    loss = loss/np.max(loss)
       
    k = np.linspace(-0.3, 0.2,1000)
    k_points = np.linspace(-0.3,0.2,len(x[i,:]))
    x_SVI = wSVI(k, a, b, rho, m, sigma)
    
    plt.figure()
    plt.grid()
    plt.plot(k,x_SVI)
    plt.plot(x[i,:],total_variance[i,:],'o')
    plt.xlabel('log moneyness')
    plt.ylabel('Volatility')
    plt.legend(('SVI volatility','Market volatility'))
    plt.title('T = '+str(T[i]))
    plt.savefig("Img_"+str(i),dpi = 600)