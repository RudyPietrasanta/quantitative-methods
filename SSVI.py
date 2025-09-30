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

'''
#Plot w(0) and total variance
for i in range(len(T)):
    plt.plot(x[i, :], total_variance[i, :], 'o-')
    plt.axvline(x = 0)
    plt.grid()'''

# %% Compute theta

x_signs = np.sign(x)
switch = np.zeros(len(T))
theta = np.zeros(len(T))

a = 1
b = 1
c = 1
 
for i in range(len(T)):
    xi = x[i, :]
    wi = total_variance[i, :]
    # trova intervallo che contiene 0
    j = np.searchsorted(xi, 0.0) - 1
    j = np.clip(j, 0, len(xi)-2)
    x0, x1 = xi[j], xi[j+1]
    w0, w1 = wi[j], wi[j+1]
    # interp lineare in x=0
    theta[i] = w0 + (w1 - w0) * (0.0 - x0) / (x1 - x0)

def ThetaModel(a,b,c,T):
    return a*(1-np.exp(-b*T))+c*T

def JacobianTheta(a,b,c,T):
    da = 1-np.exp(-b*T)
    db = a*T*np.exp(-b*T)
    dc = T
    return da,db,dc

for i in range(25):
    e = ThetaModel(a, b, c, T)-theta
    da,db,dc = JacobianTheta(a, b, c, T)
    J = np.column_stack([da,db,dc])
    param = np.array([a,b,c])
    g = np.linalg.pinv(J.T@J)@J.T@e
    param = param - g
    a = param[0]
    b = param[1]
    c = param[2]


t = np.linspace(0,2,100)
plt.Figure()
plt.plot(t,ThetaModel(a, b, c, t))
plt.plot(T,theta,'o')
plt.xlabel('T')
plt.ylabel('theta')
plt.grid()


# %% SSVI Fit

def _col(v):
    # porta scalare -> (), vettore (M,) -> (M,1); già (M,1) resta tale
    v = np.asarray(v)
    if v.ndim == 0:
        return v  # scalar
    if v.ndim == 1:
        return v[:, None]
    return v  # già con asse colonna

def phi(gamma, eta, theta):
    # φ = η * (1+θ)^(1-γ) * θ^(−γ)
    #return eta * ((1 + theta)**(1 - gamma) * theta**(-gamma))
    # Heston-like: φ(θ) = η * θ^(−γ)
    return eta * theta**(-gamma)

def SSVI(x, theta, rho, gamma, eta):
    # x: (M, N)
    ph = phi(gamma, eta, theta)                # () o (M,)
    theta_c = _col(theta)                      # () o (M,1)
    rho_c   = _col(rho)
    ph_c    = _col(ph)

    z    = ph_c * x + rho_c                    # (M,N)
    root_arg = z*z + 1 - rho_c*rho_c
    root_arg[root_arg < 0] = 1e-5
    root = np.sqrt(root_arg)                   # (M,N)

    return 0.5 * theta_c * (1 + rho_c*ph_c*x + root)

def JacobianSSVI(x, theta, rho, gamma, eta):
    ph      = phi(gamma, eta, theta)           # () o (M,)
    theta_c = _col(theta)
    rho_c   = _col(rho)
    ph_c    = _col(ph)

    z    = ph_c * x + rho_c                    # (M,N)
    root_arg = z*z + 1 - rho_c*rho_c
    root_arg[root_arg < 0] = 1e-5
    root = np.sqrt(root_arg)                   # (M,N)

    # dφ/d· (stessa shape di ph)
    # Heston-like: φ(θ)=η θ^(−γ) ⇒
    # ∂φ/∂η = φ/η ;  ∂φ/∂γ = −φ * log(θ)
    dp_deta   = ph / eta
    dp_dgamma = -ph * np.log(theta)

    dp_deta_c   = _col(dp_deta)
    dp_dgamma_c = _col(dp_dgamma)

    # ∂w/∂φ e ∂w/∂ρ (come prima)
    dw_dphi = 0.5 * theta_c * (rho_c*x + (z*x)/root)      # (M,N)
    dw_drho = 0.5 * theta_c * (ph_c*x + (z - rho_c)/root) # (M,N)

    # catena per η e γ (element-wise)
    dw_deta   = dw_dphi * dp_deta_c
    dw_dgamma = dw_dphi * dp_dgamma_c

    # Jacobiano (M,N,3): [ρ, η, γ]
    J = np.stack([dw_drho, dw_deta, dw_dgamma], axis=-1)
    return J


rho = 0
eta = 0.5
gamma = 0.5
Iter = 100
Loss = np.zeros(Iter)

for i in range(Iter):
    # residuo (MN,)
    err = (SSVI(x, theta, rho, gamma, eta) - total_variance).ravel()
    Loss[i] = np.linalg.norm(err)
    # Jacobiano (MN, 3)
    J = JacobianSSVI(x, theta, rho, gamma, eta).reshape(-1, 3)
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.linalg.pinv(J.T@J+ 0.1*np.diag(J.T@J))@J.T@err
    param = np.array([rho,eta,gamma])
    param = param - g
    rho = param[0]
    eta = param[1]
    gamma = param[2]

plt.plot(Loss)

# %% Plot con Plotly (superficie SSVI 3D)

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"   # apre nel browser di sistema

# Calcolo superficie sui punti di griglia
K_grid, T_grid = np.meshgrid(K, T)   # K strike, T maturity
F_grid = S * np.exp(r * T_grid)
x_grid = np.log(K_grid / F_grid)

# Prendi i dati reali/nodali: strikes K, scadenze T, implied vols dai nodi
K_points = np.tile(K, len(T))          # vettore strikes ripetuti per ogni T
T_points = np.repeat(T, len(K))        # vettore scadenze replicato
IV_points = ImpliedVol_data.ravel()    # flatten delle vol implicite calcolate prima

theta_vec = a * (1 - np.exp(-b*T)) + c*T   # usa il tuo modello θ(T)

def ssvi_total_variance(x, theta, rho, gamma, eta):
    ph = phi(gamma, eta, theta)
    theta_c = theta[:, None]
    ph_c = ph[:, None]
    z = ph_c * x + rho
    root = np.sqrt(z*z + 1 - rho*rho)
    return 0.5 * theta_c * (1 + rho*ph_c*x + root)

w_grid = ssvi_total_variance(x_grid, theta_vec, rho, gamma, eta)
iv_grid = np.sqrt(w_grid / T_grid)

fig = go.Figure(data=[go.Surface(
    x=K_grid, y=T_grid, z=iv_grid,
    colorscale="Viridis"
)])

fig.update_layout(
    title="SSVI Implied Vol Surface (3D)",
    scene=dict(
        xaxis_title='Strike K',
        yaxis_title='Maturity T',
        zaxis_title='Implied Volatility σ(K,T)'
    ),
    autosize=True
)

fig.add_trace(go.Scatter3d(
    x=K_points,
    y=T_points,
    z=IV_points,
    mode='markers',
    marker=dict(size=4, color='red', symbol='circle'),
    name='Market/Nodes'
))

fig.show()
