import numpy as np
import matplotlib.pyplot as plt

rho =  0.3
sigma_v = 0.2
k = 2
T = 30/365
r = 0.019
q = 0.002
theta = 0.25
v0 = 0.21
S0 = 100
K = 110

def d(rho, sigma, k, u):
    return np.sqrt((rho*sigma*1j*u - k)**2 + sigma**2*(u**2 + 1j*u))

def g(rho, sigma, k, u):
    du = d(rho, sigma, k, u)
    return (k - rho*1j*u*sigma - du) / (k - rho*1j*u*sigma + du)

def C(rho, sigma, k, u, T, r, q, theta):
    du = d(rho, sigma, k, u)
    gu = g(rho, sigma, k, u)
    return (1j*u*(r - q)*T
            + (k*theta/sigma**2) * ((k - rho*sigma*1j*u - du)*T
            - 2*np.log((1 - gu*np.exp(-du*T)) / (1 - gu))))

def D(rho, sigma, k, u, T):
    du = d(rho, sigma, k, u)
    gu = g(rho, sigma, k, u)
    return ((k - rho*sigma*1j*u - du) * (1 - np.exp(-du*T))
            / ((1 - gu*np.exp(-du*T)) * sigma**2))

def phi(rho, sigma, k, u, T, r, q, theta, v0, S0):
    Cu = C(rho, sigma, k, u, T, r, q, theta)
    Du = D(rho, sigma, k, u, T)
    return np.exp(Cu + Du*v0 + 1j*u*np.log(S0))

N = 100000
du = 0.001
f = np.zeros(N)

lnK = np.log(K)
for i in range(1, N):
    ui = du * i
    phi_val = phi(rho, sigma_v, k, ui, T, r, q, theta, v0, S0)  
    integrand = (np.exp(-1j * ui * lnK) * phi_val) / (1j * ui)  
    f[i] = integrand.real

P2 = 0.5 + (du/np.pi) * np.sum(f) 

print(P2)
