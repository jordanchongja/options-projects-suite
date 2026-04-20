import math
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad, IntegrationWarning
import scipy.stats as si


# ==============================================================================
# 0. SDE SIMULATIONS
# ==============================================================================
def simulate_gbmm(S0, mu, sigma, T, dt, num_paths):
    """
    Simulates asset price paths using Geometric Brownian Motion.
    """
    N = int(T / dt) # Number of time steps
    t = np.linspace(0, T, N)
    
    # 1. FIX: Generate N-1 increments, then prepend zeros so W_0 = 0
    dW = np.random.standard_normal(size=(num_paths, N - 1)) * np.sqrt(dt)
    W = np.zeros((num_paths, N))
    W[:, 1:] = np.cumsum(dW, axis=1) 
    
    # Calculate the paths using the analytical solution to the GBM SDE
    X = (mu - 0.5 * sigma**2) * t + sigma * W 
    S = S0 * np.exp(X) 
    return t, S

# ==============================================================================
# 1. BLACK-SCHOLES & IMPLIED VOLATILITY
# ==============================================================================
def bs_call_price(S, K, T, r, q, sigma):
    # 1. Use np.maximum instead of max to handle arrays natively
    intrinsic = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    
    # 2. To avoid dividing by zero warnings if sigma or T is exactly 0
    # we temporarily replace 0s with a tiny number
    safe_sigma = np.where(sigma <= 0, 1e-9, sigma)
    safe_T = np.where(T <= 0, 1e-9, T)
    
    # 3. Calculate normally
    d1 = (np.log(S / K) + (r - q + 0.5 * safe_sigma**2) * safe_T) / (safe_sigma * np.sqrt(safe_T))
    d2 = d1 - safe_sigma * np.sqrt(safe_T)
    price = S * np.exp(-q * safe_T) * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)
    
    # 4. Use np.where to route the logic: 
    # If sigma or T <= 0, return intrinsic. Otherwise, return the calculated price.
    return np.where((sigma <= 0) | (T <= 0), intrinsic, price)

def implied_volatility(target_price, S, K, T, r, q, option_type='C'):
    """
    Combines the safety guard and the Brent solver for production-grade IV calculation.
    """
    # 1. THE GUARD: Check for Arbitrage/Intrinsic Violations
    if option_type == 'C':
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    
    # If the market price is at or below intrinsic, IV is mathematically undefined
    if target_price <= intrinsic:
        return np.nan 

    # 2. THE SOLVER: Define what 'Zero' looks like
    def objective_function(sigma):
        if option_type == 'C':
            return bs_call_price(S, K, T, r, q, sigma) - target_price
        else:
            return bs_put_price(S, K, T, r, q, sigma) - target_price

    # 3. THE EXECUTION: Search between 0.01% and 500% vol
    try:
        # brentq is the industry standard for this task
        return brentq(objective_function, 1e-4, 5.0)
    except (ValueError, RuntimeError):
        # Returns NaN if the price is so far out of bounds the solver can't converge
        return np.nan

def d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, q, sigma):
    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

def bs_call_delta(S, K, T, r, q, sigma):
    """Rate of change of option price with respect to underlying asset."""
    return np.exp(-q * T) * si.norm.cdf(d1(S, K, T, r, q, sigma))

def bs_gamma(S, K, T, r, q, sigma):
    """Rate of change of Delta (convexity). Identical for calls and puts."""
    return (np.exp(-q * T) * si.norm.pdf(d1(S, K, T, r, q, sigma))) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, q, sigma):
    """Sensitivity to volatility. Identical for calls and puts."""
    return S * np.exp(-q * T) * si.norm.pdf(d1(S, K, T, r, q, sigma)) * np.sqrt(T)

def bs_call_theta(S, K, T, r, q, sigma):
    """Time decay of the option."""
    term1 = -(S * np.exp(-q * T) * si.norm.pdf(d1(S, K, T, r, q, sigma)) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * si.norm.cdf(d2(S, K, T, r, q, sigma))
    term3 = q * S * np.exp(-q * T) * si.norm.cdf(d1(S, K, T, r, q, sigma))
    return term1 - term2 + term3

def bs_call_rho(S, K, T, r, q, sigma):
    """Sensitivity to the risk-free interest rate."""
    return K * T * np.exp(-r * T) * si.norm.cdf(d2(S, K, T, r, q, sigma))


# ==============================================================================
# --- Put Option Pricing and Greeks ---
def bs_put_price(S, K, T, r, q, sigma):
    if sigma <= 0 or T <= 0: return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    return (K * np.exp(-r * T) * si.norm.cdf(-d2(S, K, T, r, q, sigma)) - 
            S * np.exp(-q * T) * si.norm.cdf(-d1(S, K, T, r, q, sigma)))

def bs_put_delta(S, K, T, r, q, sigma):
    return -np.exp(-q * T) * si.norm.cdf(-d1(S, K, T, r, q, sigma))

def bs_put_theta(S, K, T, r, q, sigma):
    term1 = -(S * np.exp(-q * T) * si.norm.pdf(d1(S, K, T, r, q, sigma)) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * si.norm.cdf(-d2(S, K, T, r, q, sigma))
    term3 = q * S * np.exp(-q * T) * si.norm.cdf(-d1(S, K, T, r, q, sigma))
    return term1 + term2 - term3

def bs_put_rho(S, K, T, r, q, sigma):
    return -K * T * np.exp(-r * T) * si.norm.cdf(-d2(S, K, T, r, q, sigma))


# ==============================================================================
# 2. HESTON MODEL (STOCHASTIC VOLATILITY)
# ==============================================================================
def heston_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho):
    alpha = -u**2 / 2 - 1j * u / 2
    beta = kappa - rho * xi * 1j * u
    gamma = xi**2 / 2
    
    d = np.sqrt(beta**2 - 4 * alpha * gamma)
    r_plus = (beta + d) / (xi**2)
    r_minus = (beta - d) / (xi**2)
    g = r_minus / r_plus
    
    C = kappa * (r_minus * T - (2 / (xi**2)) * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    
    return np.exp(C * theta + D * v0 + 1j * u * np.log(S0 * np.exp((r - q) * T)))

def heston_call_price(S0, K, T, r, q, v0, kappa, theta, xi, rho):
    def integrand1(u):
        cf = heston_characteristic_function(u - 1j, S0, K, T, r, q, v0, kappa, theta, xi, rho)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u * S0 * np.exp((r-q)*T))).real

    def integrand2(u):
        cf = heston_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u)).real

    limit_max = 2000 # Gibbs phenomenon fix
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        P1_int = quad(integrand1, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        P2_int = quad(integrand2, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        
    P1 = 0.5 + (1 / np.pi) * P1_int
    P2 = 0.5 + (1 / np.pi) * P2_int
    
    return max(0.0, S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)

# ==============================================================================
# 3. MERTON MODEL (JUMP DIFFUSION)
# ==============================================================================
def merton_jump_call(S, K, T, r, q, sigma, lam, mu_j, delta, max_jumps=40):
    k = np.exp(mu_j + 0.5 * delta**2) - 1
    lambda_prime = lam * (1 + k)
    price = 0.0
    
    for n in range(max_jumps):
        poisson_weight = np.exp(-lambda_prime * T) * ((lambda_prime * T)**n) / math.factorial(n)
        sigma_n = np.sqrt(sigma**2 + (n * delta**2) / T)
        r_n = r - lam * k + (n * np.log(1 + k)) / T
        
        price += poisson_weight * bs_call_price(S, K, T, r_n, q, sigma_n)
        
    return price

# ==============================================================================
# 4. BATES MODEL (STOCHASTIC VOLATILITY + JUMPS)
# ==============================================================================
def bates_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta):
    # Heston Component
    alpha = -u**2 / 2 - 1j * u / 2
    beta = kappa - rho * xi * 1j * u
    gamma = xi**2 / 2
    
    d = np.sqrt(beta**2 - 4 * alpha * gamma)
    r_plus = (beta + d) / (xi**2)
    r_minus = (beta - d) / (xi**2)
    g = r_minus / r_plus
    
    C = kappa * (r_minus * T - (2 / (xi**2)) * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    heston_cf = np.exp(C * theta + D * v0 + 1j * u * np.log(S0 * np.exp((r - q) * T)))
    
    # Merton Component
    k = np.exp(mu_j + 0.5 * delta**2) - 1 
    jump_term = np.exp(mu_j * 1j * u - 0.5 * delta**2 * u**2) - 1
    merton_cf = np.exp(lam * T * (jump_term - 1j * u * k))
    
    return heston_cf * merton_cf

def bates_call_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta):
    def integrand1(u):
        cf = bates_characteristic_function(u - 1j, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u * S0 * np.exp((r-q)*T))).real

    def integrand2(u):
        cf = bates_characteristic_function(u, S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta)
        return (np.exp(-1j * u * np.log(K)) * cf / (1j * u)).real

    limit_max = 2000 
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        P1_int = quad(integrand1, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        P2_int = quad(integrand2, 0, limit_max, epsabs=1e-4, epsrel=1e-4, limit=200)[0]
        
    P1 = 0.5 + (1 / np.pi) * P1_int
    P2 = 0.5 + (1 / np.pi) * P2_int
    
    return max(0.0, S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)

# REFACTORED: Now accepts S0, T, r, q, and data arrays as arguments
def bates_objective(params, S0, T, r, q, target_strikes, target_ivs):
    v0, kappa, theta, xi, rho, lam, mu_j, delta = params
    error = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, K in enumerate(target_strikes):
            m_price = bates_call_price(S0, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, delta)
            
            # Make sure implied_volatility is accessible in this file
            m_iv = implied_volatility(m_price, S0, K, T, r, q)
            
            if np.isnan(m_iv): 
                error += 5.0
            else: 
                error += (m_iv - target_ivs[i])**2
                
    return error / len(target_strikes)