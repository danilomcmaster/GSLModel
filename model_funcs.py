import numpy as np
import pandas as pd

## ODE SOLVER
def rk4(f, x0, t):
    """Numerically approximates the solution to the IVP:
    
    x'(t) = f(x(t),t)
    x(t0) = x0
    
    using a fourth-order Runge-Kutta method.
    Parameters:
        f (function): The right-hand side of the ODE
        x0 ((m,) ndarray): The initial condition
        t ((n,) ndarray): The array of time values
    Returns:
        ((n,m) ndarray): The approximate solution, where x[i] ≈ x(t_i)
    """
    h = t[1] - t[0]
    A = np.zeros((len(t), len(x0)))
    A[0] = np.copy(x0)
    
    #update and add row for each time step
    for i in range(len(t) - 1):

        #get RK4 variables
        K1 = f(A[i], t[i])
        K2 = f(A[i] + (h / 2) * K1, t[i] + (h / 2))
        K3 = f(A[i] + (h / 2) * K2, t[i] + (h / 2))
        K4 = f(A[i] + h * K3, t[i] + h)

        #update to next time step
        A[i + 1] = A[i] + (h / 6) * (K1 + (2 * K2) + (2 * K3) + K4)
        
    return A

def get_vol_to_sa_df(path="./data/elevation-area-volume.csv"):
    df = pd.read_csv(path)
    df["elev"] = df["elev_ft_NAVD88"]
    df = df[["elev", "volume_m3", "area_m2"]]
    return df

def get_surface_area(volume, df):
    return np.interp(volume, df["volume_m3"], df["area_m2"])

#basic inflow function
def basic_inflow(x, t):
    """Simple seasonal inflow function that varies sinusoidally with time."""
    mean = 2000000 / 365 # daily average inflow
    return mean + 1000 * np.sin(2 * np.pi * t / 365)

#basic outflow function
def basic_outflow(x, t, evap):
    """Takes a function of evaporation to determine outflow, and x, t,
    and returns outflow."""
    return evap(t)

## Evaporation function ##
## TO DO - Evaporation function as a function of time, temperature, surface area (proportional to volume), and salinity.





