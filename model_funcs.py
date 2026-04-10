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

def surface_area(volume, df):
    # Note: volume must be in m^3 and surface area will be returned in m^2
    return np.interp(volume, df["volume_m3"], df["area_m2"])

def get_vp_salinity_df(path="./data/vp-salinity.csv"):
    df = pd.read_csv(path)
    return df

def vp_reduction(salinity, df):
    return np.interp(salinity, df["salinity"], df["vp_reduction"])

def salinity(volume):
    return 1230618833073.342*(1/volume) + 171886.23798781837*(volume**(-1/3))

#basic inflow function
def basic_inflow(x, t):
    """Simple seasonal inflow function that varies sinusoidally with time."""
    mean = 2000000 / 365 # daily average inflow
    return mean + 1000 * np.sin(2 * np.pi * t / 365)

#basic outflow function
def basic_outflow(x, t, evap):
    """Takes a function of evaporation to determine outflow, and x, t,
    and returns outflow.
    x: current volume
    t: current time
    evap_func: the function used to calculate evaporation
    """
    return evap(t, x)

## Evaporation function ##
## TO DO - Evaporation function as a function of time, temperature, surface area (proportional to volume), and salinity.

# Weather Functions (Simulating Seasons)
def simulate_temperature(t):
    # Simulates temperature in Celsius over a 365 day year
    # Peaks in summer (around month 7), lowest in winter
    mean_temp = 11.0
    amplitude = 15.0
    return mean_temp + amplitude * np.sin(2 * np.pi * (t - 213) / 365)

def simulate_wind_speed(t):
    # Simulates wind speed in m/s
    mean_wind = 3.0
    amplitude = 1.5
    return mean_wind + amplitude * np.sin(2 * np.pi * t / 182.5)

def calculate_vapor_pressures(T):
    RELATIVE_HUMIDITY = 0.40 # 40% average humidity
    # Magnus-Tetens formula for saturation vapor pressure (kPa)
    es = 0.611 * np.exp((17.27 * T) / (T + 237.3))
    # Actual vapor pressure
    ea = es * RELATIVE_HUMIDITY
    return es, ea

def evap_func(t, V):
    WIND_COEFF_A = 0.001     # Empirical mass transfer coefficient
    WIND_COEFF_B = 0.0005    # Empirical mass transfer coefficient
    # Get current weather for month t
    T = simulate_temperature(t)
    u = simulate_wind_speed(t)

    # Calculate vapor pressures
    es, ea = calculate_vapor_pressures(T)

    # Calculate Dalton's Evaporation Rate (E)
    # E = f(wind) * (es - ea)
    wind_function = WIND_COEFF_A + WIND_COEFF_B * u
    E = wind_function * (es - ea)

    # Ensure evaporation doesn't go negative
    E = max(E, 0)

    # Calculate Area
    current_area = surface_area(V, df_sa)

    # The Final Differential
    dVdt = -E * current_area
    return dVdt





