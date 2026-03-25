import model_funcs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# This file can be used to test our model against the data

# Import total volume data
vol_df = pd.read_csv('data/GSLLevelVol.csv')
vol_df['Date'] = pd.to_datetime(vol_df['Date'])
vol_df = vol_df.sort_values('Date')

vol_times = vol_df['Date'].map(pd.Timestamp.timestamp).values
vol_values = vol_df['Total_vol_m3'].values

# Import inflow data
jordan = pd.read_csv('data/jordan-river.csv')
bear = pd.read_csv('data/bear-river.csv')
weber = pd.read_csv('data/weber-river.csv')

# Adjust inflow data into one dataframe with total inflow data
cols_to_drop = [
    'x', 'y', 'id', 'time_series_id', 'monitoring_location_id',
    'parameter_code', 'statistic_id',
    'approval_status', 'qualifier', 'last_modified'
]

for river in [jordan, bear, weber]:
    river['time'] = pd.to_datetime(river['time'])
    river.drop(cols_to_drop, axis=1, inplace=True)
    river.sort_values('time', inplace=True)
    river['days'] = (river['time'] - river['time'].min()).dt.days

inflow_df = jordan.merge(bear, on='time', how='outer')
inflow_df = inflow_df.merge(weber, on='time', how='outer')
inflow_df['value'] = inflow_df['value'] + inflow_df['value_x'] + inflow_df['value_y']
inflow_df['value'] = inflow_df['value'] * 0.0283168
inflow_df.drop(['unit_of_measure_x', 'unit_of_measure_y', 'value_x', 'value_y', 'days_x', 'days_y'], axis=1, inplace=True)
inflow_df.drop('unit_of_measure', axis=1, inplace=True)

# Convert dates to days since start
inflow_df['days'] = (inflow_df['time'] - inflow_df['time'].min()).dt.days
t_inflow = inflow_df['days'].values
I_values = inflow_df['value'].values

# Interpolation function for inflow
I_interp = interp1d(t_inflow, I_values, fill_value="extrapolate")

# Import volume to surface area data
vol_to_sa_df = model_funcs.get_vol_to_sa_df()

# Define evaporation function
def E(t, E0=0.003, E1=0.002, phi=200):
    return E0 + E1 * np.sin(2*np.pi*(t - phi)/365)

# Define the model
def dvdt(t, v):
    return I_interp(t) - E(t) * model_funcs.get_surface_area(v, vol_to_sa_df)

# Solve the ODE
# Initial condition
first_inflow_day = inflow_df['time'].min()
first_inflow_ts = first_inflow_day.timestamp()
v0 = np.interp(first_inflow_ts, vol_times, vol_values)
t0 = 0
t_span = (t0, (inflow_df['time'].max() - first_inflow_day).days)
t_eval = np.arange(t_span[0], t_span[1]+1)

t_inflow_rel = (inflow_df['time'] - first_inflow_day).dt.days.values
I_interp = interp1d(t_inflow_rel, inflow_df['value'].values, fill_value="extrapolate")

sol = solve_ivp(dvdt, t_span, [v0], t_eval=t_eval) # Solution

# Plot the solution vs. observed data
plt.figure(figsize=(10,6))
plt.plot(t_eval, sol.y[0], label='Model volume')
plt.plot((vol_df['Date'] - vol_df['Date'].min()).dt.days, vol_df['Total_vol_m3'], label='Observed volume')
plt.xlabel('Days since start')
plt.ylabel('Lake volume (m³)')
plt.legend()
plt.show()