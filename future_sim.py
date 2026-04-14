import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.integrate import solve_ivp
import scipy.stats as stats

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

df_sa = get_vol_to_sa_df()

# Precipitation
prcp = pd.read_csv("data/historical_precipitation_values.csv")

def precipitation_empirical(t, V):
    return surface_area(V, df_sa) * np.interp(t, np.arange(len(prcp)), prcp["Precip"].to_numpy())

def precipitation(t, V):
    A, B, phi, C = 3.62510665e-04, 1.72209593e-02, 5.55109972e-01, 1.07152734e-03
    return surface_area(V, df_sa) * (A * np.sin(B * t + phi) + C)

# Getting historical inflow values
rivers = pd.read_csv("data/historical_river_inflow.csv")
rivers['time'] = pd.to_datetime(rivers['time'])
rivers = rivers[rivers["time"] >= pd.to_datetime("2000-01-01")]
riv_inflow_vals = rivers["value"].to_numpy()

def seasonal_two_harmonics(t, A1, phi1, A2, phi2, C):
        omega = 2 * np.pi / 365
        return (
            A1 * np.sin(omega * t + phi1)
            + A2 * np.sin(2 * omega * t + phi2)
            + C)

def inflow(t, V):
    params = [-28.97393599, -23.20200893, -18.43745932, 4.29250411, 45.52330359]
    return 86400*seasonal_two_harmonics(t, *params) + precipitation(t, V)

def inflow_empircal(t, V):
    return 86400*np.interp(t, np.arange(len(riv_inflow_vals)), riv_inflow_vals) + precipitation_empirical(t, V)

a = 2.1825133333885867 
b = 0.025838829394326426
params = [0.3362724, 0.26361918, -0.1660489, 0.55571495, 0.56250287]

def river_inflow_simulated(num_years):
    draws = stats.gamma.rvs(a, scale=1/b, size=num_years)
    stretches = np.ravel(np.array([[draw]*365 for draw in draws]))
    return 86400*seasonal_two_harmonics(np.array(list(range(365))*num_years), *params)*stretches

def get_vp_salinity_df(path="./data/vp-salinity.csv"):
    df = pd.read_csv(path)
    return df

def vp_reduction(salinity, df):
    return np.interp(salinity, df["salinity"], df["vp_reduction"])

def salinity(volume):
    return 1230618833073.342*(1/volume) + 171886.23798781837*(volume**(-1/3))

def salinity_n(volume):
    return 662924936948.8833*(1/volume) + 462860.60460320744*(volume**(-1/3))

def salinity_s(volume):
    return 1564606058225.5437*(1/volume) + -4334.001932641948*(volume**(-1/3))

# north to south surface area ratio
ratio = 0.615

def vp_reduction_improved(volume, df):
    sal_n = salinity_n(volume)
    sal_s = salinity_s(volume)
    return ratio*np.interp(sal_n, df["salinity"], df["vp_reduction"]) + (1 - ratio)*np.interp(sal_s, df["salinity"], df["vp_reduction"])

df_es = get_vp_salinity_df()

# Weather Functions (Simulating Seasons)
def simulate_temperature(t, fit=False):
    # Simulates temperature in Celsius over a 365 day year
    # Peaks in summer (around month 7), lowest in winter
    mean_temp = 11.0
    amplitude = 15.0
    if not fit:
        return mean_temp + amplitude * np.sin(2 * np.pi * (t - 110) / 365)
    if fit:
        return 14.41391069*np.sin(2 * np.pi * (t - 110.99496694) / 365) + 13.51914305

def simulate_wind_speed(t, fit=False):
    # Simulates wind speed in m/s
    mean_wind = 3.0
    amplitude = 1.5
    if not fit:
        return mean_wind + amplitude * np.sin(2 * np.pi * t / 182.5)
    if fit:
        return 1.09434048*np.sin(2 * np.pi * (t - 45.16832821) / 365) + 3.92688972 + 0.57332365*np.sin(2 * np.pi * (t - 199.9250417) / 365)

def calculate_vapor_pressures(T, fit=False):
    RELATIVE_HUMIDITY = 0.2 # 40% average humidity
    if fit:
        RELATIVE_HUMIDITY = 0.3
    # Magnus-Tetens formula for saturation vapor pressure (kPa)
    es = 0.611 * np.exp((17.27 * T) / (T + 237.3))
    # Actual vapor pressure
    ea = es * RELATIVE_HUMIDITY
    return es, ea

def evap_func(t, V, include_salinity=False, improved=False, fit=False):
    WIND_COEFF_A = 0.001     # Empirical mass transfer coefficient
    WIND_COEFF_B = 0.0005    # Empirical mass transfer coefficient
    # Get current weather for month t
    T = simulate_temperature(t, fit=fit)
    u = simulate_wind_speed(t, fit=fit)

    # Calculate vapor pressures
    es, ea = calculate_vapor_pressures(T, fit=fit)
    if include_salinity:
        if not improved:
            vp_reduc = vp_reduction(salinity(V), df_es)
        if improved:
            vp_reduc = vp_reduction_improved(V, df_es)
        es = es*(1 - vp_reduc)
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

def outflow(t, V, include_salinity=False, improved=False, fit=False):
    return evap_func(t, V, include_salinity=include_salinity, improved=improved, fit=fit)

sim_years = 10
num_years = sim_years + 5
inflow_sim = river_inflow_simulated(num_years)

def inflow_simulated(t, V, inflow_sim):
    if t >= len(inflow_sim):
        t_ind = t % 365
    else:
        t_ind = t
    return inflow_sim[int(t_ind)]

# Future simiulations
# Historical prediction emprical inflow with salinity

days = 365*sim_years
# 9040 is number of days from 2000-01-01 to 2024-10-01
t_domain = (0, days) # Interval
t_eval = np.arange(days+1)

daily_dates = pd.date_range(start="2026-01-01", periods=days+1, freq="D")

def meters_cubed_per_day(KAF_per_year):
    return KAF_per_year*1000*1233.4818375475 / 365

# initial volume on 2026-01-01 is 21963728535.7859

def generate_inflow_sims(num_sims, sim_years):
    inflow_sims = []
    for i in range(num_sims):
        inflow_sims.append(river_inflow_simulated(sim_years + 2))
    return inflow_sims

def simulate(num_sims, sim_years, inflow_sims=None, intervention_level=0, plot=False, y0=np.array([7947655379]), ymax=1.8e10, ymin = 0.4e10, historical=False, true_vol=None):
    if plot:
        plt.clf()
        plt.ylim((ymin, ymax))
    first_good = True
    first_okay = True
    first_bad = True
    first_vbad = True
    good_count = 0
    okay_count = 0
    bad_count = 0
    vbad_count = 0
    width = 1
    intervention_level_dict = {0: 0, 1: meters_cubed_per_day(250), 2: meters_cubed_per_day(800)}
    if inflow_sims == None:
        inflow_sims = generate_inflow_sims(num_sims, sim_years)
    for i in range(num_sims):
        inflow_sim = inflow_sims[i]
        inflow_sim += intervention_level_dict[intervention_level]
        def ode(t, V):
            return inflow_simulated(t, V, inflow_sim) + outflow(t, V, include_salinity=True, improved=True)
        if historical:
            t_domain = (0, 9039) # Interval
            t_eval = np.arange(9040)
        sol = solve_ivp(ode, t_domain, y0, max_step=5, t_eval=t_eval)
        result = sol.y[0]
        if plot and historical:
            daily_dates = pd.date_range(start="2000-01-01", periods=9040, freq="D")
            monthly_dates = pd.date_range(start="2000-01-01", periods=298, freq="MS")
            if i == 0:
                plt.plot(daily_dates, sol.y[0], linestyle="dashed", lw=0.5, color='purple', label='Predicted Volume')
            else:
                plt.plot(daily_dates, sol.y[0], linestyle="dashed", lw=0.5, color='purple')
        if result[-1] >= 13486550741:
            good_count += 1
            if plot and not historical:
                if not first_good:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='green', lw=width, label='Healthy Outcome')
                else:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='green', lw=width)
                first_good = True
        elif result[-1] >= 10972391857:
            okay_count += 1
            if plot and not historical:
                if not first_okay:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='gold', lw=width, label='Transitory Outcome')
                else:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='gold', lw=width)
                first_okay = True
        elif result[-1] >= 8751620241:
            bad_count += 1
            if plot and not historical:
                if not first_bad:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='darkorange', lw=width, label='Adverse Effects Outcome')
                else:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='darkorange', lw=width)
                first_bad = True
        else:
            vbad_count += 1
            if plot and not historical:
                if not first_vbad:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='red', lw=width, label='Serious Adverse Effects Outcome')
                else:
                    plt.plot(daily_dates, sol.y[0], linestyle='dashed', color='red', lw=width)
                first_vbad = True

    if plot and not historical:
        plt.axhspan(13486550741, ymax, alpha=0.2, color='green', label='Healthy')
        plt.axhspan(10972391857, 13486550741, alpha=0.1, color='orange', label='Transitory')
        plt.axhspan(8751620241, 10972391857, alpha=0.1, color='red', label='Adverse Effects')
        plt.axhspan(ymin, 8751620241, alpha=0.2, color='red', label='Serious Adverse Effects')

        # plt.scatter(daily_dates[0], y0, label='Lake Volume on Jan 1st, 2026', color='red')
        plt.legend()
        if intervention_level == 0:
            plt.title("Future GSL Volume Predictions (No Intervention)")
        if intervention_level == 1:
            plt.title("Future GSL Volume Predictions (Additional 250 KAF/year)")
        if intervention_level == 2:
            plt.title("Future GSL Volume Predictions (Additional 800 KAF/year)")
        plt.xlabel("t")
        plt.ylabel("Volume")
        plt.savefig(f"./images/vol_simulation_{sim_years}years_intervention{intervention_level}.png")
        plt.clf()
    if plot and historical:
        plt.plot(monthly_dates, true_vol, label="True Volume", color='blue')
        plt.title("Historical Volume Prediction (Simulated Inflow)")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Volume")
        plt.ylim(bottom=0)
        plt.savefig("./images/vol_sim_historical.png")
    results_dict = {}
    results_dict["Healthy"] = good_count / num_sims
    results_dict["Transitory"] = okay_count / num_sims
    results_dict["Adverse Effects"] = bad_count / num_sims
    results_dict["Severe Adverse Effects"] = vbad_count / num_sims
    
    return f"In simulations for the next {sim_years} years with intervention level: {intervention_level}, Healthy Outcome count: {good_count}/{num_sims}, Transitory Outcome count: {okay_count}/{num_sims}, Adverse Effects Outcome count: {bad_count}/{num_sims}, Serious Adverse Effects Outcome count: {vbad_count}/{num_sims}", results_dict

if __name__ == "__main__":
    # intervention_level_interp_dict = {0: "No Intervention", 1: "Additional 250 KAF/year", 2: "Additional 800 KAF/year"}
    # num_sims = 500
    # sim_years = 10
    # inflow_sims = generate_inflow_sims(num_sims, sim_years)
    # intervention_levels = [0, 1, 2]
    # results_strings = []
    # results = {}
    # for level in intervention_levels:
    #     string, dict = simulate(num_sims, sim_years, inflow_sims, intervention_level=level, plot=False)
    #     results_strings.append(string)
    #     results[intervention_level_interp_dict[level]] = dict 
    # print(results)

    df = pd.read_csv('data/GSLLevelVol.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df = df[df["Date"] >= pd.to_datetime("2000-01-01")]
    true_vol = df["Total_vol_m3"].to_numpy()
    t_to_date = df["Date"].to_list()
    num_sims = 20
    sim_years = 24
    string, dict = simulate(num_sims, sim_years, intervention_level=0, plot=True, y0 = np.array([21963728535.7859]), ymax=3e10, historical=True, true_vol=true_vol)

## RESULTS (including serious adverse category)
# In simulations for the next 10 years with intervention level: 0, Healthy Outcome count: 0/30, Transitory Outcome count: 1/30, Adverse Effects Outcome count: 7/30, Serious Adverse Effects Outcome count: 22/30
# In simulations for the next 10 years with intervention level: 1, Healthy Outcome count: 0/30, Transitory Outcome count: 2/30, Adverse Effects Outcome count: 20/30, Serious Adverse Effects Outcome count: 8/30
# In simulations for the next 10 years with intervention level: 2, Healthy Outcome count: 4/30, Transitory Outcome count: 13/30, Adverse Effects Outcome count: 13/30, Serious Adverse Effects Outcome count: 0/30




## RESULTS (old)
# In simulations for the next 10 years, Healthy Level Obtained count: 0/20, No Adverse Effect Level Obtained count: 6/20, Unhealthy Level Maintained count: 14/20
# In simulations for the next 5 years, Healthy Level Obtained count: 0/100, No Adverse Effect Level Obtained count: 9/100, Unhealthy Level Maintained count: 91/100
# In simulations for the next 10 years with intervention level: 1, Healthy Level Obtained count: 1/20, No Adverse Effect Level Obtained count: 9/20, Unhealthy Level Maintained count: 10/20
# In simulations for the next 10 years with intervention level: 2, Healthy Level Obtained count: 6/30, No Adverse Effect Level Obtained count: 21/30, Unhealthy Level Maintained count: 3/30
# result_dict = {'No Intervention': {'Healthy': 0.0, 'Transitory': 0.038, 'Adverse Effects': 0.314, 'Severe Adverse Effects': 0.648}, 'Additional 250 KAF/year': {'Healthy': 0.004, 'Transitory': 0.114, 'Adverse Effects': 0.542, 'Severe Adverse Effects': 0.34}, 'Additional 800 KAF/year': {'Healthy': 0.244, 'Transitory': 0.638, 'Adverse Effects': 0.118, 'Severe Adverse Effects': 0.0}}
