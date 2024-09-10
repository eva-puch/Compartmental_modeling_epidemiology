import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


#-----  SIR model.
def lvmodel(x, t, beta, alpha):
    s, i, r = x
    dsdt = - beta * s * i
    didt = (beta * s * i) - (alpha * i)
    drdt = alpha * i
    return [dsdt, didt, drdt]

# Parameters for simulation (by days).
n = 100000                              # population size
times = np.arange(0, 71, 1)             # from 0 to 70 days  (=10 weeks)
ps0 = 0.15                              # proportion of the population likely to be ill
alpha = 2.55/7                          # rate of exit from the Infectious compartment
beta = 0.0004/7                         # effective contact rate
I0 = 1
xstart = [ps0 * n, I0, (1-ps0) * n]
parms = [beta, alpha]

# Simulation of epidemic.
solution = odeint(lvmodel, xstart, times, args=(beta, alpha))
solution = np.round(solution, 0)
out1 = pd.DataFrame(solution, columns=['s', 'i', 'r'])
out1['time'] = times
out1['newcases'] = np.concatenate(([0], -np.diff(out1['s'])))

# Graphs of evolutions : New cases, Susceptibles, Infectious.
plt.figure(figsize=(80, 8))
plt.plot(out1['time'], out1['newcases'], color='black', label='New cases')
plt.plot(out1['time'], out1['s'], color='royalblue', label='Suceptibles')
plt.plot(out1['time'], out1['i'], color='firebrick', label='Infectious')
plt.ylim(0, ps0*n)
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.title('Number of New Cases, Suceptibles and Infectious')
plt.legend()
#plt.show()

#-----  Real influenza data (http://www.sentiweb.org) import.
#data_path = "/Users/mac/Documents/Projets_data/Compartmental_modeling_epidemiology/influenza_data.csv"
data_path = "influenza_data.csv"                                # replace the path if needed
data = pd.read_csv(data_path, skiprows=1)                       # read the csv file
data = data.iloc[:, :10]                                        # keep the columns of interest
data = data[(data['week'] >= 201051) & (data['week'] <= 201107)]  # filtered the period (from 51th week of 2010 to 7th week of 2011)
data = data.sort_values(by = 'week')                            # sort by 'week'
data['week'] = range(1, len(data) + 1)                          # new column
#print(data.head())

#----  Comparison of the simulated and real epidemics, and search for the optimal
#----  parameters (alpha and beta) to best match the simulation to reality.

def opti_sim(x):
    sol = solve_ivp(lambda t, y: lvmodel(y, t, x[0], x[1]), [times[0], times[-1]], xstart, t_eval=times, method='RK45')
    out1 = pd.DataFrame(sol.y.T, columns=['s', 'i', 'r'])
    newcases = out1.iloc[:-1, 1].values - out1.iloc[1:, 1].values
    mc = np.zeros(len(out1.iloc[:, 1]) - len(inc))

    for i in range(len(out1.iloc[:, 1]) - len(inc)):
        mc[i] = np.sum((newcases[i:(i+len(inc))] - inc)**2)

    return np.min(mc[mc != 0])

# convert and extract relevant incidence column
data['inc100'] = pd.to_numeric(data['inc100'], errors='coerce')
inc = data.iloc[:, 5].values
times = np.arange(0, 41)
# define initial parameter guesses
parms0 = [400 / (n * 26), 36 / 26]

# define objective function for optimization
def objective(params):
    return opti_sim(params)

# perform the optimization
res = minimize(objective, parms0, method='Nelder-Mead')
# extract and print estimated parameters
parms_estim = res.x
print("Estimated parameters:", parms_estim)     # beta=0.00015, alpha=1.137

# SIR modeling with the estimated parameters
beta = 0.0001260318
alpha = 1.5230279631
out = pd.DataFrame(np.round(odeint(lvmodel, xstart, times, args=(beta, alpha)), 0), columns=['s', 'i', 'r'])
out['time'] = np.arange(0, len(out['s']))
#print(out)

# Calculation of incidences (in days)
out['newcases'] = [0] + list(-np.diff(out['s']))
#print(out)

# Graph of new cases per day (simulation)
plt.figure(figsize=(10, 6))
plt.plot(out['time'], out['newcases'], color='green', label="Simulated Cases")
plt.ylim(0, out['newcases'].max() + 50)
plt.title("Evolution of new cases per day \n (simulation)")
plt.xlabel("Days")
plt.ylabel("Number of new cases")
plt.grid(True)
plt.legend()
#plt.show()

# Graph of real data new cases per week (real data)
plt.figure(figsize=(10, 6))
plt.scatter(data['week'], data['inc100'], color='orange', label="New Real Cases")
plt.ylim(0, data['inc100'].max() + 50)
plt.title("Evolution of new cases per day \n (real data)")
plt.xlabel("Weeks")
plt.ylabel("Number of new cases")
plt.grid(True)
plt.legend()
#plt.show()

# alignment of peaks of simulated and real epidemics
time_max = data['week'][data['inc100'].idxmax()]        # time where the max incidence is reached in real data
time_max_s = out['time'][out['newcases'].idxmax()]      # time where the max incidence is reached in simulated data
nb_weeks = len(data['week'])                            # number of weeks
# keep ontly the 9 weeks containing the peak (4 before and 4 after the max incidence)
out = out[(out['time'] >= time_max_s - 4) & (out['time'] <= time_max_s + 4)]
out['week'] = np.arange(1, len(out) + 1)               # assign new week numbers

# combine the two previous graphs of epidemic's peak (simulation and real data)
plt.figure(figsize=(10, 6))
plt.scatter(data['week'], data['inc100'], color='orange', label="New Real Cases", zorder=3)
plt.plot(out['week'], out['newcases'], color='brown', label="Simulated Cases", zorder=2)
plt.ylim(0, max(out['newcases'].max(), data['inc100'].max()) + 50)
plt.title("Evolution of new cases per week at the peak \n (real data vs simulation)")
plt.xlabel("Weeks")
plt.ylabel("Number of new cases")
plt.grid(True)
plt.legend()
plt.show()
