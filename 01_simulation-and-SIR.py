import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model
def lvmodel(x, t, beta, alpha):
    s, i, r = x
    dsdt = - beta * s * i
    didt = (beta * s * i) - (alpha * i)
    drdt = alpha * i
    return [dsdt, didt, drdt]

# Parameters
n = 100000                              # population size
times = np.arange(0, 71, 1)             # from 0 to 70 days  (=10 weeks)
ps0 = 0.15                              # proportion of the population likely to be ill
alpha = 2.55/7                          # rate of exit from the Infectious compartment
beta = 0.0004/7                         # effective contact rate
I0 = 1
xstart = [ps0 * n, I0, (1-ps0) * n]

# Simulation of epidemic
solution = odeint(lvmodel, xstart, times, args=(beta, alpha))
solution = np.round(solution, 0)
out1 = pd.DataFrame(solution, columns=['s', 'i', 'r'])
out1['time'] = times
out1['newcases'] = np.concatenate(([0], -np.diff(out1['s'])))

# Graphs of evolutions : New cases, Susceptibles, Infectious
plt.figure(figsize=(80, 8))
plt.plot(out1['time'], out1['newcases'], color='black', label='New cases')
plt.plot(out1['time'], out1['s'], color='royalblue', label='Suceptibles')
plt.plot(out1['time'], out1['i'], color='firebrick', label='Infectious')
plt.ylim(0, ps0*n)
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.title('Number of New Cases, Suceptibles and Infectious')
plt.legend()
plt.show()

# Real influenza data (http://www.sentiweb.org)



