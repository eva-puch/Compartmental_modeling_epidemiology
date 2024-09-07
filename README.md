# Compartmental modeling of an epidemic : SIR models (Susceptible-Infectious-Recovered)
#### Project status: in progress ⏳

*This project is largely inspired by the "Compartmental Modeling" course I took during my MSc. program, which was originally implemented in R. Here, I revisit the key steps and implement them using Python.*

### 📋 Overview

Using Python, we implement three types of modeling : **simple SIR**, **SIR with vaccination**, and **stochastic SIR**.  
First, we simulate an epidemic, then retrieve actual **flu incidence data** from the 2010/2011 season to compare the simulated epidemic with the real one. The objective is to use an **optimization algorithm** to find the SIR model parameters that best fit the real epidemic data.  
In the second part, we explore **epidemic spread** in the case where **vaccination** is possible, to understand its impacts.  
Finally, we use a stochastic SIR model, which accounts for **random aspects of interactions** between individuals in the population.

