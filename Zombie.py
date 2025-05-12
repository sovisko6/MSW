import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#lotka voltrra data
a = 0.01         
beta = 0.00001   
epsilon = 0.9    
delta = 0.000005 

# SIR data
S0 = 500_000     
Z0 = 1000        
R0 = 0           
y0 = [S0, Z0, R0]

# samotné rovnce
def zombie(t, y):
    S, Z, R = y
    dSdt = a * S - beta * S * Z
    dZdt = epsilon * beta * S * Z - delta * S * Z
    dRdt = delta * S * Z
    return [dSdt, dZdt, dRdt]

# ovlineni času
t_span = (0, 20)
t_eval = np.linspace(*t_span, 20)

reseni = solve_ivp(zombie, t_span, y0, t_eval=t_eval, vectorized=True)

plt.figure(figsize=(10, 6))
plt.plot(reseni.t, reseni.y[0], label="Lidé", color="green")
plt.plot(reseni.t, reseni.y[1], label="Zombí", color="red")
plt.plot(reseni.t, reseni.y[2], label="umrtnost", color="gray")
plt.xlabel("Čas (dny)")
plt.ylabel("Populace")
plt.title("Zombie virus")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

