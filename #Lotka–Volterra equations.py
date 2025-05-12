import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parametry zvířat
alpha = 0.6   #jeleni 
beta = 0.02   # vlci 
delta = 0.03  # vlci
gamma = 0.4   # vlci
epsilon = 0.3  # medvědi lov
eta = 0.01    # medvedi beta
mu = 0.1  #medvedi gamma
# Časová osa roků
t_span = (0, 200)
t_eval = np.linspace(*t_span, 1000)
def zakladní_model(t, vars): #je potřeba udělat dvě definice proto že solve by s tím jinak nezvladl pracovat
    x, y = vars
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt] #vracíme danou derivaci se kterou si pak solve poradi

x0, y0 = 40, 9
bez_Pú = solve_ivp(zakladní_model, t_span, [x0, y0], t_eval=t_eval) #počítaní růstu / poklesu podle času
xbez_Pú, ybez_Pú = bez_Pú.y #prepis dat k vykreslení

def upraveny_model(t, vars):
    x, y, z = vars
    dxdt = alpha * x - beta * x * y - epsilon * x * z
    dydt = delta * x * y - gamma * y
    dzdt = eta * x * z - mu * z
    return [dxdt, dydt, dzdt]

z0 = 5  
s_Pú = solve_ivp(upraveny_model, t_span, [x0, y0, z0], t_eval=t_eval) #Znova Změna 
xs_Pú, ys_Pú, zs_Pú = s_Pú.y #prepis dat k vykresu
#věci co vykreslí graf
plt.figure(figsize=(14, 6)) #vytvoří prázdný graf
plt.subplot(1, 2, 1) #poloha grafu
plt.plot(t_eval, xbez_Pú, label='Jeleni') #nakreslí data 
plt.plot(t_eval, ybez_Pú, label='Vlci') #nakreslí data
plt.title("Bez medvědů") #název grafu
plt.xlabel("Čas") #pojmenování os 
plt.ylabel("Počet jedinců")
plt.legend() #vypíše legendu
plt.grid(True)

# druhý graf
plt.subplot(1, 2, 2)  #poloha grafu
plt.plot(t_eval, xs_Pú, label='Jeleni') #nakreslí data
plt.plot(t_eval, ys_Pú, label='Vlci')
plt.plot(t_eval, zs_Pú, label='Medvědi')
plt.title("S medvědy") #titulek
plt.xlabel("Čas") #jmeno os
plt.ylabel("Počet jedinců")
plt.legend() #legenda
plt.grid(True)

plt.tight_layout()  #automaticky vyrovná kde se co vykreslí
plt.show() #a všechno to vykreslí