import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px

barevna_paleta = px.colors.qualitative.Plotly #barvi pro graf 
nemoci = [["chřipka", 3, 7], ["spalničky", 17, 7], ["malárie", 100, 30], ["zarděnky", 6, 7], ["černý kašel", 16, 21]] #[jmeno, nakazlivost, jak dlouho trva uzdraveni]

def rozdeleni(nemoci, idx):
    jmeno_pro_nemoci = nemoci[idx][0]
    beta_pro_nemoci = nemoci[idx][1] * (1/nemoci[idx][2])
    alfa_pro_nemoci = 1/nemoci[idx][2]
    return jmeno_pro_nemoci, beta_pro_nemoci, alfa_pro_nemoci


def vypocet_sir(nemoci, N = 1_000_042): #rozdeli nemoci a vypocita u kazde sir
    #N je celkova populace  
    I0 = 1           
    R0 = 0           
    S0 = N - I0 - R0  
    celek = []
    for i in range(len(nemoci)):
        jmeno, beta, gamma = rozdeleni(nemoci, i)
        def sir(t, y):
            S, I, R = y
            dSdt = -beta * S * I / N  #kvuli tomu že gama a beta jsou proměne dávám definici do cyklu  
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt] #tady provadim a vracim derivaci 
        
        y0 = [S0, I0, R0]
        t_span = (0, 120) 
        t_eval = np.linspace(*t_span, 120)  
        vysledek = solve_ivp(sir, t_span, y0, t_eval=t_eval) #
        celek.append((jmeno, vysledek, beta, gamma, N))   
    return celek #vraci pro kazdou nemoc seznam s informacema jmeno, SIR, betu, gammu a celkovou populaci. 

def celek_vykresleni_grafu(nemoci):
    vysledky = vypocet_sir(nemoci)
    fig = go.Figure() #vykreslí prázdný graf 
    info_texty = []
    i = 0
    for jmeno, res, beta, gamma, N in vysledky:
        S, I, R = res.y
        t = res.t
        vrchol_idx = np.argmax(I)
        konec_idx = np.where(I < 1)[0]
        konec_epidemie = t[konec_idx[0]] if konec_idx.size > 0 else t[-1]
        onemocneli = int(R[-1])
        nenakazeni = int(S[-1])
        popis = (f"<b>{jmeno}</b><br>" #vytvoří popisky
                 f"Vrchol epidemie: den {int(t[vrchol_idx])}, nakažených: {int(I[vrchol_idx])}<br>"
                 f"Konec epidemie: den {int(konec_epidemie)}<br>"
                 f"Celkem onemocní: {onemocneli}<br>"
                 f"Nenakažených: {nenakazeni}")
        
        barva = barevna_paleta[i % len(barevna_paleta)]
        info_texty.append(popis) 
        fig.add_trace(go.Scatter(x=t, y=S, name=f"{jmeno} - Nenakažení (S)", visible=True, line=dict(dash = 'dot', color=barva))) #kreslí do grafu jednotlivé nemoci
        fig.add_trace(go.Scatter(x=t, y=I, name=f"{jmeno} - Nakažení (I)", visible=True, line=dict(dash = 'solid', width = 2, color=barva)))
        fig.add_trace(go.Scatter(x=t, y=R, name=f"{jmeno} - Uzdravení (R)", visible=True, line=dict(dash = 'dash', color=barva)))
        i += 1
    dropdown_buttons = [  #přida tlačítka
        dict(label="Zobrazit vše", method="update",
             args=[{"visible": [True] * len(vysledky) * 3},
                   {"annotations": []}])   ]

    for i, (jmeno, _, _, _, _) in enumerate(vysledky): #každé nemoci vytvoří tlačítko
        visible = [False] * len(vysledky) * 3 #zadání danou nemoc od kryje a zkryje ostatní
        visible[i*3:(i+1)*3] = [True, True, True] 
        dropdown_buttons.append(
            dict(label=jmeno, method="update",
                 args=[{"visible": visible},
                       {"annotations": [dict(
                           text=info_texty[i],
                           x=0.5, y=-0.35,
                           xref="paper", yref="paper",
                           showarrow=False,
                           align="left",
                           font=dict(size=14)   )]}]))
    fig.update_layout(  #uděla aktalizaci grafu  
        title="SIR model epidemií",
        xaxis_title="Čas (dny)",
        yaxis_title="Počet lidí",
        updatemenus=[dict(type="dropdown", buttons=dropdown_buttons)],#přejmenuje jmeno grafu na nemoc
        height=700,
        margin=dict(b=150)  )
    fig.show()

celek_vykresleni_grafu(nemoci)
