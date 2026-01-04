"""

# Hodgkin-Huxley Single Neuron Model
The Hodgkin–Huxley model is a biophysical description of how a neuron generates and propagates action potentials.
It treats a small patch of membrane as an electrical circuit where:
- The lipid membrane is a capacitor ($C_m$).
- Ion channels are voltage-dependent conductances for sodium ($g_{Na}$) and potassium ($g_K$), plus a passive leak ($g_L$).
- Each ionic current follows Ohm’s law ($I=g(V-E)$), with its own reversal potential $E_{ion}$.

The model consists of a differential equation for the membrane voltage coupled to gating-variable equations (m, h, n) that describe how channel conductances open and close with voltage and time. By solving these equations, the model reproduces the shape, threshold, and refractory behavior of real action potentials and underlies most modern conductance-based neuron models.


## The Hodgkin-Huxley Equations:
$ C \frac{dV}{dt} = I_{app} - g_{Na}m^3h(V-E_{Na}) - g_Kn^4(V-E_K) - g_L(V-E_L) $

With...
|Symbol|Formula|
|---|---|
| $I_{Na_V}$ | $g_{Na}m^3h(V-E_{Na})$ |
| $I_{K_V}$ | $g_Kn^4(V-E_K)$ |
| $I_L$ | $g_L(V-E_L)$ |


$ \frac{dm}{dt} = -\frac{m-m_\infty(V)}{\tau_m(V)} $

$ \frac{dh}{dt} = -\frac{h-h_\infty(V)}{\tau_h(V)} $

$ \frac{dn}{dt} = -\frac{n-n_\infty(V)}{\tau_n(V)} $


The description, parameters and equations for the model are taken from:
- "Theoretical Neuroscience", Dayan and Abbot (2005)
- "Cellular Biophysics and Modeling", Smith (2019)
- "Models of the Mind", Lindsay (2022)
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def run_hodgkin_huxley():
    dt = 0.01   # Δ in ms
    T = 50.0    # total time in ms
    time = np.arange(0, T, dt)

    C_m = 1.0      # membrane capacitance, in uF/cm^2
    g_Na = 120.0   # Sodium (Na) max conductance, in mS/cm^2
    g_K = 36.0     # Potassium (K) max conductance, in mS/cm^2
    g_L = 0.3      # Leak max conductance, in mS/cm^2

    E_Na = 50.0    # Sodium equilibrium potential, in mV
    E_K = -77.0    # Potassium equilibrium potential, in mV
    E_L = -54.387  # Leak equilibrium potential, in mV

    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

    I_ext = np.zeros_like(time)
    I_ext[1000:1200] = 10  # stimulus

    V = np.full_like(time, -65.0)  # membrane potential
    m = np.zeros_like(time)
    h = np.zeros_like(time)
    n = np.zeros_like(time)

    # starting steady state
    V[0] = -65
    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

    for t in range(1, len(time)):
        gNa = g_Na * m[t-1] ** 3 * h[t-1]
        gK = g_K * n[t-1] ** 4
        gL = g_L

        # Currents
        INa = gNa * (V[t-1] - E_Na)
        IK = gK * (V[t-1] - E_K)
        IL = gL * (V[t-1] - E_L)

        # update
        dV = dt * (I_ext[t-1] - INa - IK - IL) / C_m
        V[t] = V[t-1] + dV

        # update gating
        m[t] = m[t-1] + dt * (alpha_m(V[t-1]) *
                              (1 - m[t-1]) - beta_m(V[t-1]) * m[t-1])
        h[t] = h[t-1] + dt * (alpha_h(V[t-1]) *
                              (1 - h[t-1]) - beta_h(V[t-1]) * h[t-1])
        n[t] = n[t-1] + dt * (alpha_n(V[t-1]) *
                              (1 - n[t-1]) - beta_n(V[t-1]) * n[t-1])

    plt.figure(figsize=(10, 4))
    plt.plot(time, V, label='Membrane Potential (mV)')
    plt.title('Hodgkin-Huxley Neuron Model')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.grid(True)
    plt.legend()

    # Save plot
    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'imgs', 'playground')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hodgkin_huxley.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
