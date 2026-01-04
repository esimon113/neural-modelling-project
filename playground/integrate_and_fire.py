"""
Integrate and Fire Neuron Model

An integrate-and-fire neuron is a simplified model that treats a neuron’s 
membrane like a leaky RC circuit. It continuously integrates incoming current. 
When the membrane potential reaches a set threshold $V_{th}$, the model emits a 
spike and resets the potential to a value $V_{reset}$ below threshold. This 
avoids modeling the ionic mechanisms of action potentials and focuses only on 
subthreshold voltage dynamics, greatly reducing computational cost. Variants 
such as the leaky integrate-and-fire model add a passive leak term and can be 
extended to include phenomena like spike-rate adaptation for more realistic 
behavior. (Dayan & Abbot, 2001, p.162 ff.)
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def run_integrate_and_fire():
    # Parameters
    tau_m = 10.0      # Membrane time constant (in ms)
    R_m = 1.0         # Membrane resistance (in MΩ)
    V_rest = -65.0    # Resting potential (in mV)
    V_th = -50.0      # Threshold potential (in mV)
    V_reset = -70.0   # Reset potential after spike (in mV)
    t_ref = 10.0      # Refractory period (in ms)

    dt = 0.1  # Time step (in ms)
    T = 200   # Total time (in ms)
    time = np.arange(0, T, dt)

    I = np.zeros(len(time))
    I[50:150] = 24  # stimulus whihc is applied in specified period

    V = np.full(len(time), V_rest)
    spike_times = []
    refractory_counter = 0

    for t in range(1, len(time)):
        if refractory_counter > 0:
            V[t] = V_reset
            refractory_counter -= dt
        else:
            dV = (dt / tau_m) * (-(V[t-1] - V_rest) + R_m * I[t-1])
            V[t] = V[t-1] + dV

            if V[t] >= V_th:
                V[t] = 40.0  # spike
                spike_times.append(time[t])
                refractory_counter = t_ref

    plt.figure(figsize=(10, 4))
    plt.plot(time, V, label='Membrane Potential')
    plt.axhline(V_th, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Leaky Integrate-and-Fire Neuron')
    plt.legend()
    plt.grid()

    # Save plot
    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'imgs', 'playground')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'integrate_and_fire.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
