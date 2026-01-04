"""
Test the effect of varying the refractory period on neuron spiking behavior.
"""


import matplotlib.pyplot as plt
import numpy as np
from hodgkin_huxley_network import HodgkinHuxleyNeuron


def run_refractory_period():
    refractory_periods = [1.0, 5.0, 10.0, 20.0]
    colors = ['blue', 'red', 'green', 'orange']

    plt.figure(figsize=(15, 10))

    for i, ref_period in enumerate(refractory_periods):
        neuron = HodgkinHuxleyNeuron(dt=0.01, refractory_period=ref_period)
            
        I_test = np.zeros(5000) # 5000 steps = 50ms
        I_test[500:2500] = 24.0 # Apply current for 20ms
        
        # Simulate
        for j in range(len(I_test)):
            neuron.update(I_ext=I_test[j])
        
        plt.subplot(2, 2, i+1)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # voltages / spikes
        line1 = ax1.plot(neuron.get_voltage_history(), color=colors[i], linewidth=2, label='Voltage')
        ax1.set_ylabel('Voltage (mV)', color=colors[i])
        ax1.tick_params(axis='y', labelcolor=colors[i])
        ax1.grid(True, alpha=0.3)
        
        # applied current
        line2 = ax2.plot(I_test, color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='Current')
        ax2.set_ylabel('Current (nA)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f'Refractory Period = {ref_period}ms\nSpikes: {len(neuron.spike_times)}')
        plt.xlabel('Time (steps: 100 steps = 1ms)')
        
    plt.suptitle('Effect of Refractory Period on Spiking', fontsize=16)
    plt.tight_layout()
    plt.show()
