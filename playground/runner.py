"""
# Neuron Modeling Project

This notebook contains some messing around to get first contact with the modeling 
of a neuron and a somewhat biologically realistic neuronal network. 
As the filename suggests, this is just a kind of playground...
"""

from integrate_and_fire import run_integrate_and_fire
from hodgkin_huxley import run_hodgkin_huxley
from hodgkin_huxley_network import run_hodgkin_huxley_network
from refractory_period import run_refractory_period

print("Running Integrate and Fire Neuron Model...")
run_integrate_and_fire()

print("Running Hodgkin-Huxley Neuron Model...")
run_hodgkin_huxley()

print("Running Hodgkin-Huxley Network Model...")
run_hodgkin_huxley_network()

print("Running Refractory Period Model...")
run_refractory_period()

print("All models have been run successfully!")
