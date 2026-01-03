"""
Multi-Compartment Neuronal Network Model
==========================================

A somewhat biologically realistic implementation of multi-compartment neurons based on cable theory.
Each neuron consists of soma, axon, and a variable number of dendrites with corresponding morphological and electrical properties.

"""

from .neuron import Neuron
from .network import NeuronalNetwork, Connection
from .compartment import Compartment, Synapse, CompartmentState
from .parameters import (
    CompartmentType,
    CompartmentParameters,
    NeuronParameters,
    SynapticParameters,
    SOMA_PARAMETERS,
    AXON_PARAMETERS,
    DENDRITE_PARAMETERS
)

__all__ = [
    "Neuron",
    "NeuronalNetwork",
    "Connection",
    "Compartment",
    "Synapse",
    "CompartmentState",

    "CompartmentType",
    "CompartmentParameters",
    "NeuronParameters",
    "SynapticParameters",

    "SOMA_PARAMETERS",
    "AXON_PARAMETERS",
    "DENDRITE_PARAMETERS"
]
